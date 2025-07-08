# -*- coding: utf-8 -*-
"""
Created on Sun May 18 00:43:56 2025

@author: User
"""

import tensorflow as tf
import numpy as np
import logging
import os
import sys
from uuid import uuid4

from default_configurations import get_configuration, get_conversion_constants, WORKING_DIRECTORY, DEFAULT_GENERAL_CONFIG, DEFAULT_RESERVOIR_CONFIG, DEFAULT_WELLS_CONFIG, DEFAULT_SCAL_CONFIG
from complete_pvt_module import PVTModuleWithHardLayer
from relative_permeability import RelativePermeability
from welldata_processor import WellDataProcessor

project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_directory not in sys.path:
    sys.path.append(project_directory)
from data_processing import DataSummary, slice_tensor, SRMDataProcessor

logging.basicConfig(level=logging.INFO)

class WellRatesPressure:
    """
    A class to compute well rates, bottom-hole pressure, and blocking integrals for reservoir simulation.
    Uses pure TensorFlow graph functions for performance and integrates PVT and relative permeability models.
    All initialization parameters are passed via __init__ for general usability.
    """
    def __init__(
        self,
        fluid_type=None,
        use_blocking_factor=False,
        dtype=tf.float32,
        solver='newton',
        n_intervals=8,
        n_root_iter=20,
        max_iters=10,
        tol=1e-6,
        compute_mo=False,
        use_non_iterative=True,
        reservoir_config=None,
        general_config=None,
        wells_config=None,
        pvt_layer_config=None,
        scal_config=None,
        working_directory=None,
    ):
        """
        Initialize the WellRatesPressure class with all necessary parameters for reservoir, well, and computational settings.

        Args:
            fluid_type (str): Fluid type for the reservoir. Options: 'DG' (Dry Gas), 'GC' (Gas Condensate). Default: 'DG'.
            use_blocking_factor (bool): If True, use blocking factor in the well rate calculation. Default: False.
            dtype (tf.DType): TensorFlow data type for computations (e.g., tf.float32, tf.float64). Default: tf.float32.
            solver (str): Root-finding method for solving nonlinear equations. Options: 'newton', 'chandrupatla'. Default: 'newton'.
            n_intervals (int): Number of pressure intervals for numerical integration in blocking factor calculations. Default: 8.
            n_root_iter (int): Maximum iterations for root-finding algorithms (Newton or Chandrupatla). Default: 20.
            max_iters (int): Maximum iterations for bottom-hole pressure (BHP) convergence loop. Default: 10.
            tol (float): Tolerance for BHP convergence (difference between computed and target rates). Default: 1e-6.
            compute_mo (bool): Option to compute oil mobility. Default: False.
            use_non_iterative (bool): If True, use non-iterative method for BHP calculation; else, use iterative method. Default: False.
            reservoir_config (dict, optional): Reservoir configuration. Defaults to DEFAULT_RESERVOIR_CONFIG if None.
            general_config (dict, optional): General configuration. Defaults to DEFAULT_GENERAL_CONFIG if None.
            wells_config (dict, optional): Wells configuration. Defaults to DEFAULT_WELLS_CONFIG if None.
            pvt_layer_config (dict, optional): Configuration for PVT layer (e.g., spline fitting parameters). If None, derived based on fluid_type.
            scal_config (dict, optional): SCAL configuration for relative permeability. Defaults to DEFAULT_SCAL_CONFIG if None.
            working_directory (str, optional): Working directory for SRM data processing. Defaults to WORKING_DIRECTORY if None.
        """
        # Store computational parameters
        self.use_blocking_factor = use_blocking_factor
        self.dtype = dtype
        self.solver = solver
        self.n_intervals = n_intervals
        self.n_root_iter = n_root_iter
        self.max_iters = max_iters
        self.tol = tol
        self.use_non_iterative = use_non_iterative
        self.compute_mo = compute_mo

        # Use default configurations if None provided
        self.reservoir_config = reservoir_config if reservoir_config is not None else DEFAULT_RESERVOIR_CONFIG
        self.general_config = general_config if general_config is not None else DEFAULT_GENERAL_CONFIG
        self.wells_config = wells_config if wells_config is not None else DEFAULT_WELLS_CONFIG
        self.scal_config = scal_config if scal_config is not None else DEFAULT_SCAL_CONFIG
        self.working_directory = working_directory if working_directory is not None else WORKING_DIRECTORY

        # Store general configurations
        self.units = get_conversion_constants(self.general_config['srm_units'])  # Conversion constants for unit system
        self.C = self.units['C']  # Well productivity constant
        self.D = self.units['D']  # Another unit-specific constant
        self.unit_target_shape = self.general_config['unit_target_shape']  # Grid shape
        self.initial_wells = self.wells_config['connections']  # Well configurations

        # Store reservoir properties as TensorFlow constants for graph compatibility
        self.phi = tf.constant(self.reservoir_config['porosity'], dtype=self.dtype)  # Porosity
        self.kx_ky = tf.constant(self.reservoir_config['horizontal_anisotropy'], dtype=self.dtype)  # ky/kx ratio
        self.kv_kh = tf.constant(self.reservoir_config['vertical_anisotropy'], dtype=self.dtype)  # kz/kx ratio
        self.reservoir_depth = tf.constant(self.reservoir_config['depth'], dtype=self.dtype)  # z-dimension
        self.reservoir_length = tf.constant(self.reservoir_config['length'], dtype=self.dtype)  # x-dimension
        self.reservoir_width = tf.constant(self.reservoir_config['width'], dtype=self.dtype)  # y-dimension
        self.reservoir_thickness = tf.constant(self.reservoir_config['thickness'], dtype=self.dtype)  # z-dimension (alternative to depth)
        self.Nx = self.reservoir_config['Nx']  # Grid cells in x
        self.Ny = self.reservoir_config['Ny']  # Grid cells in y
        self.Nz = self.reservoir_config['Nz']  # Grid cells in z
        self.Pi = tf.constant(self.reservoir_config['initialization']['Pi'], shape=self.unit_target_shape, dtype=self.dtype)  # Initial pressure field

        # Compute grid cell sizes
        # dx = length / Nx, dy = width / Ny, dz = depth / Nz (or thickness / Nz, depending on context)
        self.dx = self.reservoir_length / self.Nx
        self.dy = self.reservoir_width / self.Ny
        self.dz = self.reservoir_thickness / self.Nz

        # Initialize well data processor
        logging.info("Initializing WellDataProcessor with provided well configurations")
        self.well_data_processor = WellDataProcessor(self.initial_wells, dtype=self.dtype)
        self.well_data = self.well_data_processor.get_well_data()

        # Scatter well properties onto the computational grid
        # well_id: Binary mask indicating well locations
        # rw: Wellbore radius at each grid cell
        # q0: Target well rate (control mode value)
        # pwf_min: Minimum bottom-hole pressure
        # completion_ratio: Fraction of grid cell open to flow
        self.well_id = self.well_data_processor.scatter_y(self.unit_target_shape, self.well_data['connection_index'], 1.0)
        self.rw = self.well_data_processor.scatter_y(self.unit_target_shape, self.well_data['connection_index'], self.well_data['wellbore_radius'])
        self.q0 = self.well_data_processor.scatter_y(self.unit_target_shape, self.well_data['connection_index'], self.well_data['control_mode_value'])
        self.pwf_min = self.well_data_processor.scatter_y(self.unit_target_shape, self.well_data['connection_index'], self.well_data['minimum_bhp'])
        self.completion_ratio = self.well_data_processor.scatter_y(self.unit_target_shape, self.well_data['connection_index'], self.well_data['completion_ratio'])

        # Initialize SRM data processor for permeability statistics
        self.srm_processor = SRMDataProcessor(base_dir=self.working_directory)
        try:
            train_config_hash = self.srm_processor._generate_full_config_hash()[1]
            statistics = self.srm_processor.load_training_statistics(train_config_hash)
            self.data_summary = DataSummary([statistics], dtype=self.dtype)
        except FileNotFoundError as e:
            logging.warning(f"Could not load statistics: {e}")
            self.data_summary = None

        # Get normalization configuration from DEFAULT_GENERAL_CONFIG
        self.norm_config = self.general_config['data_normalization']
        
        # Build PVT model
        # If pvt_layer_config is not provided, derive it based on fluid_type
        if fluid_type is None:
            fluid_type = self.general_config['fluid_type']
        self.fluid_type = fluid_type
        if pvt_layer_config is None:
            pvt_layer_config = get_configuration('pvt_layer', fluid_type=fluid_type, fitting_method='spline')
        self.pvt_model = self._build_pvt_model_without_hard(
            input_shape=(None, *self.unit_target_shape[1:]),
            name="well_rate_bhp_pvt_model",
            fluid_type=fluid_type,
            pvt_layer_config=pvt_layer_config
        )

        # Initialize relative permeability model with provided SCAL parameters
        self.relperm = RelativePermeability(
            end_points=self.scal_config['end_points'],
            corey_exponents=self.scal_config['corey_exponents'],
            dtype=self.dtype
        )


    def _build_pvt_model_without_hard(self, input_shape, name, fluid_type, pvt_layer_config):
        """
        Build a PVT model without hard layer, replicating network_architecture_case.py structure.
        
        Args:
            input_shape (tuple): Shape for the input layer (e.g., (None, x, y, z)).
            name (str): Model name.
            fluid_type (str): Fluid type ('DG' or 'GC').
            pvt_layer_config (dict): Configuration for PVT layer (e.g., spline fitting parameters).
        
        Returns:
            tf.keras.Model: PVT model for computing fluid properties.
        """
        logging.info(f"\n{'='*50}\n=== Building Standalone PVT Model WITHOUT Hard Layer ({name}) ===\n{'='*50}")
        logging.info("Configuring PVT Layer with provided configuration")
        
        pvt_module = PVTModuleWithHardLayer(use_hard_layer=False, pvt_layer_config=pvt_layer_config)
        
        inputs = tf.keras.layers.Input(shape=input_shape[1:])
        outputs = pvt_module(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        
        model.summary()
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        logging.info(f"Total trainable parameters: {trainable_params}")
        
        return model

    @tf.function
    def extract_pvt_properties(self, pvt_tensor, enable_logging=True):
        """
        Extract PVT properties based on fluid type.
        
        Args:
            pvt_tensor: Tensor of shape [2, n_properties, ...], where index 0 is values, 1 is derivatives.
            enable_logging: Bool, whether to log property shapes.
        
        Returns:
            Tuple: (invBg, invBo, invug, invuo, Rs, Rv) containing inverse formation volume factors,
                   inverse viscosities, and solution gas-oil/gas ratios.
        """
        zero_shape = tf.shape(pvt_tensor)[2:]
        zero_tensor = tf.zeros(zero_shape, dtype=self.dtype)
        
        if self.fluid_type == 'DG':
            invBg = pvt_tensor[0, 0]
            invug = pvt_tensor[0, 1]
            if enable_logging:
                logging.info(f"DG: invBg shape: {invBg.shape}, invug shape: {invug.shape}")
            return invBg, zero_tensor, invug, zero_tensor, zero_tensor, zero_tensor
        elif self.fluid_type == 'GC':
            invBg = pvt_tensor[0, 0]
            invBo = pvt_tensor[0, 1]
            invug = pvt_tensor[0, 2]
            invuo = pvt_tensor[0, 3]
            Rs = pvt_tensor[0, 4]
            Rv = pvt_tensor[0, 5]
            if enable_logging:
                logging.info(f"GC: invBg shape: {invBg.shape}, invBo shape: {invBo.shape}")
                logging.info(f"GC: invug shape: {invug.shape}, invuo shape: {invuo.shape}")
                logging.info(f"GC: Rs shape: {Rs.shape}, Rv shape: {Rv.shape}")
            return invBg, invBo, invug, invuo, Rs, Rv
        else:
            logging.warning(f"Unknown fluid type: {self.fluid_type}. Defaulting to DG.")
            return self.extract_pvt_properties(pvt_tensor, enable_logging=enable_logging)

    @tf.function
    def _solve_newton(self, cost, ref, max_iters=20, max_value=1.):
        """
        Newton-Raphson method to find root of cost(Sg)=0.
        
        Args:
            cost: Function that computes cost(Sg).
            ref: Tensor to match output shape.
            max_iters: Maximum iterations.
            max_value: Maximum value for clipping (e.g., maximum gas saturation).
        
        Returns:
            Tensor: Root of the cost function (gas saturation).
        """
        zero = tf.constant(0, dtype=tf.int32)
        Sg = tf.fill(tf.shape(ref), 0.1)

        def cond(it, Sg):
            return tf.less(it, max_iters)

        def body(it, Sg):
            with tf.GradientTape() as g:
                g.watch(Sg)
                f = cost(Sg)
            df = g.gradient(f, Sg)
            Sg_new = Sg - f / (df + 1e-12)
            Sg_new = tf.clip_by_value(Sg_new, 0.0, max_value)
            return it + 1, Sg_new

        _, Sg = tf.while_loop(
            cond, body,
            [zero, Sg],
            shape_invariants=[zero.get_shape(), ref.get_shape()]
        )
        return Sg

    @tf.function
    def _solve_chandrupatla(self, cost, ref, max_iters=20, tol=1e-6, max_value=1.):
        """
        Chandrupatla's method to find root of cost(Sg)=0 in [0, max_value].
        
        Args:
            cost: Function that computes cost(Sg).
            ref: Tensor to match output shape.
            max_iters: Maximum iterations.
            tol: Tolerance for convergence.
            max_value: Maximum value for clipping (e.g., maximum gas saturation).
        
        Returns:
            Tensor: Root of the cost function (gas saturation).
        """
        zero = tf.constant(0, dtype=tf.int32)
        lo = tf.zeros_like(ref)
        hi = tf.ones_like(ref) * max_value
        f_lo = cost(lo)
        f_hi = cost(hi)
        # Ensure bracket: if f_lo * f_hi > 0, nudge hi
        bad = tf.greater(f_lo * f_hi, 0.0)
        hi = tf.where(bad, lo + 1e-3, hi)
        f_hi = tf.where(bad, cost(hi), f_hi)

        def cond(lo, hi, f_lo, f_hi, it):
            return tf.logical_and(
                tf.reduce_any(hi - lo > tol),
                tf.less(it, max_iters)
            )

        def body(lo, hi, f_lo, f_hi, it):
            d = (f_hi - f_lo) / (hi - lo + 1e-12)
            guess = hi - f_hi / d
            f_guess = cost(guess)
            replace_lo = tf.less(f_lo * f_guess, 0.0)
            new_lo = tf.where(replace_lo, lo, guess)
            new_f_lo = tf.where(replace_lo, f_lo, f_guess)
            new_hi = tf.where(replace_lo, guess, hi)
            new_f_hi = tf.where(replace_lo, f_guess, f_hi)
            return new_lo, new_hi, new_f_lo, new_f_hi, it + 1

        lo, hi, _, _, _ = tf.while_loop(
            cond, body,
            [lo, hi, f_lo, f_hi, zero],
            shape_invariants=[
                ref.get_shape(),
                ref.get_shape(),
                ref.get_shape(),
                ref.get_shape(),
                zero.get_shape()
            ]
        )
        return 0.5 * (lo + hi)
    
    @tf.function
    def log_tensor_to_file(self, tensor_history_stacked, it_final, final_tensor, tensor_name, file_prefix, values_per_line=10, well_specific=False, method_dir=None):
        """
        Logs tensor values from a stacked TensorArray to a file in the method's directory.
        Designed for use by iterative and non-iterative methods to log tensors like pwf.
    
        Args:
            tensor_history_stacked: Stacked TensorArray containing tensor values for each iteration.
            it_final: Number of valid iterations (scalar tensor).
            final_tensor: Final tensor value (e.g., pwf_final).
            tensor_name: Name of the tensor for log formatting (e.g., 'pwf').
            file_prefix: Prefix for the log file name (e.g., 'pwf_log').
            values_per_line: Number of tensor values to include per line in the log file (default: 10).
            well_specific: If True, logs only values at well locations using connection_index (default: False).
    
        Returns:
            None
        """
        if method_dir is None:
            method_dir = project_directory
        # Get the full shape of tensor_history_stacked
        full_shape = tf.shape(tensor_history_stacked)
        
        # Initialize log content
        log_strings = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True)
        log_index = tf.constant(0, dtype=tf.int32)
        
        # Handle well-specific indexing
        if well_specific:
            well_indices = self.well_data['connection_index']
            num_wells = tf.shape(well_indices)[0]
            if num_wells > 0:
                batch_size = full_shape[1]
                batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int32), num_wells)
                well_indices_tiled = tf.tile(well_indices, [batch_size, 1])
                channel_indices = tf.zeros([batch_size * num_wells, 1], dtype=tf.int32)
                indices = tf.concat([tf.expand_dims(batch_indices, -1), well_indices_tiled, channel_indices], axis=1)
                final_tensor = tf.gather_nd(final_tensor, indices)
                if it_final > 0:
                    slice_size = tf.concat([tf.expand_dims(it_final, 0), full_shape[1:]], axis=0)
                    tensor_history_valid = tf.slice(tensor_history_stacked, tf.zeros_like(full_shape), slice_size)
                    tensor_history_valid = tf.gather_nd(tensor_history_valid, indices)
                else:
                    tensor_history_valid = tf.zeros([0], dtype=self.dtype)
            else:
                warning_string = tf.constant("Warning: No well connections found for ") + tensor_name + tf.constant(".\n")
                log_strings = log_strings.write(log_index, warning_string)
                log_index += 1
                # Initialize tensor_history_valid as in well_specific=False case
                if it_final > 0:
                    slice_size = tf.concat([tf.expand_dims(it_final, 0), full_shape[1:]], axis=0)
                    tensor_history_valid = tf.slice(tensor_history_stacked, tf.zeros_like(full_shape), slice_size)
                else:
                    tensor_history_valid = tf.zeros([0], dtype=self.dtype)
                # Keep original final_tensor since no well-specific indexing
                well_specific = False
        else:
            if it_final > 0:
                slice_size = tf.concat([tf.expand_dims(it_final, 0), full_shape[1:]], axis=0)
                tensor_history_valid = tf.slice(tensor_history_stacked, tf.zeros_like(full_shape), slice_size)
            else:
                tensor_history_valid = tf.zeros([0], dtype=self.dtype)
        
        # Flatten tensors for logging
        if it_final > 0:
            tensor_history_flat = tf.reshape(tensor_history_valid, [it_final, -1])
        else:
            tensor_history_flat = tf.zeros([0, 0], dtype=self.dtype)
        final_tensor_flat = tf.reshape(final_tensor, [-1])
        
        # Format tensor values for each iteration
        values_per_line = tf.constant(values_per_line, dtype=tf.int32)
        
        def format_iteration(i, log_strings, log_index):
            values = tensor_history_flat[i]
            num_values = tf.shape(values)[0]
            num_lines = tf.cast(tf.math.ceil(tf.cast(num_values, tf.float32) / tf.cast(values_per_line, tf.float32)), tf.int32)
            
            def format_line(j):
                start_idx = j * values_per_line
                end_idx = tf.minimum(start_idx + values_per_line, num_values)
                line_values = values[start_idx:end_idx]
                # Format all values in the line vectorized
                formatted_values = tf.strings.as_string(line_values, precision=6)
                # Join with commas
                joined_values = tf.strings.reduce_join(formatted_values, separator=", ")
                return tf.strings.format("  Values[{}:{}]: {}\n", (start_idx, end_idx, joined_values))
            
            line_indices = tf.range(num_lines)
            line_strings = tf.map_fn(format_line, line_indices, dtype=tf.string)
            iteration_string = tf.strings.join([
                tf.strings.format("Iteration {}: {} =\n", (i, tensor_name)),
                tf.strings.reduce_join(line_strings, separator="")
            ])
            log_strings = log_strings.write(log_index, iteration_string)
            return i + 1, log_strings, log_index + 1
        
        # Process iterations only if it_final > 0
        if it_final > 0:
            _, log_strings, log_index = tf.while_loop(
                lambda i, ls, idx: tf.less(i, it_final),
                format_iteration,
                [tf.constant(0, dtype=tf.int32), log_strings, log_index],
                shape_invariants=[tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape([])]
            )
        
        # Format final tensor values
        num_values_final = tf.shape(final_tensor_flat)[0]
        num_lines_final = tf.cast(tf.math.ceil(tf.cast(num_values_final, tf.float32) / tf.cast(values_per_line, tf.float32)), tf.int32)
        
        def format_final_line(j):
            start_idx = j * values_per_line
            end_idx = tf.minimum(start_idx + values_per_line, num_values_final)
            line_values = final_tensor_flat[start_idx:end_idx]
            # Format all values in the line vectorized
            formatted_values = tf.strings.as_string(line_values, precision=6)
            # Join with commas
            joined_values = tf.strings.reduce_join(formatted_values, separator=", ")
            return tf.strings.format("  Values[{}:{}]: {}\n", (start_idx, end_idx, joined_values))
        
        final_line_strings = tf.map_fn(format_final_line, tf.range(num_lines_final), dtype=tf.string)
        final_log = tf.strings.join([
            tf.strings.format("Final: iterations = {}, {} =\n", (it_final, tensor_name)),
            tf.strings.reduce_join(final_line_strings, separator="")
        ])
        
        # Add final log
        log_strings = log_strings.write(log_index, final_log)
        log_index += 1
        
        # Join all log content
        log_content = tf.strings.reduce_join(log_strings.stack(), separator="")
        
        # Generate unique file name using UUID
        import os
        from uuid import uuid4
        log_file = os.path.join(method_dir, f"{file_prefix}_{uuid4()}.txt")
        tf.io.write_file(log_file, log_content)
            
    @tf.function
    def _solve_secant_for_scaling(self, cost, ref, x0=0.5, x1=1.0, max_iters=10, tol=1e-6):
        """
        Secant method to find the root of cost(x)=0, where x is the scaling factor.
        
        Args:
            cost: Function that computes the difference between computed and target rate.
            ref: Tensor to match the shape of the output.
            x0, x1: Initial guesses for the scaling factor (e.g., 0.5 and 1.0).
            max_iters: Maximum iterations for convergence.
            tol: Tolerance for convergence.
        
        Returns:
            Scaling factor tensor.
        """
        zero = tf.constant(0, dtype=tf.int32)
        x0 = tf.ones_like(ref) * x0
        x1 = tf.ones_like(ref) * x1
        f0 = cost(x0)
        f1 = cost(x1)

        def cond(x0, x1, f0, f1, it):
            return tf.logical_and(
                tf.reduce_any(tf.abs(f1) > tol),
                tf.less(it, max_iters)
            )

        def body(x0, x1, f0, f1, it):
            # Secant method update: x_new = x1 - f1 * (x1 - x0) / (f1 - f0 + eps)
            eps = 1e-12
            denom = f1 - f0 + eps
            x_new = x1 - f1 * (x1 - x0) / denom
            x_new = tf.clip_by_value(x_new, 0.0, 1.0)  # Ensure scaling factor in [0, 1]
            f_new = cost(x_new)
            return x1, x_new, f1, f_new, it + 1

        _, x_final, _, _, _ = tf.while_loop(
            cond, body,
            [x0, x1, f0, f1, zero],
            shape_invariants=[
                ref.get_shape(),
                ref.get_shape(),
                ref.get_shape(),
                ref.get_shape(),
                zero.get_shape()
            ]
        )
        return x_final

    @tf.function
    def _iterative_method(self, p_n1, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1, model_PVT, relperm_model, Ck, q_target, min_bhp, Sg_max):
        """
        Iterative method to compute bottomhole pressure (BHP) using Newton-Raphson to match the target gas rate.
        Writes pwf values and diagnostics for each iteration to a file in the method's directory.
    
        Args:
            p_n1: Gridblock pressure.
            Sg_n1: Gas saturation.
            mg_n1, mo_n1: Gas and oil mobilities.
            invBg_n1, invBo_n1: Inverse formation volume factors for gas and oil.
            invug_n1, invuo_n1: Inverse viscosities for gas and oil.
            Rs_n1, Rv_n1: Solution gas-oil and oil-gas ratios.
            model_PVT: TensorFlow model for PVT properties.
            relperm_model: Function to compute relative permeabilities.
            Ck: Well productivity constant.
            q_target: Target gas rate.
            min_bhp: Minimum allowable bottomhole pressure.
            Sg_max: Maximum gas saturation.
    
        Returns:
            Computed bottomhole pressure (pwf).
        """
        # Initial guess for BHP
        pwf = min_bhp + 0.5 * (p_n1 - min_bhp)
        it = tf.constant(0, dtype=tf.int32)
        eps = tf.constant(14.7, dtype=self.dtype)  # Perturbation for derivative approximation
        tol = tf.constant(self.tol, dtype=self.dtype)
        max_iters = tf.constant(self.max_iters, dtype=tf.int32)
        
        # Initialize TensorArray to collect pwf values and diagnostics
        pwf_history = tf.TensorArray(dtype=self.dtype, size=max_iters, dynamic_size=False, clear_after_read=False)
        # error_history = tf.TensorArray(dtype=self.dtype, size=max_iters, dynamic_size=False, clear_after_read=False)
        # dqg_dpwf_history = tf.TensorArray(dtype=self.dtype, size=max_iters, dynamic_size=False, clear_after_read=False)
    
        def cond(pwf, it):
            qg, _ = self._compute_phase_rates(
                p_n1, pwf, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
                model_PVT, relperm_model, Ck, q_target, self.well_id, Sg_max
            )
            error = tf.abs(qg - q_target)
            continue_loop = tf.logical_and(
                it < max_iters,
                tf.reduce_any(error > tol)
            )
            # Log condition diagnostics
            log_string = tf.strings.format(
                "Condition check at iteration {}: |qg - q_target| = {}, tol = {}, continue = {}\n",
                (it, tf.reduce_max(error), tol, continue_loop)
            )
            return continue_loop
    
        def body(pwf, it):
            # Compute gas rate at current BHP
            qg, _ = self._compute_phase_rates(
                p_n1, pwf, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
                model_PVT, relperm_model, Ck, q_target, self.well_id, Sg_max
            )
            # Compute gas rate at perturbed BHP
            qg_plus, _ = self._compute_phase_rates(
                p_n1, pwf + eps, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
                model_PVT, relperm_model, Ck, q_target, self.well_id, Sg_max
            )
            # Approximate derivative: dqg/dpwf
            dqg_dpwf = (qg_plus - qg) / eps
            # Store pwf in TensorArray
            pwf_history_it = pwf_history.write(it, pwf)
            # error = tf.abs(qg - q_target)
            # error_history_it = error_history.write(it, error)
            # dqg_dpwf_history_it = dqg_dpwf_history.write(it, dqg_dpwf)
            # Newton-Raphson update with small constant to avoid division by zero
            pwf_new = pwf - (qg - q_target) / (dqg_dpwf + 1e-12)
            # Ensure BHP stays within physical bounds
            pwf_new = tf.clip_by_value(pwf_new, min_bhp, p_n1)
            return pwf_new, it + 1
    
        # Run the iteration loop
        pwf_final, it_final = tf.while_loop(
            cond, body,
            [pwf, it],
            shape_invariants=[p_n1.get_shape(), it.get_shape()]
        )
        # Write diagnostics and pwf values to a file
        pwf_history_stacked = pwf_history.stack()
        # error_history_stacked = error_history.stack()
        # dqg_dpwf_history_stacked = dqg_dpwf_history.stack()
        self.log_tensor_to_file(
            pwf_history_stacked, it_final, pwf_final,
            tensor_name="pwf", file_prefix="pwf_log", values_per_line=10, well_specific=True
        )
        # self.log_tensor_to_file(error_history_stacked, it_final, tf.abs(self._compute_phase_rates(
        #     p_n1, pwf_final, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1,
        #     Rs_n1, Rv_n1, model_PVT, relperm_model, Ck, q_target, self.well_id, Sg_max
        # )[0] - q_target), tensor_name="error", file_prefix="error_log")
        # self.log_tensor_to_file(dqg_dpwf_history_stacked, it_final, tf.zeros_like(pwf_final),
        #                        tensor_name="dqg_dpwf", file_prefix="dqg_dpwf_log")
        
        return pwf_final
    
    @tf.function
    def _non_iterative_method(self, p_n1, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1, model_PVT, relperm_model, Ck, q_target, min_bhp, Sg_max, krog_n1):
        """
        Non-iterative method to compute BHP using a secant method for scaling.
        
        Args:
            p_n1: Gridblock pressure.
            Sg_n1: Gas saturation.
            mg_n1: Gas mobility.
            mo_n1: Oil mobility.
            invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1: PVT properties.
            model_PVT: PVT model function.
            relperm_model: Relative permeability function.
            Ck: Well productivity constant.
            q_target: Target gas rate.
            min_bhp: Minimum BHP.
            Sg_max: Maximum gas saturation.
            krog_n1: Relative permeability to oil.
        
        Returns:
            Tensor: Computed BHP (pwf).
        """
        # Set compute_mo based on fluid_type
        compute_mo = self.fluid_type == 'GC'

        # For dry gas, set oil mobility to zero
        if self.fluid_type == 'DG':
            mo_n1 = tf.zeros_like(mo_n1)

        # Compute maximum blocking integral at min_bhp
        Ig_max, Io_max = self.compute_blocking_integral_and_factor(
            p_n1=p_n1,
            Sg_n1=Sg_n1,
            relperm_model=relperm_model,
            model_PVT=model_PVT,
            pwf_n1=min_bhp
        )[:2]
        dp_max = p_n1 - min_bhp + 1e-12

        # Compute maximum blocking factors
        if self.use_blocking_factor:
            blk_fac_g_max = tf.math.divide_no_nan(Ig_max, mg_n1 * dp_max)
            div_result = tf.math.divide_no_nan(Io_max, mo_n1 * dp_max)
        else:
            blk_fac_g_max = Ig_max
            div_result = Io_max
        blk_fac_o_max = tf.where(compute_mo, div_result, tf.ones_like(div_result))

        # Compute maximum rates
        qg_max = self.well_id * Ck * blk_fac_g_max * mg_n1 * dp_max
        qo_max = self.well_id * Ck * blk_fac_o_max * mo_n1 * dp_max

        # (TODO) Determine optimum rates - avoid minimum clipping during injection
        qg_opt = tf.maximum(tf.minimum(q_target, qg_max), 0.) #tf.minimum(q_target, qg_max) 
        qo_opt = tf.zeros_like(qg_opt)
        if self.fluid_type == 'GC':
            qo_target = qg_opt * (1.0 / (Rv_n1 + 1e-12))
            qo_opt = tf.maximum(tf.minimum(qo_target, qo_max), 0.) #tf.minimum(qo_target, qo_max)

        # Define cost function for scaling factor (lambda in [0, 1])
        # def cost(lambda_):
        #     # Scale the pressure difference
        #     dp_scaled = lambda_ * dp_max
        #     Ig_scaled = lambda_ * Ig_max
        #     Io_scaled = lambda_ * Io_max
        #     blk_fac_g = tf.math.divide_no_nan(Ig_scaled, mg_n1 * dp_scaled + 1e-12)
        #     blk_fac_o = tf.math.divide_no_nan(Io_scaled, mo_n1 * dp_scaled + 1e-12)
        #     qg_scaled = self.well_id * Ck * blk_fac_g * mg_n1 * dp_scaled
        #     qo_scaled = self.well_id * Ck * blk_fac_o * mo_n1 * dp_scaled
        #     # Cost is the difference from optimum rates
        #     return qg_scaled - qg_opt  # Focus on gas rate for consistency

        # # Solve for scaling factor using secant method
        # lambda_opt = self._solve_secant_for_scaling(
        #     cost=cost,
        #     ref=p_n1,
        #     x0=0.5,
        #     x1=1.0,
        #     max_iters=self.max_iters,
        #     tol=self.tol
        # )
        
        # # Initialize TensorArray to collect pwf values and diagnostics
        lambda_opt_history = tf.TensorArray(dtype=self.dtype, size=1, dynamic_size=False, clear_after_read=False)
        
        # # Directly solve for lambda_opt using the gas rate 
        lambda_opt = tf.clip_by_value(tf.math.divide_no_nan(qg_opt, self.well_id * Ck * blk_fac_g_max * mg_n1), 0, blk_fac_g_max)
        
        # # Write to history file
        lambda_opt_history_it = lambda_opt_history.write(0, lambda_opt)
        
        # # Write diagnostics and lambda_opt values to a file
        lambda_opt_history_stacked = lambda_opt_history.stack()
               
        # Debug shapes
        # tf.print("lambda_opt shape:", tf.shape(lambda_opt))
        # tf.print("lambda_opt_history_stacked shape:", tf.shape(lambda_opt_history_stacked))
        
        # Write diagnostics and lambda_opt values to a file
        self.log_tensor_to_file(
            lambda_opt_history_stacked,
            it_final=0,  # Non-iterative, no iterations
            final_tensor=lambda_opt,
            tensor_name="lambda_opt",
            file_prefix="lambda_opt_log",
            well_specific=True  # Assuming well-specific logging for consistency
        )        
        # Compute final bottomhole pressure
        dp_opt = lambda_opt * dp_max
        pwf_final = p_n1 - dp_opt
        pwf_final = self.well_id * tf.clip_by_value(pwf_final, min_bhp, p_n1)
        return pwf_final

    # @tf.function - disabled to prevent graph topological sort error as sub methods are already hooked up to their respective static graph nodes.
    def compute_rates_and_bhp(self, x_n1, p_n1, Sg_n1, relperm_model, model_PVT, q_target=None, shutin_days=None):
        """
        Compute well rates and bottom-hole pressure (BHP) using either iterative or non-iterative method.
        
        Args:
            x_n1: Tensor of input features [batch,...,Nf], Nf is the number of input features = (..., t, kx)
            p_n1: Tensor, gridblock pressure (in units consistent with srm_units).
            Sg_n1: Tensor, gas saturation (fraction, 0 to 1).
            relperm_model: Function, computes relative permeabilities (krog, krgo).
            model_PVT: Function, PVT model returning properties tensor.
            q_target: Tensor, target gas rate. If None, defaults to self.q0.
            shutin_days: Three-level nested list [[[c1_t1,c1_t2],[c1_t5,c1_t6]],[[c2_t1,c2_t2],[c2_t5,c2_t6]]] 
                        indicating the connections-times where batch shutins occurred.
                        if None, default shutins are obtained from the initialization and used to compute the connection-time shutin identity.
        Returns:
            For DG: (qg, pwf) - gas rate and BHP.
            For GC: ((qgg, qgo, qoo, qog), pwf) - gas and oil component rates and BHP.
        """
        
        norm_t_n1 = slice_tensor(x_n1, [self.data_summary.get_key_index('time')])

        stats_indices_map_t = tf.constant([[0,],[self.data_summary.get_key_index('time')]], dtype=tf.int32)  # norm dimension indices - normalization statistics indices (perm, time, etc.)
        t_n1 = self.data_summary.nonormalize(
                    norm_t_n1,
                    norm_config=self.norm_config,
                    statistics_index=stats_indices_map_t,
                    compute=True,
                    nonormalization_dimension=-1,  # Normalize along the last dimension
                    dtype=self.dtype
                )
        Sg_n1 = Sg_n1 if Sg_n1 is not None else 1-self.scal_config['end_points']['Swmin']

        # Set default q_target to self.q0 if q_target is None
        if q_target is None:
            # no tensor was passed → use our default
            q_target = self.q0
            
        if shutin_days is None:
            # Compute the shutins identity - allow dynamic shutins at a later date
            shutins_id_n1 = self.well_data_processor.conn_shutins_idx(t_n1, self.well_data['connection_index'], self.well_data['shutin_days'], time_axis=0)
        else:
            shutins_id_n1 = self.well_data_processor.conn_shutins_idx(t_n1, self.well_data['connection_index'], shutin_days, time_axis=0)
        
        # Compute other dynamic variables and well factor (Peaceman formula)
        norm_kx_n1 = slice_tensor(x_n1, [self.data_summary.get_key_index('permx')])
        stats_indices_map_kx = tf.constant([[0,],[self.data_summary.get_key_index('permx')]], dtype=tf.int32)  # norm dimension indices - normalization statistics indices (perm, time, etc.)
        kx_n1 = self.data_summary.nonormalize(
                    norm_kx_n1,
                    norm_config=self.norm_config,
                    statistics_index=stats_indices_map_kx,
                    compute=True,
                    nonormalization_dimension=-1,  # Normalize along the last dimension
                    dtype=self.dtype
                )

        ky_n1 = self.kx_ky * kx_n1
        ro = 0.28 * tf.math.pow(
            (tf.math.pow(ky_n1 / kx_n1, 0.5) * tf.math.pow(self.dx, 2) +
             tf.math.pow(kx_n1 / ky_n1, 0.5) * tf.math.pow(self.dy, 2)), 0.5
        ) / (tf.math.pow(ky_n1 / kx_n1, 0.25) + tf.math.pow(kx_n1 / ky_n1, 0.25))

        Ck = tf.cast(shutins_id_n1, x_n1.dtype) * (2 * np.pi * self.completion_ratio * kx_n1 * self.dz * self.C) / tf.math.log(ro / self.rw)
       
        # Compute relative permeabilities
        krog_n1, krgo_n1 = relperm_model(Sg_n1)
        
        # Compute PVT properties
        pvt_tensor = model_PVT(p_n1)
        invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1 = self.extract_pvt_properties(pvt_tensor, enable_logging=False)

        # Compute mobilities
        if self.fluid_type == 'DG':
            mg_n1 = krgo_n1 * invBg_n1 * invug_n1
            mo_n1 = tf.zeros_like(mg_n1)
        else:
            mgg_n1 = krgo_n1 * invBg_n1 * invug_n1
            mgo_n1 = krog_n1 * invBo_n1 * invuo_n1 * Rs_n1
            moo_n1 = krog_n1 * invBo_n1 * invuo_n1
            mog_n1 = krgo_n1 * invBg_n1 * invug_n1 * Rv_n1
            mg_n1 = mgg_n1 + mgo_n1
            mo_n1 = moo_n1 + mog_n1
        
        # Compute BHP using iterative or non-iterative method
        min_bhp = self.pwf_min 
        Sg_max = 1.0 - self.relperm.end_points['Swmin']  # Use initialized end_points
        
        pwf_final = tf.cond(
            tf.cast(self.use_non_iterative, tf.bool),
            lambda: self._non_iterative_method(
                p_n1, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
                model_PVT, relperm_model, Ck, q_target, min_bhp, Sg_max, krog_n1
            ),
            lambda: self._iterative_method(
                p_n1, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
                model_PVT, relperm_model, Ck, q_target, min_bhp, Sg_max
            )
        )

        # Compute final rates with computed BHP
        qg, qo = self._compute_phase_rates(
            p_n1, pwf_final, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
            model_PVT, relperm_model, Ck, q_target, self.well_id, Sg_max
        )
        
        if self.fluid_type == 'DG':
            return qg, pwf_final
        else:
            qgg, qgo, qoo, qog = self._split_condensate_components(
                qg, qo, Sg_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1, relperm_model
            )
            return (qgg, qgo, qoo, qog), pwf_final

    @tf.function
    def compute_blocking_integral_and_factor(self, p_n1, Sg_n1, relperm_model, model_PVT, pwf_n1, eps=1e-12):
        """
        Compute blocking integral and blocking factor for phase flow.
        
        Args:
            p_n1: Tensor, gridblock pressure.
            Sg_n1: Tensor, gas saturation.
            relperm_model: Function, computes relative permeabilities (krog, krgo).
            model_PVT: Function, PVT model returning properties tensor.
            pwf_n1: Tensor, bottom-hole pressure.
            eps: Float, small value for numerical stability to avoid division by zero.
        
        Returns:
            Tuple: (Ig, Io, blk_fac_g, blk_fac_o) - gas and oil blocking integrals and factors.
        """
        # Pythonic statments in the static graph (@tf.function), i.e.,  if self.use_blocking_factor: and self.fluid_type == 'DG': are evaluated once during the trace
        # and not dynamically updated during the execution. 

        if self.use_blocking_factor:            
            # Compute mobilities
            krog_n1, krgo_n1 = relperm_model(Sg_n1)
            pvt_tensor = model_PVT(p_n1)
            invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1 = self.extract_pvt_properties(pvt_tensor, enable_logging=False)
            
            if self.fluid_type == 'DG':
                mg_n1 = krgo_n1 * invBg_n1 * invug_n1
                mo_n1 = tf.zeros_like(mg_n1)
            else:
                mgg_n1 = krgo_n1 * invBg_n1 * invug_n1
                mgo_n1 = krog_n1 * invBo_n1 * invuo_n1 * Rs_n1
                moo_n1 = krog_n1 * invBo_n1 * invuo_n1
                mog_n1 = krgo_n1 * invBg_n1 * invug_n1 * Rv_n1
                mg_n1 = mgg_n1 + mgo_n1
                mo_n1 = moo_n1 + mog_n1
            
            # Compute blocking integrals
            p_grid = tf.linspace(p_n1, pwf_n1, self.n_intervals + 1)
            new_shape = tf.concat([[self.n_intervals + 1], tf.shape(p_n1)], axis=0)
            p_grid = tf.reshape(p_grid, new_shape)
            
            sum_g = tf.zeros_like(p_n1)
            sum_o = tf.zeros_like(p_n1)
            mg_prev = mg_n1
            mo_prev = mo_n1
            i0 = tf.constant(0)

            def outer_cond(i, *_):
                return tf.less(i, self.n_intervals)

            def body(i, sum_g, sum_o, mg_prev, mo_prev):
                p0 = tf.gather(p_grid, i, axis=0)
                p1 = tf.gather(p_grid, i + 1, axis=0)
                pvt_tensor = model_PVT(p1)
                invBg1, invBo1, invug1, invuo1, Rs1, Rv1 = self.extract_pvt_properties(pvt_tensor, enable_logging=False)
                
                cond = tf.logical_or(tf.equal(self.fluid_type, 'DG'), krog_n1 < 1e-3)
                
                def cost(Sg):
                    krog, krgo = relperm_model(Sg)
                    mgg = krgo * invBg1 * invug1
                    mgo = krog * invBo1 * invuo1 * Rs1
                    moo = krog * invBo1 * invuo1
                    mog = krgo * invBg1 * invug1 * Rv1
                    mg = mgg + mgo
                    mo = tf.where(self.compute_mo, moo + mog, tf.zeros_like(mg))
                    return self.well_id * (mo * mg_n1 - mo_n1 * mg)
                
                Sg1 = tf.cond(
                    tf.equal(self.solver, 'newton'),
                    lambda: self._solve_newton(cost, Sg_n1, self.n_root_iter, max_value=1.0 - self.relperm.end_points['Swmin']),
                    lambda: self._solve_chandrupatla(cost, Sg_n1, self.n_root_iter, tol=1e-6, max_value=1.0 - self.relperm.end_points['Swmin'])
                )
                Sg1 = tf.where(cond, tf.ones_like(Sg1) * (1.0 - self.relperm.end_points['Swmin']), Sg1)
                krog1, krgo1 = relperm_model(Sg1)
                
                if self.fluid_type == 'DG':
                    mgg1 = krgo1 * invBg1 * invug1
                    mg1 = mgg1
                    mo1 = tf.zeros_like(mg1)
                    dp = p0 - p1
                    sum_g_new = sum_g + 0.5 * (mg_prev + mg1) * dp
                    sum_o_new = sum_o
                else:
                    mgg1 = krgo1 * invBg1 * invug1
                    mgo1 = krog1 * invBo1 * invuo1 * Rs1
                    moo1 = krog1 * invBo1 * invuo1
                    mog1 = krgo1 * invBg1 * invug1 * Rv1
                    mg1 = mgg1 + mgo1
                    mo1 = tf.cond(
                        tf.cast(self.compute_mo, tf.bool),
                        lambda: moo1 + mog1,
                        lambda: tf.zeros_like(mg1)
                    )
                    dp = p0 - p1
                    sum_g_new = sum_g + 0.5 * (mg_prev + mg1) * dp
                    sum_o_new = sum_o + 0.5 * (mo_prev + mo1) * dp * tf.cast(self.compute_mo, sum_g_new.dtype)
                
                return i + 1, sum_g_new, sum_o_new, mg1, mo1

            i, Ig, Io, mg_final, mo_final = tf.while_loop(
                outer_cond, body,
                [i0, sum_g, sum_o, mg_prev, mo_prev],
                shape_invariants=[
                    i0.get_shape(),
                    p_n1.get_shape(),
                    p_n1.get_shape(),
                    p_n1.get_shape(),
                    p_n1.get_shape(),
                ]
            )
            
            # Compute blocking factors
            dp = p_n1 - pwf_n1 + eps
            blk_fac_g = tf.math.divide_no_nan(Ig, mg_n1 * dp)
            blk_fac_o = tf.math.divide_no_nan(Io, mo_n1 * dp)
        else:
            def return_ones():
                ones = tf.ones_like(p_n1, dtype=self.dtype)
                return ones, ones, ones, ones
            Ig, Io, blk_fac_g, blk_fac_o = return_ones()
        return Ig, Io, blk_fac_g, blk_fac_o

    @tf.function
    def _compute_phase_rates(self, p_n1, pwf, Sg_n1, mg_n1, mo_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1, model_PVT, relperm_model, Ck, q_target, well_id, Sg_max):
        """
        Helper method to compute phase rates.
        
        Args:
            p_n1: Gridblock pressure.
            pwf: Bottom-hole pressure.
            Sg_n1: Gas saturation.
            mg_n1: Gas mobility.
            mo_n1: Oil mobility.
            invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1: PVT properties.
            model_PVT: PVT model function.
            relperm_model: Relative permeability function.
            Ck: Well productivity constant (well factor).
            q_target: Target gas rate.
            well_id: Well identifier.
            Sg_max: Maximum gas saturation.
        
        Returns:
            Tuple: (qg, qo) - gas and oil phase rates.
        """
        Ig, Io = self.compute_blocking_integral_and_factor(
            p_n1, Sg_n1, relperm_model, model_PVT, pwf
        )[:2]
        dp = p_n1 - pwf + 1e-12
        
        # Compute maximum blocking factors
        if self.use_blocking_factor:
            blk_fac_g = tf.math.divide_no_nan(Ig, mg_n1 * dp)
            div_result = tf.math.divide_no_nan(Io, mo_n1 * dp)
        else:
            blk_fac_g = Ig
            div_result = Io  
        blk_fac_o = tf.where(self.compute_mo, div_result, tf.ones_like(div_result))  

        qg_max = well_id * Ck * blk_fac_g * mg_n1 * dp
        qo_max = well_id * Ck * blk_fac_o * mo_n1 * dp
        
        qg = tf.maximum(tf.minimum(q_target, qg_max), 0.0) #tf.minimum(q_target, qg_max)
        qo = tf.zeros_like(qg)
        if self.fluid_type == 'GC':
            qo_target = qg * (1.0 / (Rv_n1 + 1e-12))
            qo = tf.maximum(tf.minimum(qo_target, qo_max), 0.0) #tf.minimum(qo_target, qo_max)
        
        return qg, qo

    @tf.function
    def _split_condensate_components(self, qg, qo, Sg_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1, relperm_model):
        """
        Split condensate rates into components (qgg, qgo, qoo, qog).
        
        Args:
            qg, qo: Gas and oil rates.
            Sg_n1: Gas saturation.
            invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1: PVT properties.
            relperm_model: Relative permeability model.
        
        Returns:
            Tuple: (qgg, qgo, qoo, qog) - gas and oil component rates.
        """
        krog, krgo = relperm_model(Sg_n1)
        mgg = krgo * invBg_n1 * invug_n1
        mgo = krog * invBo_n1 * invuo_n1 * Rs_n1
        moo = krog * invBo_n1 * invuo_n1
        mog = krgo * invBg_n1 * invug_n1 * Rv_n1
        denom_g = mgg + mgo + 1e-12
        denom_o = moo + mog + 1e-12
        qgg = qg * (mgg / denom_g)
        qgo = qg * (mgo / denom_g)
        qoo = qo * (moo / denom_o)
        qog = qo * (mog / denom_o)
        return qgg, qgo, qoo, qog

if __name__ == "__main__":
    tf.random.set_seed(42)
    
    # Test inputs
    unit_target_shape = DEFAULT_GENERAL_CONFIG['unit_target_shape']
    shape = (24, *unit_target_shape[1:])
    p_n1 = tf.random.uniform(shape, minval=5000.0, maxval=5000.0, dtype=tf.float32)
    Sg_n1 = tf.random.uniform(shape, minval=0.78, maxval=0.78, dtype=tf.float32)
    
    # Initialize WellRatesPressure to access data_summary for feature indices
    well_rates_init = WellRatesPressure(
        fluid_type='GC',
        dtype=tf.float32,
        solver='newton',
        n_intervals=8,
        n_root_iter=10,
        max_iters=2,
        tol=1e-6,
        compute_mo=False,
        use_non_iterative=False,
        reservoir_config=DEFAULT_RESERVOIR_CONFIG,
        general_config=DEFAULT_GENERAL_CONFIG,
        wells_config=DEFAULT_WELLS_CONFIG,
        pvt_layer_config=None,
        scal_config=DEFAULT_SCAL_CONFIG
    )
    
    # Assign realistic values to permx and time channels
    permx_idx = well_rates_init.data_summary.get_key_index('permx') if well_rates_init.data_summary else 0
    time_idx = well_rates_init.data_summary.get_key_index('time') if well_rates_init.data_summary else 1
    logging.info(f"Using permx_idx={permx_idx}, time_idx={time_idx} for x_n1")
    if well_rates_init.data_summary is None:
        logging.warning("data_summary is None; using default indices for permx and time")
    
    # Create x_n1 with specific ranges for permx (10-1000 mD) and time (0-1000 days)
    x_n1_list = [tf.random.uniform(shape[:-1], minval=-1, maxval=1, dtype=tf.float32) for _ in range(5)]
    x_n1_list[permx_idx] = tf.random.uniform(shape[:-1], minval=-1, maxval=1, dtype=tf.float32)  # permx in mD
    x_n1_list[time_idx] = tf.random.uniform(shape[:-1], minval=-1, maxval=1, dtype=tf.float32)  # time in days
    x_n1 = tf.stack(x_n1_list, axis=-1)
    
    # Test both iterative and non-iterative methods (only False as per provided code)
    for use_non_iterative in [False]:
        logging.info(f"\n=== Testing with use_non_iterative={use_non_iterative} ===")
        
        # Initialize WellRatesPressure with configuration dictionaries
        well_rates = WellRatesPressure(
            fluid_type='DG',
            use_blocking_factor=False,
            dtype=tf.float32,
            solver='newton',
            n_intervals=6,
            n_root_iter=10,
            max_iters=4,
            tol=1e-6,
            compute_mo=False,  # Disable oil mobility for GC
            use_non_iterative=use_non_iterative,
            reservoir_config=DEFAULT_RESERVOIR_CONFIG,
            general_config=DEFAULT_GENERAL_CONFIG,
            wells_config=DEFAULT_WELLS_CONFIG,
            pvt_layer_config=None,  # Will be derived based on fluid_type
            scal_config=DEFAULT_SCAL_CONFIG
        )
        
        # Compute rates and BHP with x_n1
        result = well_rates.compute_rates_and_bhp(
            x_n1=x_n1,
            p_n1=p_n1,
            Sg_n1=Sg_n1,
            relperm_model=well_rates.relperm.compute_krog_krgo,
            model_PVT=well_rates.pvt_model
        )
        
        # Prepare indices for gathering values at well locations across all batches
        well_data = well_rates.well_data_processor.get_well_data()
        num_wells = tf.shape(well_data['connection_index'])[0]
        batch_indices = tf.repeat(tf.range(24, dtype=tf.int32), num_wells)
        well_indices = tf.tile(well_data['connection_index'], [24, 1])
        channel_indices = tf.zeros([24 * num_wells, 1], dtype=tf.int32)
        verify_positions = tf.concat([
            tf.expand_dims(batch_indices, -1),
            well_indices,
            channel_indices
        ], axis=1)

        if well_rates.fluid_type == 'DG':
            qg, pwf = result
            logging.info(f"DG: qg shape = {tf.shape(qg)}, pwf shape = {tf.shape(pwf)}")
            qg_gathered = tf.gather_nd(qg, verify_positions)
            pwf_gathered = tf.gather_nd(pwf, verify_positions)
            logging.info(f"DG: qg at well locations: {qg_gathered.numpy()}")
            logging.info(f"DG: pwf at well locations: {pwf_gathered.numpy()}")
        else:
            (qgg, qgo, qoo, qog), pwf = result
            logging.info(f"GC: qgg shape = {tf.shape(qgg)}, qgo shape = {tf.shape(qgo)}, qoo shape = {tf.shape(qoo)}, qog shape = {tf.shape(qog)}, pwf shape = {tf.shape(pwf)}")
            qgg_gathered = tf.gather_nd(qgg, verify_positions)
            qgo_gathered = tf.gather_nd(qgo, verify_positions)
            qoo_gathered = tf.gather_nd(qoo, verify_positions)
            qog_gathered = tf.gather_nd(qog, verify_positions)
            pwf_gathered = tf.gather_nd(pwf, verify_positions)
            # Print gathered rates and pressure for each batch on a separate line
            for i in [5]:# range(24):
               start_idx = i * num_wells
               end_idx = (i + 1) * num_wells
               logging.info(f"Batch {i}:")
               logging.info(f"  qgg = {qgg_gathered[start_idx:end_idx].numpy()}")
               logging.info(f"  qgo = {qgo_gathered[start_idx:end_idx].numpy()}")
               logging.info(f"  qoo = {qoo_gathered[start_idx:end_idx].numpy()}")
               logging.info(f"  qog = {qog_gathered[start_idx:end_idx].numpy()}")
               logging.info(f"  pwf = {pwf_gathered[start_idx:end_idx].numpy()}")                   
       
        # Compute blocking integral and factor
        Ig, Io, blk_fac_g, blk_fac_o = well_rates.compute_blocking_integral_and_factor(
            p_n1=p_n1,
            Sg_n1=Sg_n1,
            relperm_model=well_rates.relperm.compute_krog_krgo,
            model_PVT=well_rates.pvt_model,
            pwf_n1=well_rates.pwf_min
        )
        logging.info(f"Blocking integrals: Ig shape = {tf.shape(Ig)}, Io shape = {tf.shape(Io)}")
        logging.info(f"Blocking factors: blk_fac_g shape = {tf.shape(blk_fac_g)}, blk_fac_o shape = {tf.shape(blk_fac_o)}")
        Ig_gathered = tf.gather_nd(Ig, verify_positions)
        Io_gathered = tf.gather_nd(Io, verify_positions)
        blk_fac_g_gathered = tf.gather_nd(blk_fac_g, verify_positions)
        blk_fac_o_gathered = tf.gather_nd(blk_fac_o, verify_positions)
        logging.info(f"Ig at well locations: {Ig_gathered.numpy()}")
        logging.info(f"Io at well locations: {Io_gathered.numpy()}")
        logging.info(f"blk_fac_g at well locations: {blk_fac_g_gathered.numpy()}")
        logging.info(f"blk_fac_o at well locations: {blk_fac_o_gathered.numpy()}")