#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import numpy as np
import tensorflow as tf
import hashlib
import warnings
import re
import logging

# Set up logging to show INFO level by default
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the current directory to the system path for importing default configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from default_configurations import WORKING_DIRECTORY, DEFAULT_GENERAL_CONFIG, DEFAULT_RESERVOIR_CONFIG, DEFAULT_WELLS_CONFIG, DEFAULT_SCAL_CONFIG, DEFAULT_SIMDATA_PROCESS_CONFIG
    from data_processing import create_positional_grids, DataSummary, weave_tensors, align_and_trim_pair_lists, split_tensor_sequence, slice_statistics
    from simulation_data_process_pipeline import run_pipeline_from_config
except ImportError as e:
    logging.error(f"ERROR: Could not import required modules: {e}")
    logging.info("Make sure default_configurations.py and data_processing.py exist in the same directory.")
    WORKING_DIRECTORY = None
    DEFAULT_GENERAL_CONFIG = None
    DEFAULT_RESERVOIR_CONFIG = None
    DEFAULT_WELLS_CONFIG = None
    DEFAULT_SCAL_CONFIG = None
    DEFAULT_SIMDATA_PROCESS_CONFIG = None


class SRMDataProcessor:
    """Class for processing data for Surrogate Reservoir Models (SRM) based on default configurations."""
    
    def __init__(self, base_dir=None, dtype=tf.float32):
        """
        Initialize the SRM data processor.
        
        Args:
            base_dir (str, optional): Base directory for data. If None, uses the directory of this script.
        """
        self.base_dir = base_dir or WORKING_DIRECTORY
        self.kle_dir = os.path.join(self.base_dir, "KL_Realizations")
        
        # Verify that the required configurations are available
        if None in (WORKING_DIRECTORY, DEFAULT_GENERAL_CONFIG, DEFAULT_RESERVOIR_CONFIG, DEFAULT_WELLS_CONFIG):
            raise ValueError("Required default configurations are not available")

        # general config
        self.general_config = DEFAULT_GENERAL_CONFIG
        
        # Reservoir config
        self.reservoir_config = DEFAULT_RESERVOIR_CONFIG
        
        # Wells config
        self.wells_config = DEFAULT_WELLS_CONFIG

        # SCAL config
        self.scal_config = DEFAULT_SCAL_CONFIG

        # Extract configuration values
        self.srm_start_time = self.general_config['srm_start_time']
        self.srm_end_time = self.general_config['srm_end_time']
        self.srm_timestep = self.general_config['srm_timestep']
        
        # Check for porosity realizations
        self.has_porosity = self._check_porosity_realizations()
        
        # Store KLE configuration hash for folder identification
        self.full_config_hash = self._generate_full_config_hash()
        
        # Initialize data holders
        self.kle_data = None
        self.time_tensor = None
        self.summary_info = None
        self.x_grid = None
        self.y_grid = None
        self.z_grid = None
        
        # Get dtype
        self.dtype = self.general_config['dtype']
        if self.dtype is None: 
            self.dtype = dtype

        # Default split keys for the data
        self.split_keys = self.general_config['split_keys']
        self.split_axis = self.general_config['split_axis']
        self.seed = self.general_config['seed']
        # Renormalize all split_ratio keys if needed and store the result in self.split_ratio
        split_ratio_dict = {}
        for key, ratio_tuple in self.general_config['split_ratio'].items():
            ratio_list = list(ratio_tuple)
            total = sum(ratio_list)
            if total > 1.0:
                ratio_list = [r / total for r in ratio_list]
                logging.warning(f"Split ratio for key {key} sums to {total:.3f} > 1.0, renormalizing to: {ratio_list}")
            split_ratio_dict[key] = ratio_list
        self.split_ratio = split_ratio_dict

    def _generate_full_config_hash(self):
        """
        Generate a unique identifier and hash for the full configuration using the canonical hash logic from kl_realizations_generator.py.
        Returns:
            list: [readable_name, config_hash]
        """
        from kl_realizations_generator import generate_full_config_hash
        from default_configurations import get_configuration
        config_str, config_hash = generate_full_config_hash(
            self.general_config,
            self.reservoir_config,
            self.wells_config,
            self.scal_config,
            get_configuration
        )
        Nx = self.reservoir_config.get('Nx')
        Ny = self.reservoir_config.get('Ny')
        Nz = self.reservoir_config.get('Nz')
        number_of_realizations = self.reservoir_config.get('realizations', {}).get('permx', {}).get('number')
        readable_name = f"KLE_{Nx}x{Ny}x{Nz}_R{number_of_realizations}_"
        return [readable_name, config_hash]
    
    def _check_porosity_realizations(self):
        """
        Check if porosity realizations are available in the reservoir configuration.
        
        Returns:
            bool: True if porosity realizations are configured
        """
        if 'realizations' in self.reservoir_config and 'poro' in self.reservoir_config['realizations']:
            poro_config = self.reservoir_config['realizations']['poro']
            if poro_config is not None and poro_config != {None}:
                logging.info("Porosity realizations are available in configuration. These will be processed.")
                return True
        
        return False
    
    def find_kle_folder(self):
        """
        Returns the path to the current KLE folder in the Static and Dynamic Properties directory using the full configuration hash and readable name.
        """
        readable_name, config_hash = self._generate_full_config_hash()
        folder_name = f"{readable_name}{config_hash}"
        kle_folder = os.path.join(WORKING_DIRECTORY, "Static and Dynamic Properties", folder_name)
        if os.path.isdir(kle_folder):
            return kle_folder
        raise FileNotFoundError(f"KLE folder not found: {kle_folder}")

    def find_simulation_folder(self, sim_type='test'):
        """
        Find the simulation folder in the Static and Dynamic Properties directory for the given sim_type (train, val, test).
        The path is:
        WORKING_DIRECTORY/Static and Dynamic Properties/<readable_name><full_config_hash>/dat_files_<sim_type>_<full_config_hash>/dynamic

        Args:
            sim_type (str): One of 'train', 'val', or 'test'. Default is 'test'.
        Returns:
            str: Path to the simulation folder.
        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        readable_name, config_hash = self._generate_full_config_hash()
        folder_name = f"{readable_name}{config_hash}"

        dat_files_folder = f"dat_files_{sim_type}_{config_hash}"
        sim_folder = os.path.join(
            WORKING_DIRECTORY,
            "Static and Dynamic Properties",
            folder_name,
            dat_files_folder,
            "dynamic"
        )
        if os.path.isdir(sim_folder):
            return sim_folder
        raise FileNotFoundError(f"Simulations folder not found: {sim_folder}")

    def load_kle_data(self, load_flag="train", load_compressed=False):
        """
        Load KLE data (permeability realizations) and summary info from the KLE folder.
        Args:
            load_flag (str): Which split to load ('all', 'train', 'val', 'test'). Defaults to 'train'.
            load_compressed (bool): Whether to load the compressed .npz version. Defaults to False.
        Returns:
            np.ndarray: Loaded KLE data
        """
        try:
            kle_folder = self.find_kle_folder()

            # Determine file names based on flag
            base_name = f"realizations_{load_flag}"
            kle_data_path = os.path.join(kle_folder, f"{base_name}.npy")
            compressed_path = os.path.join(kle_folder, f"{base_name}.npz")

            # Ensure self.kle_data is a dictionary
            if not hasattr(self, 'kle_data') or not isinstance(self.kle_data, dict):
                self.kle_data = {}

            # Determine which file to load
            if load_compressed:
                kle_path = compressed_path
                log_msg = f"Loaded KLE data (compressed) from {kle_path}"
            else:
                kle_path = kle_data_path
                log_msg = f"Loaded KLE data from {kle_path}"

            # Load the file or raise an error
            if os.path.exists(kle_path):
                self.kle_data[load_flag] = np.load(kle_path)
                logging.info(f"{log_msg}, shape: {self.kle_data[load_flag].shape}")
            else:
                raise FileNotFoundError(f"Could not find KLE data files for flag '{load_flag}' in {kle_folder}")

            # Always update/replace summary for the split
            summary_path = os.path.join(kle_folder, f"summary_{load_flag}.json")
            if not hasattr(self, 'summary_info') or not isinstance(self.summary_info, dict):
                self.summary_info = {}
            
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    self.summary_info[load_flag] = json.load(f)
                logging.info(f"Loaded summary_{load_flag} information from {summary_path}")
            else:
                self.summary_info[load_flag] = None
                logging.warning(f"No summary file found for flag '{load_flag}' in {kle_folder}")

            # Store the KLE folder path for later use
            self.kle_folder = kle_folder

            return self.kle_data[load_flag]
        except Exception as e:
            logging.error(f"Error loading KLE data: {str(e)}")
            raise
    
    def generate_time_tensor(self, load_flag = "train"):
        """
        Generate a 1D time tensor based on SRM time settings and well connection shutin times.
        {{ ... }}
        Returns:
            np.ndarray: Time tensor of shape [N, 1]
        """
        # Create base time points from start to end with specified timestep
        num_steps = int((self.srm_end_time - self.srm_start_time) / self.srm_timestep) + 1
        base_times = np.linspace(self.srm_start_time, self.srm_end_time, num_steps, dtype=self.dtype)
        
        # Collect all unique shutin times from well configuration
        shutin_times = set()
        # Process connection_shutin_days
        if 'connection_shutin_days' in self.wells_config:
            for well_name, intervals in self.wells_config['connection_shutin_days'].items():
                for interval in intervals:
                    if len(interval) == 2:
                        shutin_start, shutin_end = interval
                        if shutin_start <= self.srm_end_time:
                            shutin_times.add(shutin_start)
                            # print(shutin_start)
                        if shutin_end <= self.srm_end_time:
                            shutin_times.add(shutin_end)

        # Combine base times with shutin times and sort
        all_times = np.sort(np.unique(np.concatenate([base_times, np.array(list(shutin_times), dtype=self.dtype)])))
        
        # Filter to ensure all times are <= srm_end_time
        all_times = all_times[all_times <= self.srm_end_time]
        
        # Reshape to [N, 1] tensor
        time_tensor = all_times.reshape(-1, 1)

        # Store in dictionary by split
        if not hasattr(self, 'time_tensor') or not isinstance(self.time_tensor, dict):
            self.time_tensor = {}

        # Always split the time tensor sequentially according to split_ratio and split_keys
        split_keys = self.split_keys
        split_ratio = self.split_ratio[1]  # Use key 1 for time splitting by default
        n = time_tensor.shape[0]

        # Computes the end indices for each split in a sequential split of an array (e.g., time tensor) based on the 
        # provided split ratios 
        idx = [int(n * sum(split_ratio[:i+1])) for i in range(len(split_ratio))]
        idx = [0] + idx                                                    # [0, ...]

        # Create all splits
        for i, key in enumerate(split_keys):
            start, end = idx[i], idx[i+1] if i+1 < len(idx) else n
            if key in ('val', 'test'):
                self.time_tensor[key] = time_tensor
                logging.info(f"For split '{key}', using the full time tensor with shape {time_tensor.shape}")
                if len(time_tensor) > 0:
                    logging.info(f"Time range: {time_tensor.min()} to {time_tensor.max()}")
                logging.info(f"Number of time points: {len(time_tensor)}")
            else:
                split_tensor = time_tensor[start:end]
                self.time_tensor[key] = split_tensor
                if load_flag == 'all' or load_flag == key:
                    logging.info(f"Generated time tensor for split '{key}' with shape {split_tensor.shape}")
                    if len(split_tensor) > 0:
                        logging.info(f"Time range: {split_tensor.min()} to {split_tensor.max()}")
                    logging.info(f"Number of time points: {len(split_tensor)}")


        # Return only the requested split if not 'all'
        if load_flag == 'all':
            return self.time_tensor
        else:
            return self.time_tensor[load_flag]


    def create_positional_tensors(self, load_flag = "train"):
        """
        Create positional tensors (x, y, z distances) or additional patterns using the create_positional_grids function.
        
        Returns:
            tuple: (x_tensor, y_tensor, z_tensor) with shapes [1, Nz, Ny, Nx]
        """
        logging.info(f"Generating positional tensors for split '{load_flag}'")
        
        # Get dimensions from reservoir config
        Nx = self.reservoir_config.get('Nx')
        Ny = self.reservoir_config.get('Ny')
        Nz = self.reservoir_config.get('Nz')
        Lx = self.reservoir_config.get('length')
        Ly = self.reservoir_config.get('width')
        Lz = self.reservoir_config.get('thickness')
        
        logging.info(f"Creating positional grids for dimensions: [{Nx}, {Ny}, {Nz}]")
        logging.info(f"Physical dimensions: [{Lx}, {Ly}, {Lz}]")
        
        # Create midpoint grids using the create_positional_grids function
        D = [Lx, Ly, Lz]  # Physical dimensions
        N = [Nx, Ny, Nz]  # Number of grid cells
        grids = create_positional_grids(D, N, indexing='ij', transpose_order=[2, 1, 0])                 # Transpose order for the array grid to be consistent with Fortran cyclying
        
        # Extract individual grids
        x_grid, y_grid, z_grid = grids
        
        # Reshape to add leading dimension of 1 for batch processing
        x_tensor = tf.cast(np.expand_dims(x_grid, axis=0),self.dtype)  # Shape: [1, Nz, Ny, Nx]
        y_tensor = tf.cast(np.expand_dims(y_grid, axis=0),self.dtype)  # Shape: [1, Nz, Ny, Nx]
        z_tensor = tf.cast(np.expand_dims(z_grid, axis=0),self.dtype)  # Shape: [1, Nz, Ny, Nx]
        
        # Store tensors as dictionaries keyed by split
        if not hasattr(self, 'x_grid') or not isinstance(self.x_grid, dict):
            self.x_grid = {}
        if not hasattr(self, 'y_grid') or not isinstance(self.y_grid, dict):
            self.y_grid = {}
        if not hasattr(self, 'z_grid') or not isinstance(self.z_grid, dict):
            self.z_grid = {}
        
        self.x_grid[load_flag] = x_tensor
        self.y_grid[load_flag] = y_tensor
        self.z_grid[load_flag] = z_tensor
        
        logging.info(f"Created positional tensors with shapes: x={x_tensor.shape}, y={y_tensor.shape}, z={z_tensor.shape}")
        
        return x_tensor, y_tensor, z_tensor
    
    def weave_tensor(self, data, flatten_first_axes=False, merge_consecutive_singleton_dims=True):
        """
        Weave the various tensors in the data dictionary into a single tensor.{{ ... }}
        The function uses the weave_tensors function from data_processing to combine:

        - x_distances: shape (1, Nz, Ny, Nx)
        - y_distances: shape (1, Nz, Ny, Nx)
        - z_distances: shape (1, Nz, Ny, Nx)
        - time: shape (B, 1)
        - permx: shape (A, Nz, Ny, Nx)
        
        Args:
            data (dict): Dictionary containing data arrays
            
        Returns:
            np.ndarray: Woven tensor with shape (A*B*1*1*1, Nz, Ny, Nx, 5)
        """
        # Prepare the list of tensors for weaving
        tensor_list = list(data.values())

        # Debug: print shapes of all tensors before weaving
        logging.debug("Debug woven tensor shapes:")
        logging.debug(f"  permx: {data['permx'].shape}")
        logging.debug(f"  time: {data['time'].shape}")
        logging.debug(f"  x: {data['x'].shape}")
        logging.debug(f"  y: {data['y'].shape}")
        logging.debug(f"  z: {data['z'].shape}")
        
        # Get target trailing shape from permx (Nz, Ny, Nx)
        permx_shape = data['permx'].shape
        target_trailing_shape = permx_shape[1:]  # (Nz, Ny, Nx)
        logging.debug(f"  target_trailing_shape: {target_trailing_shape}")
        # Call weave_tensors to combine all tensors, flattening the first axes
        woven_tensor = weave_tensors(
            tensor_list=tensor_list, 
            target_trailing_shape=target_trailing_shape, 
            flatten_first_axes=flatten_first_axes,
            merge_consecutive_singleton_dims=merge_consecutive_singleton_dims
        )
        logging.info(f"Woven tensor shape: {woven_tensor.shape}")
        return woven_tensor
    
    def create_tensorflow_tensor(self, data, dtype=tf.float32):
        """
        Convert numpy array to TensorFlow tensor.
        
        Args:
            data (np.ndarray): Data to convert
            dtype: TensorFlow data type
            
        Returns:
            tf.Tensor: TensorFlow tensor
        """
        return tf.convert_to_tensor(data, dtype=dtype)
    
    def process_data(self, apply_normalization=True, create_woven_tensor=True, process_flag="test"):
        """
        Process all data: load KLE data, generate time tensor, create positional tensors, 
        normalize data for SRM, run simulation data processing pipeline, and optionally create a woven tensor.
        
        Args:
            apply_normalization (bool): Whether to apply normalization
            create_woven_tensor (bool): Whether to create a woven tensor
            
        Returns:
            tuple: (train_groups, val_groups, test_groups) - Dictionary of tensors
        """
        # Load KLE data for all splits if any are missing
        if not isinstance(self.kle_data, dict):
            self.kle_data = {}
        for split in self.split_keys:
            self.load_kle_data(load_flag=split)

        
        # Generate time tensor for all splits if not already generated
        if not isinstance(self.time_tensor, dict):
            self.time_tensor = {}
        for split in self.split_keys:
            self.generate_time_tensor(load_flag=split)
        
        # Create positional tensors for all splits if not already created
        if not isinstance(self.x_grid, dict):
            self.x_grid = {}
        if not isinstance(self.y_grid, dict):
            self.y_grid = {}
        if not isinstance(self.z_grid, dict):
            self.z_grid = {}
        for split in self.split_keys:
            self.create_positional_tensors(load_flag=split)
        
        # Run simulation data processing pipeline if KLE folder exists and has a hash
        sim_data = None
        if hasattr(self, 'kle_folder') and self.kle_folder:
            # Construct sim_folder_path based on the required convention
            _, full_config_hash = self._generate_full_config_hash()  # returns (readable_name, hash)
            dat_files_dir = f"dat_files_{process_flag}_{full_config_hash}"
            sim_folder_path = os.path.join(self.kle_folder, dat_files_dir, "dynamic")

            if os.path.isdir(sim_folder_path):
                sim_folder = sim_folder_path
                logging.info(f"Found simulation data folder: {sim_folder}")
                
                # Get or create simulation data processing configuration
                sim_config = DEFAULT_SIMDATA_PROCESS_CONFIG.copy() if DEFAULT_SIMDATA_PROCESS_CONFIG else {}
                
                # Set up input and output folders
                output_folder = os.path.join(sim_folder, "output")
                os.makedirs(output_folder, exist_ok=True)

                # Check for pre-existing combined_results.npz
                combined_results_path = os.path.join(output_folder, "combined_results.npz")
                combined_exists = os.path.isfile(combined_results_path)

                # Update the configuration for the simulation pipeline
                if "simulation_pipeline" in sim_config and sim_config["simulation_pipeline"].get("enabled", False):
                    sim_config["simulation_pipeline"]["input_folder"] = sim_folder
                    sim_config["simulation_pipeline"]["output_folder"] = output_folder

                    # Shape should be updated from the KLE data dimensions (optional to ensure consistency)
                    # Note the shape is (Nx, Ny, Nz) while the KLE data is already reversed (Nz, Ny, Nx)
                    sim_config["simulation_pipeline"]["shape"] = (self.kle_data[process_flag].shape[3], self.kle_data[process_flag].shape[2], self.kle_data[process_flag].shape[1])
                    
                    # Disable simulation pipeline if combined_results.npz exists
                    if combined_exists:
                        sim_config["simulation_pipeline"]["enabled"] = False
                        logging.info(f"combined_results.npz found in {output_folder}. Skipping simulation pipeline.")
                else:
                    raise RuntimeError("DEFAULT_SIMDATA_PROCESS_CONFIG must contain 'simulation_pipeline' key.")

                # Set up array pipeline config; only override 'slices' from current context
                if "array_pipeline" in sim_config and sim_config["array_pipeline"].get("enabled", False):
                    sim_config["array_pipeline"]["directory"] = output_folder
                    sim_config["array_pipeline"]["slices"] = list(tf.cast(tf.reshape(self.time_tensor[process_flag],(-1,)), tf.int32).numpy())
                else:
                    raise RuntimeError("DEFAULT_SIMDATA_PROCESS_CONFIG must contain 'array_pipeline' key.")

                # Run the simulation data processing pipeline
                logging.info(f"Running simulation data processing pipeline for {sim_folder}")
                try:
                    sim_data = run_pipeline_from_config(sim_config)
                    logging.info("Simulation data processing complete.")
                    SRMDataProcessor.free_gpu_memory()  # Free GPU after simulation pipeline
                    # Debug: print shape/type of sim_data
                    if sim_data is not None:
                        # Print each key on a new line
                        for key, value in sim_data.items():
                            logging.info(f"{key}: {value.shape}")
                except Exception as e:
                    logging.warning(f"Failed to process simulation data: {str(e)}")
            else:
                logging.info(f"No simulation data folder found for expected path: {sim_folder_path}")
    
        # --- REMOVE normalization: operate on raw data ---
        # Prepare raw (unnormalized) data dict for each split
        raw_data = {}
        for split in self.split_keys:
            raw_data[split] = {
                'permx': self.kle_data[split],
                'time': self.time_tensor[split],
                'x': self.x_grid[split],
                'y': self.y_grid[split],
                'z': self.z_grid[split],
            }

        # Create woven tensor from unnormalized data for each split if requested
        woven_tensor = {}
        if create_woven_tensor:
            for split in self.split_keys:
                try:                    
                    woven_tensor[split] = self.weave_tensor(raw_data[split], flatten_first_axes=False, merge_consecutive_singleton_dims=True)   
                    SRMDataProcessor.free_gpu_memory()  # Free GPU after weaving tensor for this split
                except Exception as e:
                    logging.warning(f"Failed to create woven tensor for split '{split}': {str(e)}")

        # Create a dictionary with zero-like values matching each woven_tensor as labels for the train and validation data
        # The innermost dimension is excluded as it is a feature

        seed = self.general_config['seed']
        physics_mode_fraction = self.general_config['physics_mode_fraction']
        if physics_mode_fraction>=1.0:
            v_train = tf.zeros_like(woven_tensor['train'][...,0])
            v_val = tf.zeros_like(woven_tensor['val'][...,0])
            train_labels = {key:v_train for key in sim_data.keys()}
            val_labels = {key:v_val for key in sim_data.keys()}

        # For the test dataset, the simulation data is used as the label.
        # Compare woven_tensor['test'][...,0].shape[0] with the simulation data's shape for the first key at axes [0,1].
        # If not equal, pad woven_tensor['test'] (along axis 0) with zeros or trim as needed to match the sim_data shape.
        test_tensor = woven_tensor['test']
        # Get the first key from sim_data (assume all keys have the same shape at axes [0,1])

        first_sim_key = next(iter(sim_data.keys()))
        sim_shape = sim_data[first_sim_key].shape
        test_shape_0 = test_tensor[...,0].shape[0]
        sim_shape_0 = sim_shape[0]
        # Pad or trim woven_tensor['test'] if necessary
        if test_shape_0 < sim_shape_0:
            # Pad with zeros along axis 0
            pad_width = sim_shape_0 - test_shape_0
            pad_shape = list(test_tensor.shape)
            pad_shape[0] = pad_width
            pad_tensor = tf.zeros(pad_shape, dtype=test_tensor.dtype)
            test_tensor = tf.concat([test_tensor, pad_tensor], axis=0)
        elif test_shape_0 > sim_shape_0:
            # Trim along axis 0
            test_tensor = test_tensor[:sim_shape_0]

        # Update woven_tensor['test'] to use the shape-aligned test_tensor
        woven_tensor['test'] = test_tensor

        # Generate a feature_label prediction dataset for the prediction period, i.e., test permeability samples at prediction times (tp,... tn) 
        # instead of using all time points (0, 1, ..., tn) as used in the fearture_label_test dataset

        import copy
        # For the prediction dataset, split_ratio: {0--permeability: (0.0, 0.0, 1.0), 1--time: (retain original)}
        split_ratio_pred = copy.deepcopy(self.split_ratio)
        split_ratio_pred[0] = (0.0, 0.0, 1.0)

        # Create the raw (unnormalized) data and weave the tensor
        raw_data_pred = {
                'permx': self.kle_data['test'],
                'time': self.time_tensor['test'],
                'x': self.x_grid['test'],
                'y': self.y_grid['test'],
                'z': self.z_grid['test'],
            }

        # Create woven tensor from unnormalized pred data for each split if requested
        if create_woven_tensor:
            try:
                woven_tensor_pred = self.weave_tensor(raw_data_pred, flatten_first_axes=False, merge_consecutive_singleton_dims=True)
                SRMDataProcessor.free_gpu_memory()  # Free GPU after weaving tensor for this split
            except Exception as e:
                logging.warning(f"Failed to create woven tensor for split 'pred': {str(e)}")
        
        _, _, pred_features = split_tensor_sequence(
            [woven_tensor_pred], split_ratio_pred, self.split_axis, seed=self.seed, merge_consecutive_singleton_dims=True)
        _, _, pred_labels = split_tensor_sequence([sim_data], split_ratio_pred, self.split_axis, seed=self.seed, merge_consecutive_singleton_dims=True)

        pred_features, pred_labels = align_and_trim_pair_lists(pred_features[0], pred_labels, dims=[0,1], trim_target="b")
  
        @tf.function
        def filter_empty(features, labels):
            """
            Wrap features and labels into lists, handling single items, lists, TensorFlow tensors, or dictionaries.
            No filtering is performed; all inputs are retained as-is.
            Handles length mismatches by repeating single items to match the length of the other input.
            Always returns lists for both outputs.
        
            Args:
                features: tf.Tensor, dict, list, or single item containing feature data.
                labels: tf.Tensor, dict, list, or single item containing label data.
        
            Returns:
                tuple: (filtered_features, filtered_labels), lists containing the original inputs unchanged.
            """
            # Wrap as list if not already
            features_list = features if isinstance(features, (list, tuple)) else [features]
            labels_list = labels if isinstance(labels, (list, tuple)) else [labels]
        
            # Handle empty inputs
            if not features_list and not labels_list:
                return [], []
        
            # Repeat if only one is a list
            if len(features_list) != len(labels_list):
                if len(features_list) == 1:
                    features_list = features_list * len(labels_list)
                elif len(labels_list) == 1:
                    labels_list = labels_list * len(features_list)
                else:
                    raise ValueError("Length mismatch between features and labels.")
        
            # Return inputs as-is (no filtering)
            return features_list, labels_list

        train_features, train_labels = filter_empty(woven_tensor['train'], train_labels)
        val_features, val_labels = filter_empty(woven_tensor['val'], val_labels)
        test_features, test_labels = filter_empty(woven_tensor['test'], sim_data)
        pred_features, pred_labels = filter_empty(pred_features, pred_labels)

        # Create initial groups for statistics calculation
        initial_train_groups = list(zip(train_features, train_labels))
        # Save training statistics with raw_data keys
        train_config_hash = self._generate_full_config_hash()[1]
        raw_data_keys = list(raw_data['train'].keys())[::-1] if isinstance(raw_data['train'], dict) else None
        statistics, stats_path = self.save_training_statistics(initial_train_groups, train_config_hash, raw_data_keys=raw_data_keys)
        
        # Create DataSummary from the statistics for normalization
        from data_processing import DataSummary
        data_summary = DataSummary([statistics], dtype=self.dtype)
        
        # Get normalization configuration from DEFAULT_GENERAL_CONFIG
        norm_config = self.general_config['data_normalization']

        # Normalize features in all groups
        def normalize_features(groups):
            normalized_groups = []
            for features, labels in groups:
                # Convert to tensor if needed
                if not isinstance(features, tf.Tensor):
                    features = tf.convert_to_tensor(features, dtype=self.dtype)
                
                # Normalize features (index 0) using DataSummary
                # Use statistics index 0 for all features or create appropriate mapping
                stats_idx_map = tf.constant([[0,1,2,3,4],[0,1,2,3,4]], dtype=tf.int32)  # Mapping: [norm dimension indices] => [normalization statistics indices (perm, time, etc.)]

                normalized_features = data_summary.normalize(
                    features,
                    norm_config=norm_config,
                    statistics_index=stats_idx_map,
                    compute=True,
                    normalization_dimension=-1,  # Normalize along the last dimension
                    dtype=self.dtype
                )
                
                normalized_groups.append((normalized_features, labels))
            return normalized_groups

        # Apply normalization to all groups
        train_groups = normalize_features(list(zip(train_features, train_labels)))
        val_groups = normalize_features(list(zip(val_features, val_labels)))
        test_groups = normalize_features(list(zip(test_features, test_labels)))
        pred_groups = normalize_features(list(zip(pred_features, pred_labels)))
        
        logging.info(f"Features normalized using {norm_config['feature_normalization_method']} with range {norm_config['normalization_limits']}")
        
        # Save [train_groups, val_groups, test_groups, pred_groups] as a list
        self.save_data_groups_list([train_groups, val_groups, test_groups, pred_groups], train_config_hash, dtype=self.dtype)
        
        return train_groups, val_groups, test_groups, pred_groups, statistics, stats_path

    def save_data_groups_list(self, groups_list, train_config_hash=None, dtype=None):
        import pickle
        """
        Saves the [train_groups, val_groups, test_groups] list as a .pkl file in the KLE folder.
        Optionally casts all arrays to the specified dtype (e.g., np.float32).
        """
        kle_folder = self.find_kle_folder()
        if train_config_hash is None:
            train_config_hash = self._generate_full_config_hash()[1]
        save_path = os.path.join(kle_folder, f"training_data_{train_config_hash}.pkl")
        groups_list_np = []
    
        for group in groups_list:
            group_np = []
            for t in group:
                # first, get a numpy-compatible object
                arr = t.numpy() if isinstance(t, tf.Tensor) else t
    
                # if it's a dict, convert its values individually
                if isinstance(arr, dict):
                    converted = {}
                    for k, v in arr.items():
                        v_arr = v.numpy() if isinstance(v, tf.Tensor) else v
                        if dtype is not None and hasattr(v_arr, "astype"):
                            converted[k] = v_arr.astype(dtype)
                        else:
                            converted[k] = v_arr
                    arr = converted
    
                # otherwise, if it has astype (e.g. numpy array), cast it
                elif dtype is not None and hasattr(arr, "astype"):
                    arr = arr.astype(dtype)
    
                # leave any other types (lists, scalars, etc.) as-is
                group_np.append(arr)
            groups_list_np.append(group_np)
    
        with open(save_path, 'wb') as f:
            pickle.dump(groups_list_np, f)
    
        logging.info(f"Saved [train_groups, val_groups, test_groups] list to {save_path}")
        return save_path
        
    def save_training_statistics(self, train_groups, train_config_hash=None, raw_data_keys=None):
        """
        Calculate and save training statistics [min, max, mean, std, shape] for features and labels in train_groups.
        
        Args:
            train_groups (list): List of tuples containing (features, labels) from training data
            train_config_hash (str, optional): Hash for the training data filename
            raw_data_keys (list, optional): List of keys for the feature channels. If None, uses default keys or generates sequential names
        Returns:
            str: Path to the saved statistics file
        """
        import json
        import numpy as np
        kle_folder = self.find_kle_folder()
        if train_config_hash is None:
            train_config_hash = self._generate_full_config_hash()[1]
        # Generate statistics filename
        stats_filename = f"training_statistics_summary_{train_config_hash}.json"
        stats_path = os.path.join(kle_folder, stats_filename)
        
        # Check if train_groups is a list of tuples or a single tuple
        if isinstance(train_groups, list):
            if len(train_groups) == 0:
                logging.error("Empty train_groups list provided")
                return None
            # Take the first tuple in the list
            features, labels = train_groups[0]
        else:
            # Assume it's already a tuple of (features, labels)
            features, labels = train_groups
        
        # Initialize statistics dictionary
        statistics = {}
        
        # Process features
        if isinstance(features, np.ndarray) or isinstance(features, tf.Tensor):
            # Convert TensorFlow tensor to numpy if needed
            if isinstance(features, tf.Tensor):
                features_np = features.numpy()
            else:
                features_np = features
                
            # Determine feature keys
            if raw_data_keys is not None:
                # Use provided raw_data_keys
                feature_keys = raw_data_keys
            else:
                # Default feature keys from raw_data if available
                default_keys = [ 'z', 'y', 'x', 'time', 'permx']
                
                # Check if the number of channels matches default keys
                if features_np.ndim > 4 and features_np.shape[-1] == len(default_keys):
                    feature_keys = default_keys
                else:
                    # Generate sequential feature names based on channel count
                    num_channels = features_np.shape[-1] if features_np.ndim > 4 else 1
                    feature_keys = [f'feature_{i+1}' for i in range(num_channels)]
            
            # Calculate statistics for each feature dimension
            for i, key in enumerate(feature_keys):
                if features_np.ndim > 4:  # If features have channel dimension
                    # Extract the specific feature channel
                    feature_channel = features_np[..., i]
                    
                    # Calculate statistics
                    statistics[key] = {
                        'min': float(np.min(feature_channel)),
                        'max': float(np.max(feature_channel)),
                        'mean': float(np.mean(feature_channel)),
                        'std': float(np.std(feature_channel)),
                        'shape': list(feature_channel.shape)
                    }
                else:
                    logging.warning(f"Features don't have expected channel dimension. Shape: {features_np.shape}")
        else:
            # Handle dictionary of features
            for key, value in features.items():
                if isinstance(value, tf.Tensor):
                    value = value.numpy()
                    
                statistics[key] = {
                    'min': float(np.min(value)),
                    'max': float(np.max(value)),
                    'mean': float(np.mean(value)),
                    'std': float(np.std(value)),
                    'shape': list(value.shape)
                }
        
        # Process labels
        if isinstance(labels, np.ndarray) or isinstance(labels, tf.Tensor):
            # Convert TensorFlow tensor to numpy if needed
            if isinstance(labels, tf.Tensor):
                labels_np = labels.numpy()
            else:
                labels_np = labels
                
            # For labels, we'll use generic names if specific keys aren't provided
            if labels_np.ndim > 4:  # If labels have channel dimension
                for i in range(labels_np.shape[-1]):
                    label_key = f"label_{i}"
                    label_channel = labels_np[..., i]
                    
                    statistics[label_key] = {
                        'min': float(np.min(label_channel)),
                        'max': float(np.max(label_channel)),
                        'mean': float(np.mean(label_channel)),
                        'std': float(np.std(label_channel)),
                        'shape': list(label_channel.shape)
                    }
            else:
                statistics['label'] = {
                    'min': float(np.min(labels_np)),
                    'max': float(np.max(labels_np)),
                    'mean': float(np.mean(labels_np)),
                    'std': float(np.std(labels_np)),
                    'shape': list(labels_np.shape)
                }
        else:
            # Handle dictionary of labels
            for key, value in labels.items():
                if isinstance(value, tf.Tensor):
                    value = value.numpy()
                
                # Use the key directly (in lowercase) without adding 'label_' prefix
                statistics[key.lower()] = {
                    'min': float(np.min(value)),
                    'max': float(np.max(value)),
                    'mean': float(np.mean(value)),
                    'std': float(np.std(value)),
                    'shape': list(value.shape)
                }
        
        # Save statistics to JSON file
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=4)
            
        logging.info(f"Saved training statistics to {stats_path}")
        return statistics, stats_path
    
    def get_or_generate_training_data(self, train_config_hash=None, train_silent=True):
        import pickle
        """
        Checks if the training data file exists for the current KLE config. If not, runs SRMDataProcessor to generate it.
        Returns:
            training_data_path (str): Path to the training data .pkl file
            train_groups, val_groups, test_groups: Loaded training data
        """
        kle_folder = self.find_kle_folder()
        if train_config_hash is None:
            train_config_hash = self._generate_full_config_hash()[1]
        training_data_path = os.path.join(kle_folder, f"training_data_{train_config_hash}.pkl")

        if os.path.exists(training_data_path):
            if not train_silent:
                logging.info(f"Training data already exists at {training_data_path}. Loading...")
            with open(training_data_path, 'rb') as f:
                train_groups, val_groups, test_groups, pred_groups = pickle.load(f)
            return training_data_path, train_groups, val_groups, test_groups, pred_groups
        else:
            if not train_silent:
                logging.info(f"Training data not found at {training_data_path}. Generating with SRMDataProcessor...")
            # Generate and save training data
            train_groups, val_groups, test_groups, pred_groups, _, _ = self.process_data()
            # Save using the processor's method
            self.save_data_groups_list(
                [train_groups, val_groups, test_groups, pred_groups], train_config_hash, dtype=None
            )
            return training_data_path, train_groups, val_groups, test_groups, pred_groups
        
    def load_training_statistics(self, train_config_hash=None):
        """
        Load training statistics from a JSON file in the KLE folder.
        
        Args:
            kle_folder (str): Path to the KLE folder where statistics are saved
            train_data_name_hash (str, optional): Hash for the training data filename
            
        Returns:
            dict: Dictionary containing the training statistics
        """
        import json

        kle_folder = self.find_kle_folder()
        if train_config_hash is None:
            train_config_hash = self._generate_full_config_hash()[1]
        
        # Generate training statistics filename
        stats_filename = f"training_statistics_summary_{train_config_hash}.json"
        stats_path = os.path.join(kle_folder, stats_filename)
            
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Training statistics file not found: {stats_path}")
        
        # Load statistics from JSON file
        try:
            with open(stats_path, 'r') as f:
                statistics = json.load(f)
                
            logging.info(f"Loaded training statistics from {stats_path}")
            return statistics
        except Exception as e:
            logging.error(f"Error loading training statistics: {str(e)}")
            raise    

    @staticmethod
    def free_gpu_memory(variables_to_delete=None):
        """
        Attempts to free GPU memory by clearing TensorFlow and Python garbage collector resources.
        Optionally deletes provided variables. Note: TensorFlow may not fully release GPU memory until process exit.
        """
        import gc
        import tensorflow as tf
        import logging
        try:
            if variables_to_delete is not None:
                for var in variables_to_delete:
                    try:
                        del var
                    except Exception as e:
                        logging.debug(f"Could not delete variable: {e}")
            tf.keras.backend.clear_session()
            gc.collect()
            # For TensorFlow 2.x, try to enable memory growth (prevents full allocation)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as e:
                        logging.debug(f"Could not set memory growth for {gpu}: {e}")
            logging.info("[INFO] Attempted to clear GPU and Python memory. Note: TensorFlow may not fully release GPU memory until process exit.")
        except Exception as e:
            logging.warning(f"[WARN] Could not free GPU memory: {e}")
    
    

def validate_split_ratios(ratios):
    # Check for type1: a dictionary with integer keys and tuple/list values
    if isinstance(ratios, dict):
        for key, value in ratios.items():
            if not isinstance(key, int):
                raise ValueError(f"Invalid key type: {key} (expected int)")
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"Invalid value type for key {key}: {value} (expected tuple or list)")
        return  # Valid type1 format

    # Check for type2: a tuple or list
    elif isinstance(ratios, (tuple, list)):
        return  # Valid type2 format

    else:
        raise ValueError("Invalid format: ratios must be either a dict with int keys and tuple/list values, or a tuple/list.")

def load_srm_data(base_dir=None, apply_normalization=True, create_woven_tensor=True):
    """
    Convenience function to load SRM data from default configurations.
    
    Args:
        base_dir (str, optional): Base directory for data
        apply_normalization (bool): Whether to apply normalization
        create_woven_tensor (bool): Whether to create a woven tensor
        
    Returns:
        tuple: (train_groups, val_groups, test_groups) - Dictionary of tensors
    """
    processor = SRMDataProcessor(base_dir)
    return processor.process_data(apply_normalization, create_woven_tensor)


if __name__ == "__main__":
    # Example usage
    try:
        logging.info("Initializing SRM Data Processor...")
        processor = SRMDataProcessor()
        
        # Check if porosity realizations are available
        if processor.has_porosity:
            logging.info("Porosity realizations will be processed along with permeability.")
        else:
            logging.info("No porosity realizations found in configuration.")
        
        logging.info("\nLoading KLE Data...")
        kle_data = processor.load_kle_data()
        logging.info(f"KLE Data shape: {kle_data.shape}")
        
        logging.info("\nGenerating Time Tensor...")
        time_tensor = processor.generate_time_tensor()
        logging.info(f"Time Tensor shape: {time_tensor.shape}")
        logging.info(f"Time Tensor values: {time_tensor[:].flatten()}")
        
        logging.info("\nCreating Positional Tensors...")
        x_tensor, y_tensor, z_tensor = processor.create_positional_tensors()

        # Run the main data processing pipeline
        train_groups, val_groups, test_groups, pred_groups, _, _ = processor.process_data(apply_normalization=True, create_woven_tensor=True)

        logging.info("\nProcessing Complete! Data is ready for SRM processing.")
        # Print normalization config from default_configurations
        from default_configurations import DEFAULT_GENERAL_CONFIG
        norm_cfg = DEFAULT_GENERAL_CONFIG['data_normalization']
        logging.info(f"Normalization method: {norm_cfg['feature_normalization_method']}")
        logging.info(f"Normalization range: {norm_cfg['normalization_limits']}")
    except Exception as e:
        logging.error(f"Error in SRM data processing: {str(e)}")
        import traceback
        traceback.print_exc()
