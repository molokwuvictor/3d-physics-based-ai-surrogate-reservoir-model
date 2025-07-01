# -*- coding: utf-8 -*-
"""
Created on Tue 01 07 2025 14:57:18

@author: User
"""

import tensorflow as tf

def _normalize_mode_key(key):
    """Normalize control mode key by stripping and converting to uppercase."""
    return key.strip().upper()

def _normalize_type_key(key):
    """Normalize well type key by stripping and converting to lowercase."""
    return key.strip().lower()

class WellDataProcessor:
    """
    Processes and dynamically updates well definitions.
    Supports dynamic addition by rebuilding tensors on updates.
    Prevents duplicate coordinates in well_list.
    All floating-point operations use a specified dtype (default tf.float32).
    shutin_days is stored as a list of lists of lists: [[[float, float]], ...].
    """
    def __init__(self, well_list, mode_order=('k', 'j', 'i'), control_modes=('ORAT', 'GRAT', 'WRAT', 'BHP', 'LRAT'), dtype=tf.float32):
        self.mode_keys = [m.upper() for m in control_modes]
        self.mode_count = len(self.mode_keys)
        self.bhp_idx = self.mode_keys.index('BHP') if 'BHP' in self.mode_keys else None
        self.perm = [{'i': 0, 'j': 1, 'k': 2}[d] for d in mode_order]
        self.dtype = dtype
        # Python-side list
        self.well_list = list(well_list)
        # Initialize tensors
        self._rebuild_tensors()

    def _rebuild_tensors(self):
        """Recreate all tensors from current well_list."""
        coords = tf.constant([[w['i'], w['j'], w['k']] for w in self.well_list], dtype=tf.int32)
        self.connection_indices = tf.gather(coords, self.perm, axis=1)

        ctrl_rows = []
        signs = []
        names = []
        wellbore_radii = []
        completion_ratios = []
        minimum_bhp_values = []
        shutin_days_values = []
        for w in self.well_list:
            mode = _normalize_mode_key(w.get('control', ''))
            val = float(w.get('value', 0.0))
            ctrl_rows.append([val if m == mode else 0.0 for m in self.mode_keys])
            t = _normalize_type_key(w.get('type', ''))
            signs.append(1.0 if t == 'producer' else -1.0)
            names.append(w.get('name', ''))
            wellbore_radii.append(float(w.get('wellbore_radius', 0.0)))
            completion_ratios.append(float(w.get('completion_ratio', 0.0)))
            minimum_bhp_values.append(float(w.get('minimum_bhp', 0.0)))
            # Extract shutin_days as [[[float, float]], ...], default to [[0.0, 0.0]]
            shutin = w.get('shutin_days', [[0.0, 0.0]])
            if shutin and len(shutin) == 1 and len(shutin[0]) == 2:
                shutin_days_values.append([[float(shutin[0][0]), float(shutin[0][1])]])
            else:
                shutin_days_values.append([[0.0, 0.0]])

        self.control_matrix = tf.cast(tf.constant(ctrl_rows), dtype=self.dtype)
        self.type_sign = tf.cast(tf.constant(signs), dtype=self.dtype)
        self.names = tf.constant(names, dtype=tf.string)
        self.wellbore_radius = tf.cast(tf.constant(wellbore_radii), dtype=self.dtype)
        self.completion_ratio = tf.cast(tf.constant(completion_ratios), dtype=self.dtype)
        self.minimum_bhp = tf.cast(tf.constant(minimum_bhp_values), dtype=self.dtype)
        self.shutin_days = tf.cast(tf.constant(shutin_days_values), dtype=self.dtype)

    def get_well_data(self):
        """
        Returns dict:
            'connection_index': tf.Tensor [N,3]
            'control_mode_value': tf.Tensor [N]
            'names': tf.Tensor [N]
            'wellbore_radius': tf.Tensor [N]
            'completion_ratio': tf.Tensor [N]
            'minimum_bhp': tf.Tensor [N]
            'shutin_days': tf.Tensor [N,1,2]
        Applies sign rules, BHP always positive.
        """
        # Rebuild to catch any updates
        self._rebuild_tensors()

        raw = self.control_matrix
        signed = raw * tf.expand_dims(self.type_sign, 1)
        if self.bhp_idx is not None:
            idx = self.bhp_idx
            mask = tf.equal(tf.range(self.mode_count), idx)
            mask2d = tf.broadcast_to(tf.expand_dims(mask, 0), tf.shape(signed))
            signed = tf.reduce_sum(tf.where(mask2d, tf.abs(raw), signed), 1)
        else:
            signed = tf.reduce_sum(signed, 1)

        return {
            'connection_index': self.connection_indices,
            'control_mode_value': signed,
            'names': self.names,
            'wellbore_radius': self.wellbore_radius,
            'completion_ratio': self.completion_ratio,
            'minimum_bhp': self.minimum_bhp,
            'shutin_days': self.shutin_days
        }

    def update_control(self, well_idx, mode_key, new_value):
        """Updates control mode and value for a well at well_idx."""
        self.well_list[well_idx]['control'] = mode_key
        self.well_list[well_idx]['value'] = tf.cast(new_value, self.dtype).numpy()

    def update_type(self, well_idx, new_type):
        """Updates type for a well at well_idx."""
        self.well_list[well_idx]['type'] = new_type

    def update_shutin_days(self, well_idx, new_shutin_days):
        """Updates shutin_days for a well at well_idx."""
        # Validate new_shutin_days as [[float, float]]
        if not (isinstance(new_shutin_days, (list, tuple)) and len(new_shutin_days) == 1 and isinstance(new_shutin_days[0], (list, tuple)) and len(new_shutin_days[0]) == 2):
            raise ValueError("new_shutin_days must be a list of one list of 2 floats, e.g., [[1000.0, 0.0]]")
        shutin = [[float(new_shutin_days[0][0]), float(new_shutin_days[0][1])]]
        self.well_list[well_idx]['shutin_days'] = shutin

    def update_well_list(self, new_wells):
        """Updates or adds wells based on coordinates (i, j, k)."""
        # Create a dictionary mapping coordinates to well indices
        coord_to_index = {(w['i'], w['j'], w['k']): idx for idx, w in enumerate(self.well_list)}
        
        for w in new_wells:
            coord_key = (w['i'], w['j'], w['k'])
            shutin = w.get('shutin_days', [[0.0, 0.0]])
            if not (shutin and len(shutin) == 1 and len(shutin[0]) == 2):
                shutin = [[0.0, 0.0]]
            shutin = [[float(shutin[0][0]), float(shutin[0][1])]]
            if coord_key in coord_to_index:
                # Update existing well
                idx = coord_to_index[coord_key]
                self.well_list[idx].update({
                    'name': w.get('name', ''),
                    'i': w['i'],
                    'j': w['j'],
                    'k': w['k'],
                    'type': w.get('type', ''),
                    'control': w.get('control', ''),
                    'value': tf.cast(w.get('value', 0.0), self.dtype).numpy(),
                    'wellbore_radius': tf.cast(w.get('wellbore_radius', 0.0), self.dtype).numpy(),
                    'completion_ratio': tf.cast(w.get('completion_ratio', 0.0), self.dtype).numpy(),
                    'minimum_bhp': tf.cast(w.get('minimum_bhp', 0.0), self.dtype).numpy(),
                    'shutin_days': shutin
                })
            else:
                # Add new well
                self.well_list.append({
                    'name': w.get('name', ''),
                    'i': w['i'],
                    'j': w['j'],
                    'k': w['k'],
                    'type': w.get('type', ''),
                    'control': w.get('control', ''),
                    'value': tf.cast(w.get('value', 0.0), self.dtype).numpy(),
                    'wellbore_radius': tf.cast(w.get('wellbore_radius', 0.0), self.dtype).numpy(),
                    'completion_ratio': tf.cast(w.get('completion_ratio', 0.0), self.dtype).numpy(),
                    'minimum_bhp': tf.cast(w.get('minimum_bhp', 0.0), self.dtype).numpy(),
                    'shutin_days': shutin
                })

    @tf.function
    def scatter_y(self, target_shape, index_list, y, start_dim=1):
        """
        Scatter values into an N‑dimensional TensorFlow tensor using pure graph ops.

        This function builds a tensor of zeros with shape `target_shape` and places
        values from `y` at specified coordinate indices, supporting arbitrary
        outer and inner dimensions around the indexed axes.

        Args:
            target_shape (Sequence[int]): Desired output shape, e.g. [B, C, H, W, ...].
            index_list (Sequence[Tuple[int, ...]]): List of index tuples of length k,
                referring to positions along dimensions [start_dim, start_dim+k).
            y (Union[float, Sequence[float]]): A scalar or 1D sequence of length N
                (number of indices). If scalar, that value is broadcast to all indices.
            start_dim (int): The first dimension in `target_shape` at which indices apply.
                Dimensions [0, ..., start_dim-1] are treated as outer dims.

        Returns:
            tf.Tensor: A tensor of shape `target_shape` with `y` scattered into zeros.

        Example:
            # Scatter control_mode_value into a 5D tensor
            well_data = self.get_well_data()
            target_shape = (1, 1, 39, 39, 1)
            result = self.scatter_y(target_shape, well_data['connection_index'], well_data['control_mode_value'])
        """
        # Convert shape and indices to tensors
        target_shape = tf.convert_to_tensor(target_shape, dtype=tf.int32)
        rank = tf.size(target_shape)  # total number of dimensions

        indices = tf.convert_to_tensor(index_list, dtype=tf.int32)
        num_indices = tf.shape(indices)[0]  # N
        k = tf.shape(indices)[1]           # length of each index tuple

        # Build zero-padding for dimensions before and after the indexed dims
        outer_pad = tf.zeros([num_indices, start_dim], dtype=tf.int32)
        inner_pad_len = rank - (start_dim + k)
        inner_pad = tf.zeros([num_indices, inner_pad_len], dtype=tf.int32)

        # Concatenate pads and actual indices into full N-D index tuples
        full_indices = tf.concat([outer_pad, indices, inner_pad], axis=1)

        # Prepare the updates: broadcast scalar or align vector
        y_tensor = tf.reshape(tf.convert_to_tensor(y, dtype=self.dtype), [-1])  # shape [m]
        ones = tf.ones([num_indices], dtype=self.dtype)
        candidate = ones * y_tensor
        updates = tf.where(
            tf.equal(tf.size(y_tensor), 1),  # if y is scalar
            ones * y_tensor[0],               # broadcast that scalar
            candidate                         # otherwise per-index values
        )

        # Perform scatter_nd into a zero tensor
        scattered = tf.scatter_nd(full_indices, updates, tf.cast(target_shape, tf.int32))
        return tf.cast(scattered, self.dtype)
    
    # Multi-Dimensional Tensors
    @tf.function
    def conn_shutins_idx(self, time_tensor, index_list, range_conditions, time_axis=0):
        """
        Parameters:
          time_tensor: tf.Tensor with shape [*outer_dims, T, C, H, W, *inner_dims].
          index_list: List of tuples in (channel, height, width) format; each index refers to a spatial cell.
          range_conditions: A list (per spatial index) where each element is a list of [start, stop] pairs.
                            For example: 
                               [
                                 [[50, 200], [300, 400]],      # For first index
                                 [[200, 500], [600, 800]],      # For second index
                                 ...  
                               ]
                             Each condition pair is applied to the values at the corresponding spatial cell.
    
                            For any spatial index, if the representative value does not fall in any range, update is 1.
          time_axis: Integer that points to the T (time) dimension in the input tensor.
    
        Returns:
          A binary tensor (of tf.int32) with the same shape as time_tensor. For a given spatial location (from index_list),
          at every time step the cell remains 0 if the (representative) value is in any range; otherwise it is set to 1.
        """
        # Get the full dynamic shape.
        input_shape = tf.shape(time_tensor)  # shape: [*outer_dims, T, C, H, W, *inner_dims]
        rank = tf.size(input_shape)
    
        # Assume time_axis is valid. Split shape into:
        # outer_dims = dims before time_axis.
        # T = dimension at time_axis.
        # Spatial dims: next three dims (C, H, W).
        # inner_dims: all dims after that.
        outer_dims = input_shape[:time_axis]              # shape: [n1, n2, ...]
        T = input_shape[time_axis]                        # scalar T
        spatial_dims = input_shape[time_axis+1: time_axis+4]  # [C, H, W]
        inner_dims = input_shape[time_axis+4:]             # [*inner_dims] (if none, will be empty)
    
        # Compute the product of the outer dimensions.
        outer_prod = tf.reduce_prod(outer_dims)
        # Compute product of inner dimensions; if none, set inner_prod to 1.
        inner_prod = tf.cond(tf.equal(tf.size(inner_dims), 0),
                             lambda: tf.constant(1, dtype=tf.int32),
                             lambda: tf.reduce_prod(inner_dims))
        
        # Unpack spatial dims.
        C = spatial_dims[0]
        H = spatial_dims[1]
        W = spatial_dims[2]
    
        # Reshape the tensor:
        # New batch: outer_prod * T
        # Keep spatial channels C and H as is.
        # Merge W and inner dims: new_W = W * inner_prod.
        new_batch = outer_prod * T
        new_W = W * inner_prod
        reshaped_tensor = tf.reshape(time_tensor, tf.stack([new_batch, C, H, new_W]))
    
        # Initialize the binary tensor (all zeros) with same shape as reshaped_tensor.
        binary_tensor = tf.zeros_like(reshaped_tensor, dtype=tf.int32)
    
        # Convert index_list and range_conditions into constant tensors.
        # index_list is expected to have shape [num_indices, 3]
        spatial_indices = tf.convert_to_tensor(index_list, dtype=tf.int32)
        range_conditions_tensor = tf.convert_to_tensor(range_conditions, dtype=tf.float32)
    
        # Validate shapes.
        tf.debugging.assert_shapes([(spatial_indices, (None, 3))])
        tf.debugging.assert_shapes([(range_conditions_tensor, (None, None, 2))])
        tf.debugging.assert_equal(tf.shape(spatial_indices)[0], tf.shape(range_conditions_tensor)[0],
                                  message="index_list and range_conditions must have the same number of indices")
        num_indices = tf.shape(spatial_indices)[0]  # number of indices
        # shape: [num_indices, num_conditions, 2]
    
        # ----------------------------------------------------------------------
        # For each spatial index we update the binary tensor.
        # Because we are not allowed to use tf.range, we create a sequence for the batch dimension
        # by using cumulative sum on a ones vector.
        batch_size_new = tf.shape(reshaped_tensor)[0]
        ones_batch = tf.ones([batch_size_new], dtype=tf.int32)
        # This produces a tensor [0, 1, 2, ..., batch_size_new-1]
        batch_idx = tf.cast(tf.math.cumsum(ones_batch) - 1, tf.int32)  # shape: [batch_size_new]
    
        # Expand batch indices for each spatial index.
        batch_idx_exp = tf.reshape(batch_idx, [-1, 1])             # shape: [batch_size_new, 1]
        batch_idx_tiled = tf.tile(batch_idx_exp, [1, num_indices])   # shape: [batch_size_new, num_indices]
    
        # Expand the spatial_indices so that they are repeated along the batch dimension.
        spatial_indices_exp = tf.reshape(spatial_indices, [1, num_indices, 3])  # shape: [1, num_indices, 3]
        spatial_indices_tiled = tf.tile(spatial_indices_exp, [batch_size_new, 1, 1])  # shape: [batch_size_new, num_indices, 3]
    
        # For each spatial index, extract the (channel, height, original-width) values.
        # Later, we will update the corresponding block along the last dimension.
        c_val = spatial_indices_tiled[..., 0]  # shape: [batch_size_new, num_indices]
        h_val = spatial_indices_tiled[..., 1]  # shape: [batch_size_new, num_indices]
        w_orig = spatial_indices_tiled[..., 2] # shape: [batch_size_new, num_indices]
        
        # For each provided (w_orig), compute the starting column in the reshaped tensor.
        # Each block corresponds to inner_prod contiguous columns.
        block_start = w_orig * inner_prod     # shape: [batch_size_new, num_indices]
    
        # Now build the full indices for the representative element of each block.
        # We will use the first column in each block as the representative value.
        # The indices are: [batch, channel, height, block_start]
        full_indices = tf.stack([batch_idx_tiled, c_val, h_val, block_start], axis=-1)  # shape: [batch_size_new, num_indices, 4]
        full_indices_reshaped = tf.reshape(full_indices, [-1, 4])  # shape: [batch_size_new * num_indices, 4]
    
        # Gather representative values from reshaped_tensor.
        values = tf.gather_nd(reshaped_tensor, full_indices_reshaped)  # shape: [batch_size_new * num_indices]
        values = tf.reshape(values, [batch_size_new, num_indices])       # shape: [batch_size_new, num_indices]
    
        # Expand values for comparison.
        values_expanded = tf.expand_dims(values, axis=-1)  # shape: [batch_size_new, num_indices, 1]
    
        # Tile the range conditions across the batch dimension.
        range_conditions_exp = tf.expand_dims(range_conditions_tensor, axis=0)
        range_conditions_exp = tf.tile(range_conditions_exp, [batch_size_new, 1, 1, 1])  
        # now shape: [batch_size_new, num_indices, num_conditions, 2]
    
        # Split the start and stop values.
        start_values = range_conditions_exp[..., 0]  # shape: [batch_size_new, num_indices, num_conditions]
        stop_values  = range_conditions_exp[..., 1]  # shape: [batch_size_new, num_indices, num_conditions]
    
        # Check conditions: For each condition, compare whether the value lies within [start, stop].
        condition_check = tf.logical_and(values_expanded >= start_values, values_expanded <= stop_values)
        # If any condition is met for a given batch and index, consider it satisfied.
        condition_satisfied = tf.reduce_any(condition_check, axis=-1)  # shape: [batch_size_new, num_indices]
        
        # Determine update values: if condition is NOT satisfied, we want to mark that location as 1.
        updates = tf.cast(tf.logical_not(condition_satisfied), tf.int32)  # shape: [batch_size_new, num_indices]
    
        # --- Now update binary_tensor over the entire block.
        # For every spatial index, the update must occur for the entire block of size [inner_prod] along the last axis.
        # First, create a block of update values for each spatial index (by repeating the scalar update across inner_prod columns).
        updates_block = tf.tile(tf.expand_dims(updates, axis=-1), [1, 1, inner_prod])  # shape: [batch_size_new, num_indices, inner_prod]
        updates_block_flat = tf.reshape(updates_block, [-1])  # flatten updates
    
        # Next, we must construct full indices for every element in each block.
        # For the inner block offsets, we again create a [0, 1, ..., inner_prod-1] sequence by using cumulative sum.
        ones_inner = tf.ones([inner_prod], dtype=tf.int32)
        inner_offset = tf.cast(tf.math.cumsum(ones_inner) - 1, tf.int32)  # shape: [inner_prod]
        # Tile this offset so that it applies to every (batch, index) pair.
        inner_offset_tiled = tf.tile(tf.reshape(inner_offset, [1, 1, inner_prod]), [batch_size_new, num_indices, 1])
        # The final column index is the base block_start plus the inner offset.
        col_indices = tf.expand_dims(block_start, axis=-1) + inner_offset_tiled  # shape: [batch_size_new, num_indices, inner_prod]
    
        # Prepare the other indices (batch, channel, height) for the block.
        batch_indices_block = tf.tile(tf.expand_dims(batch_idx_tiled, axis=-1), [1, 1, inner_prod])
        c_block = tf.tile(tf.expand_dims(c_val, axis=-1), [1, 1, inner_prod])
        h_block = tf.tile(tf.expand_dims(h_val, axis=-1), [1, 1, inner_prod])
        
        # Stack these indices together to form the full indices for every element in each update block.
        full_indices_block = tf.stack([batch_indices_block, c_block, h_block, col_indices], axis=-1)  
        # Shape: [batch_size_new, num_indices, inner_prod, 4]
        full_indices_block_reshaped = tf.reshape(full_indices_block, [-1, 4])
        
        # Finally, update the binary_tensor (which is a tensor of zeros) using tensor_scatter_nd_update.
        binary_tensor_updated = tf.tensor_scatter_nd_update(binary_tensor, full_indices_block_reshaped, updates_block_flat)
        
        # Reshape the updated tensor back to the original shape:
        # [ *outer_dims, T, C, H, W, *inner_dims ]
        # (Note: if there were no inner dims originally, inner_prod is 1.)
        original_shape = tf.concat([outer_dims, [T, C, H, W], inner_dims], axis=0)
        updated_tensor = tf.reshape(binary_tensor_updated, original_shape)
        return updated_tensor

class WellDataProcessorStaticMode:
    """
    Processes and dynamically updates well definitions using pure TensorFlow.
    Supports a fixed maximum number of wells, with tensors preallocated at initialization.
    All floating-point operations use a specified dtype (default tf.float32).
    All updates are performed in TensorFlow graph mode.
    shutin_days is stored as a list of lists of lists: [[[float, float]], ...].
    """
    def __init__(self, well_list, max_wells=10, mode_order=('k', 'j', 'i'), control_modes=('ORAT', 'GRAT', 'WRAT', 'BHP', 'LRAT'), dtype=tf.float32):
        self.max_wells = max_wells
        self.dtype = dtype
        self.mode_keys = [m.upper() for m in control_modes]
        self.mode_count = len(self.mode_keys)
        self.bhp_idx = self.mode_keys.index('BHP') if 'BHP' in self.mode_keys else None
        self.perm = tf.constant([{'i': 0, 'j': 1, 'k': 2}[d] for d in mode_order], dtype=tf.int32)

        # Lookup tables for mode and type conversion
        self.mode_key_to_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(self.mode_keys),
                values=tf.range(len(self.mode_keys), dtype=tf.int32)
            ),
            default_value=-1
        )
        self.type_to_sign = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(['producer', 'injector']),
                values=tf.constant([1.0, -1.0], dtype=self.dtype)
            ),
            default_value=tf.constant(0.0, dtype=self.dtype)
        )

        # Preallocate TensorFlow variables
        self.coordinates = tf.Variable(tf.zeros([max_wells, 3], dtype=tf.int32), trainable=False)
        self.control_mode_indices = tf.Variable(tf.zeros([max_wells], dtype=tf.int32), trainable=False)
        self.control_values = tf.Variable(tf.zeros([max_wells], dtype=self.dtype), trainable=False)
        self.types = tf.Variable(tf.zeros([max_wells], dtype=self.dtype), trainable=False)
        self.names = tf.Variable([''] * max_wells, dtype=tf.string, trainable=False)
        self.wellbore_radius = tf.Variable(tf.zeros([max_wells], dtype=self.dtype), trainable=False)
        self.completion_ratio = tf.Variable(tf.zeros([max_wells], dtype=self.dtype), trainable=False)
        self.minimum_bhp = tf.Variable(tf.zeros([max_wells], dtype=self.dtype), trainable=False)
        self.shutin_days = tf.Variable(tf.zeros([max_wells, 1, 2], dtype=self.dtype), trainable=False)
        self.N_current = tf.Variable(0, dtype=tf.int32, trainable=False)

        # Initialize with provided well_list
        if well_list:
            num_wells = len(well_list)
            if num_wells > max_wells:
                raise ValueError(f"Initial wells {num_wells} exceed max_wells {max_wells}")

            coords = []
            control_modes = []
            ctrl_values = []
            types = []
            names = []
            wellbore_radii = []
            completion_ratios = []
            minimum_bhp_values = []
            shutin_days_values = []
            for well in well_list:
                coords.append([well['i'], well['j'], well['k']])
                control_modes.append(_normalize_mode_key(well.get('control', '')))
                ctrl_values.append(float(well.get('value', 0.0)))
                types.append(_normalize_type_key(well.get('type', '')))
                names.append(well.get('name', ''))
                wellbore_radii.append(float(well.get('wellbore_radius', 0.0)))
                completion_ratios.append(float(well.get('completion_ratio', 0.0)))
                minimum_bhp_values.append(float(well.get('minimum_bhp', 0.0)))
                # Extract shutin_days as [[[float, float]], ...], default to [[0.0, 0.0]]
                shutin = well.get('shutin_days', [[0.0, 0.0]])
                if shutin and len(shutin) == 1 and len(shutin[0]) == 2:
                    shutin_days_values.append([[float(shutin[0][0]), float(shutin[0][1])]])
                else:
                    shutin_days_values.append([[0.0, 0.0]])

            # Convert to tensors with specified dtype
            coords = tf.constant(coords, dtype=tf.int32)
            control_modes = tf.constant(control_modes, dtype=tf.string)
            ctrl_values = tf.cast(tf.constant(ctrl_values), dtype=self.dtype)
            types = tf.constant(types, dtype=tf.string)
            names = tf.constant(names, dtype=tf.string)
            wellbore_radii = tf.cast(tf.constant(wellbore_radii), dtype=self.dtype)
            completion_ratios = tf.cast(tf.constant(completion_ratios), dtype=self.dtype)
            minimum_bhp_values = tf.cast(tf.constant(minimum_bhp_values), dtype=self.dtype)
            shutin_days_values = tf.cast(tf.constant(shutin_days_values), dtype=self.dtype)

            # Validate and map using lookup tables
            mode_indices = self.mode_key_to_index.lookup(control_modes)
            tf.debugging.assert_none_equal(mode_indices, -1, message="Invalid control mode in initial well_list")
            type_signs = self.type_to_sign.lookup(types)
            tf.debugging.assert_none_equal(type_signs, tf.constant(0.0, dtype=self.dtype), message="Invalid type in initial well_list")

            # Assign to variables
            self.coordinates[:num_wells].assign(coords)
            self.control_mode_indices[:num_wells].assign(mode_indices)
            self.control_values[:num_wells].assign(ctrl_values)
            self.types[:num_wells].assign(type_signs)
            self.names[:num_wells].assign(names)
            self.wellbore_radius[:num_wells].assign(wellbore_radii)
            self.completion_ratio[:num_wells].assign(completion_ratios)
            self.minimum_bhp[:num_wells].assign(minimum_bhp_values)
            self.shutin_days[:num_wells].assign(shutin_days_values)
            self.N_current.assign(num_wells)

    @tf.function
    def get_well_data(self):
        """
        Returns dict:
            'connection_index': tf.Tensor [N,3]
            'control_mode_value': tf.Tensor [N]
            'names': tf.Tensor [N]
            'wellbore_radius': tf.Tensor [N]
            'completion_ratio': tf.Tensor [N]
            'minimum_bhp': tf.Tensor [N]
            'shutin_days': tf.Tensor [N,1,2]
        Applies sign rules, BHP always positive.
        """
        N = self.N_current
        coords = self.coordinates[:N]
        connection_indices = tf.gather(coords, self.perm, axis=1)

        # Generate control matrix from indices and values
        mode_one_hot = tf.one_hot(self.control_mode_indices[:N], self.mode_count, dtype=self.dtype)
        ctrl_vals = tf.expand_dims(self.control_values[:N], 1)
        control_matrix = mode_one_hot * ctrl_vals

        # Apply type signs
        signs = self.types[:N]
        signed_control = control_matrix * tf.expand_dims(signs, 1)

        # Handle BHP (absolute value for BHP column)
        if self.bhp_idx is not None:
            bhp_mask = tf.equal(tf.range(self.mode_count), self.bhp_idx)
            bhp_mask_2d = tf.broadcast_to(tf.expand_dims(bhp_mask, 0), tf.shape(signed_control))
            signed_control = tf.where(bhp_mask_2d, tf.abs(control_matrix), signed_control)

        return {
            'connection_index': connection_indices,
            'control_mode_value': tf.reduce_sum(signed_control, 1),
            'names': self.names[:N],
            'wellbore_radius': self.wellbore_radius[:N],
            'completion_ratio': self.completion_ratio[:N],
            'minimum_bhp': self.minimum_bhp[:N],
            'shutin_days': self.shutin_days[:N]
        }

    @tf.function
    def update_control(self, well_idx, mode_key, new_value):
        """Updates control mode and value for a well at well_idx."""
        well_idx = tf.cast(well_idx, tf.int32)
        mode_key = tf.strings.strip(tf.strings.upper(mode_key))
        mode_idx = self.mode_key_to_index.lookup(mode_key)
        tf.debugging.assert_greater_equal(mode_idx, 0, message="Invalid control mode")

        # Update mode and value
        self.control_mode_indices.scatter_nd_update([[well_idx]], [mode_idx])
        self.control_values.scatter_nd_update([[well_idx]], [tf.cast(new_value, self.dtype)])

    @tf.function
    def update_type(self, well_idx, new_type):
        """Updates type for a well at well_idx."""
        well_idx = tf.cast(well_idx, tf.int32)
        new_type = tf.strings.strip(tf.strings.lower(new_type))
        sign = self.type_to_sign.lookup(new_type)
        tf.debugging.assert_none_equal(sign, tf.constant(0.0, dtype=self.dtype), message="Invalid type")

        self.types.scatter_nd_update([[well_idx]], [sign])

    @tf.function
    def update_shutin_days(self, well_idx, new_shutin_days):
        """Updates shutin_days for a well at well_idx."""
        well_idx = tf.cast(well_idx, tf.int32)
        new_shutin = tf.cast(new_shutin_days, self.dtype)
        tf.debugging.assert_equal(tf.shape(new_shutin), [1, 2], message="new_shutin_days must be a tensor of shape [1, 2]")
        self.shutin_days.scatter_nd_update([[well_idx]], [new_shutin])

    @tf.function
    def update_well_list(self, new_wells):
        """Adds or updates wells from a list of dictionaries."""
        # Extract fields from new_wells in eager mode
        with tf.init_scope():
            num_new = len(new_wells)
            current_N = self.N_current.numpy()
            coords = []
            control_modes = []
            ctrl_values = []
            types = []
            names = []
            wellbore_radii = []
            completion_ratios = []
            minimum_bhp_values = []
            shutin_days_values = []
            for well in new_wells:
                coords.append([well['i'], well['j'], well['k']])
                control_modes.append(_normalize_mode_key(well.get('control', '')))
                ctrl_values.append(float(well.get('value', 0.0)))
                types.append(_normalize_type_key(well.get('type', '')))
                names.append(well.get('name', ''))
                wellbore_radii.append(float(well.get('wellbore_radius', 0.0)))
                completion_ratios.append(float(well.get('completion_ratio', 0.0)))
                minimum_bhp_values.append(float(well.get('minimum_bhp', 0.0)))
                # Extract shutin_days as [[[float, float]], ...], default to [[0.0, 0.0]]
                shutin = well.get('shutin_days', [[0.0, 0.0]])
                if shutin and len(shutin) == 1 and len(shutin[0]) == 2:
                    shutin_days_values.append([[float(shutin[0][0]), float(shutin[0][1])]])
                else:
                    shutin_days_values.append([[0.0, 0.0]])

            # Convert to tensors with specified dtype
            new_coords = tf.constant(coords, dtype=tf.int32)
            new_modes = tf.constant(control_modes, dtype=tf.string)
            new_values = tf.cast(tf.constant(ctrl_values), dtype=self.dtype)
            new_types = tf.constant(types, dtype=tf.string)
            new_names = tf.constant(names, dtype=tf.string)
            new_wellbore_radii = tf.cast(tf.constant(wellbore_radii), dtype=self.dtype)
            new_completion_ratios = tf.cast(tf.constant(completion_ratios), dtype=self.dtype)
            new_minimum_bhp_values = tf.cast(tf.constant(minimum_bhp_values), dtype=self.dtype)
            new_shutin_days = tf.cast(tf.constant(shutin_days_values), dtype=self.dtype)

        # Validate inputs
        new_mode_indices = self.mode_key_to_index.lookup(new_modes)
        tf.debugging.assert_greater_equal(tf.reduce_min(new_mode_indices), 0, message="Invalid control mode in new_wells")
        new_signs = self.type_to_sign.lookup(new_types)
        tf.debugging.assert_none_equal(new_signs, tf.constant(0.0, dtype=self.dtype), message="Invalid type in new_wells")

        # Check for matching coordinates
        existing_coords = self.coordinates[:current_N]
        matches = tf.reduce_all(tf.equal(existing_coords[:, tf.newaxis, :], new_coords[tf.newaxis, :, :]), axis=2)
        has_match = tf.reduce_any(matches, axis=0)
        matching_indices = tf.argmax(matches, axis=0, output_type=tf.int32)

        # Compute number of wells to add (those without matches)
        num_adds = tf.reduce_sum(tf.cast(~has_match, tf.int32))
        tf.debugging.assert_less_equal(current_N + num_adds, tf.constant(self.max_wells, dtype=tf.int32), message="Exceeds max_wells after additions")

        # Update existing wells where coordinates match
        update_indices_j = tf.where(has_match)[:, 0]
        update_indices_i = tf.gather(matching_indices, update_indices_j)
        update_mode_indices = tf.gather(new_mode_indices, update_indices_j)
        update_values = tf.gather(new_values, update_indices_j)
        update_types = tf.gather(new_signs, update_indices_j)
        update_names = tf.gather(new_names, update_indices_j)
        update_wellbore_radii = tf.gather(new_wellbore_radii, update_indices_j)
        update_completion_ratios = tf.gather(new_completion_ratios, update_indices_j)
        update_minimum_bhp_values = tf.gather(new_minimum_bhp_values, update_indices_j)
        update_shutin_days = tf.gather(new_shutin_days, update_indices_j)
        self.control_mode_indices.scatter_nd_update(tf.expand_dims(update_indices_i, 1), update_mode_indices)
        self.control_values.scatter_nd_update(tf.expand_dims(update_indices_i, 1), update_values)
        self.types.scatter_nd_update(tf.expand_dims(update_indices_i, 1), update_types)
        self.names.scatter_nd_update(tf.expand_dims(update_indices_i, 1), update_names)
        self.wellbore_radius.scatter_nd_update(tf.expand_dims(update_indices_i, 1), update_wellbore_radii)
        self.completion_ratio.scatter_nd_update(tf.expand_dims(update_indices_i, 1), update_completion_ratios)
        self.minimum_bhp.scatter_nd_update(tf.expand_dims(update_indices_i, 1), update_minimum_bhp_values)
        self.shutin_days.scatter_nd_update(tf.expand_dims(update_indices_i, 1), update_shutin_days)

        # Add new wells where coordinates do not match
        add_indices_j = tf.where(~has_match)[:, 0]
        add_indices = tf.range(current_N, current_N + num_adds)
        add_coords = tf.gather(new_coords, add_indices_j)
        add_mode_indices = tf.gather(new_mode_indices, add_indices_j)
        add_values = tf.gather(new_values, add_indices_j)
        add_types = tf.gather(new_signs, add_indices_j)
        add_names = tf.gather(new_names, add_indices_j)
        add_wellbore_radii = tf.gather(new_wellbore_radii, add_indices_j)
        add_completion_ratios = tf.gather(new_completion_ratios, add_indices_j)
        add_minimum_bhp_values = tf.gather(new_minimum_bhp_values, add_indices_j)
        add_shutin_days = tf.gather(new_shutin_days, add_indices_j)
        self.coordinates.scatter_nd_update(tf.expand_dims(add_indices, 1), add_coords)
        self.control_mode_indices.scatter_nd_update(tf.expand_dims(add_indices, 1), add_mode_indices)
        self.control_values.scatter_nd_update(tf.expand_dims(add_indices, 1), add_values)
        self.types.scatter_nd_update(tf.expand_dims(add_indices, 1), add_types)
        self.names.scatter_nd_update(tf.expand_dims(add_indices, 1), add_names)
        self.wellbore_radius.scatter_nd_update(tf.expand_dims(add_indices, 1), add_wellbore_radii)
        self.completion_ratio.scatter_nd_update(tf.expand_dims(add_indices, 1), add_completion_ratios)
        self.minimum_bhp.scatter_nd_update(tf.expand_dims(add_indices, 1), add_minimum_bhp_values)
        self.shutin_days.scatter_nd_update(tf.expand_dims(add_indices, 1), add_shutin_days)
        self.N_current.assign(current_N + num_adds)

    @tf.function
    def scatter_y(self, target_shape, index_list, y, start_dim=1):
        """
        Scatter values into an N‑dimensional TensorFlow tensor using pure graph ops.

        This function builds a tensor of zeros with shape `target_shape` and places
        values from `y` at specified coordinate indices, supporting arbitrary
        outer and inner dimensions around the indexed axes.

        Args:
            target_shape (Sequence[int]): Desired output shape, e.g. [B, C, H, W, ...].
            index_list (Sequence[Tuple[int, ...]]): List of index tuples of length k,
                referring to positions along dimensions [start_dim, start_dim+k).
            y (Union[float, Sequence[float]]): A scalar or 1D sequence of length N
                (number of indices). If scalar, that value is broadcast to all indices.
            start_dim (int): The first dimension in `target_shape` at which indices apply.
                Dimensions [0, ..., start_dim-1] are treated as outer dims.

        Returns:
            tf.Tensor: A tensor of shape `target_shape` with `y` scattered into zeros.

        Example:
            # Scatter control_mode_value into a 5D tensor
            well_data = self.get_well_data()
            target_shape = (1, 1, 39, 39, 1)
            result = self.scatter_y(target_shape, well_data['connection_index'], well_data['control_mode_value'])
        """
        # Convert shape and indices to tensors
        target_shape = tf.convert_to_tensor(target_shape, dtype=tf.int32)
        rank = tf.size(target_shape)  # total number of dimensions

        indices = tf.convert_to_tensor(index_list, dtype=tf.int32)
        num_indices = tf.shape(indices)[0]  # N
        k = tf.shape(indices)[1]           # length of each index tuple

        # Build zero-padding for dimensions before and after the indexed dims
        outer_pad = tf.zeros([num_indices, start_dim], dtype=tf.int32)
        inner_pad_len = rank - (start_dim + k)
        inner_pad = tf.zeros([num_indices, inner_pad_len], dtype=tf.int32)

        # Concatenate pads and actual indices into full N-D index tuples
        full_indices = tf.concat([outer_pad, indices, inner_pad], axis=1)

        # Prepare the updates: broadcast scalar or align vector
        y_tensor = tf.reshape(tf.convert_to_tensor(y, dtype=self.dtype), [-1])  # shape [m]
        ones = tf.ones([num_indices], dtype=self.dtype)
        candidate = ones * y_tensor
        updates = tf.where(
            tf.equal(tf.size(y_tensor), 1),  # if y is scalar
            ones * y_tensor[0],               # broadcast that scalar
            candidate                         # otherwise per-index values
        )

        # Perform scatter_nd into a zero tensor
        scattered = tf.scatter_nd(full_indices, updates, tf.cast(target_shape, tf.int32))
        return tf.cast(scattered, self.dtype)
    
    # Multi-Dimensional Tensors
    @tf.function
    def conn_shutins_idx(self, time_tensor, index_list, range_conditions, time_axis=0):
        """
        Parameters:
          time_tensor: tf.Tensor with shape [*outer_dims, T, C, H, W, *inner_dims].
          index_list: List of tuples in (channel, height, width) format; each index refers to a spatial cell.
          range_conditions: A list (per spatial index) where each element is a list of [start, stop] pairs.
                            For example: 
                               [
                                 [[50, 200], [300, 400]],      # For first index
                                 [[200, 500], [600, 800]],      # For second index
                                 ...  
                               ]
                             Each condition pair is applied to the values at the corresponding spatial cell.
    
                            For any spatial index, if the representative value does not fall in any range, update is 1.
          time_axis: Integer that points to the T (time) dimension in the input tensor.
    
        Returns:
          A binary tensor (of tf.int32) with the same shape as time_tensor. For a given spatial location (from index_list),
          at every time step the cell remains 0 if the (representative) value is in any range; otherwise it is set to 1.
        """
        # Get the full dynamic shape.
        input_shape = tf.shape(time_tensor)  # shape: [*outer_dims, T, C, H, W, *inner_dims]
        rank = tf.size(input_shape)
    
        # Assume time_axis is valid. Split shape into:
        # outer_dims = dims before time_axis.
        # T = dimension at time_axis.
        # Spatial dims: next three dims (C, H, W).
        # inner_dims: all dims after that.
        outer_dims = input_shape[:time_axis]              # shape: [n1, n2, ...]
        T = input_shape[time_axis]                        # scalar T
        spatial_dims = input_shape[time_axis+1: time_axis+4]  # [C, H, W]
        inner_dims = input_shape[time_axis+4:]             # [*inner_dims] (if none, will be empty)
    
        # Compute the product of the outer dimensions.
        outer_prod = tf.reduce_prod(outer_dims)
        # Compute product of inner dimensions; if none, set inner_prod to 1.
        inner_prod = tf.cond(tf.equal(tf.size(inner_dims), 0),
                             lambda: tf.constant(1, dtype=tf.int32),
                             lambda: tf.reduce_prod(inner_dims))
        
        # Unpack spatial dims.
        C = spatial_dims[0]
        H = spatial_dims[1]
        W = spatial_dims[2]
    
        # Reshape the tensor:
        # New batch: outer_prod * T
        # Keep spatial channels C and H as is.
        # Merge W and inner dims: new_W = W * inner_prod.
        new_batch = outer_prod * T
        new_W = W * inner_prod
        reshaped_tensor = tf.reshape(time_tensor, tf.stack([new_batch, C, H, new_W]))
    
        # Initialize the binary tensor (all zeros) with same shape as reshaped_tensor.
        binary_tensor = tf.zeros_like(reshaped_tensor, dtype=tf.int32)
    
        # Convert index_list and range_conditions into constant tensors.
        # index_list is expected to have shape [num_indices, 3]
        spatial_indices = tf.convert_to_tensor(index_list, dtype=tf.int32)
        range_conditions_tensor = tf.convert_to_tensor(range_conditions, dtype=tf.float32)
    
        # Validate shapes.
        tf.debugging.assert_shapes([(spatial_indices, (None, 3))])
        tf.debugging.assert_shapes([(range_conditions_tensor, (None, None, 2))])
        tf.debugging.assert_equal(tf.shape(spatial_indices)[0], tf.shape(range_conditions_tensor)[0],
                                  message="index_list and range_conditions must have the same number of indices")
        num_indices = tf.shape(spatial_indices)[0]  # number of indices  
        # shape: [num_indices, num_conditions, 2]
    
        # ----------------------------------------------------------------------
        # For each spatial index we update the binary tensor.
        # Because we are not allowed to use tf.range, we create a sequence for the batch dimension
        # by using cumulative sum on a ones vector.
        batch_size_new = tf.shape(reshaped_tensor)[0]
        ones_batch = tf.ones([batch_size_new], dtype=tf.int32)
        # This produces a tensor [0, 1, 2, ..., batch_size_new-1]
        batch_idx = tf.cast(tf.math.cumsum(ones_batch) - 1, tf.int32)  # shape: [batch_size_new]
    
        # Expand batch indices for each spatial index.
        batch_idx_exp = tf.reshape(batch_idx, [-1, 1])             # shape: [batch_size_new, 1]
        batch_idx_tiled = tf.tile(batch_idx_exp, [1, num_indices])   # shape: [batch_size_new, num_indices]
    
        # Expand the spatial_indices so that they are repeated along the batch dimension.
        spatial_indices_exp = tf.reshape(spatial_indices, [1, num_indices, 3])  # shape: [1, num_indices, 3]
        spatial_indices_tiled = tf.tile(spatial_indices_exp, [batch_size_new, 1, 1])  # shape: [batch_size_new, num_indices, 3]
    
        # For each spatial index, extract the (channel, height, original-width) values.
        # Later, we will update the corresponding block along the last dimension.
        c_val = spatial_indices_tiled[..., 0]  # shape: [batch_size_new, num_indices]
        h_val = spatial_indices_tiled[..., 1]  # shape: [batch_size_new, num_indices]
        w_orig = spatial_indices_tiled[..., 2] # shape: [batch_size_new, num_indices]
        
        # For each provided (w_orig), compute the starting column in the reshaped tensor.
        # Each block corresponds to inner_prod contiguous columns.
        block_start = w_orig * inner_prod     # shape: [batch_size_new, num_indices]
    
        # Now build the full indices for the representative element of each block.
        # We will use the first column in each block as the representative value.
        # The indices are: [batch, channel, height, block_start]
        full_indices = tf.stack([batch_idx_tiled, c_val, h_val, block_start], axis=-1)  # shape: [batch_size_new, num_indices, 4]
        full_indices_reshaped = tf.reshape(full_indices, [-1, 4])  # shape: [batch_size_new * num_indices, 4]
    
        # Gather representative values from reshaped_tensor.
        values = tf.gather_nd(reshaped_tensor, full_indices_reshaped)  # shape: [batch_size_new * num_indices]
        values = tf.reshape(values, [batch_size_new, num_indices])       # shape: [batch_size_new, num_indices]
    
        # Expand values for comparison.
        values_expanded = tf.expand_dims(values, axis=-1)  # shape: [batch_size_new, num_indices, 1]
    
        # Tile the range conditions across the batch dimension.
        range_conditions_exp = tf.expand_dims(range_conditions_tensor, axis=0)
        range_conditions_exp = tf.tile(range_conditions_exp, [batch_size_new, 1, 1, 1])  
        # now shape: [batch_size_new, num_indices, num_conditions, 2]
    
        # Split the start and stop values.
        start_values = range_conditions_exp[..., 0]  # shape: [batch_size_new, num_indices, num_conditions]
        stop_values  = range_conditions_exp[..., 1]  # shape: [batch_size_new, num_indices, num_conditions]
    
        # Check conditions: For each condition, compare whether the value lies within [start, stop].
        condition_check = tf.logical_and(values_expanded >= start_values, values_expanded <= stop_values)
        # If any condition is met for a given batch and index, consider it satisfied.
        condition_satisfied = tf.reduce_any(condition_check, axis=-1)  # shape: [batch_size_new, num_indices]
        
        # Determine update values: if condition is NOT satisfied, we want to mark that location as 1.
        updates = tf.cast(tf.logical_not(condition_satisfied), tf.int32)  # shape: [batch_size_new, num_indices]
    
        # --- Now update binary_tensor over the entire block.
        # For every spatial index, the update must occur for the entire block of size [inner_prod] along the last axis.
        # First, create a block of update values for each spatial index (by repeating the scalar update across inner_prod columns).
        updates_block = tf.tile(tf.expand_dims(updates, axis=-1), [1, 1, inner_prod])  # shape: [batch_size_new, num_indices, inner_prod]
        updates_block_flat = tf.reshape(updates_block, [-1])  # flatten updates
    
        # Next, we must construct full indices for every element in each block.
        # For the inner block offsets, we again create a [0, 1, ..., inner_prod-1] sequence by using cumulative sum.
        ones_inner = tf.ones([inner_prod], dtype=tf.int32)
        inner_offset = tf.cast(tf.math.cumsum(ones_inner) - 1, tf.int32)  # shape: [inner_prod]
        # Tile this offset so that it applies to every (batch, index) pair.
        inner_offset_tiled = tf.tile(tf.reshape(inner_offset, [1, 1, inner_prod]), [batch_size_new, num_indices, 1])
        # The final column index is the base block_start plus the inner offset.
        col_indices = tf.expand_dims(block_start, axis=-1) + inner_offset_tiled  # shape: [batch_size_new, num_indices, inner_prod]
    
        # Prepare the other indices (batch, channel, height) for the block.
        batch_indices_block = tf.tile(tf.expand_dims(batch_idx_tiled, axis=-1), [1, 1, inner_prod])
        c_block = tf.tile(tf.expand_dims(c_val, axis=-1), [1, 1, inner_prod])
        h_block = tf.tile(tf.expand_dims(h_val, axis=-1), [1, 1, inner_prod])
        
        # Stack these indices together to form the full indices for every element in each update block.
        full_indices_block = tf.stack([batch_indices_block, c_block, h_block, col_indices], axis=-1)  
        # Shape: [batch_size_new, num_indices, inner_prod, 4]
        full_indices_block_reshaped = tf.reshape(full_indices_block, [-1, 4])
        
        # Finally, update the binary_tensor (which is a tensor of zeros) using tensor_scatter_nd_update.
        binary_tensor_updated = tf.tensor_scatter_nd_update(binary_tensor, full_indices_block_reshaped, updates_block_flat)
        
        # Reshape the updated tensor back to the original shape:
        # [ *outer_dims, T, C, H, W, *inner_dims ]
        # (Note: if there were no inner dims originally, inner_prod is 1.)
        original_shape = tf.concat([outer_dims, [T, C, H, W], inner_dims], axis=0)
        updated_tensor = tf.reshape(binary_tensor_updated, original_shape)
        return updated_tensor
    
# Example usage
if __name__ == '__main__':
    initial = [
        {'name': 'P1', 'i': 29, 'j': 29, 'k': 0, 'type': 'producer', 'control': 'ORAT', 'value': 500.0, 'minimum_bhp': 1000.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.5, 'shutin_days': [[1000.0, 0.0]]},
        {'name': 'P2', 'i': 29, 'j': 9, 'k': 0, 'type': 'producer', 'control': 'ORAT', 'value': 10000.0, 'minimum_bhp': 1000.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.5, 'shutin_days': [[1000.0, 0.0]]},
        {'name': 'P3', 'i': 9, 'j': 9, 'k': 0, 'type': 'producer', 'control': 'ORAT', 'value': 1500.0, 'minimum_bhp': 1000.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.5, 'shutin_days': [[1000.0, 0.0]]},
        {'name': 'P4', 'i': 9, 'j': 29, 'k': 0, 'type': 'producer', 'control': 'ORAT', 'value': 2000.0, 'minimum_bhp': 1000.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.5, 'shutin_days': [[1000.0, 0.0]]},
        {'name': 'I1', 'i': 19, 'j': 19, 'k': 0, 'type': 'injector', 'control': 'ORAT', 'value': 0.0, 'minimum_bhp': 1000.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.5, 'shutin_days': [[1000.0, 0.0]]}
    ]

    # Test WellDataProcessor (eager mode)
    print("Testing WellDataProcessor (Eager Mode)")
    proc_eager = WellDataProcessor(initial, dtype=tf.float32)
    well_data = proc_eager.get_well_data()
    print("Initial wells:")
    print(well_data)

    # Scatter control_mode_value into grid
    target_shape = (1, 1, 39, 39, 1)
    result = proc_eager.scatter_y(target_shape, well_data['connection_index'], well_data['control_mode_value'],)
    # Verify scattered values
    verify_positions = tf.concat([
        tf.zeros([tf.shape(well_data['connection_index'])[0], 1], dtype=tf.int32),
        well_data['connection_index'],tf.zeros([tf.shape(well_data['connection_index'])[0], 1], dtype=tf.int32)
    ], axis=1)
    gathered = tf.gather_nd(result, verify_positions)
    tf.print("Scattered control_mode_value:", gathered)

    # Change control of P1
    proc_eager.update_control(0, 'WRAT', 1000.0)
    print("\nAfter updating P1 control:")
    print(proc_eager.get_well_data())

    # Update shutin_days for P1
    proc_eager.update_shutin_days(0, [[1000.0, 5.0]])
    print("\nAfter updating P1 shutin_days:")
    print(proc_eager.get_well_data())

    # Attempt to add well with same coordinates as P1 (should update)
    new_wells = [{'name': 'P1_updated', 'i': 29, 'j': 29, 'k': 0, 'type': 'injector', 'control': 'ORAT', 'value': 750.0, 'minimum_bhp': 1200.0, 'wellbore_radius': 0.1, 'completion_ratio': 0.75, 'shutin_days': [[1000.0, 10.0]]}]
    proc_eager.update_well_list(new_wells)
    print("\nAfter updating well at [29, 29, 0]:")
    print(proc_eager.get_well_data())

    # Add new well with unique coordinates
    new_wells = [{'name': 'P5', 'i': 19, 'j': 9, 'k': 0, 'type': 'injector', 'control': 'ORAT', 'value': 10000.0, 'minimum_bhp': 1000.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.5, 'shutin_days': [[1000.0, 0.0]]}]
    proc_eager.update_well_list(new_wells)
    print("\nAfter adding new well at [19, 9, 0]:")
    print(proc_eager.get_well_data())

    # Attempt to re-update P1 (should update, no duplicate)
    new_wells = [{'name': 'P1_again', 'i': 29, 'j': 29, 'k': 0, 'type': 'injector', 'control': 'WRAT', 'value': 500.0, 'minimum_bhp': 1100.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.75, 'shutin_days': [[1000.0, 15.0]]}]
    proc_eager.update_well_list(new_wells)
    print("\nAfter re-updating well at [29, 29, 0]:")
    print(proc_eager.get_well_data())

    # Test WellDataProcessorStaticMode (static mode)
    print("\nTesting WellDataProcessorStaticMode (Static Mode)")
    proc_static = WellDataProcessorStaticMode(initial, dtype=tf.float32)
    well_data = proc_static.get_well_data()
    print("Initial wells:")
    print(well_data)

    # Scatter control_mode_value into grid
    result = proc_static.scatter_y(target_shape, well_data['connection_index'], well_data['control_mode_value'])
    # Verify scattered values
    verify_positions = tf.concat([
        tf.zeros([tf.shape(well_data['connection_index'])[0], 2], dtype=tf.int32),
        well_data['connection_index']
    ], axis=1)
    gathered = tf.gather_nd(result, verify_positions)
    tf.print("Scattered control_mode_value:", gathered)

    # Change control of P1
    proc_static.update_control(0, 'WRAT', 1000.0)
    print("\nAfter updating P1 control:")
    print(proc_static.get_well_data())

    # Update shutin_days for P1
    proc_static.update_shutin_days(0, [[1000.0, 5.0]])
    print("\nAfter updating P1 shutin_days:")
    print(proc_static.get_well_data())

    # Attempt to add well with same coordinates as P1 (should update)
    new_wells = [{'name': 'P1_updated', 'i': 29, 'j': 29, 'k': 0, 'type': 'injector', 'control': 'ORAT', 'value': 750.0, 'minimum_bhp': 1200.0, 'wellbore_radius': 0.1, 'completion_ratio': 0.75, 'shutin_days': [[1000.0, 10.0]]}]
    proc_static.update_well_list(new_wells)
    print("\nAfter updating well at [29, 29, 0]:")
    print(proc_static.get_well_data())

    # Add new well with unique coordinates
    new_wells = [{'name': 'P5', 'i': 19, 'j': 9, 'k': 0, 'type': 'injector', 'control': 'ORAT', 'value': 10000.0, 'minimum_bhp': 1000.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.5, 'shutin_days': [[1000.0, 0.0]]}]
    proc_static.update_well_list(new_wells)
    print("\nAfter adding new well at [19, 9, 0]:")
    print(proc_static.get_well_data())

    # Attempt to re-update P1 (should update, no duplicate)
    new_wells = [{'name': 'P1_again', 'i': 29, 'j': 29, 'k': 0, 'type': 'injector', 'control': 'WRAT', 'value': 500.0, 'minimum_bhp': 1100.0, 'wellbore_radius': 0.09525, 'completion_ratio': 0.75, 'shutin_days': [[1000.0, 15.0]]}]
    proc_static.update_well_list(new_wells)
    print("\nAfter re-updating well at [29, 29, 0]:")
    print(proc_static.get_well_data())
    
    # Test the connection_shutin_idx
    
    # For reproducibility.
    tf.random.set_seed(42)
    
    # Example: Create a 5D tensor.
    # Here we test a modified shape: [10, 1, 39, 39, 3]
    # When time_axis==0 the tensor shape is interpreted as:
    #   outer_dims: [] (none),
    #   T: 10,
    #   spatial dims: (1, 39, 39)
    #   inner_dims: [3]
    time_tensor = tf.random.uniform((10, 1, 39, 39, 3), minval=0, maxval=1000, dtype=tf.float32)
    
    # Define spatial index_list.
    # Each tuple is (channel, height, width) where "width" refers to the value in the original W dimension.
    index_list = [
        (0, 15, 15),   # A central-ish location.
        (0, 10, 10),   # Top left area.
        (0, 20, 20),   # Another center location.
        (0, 30, 30)    # Bottom right area.
    ]
    
    # index_list_1 = [
    #     (0, 15, 1),   # A central-ish location.
    #     (0, 10, 1),   # Top left area.
    #     (0, 20, 2),   # Another center location.
    #     (0, 30, 0)    # Bottom right area.
    # ]
    # Define range_conditions corresponding to each index.
    range_conditions = [
        [[50, 200], [300, 400]],    # For index (0, 15, 15)
        [[200, 500], [600, 800]],    # For index (0, 10, 10)
        [[0, 100], [500, 700]],      # For index (0, 20, 20)
        [[400, 800], [900, 1000]]     # For index (0, 30, 30)
    ]
    
    # Compute the updated binary tensor.
    # Try with time_axis=0 (as in our example) ...
    updated_tensor = proc_static.conn_shutins_idx(time_tensor, index_list, range_conditions, time_axis=0)
    #updated_tensor_ = conn_shutins_idx(time_tensor, index_list_, range_conditions, time_axis=1)
    # For additional testing, you can move the time axis.
    # For example, if you want time_axis==1, first permute:
    #   From shape [B, C, H, W, inner] to [C, B, H, W, inner]
    # time_tensor_t = tf.transpose(time_tensor, perm=[1, 0, 2, 3, 4])
    # updated_tensor2 = conn_shutins_idx(time_tensor_t, index_list, range_conditions, time_axis=1)
    
    # To verify the results, we will print:
    # 1. The representative original values at the locations corresponding to index_list.
    # 2. Their updated binary values.
    #
    # We first reconstruct the indices used to sample the representative values.
    
    # Compute new reshaped dimensions.
    # With time_axis==0, outer_dims is empty, T==10, spatial dims: (1, 39, 39) and inner_dims: [3]
    # so reshaped shape is [10, 1, 39, 39*3].
    outer_dims = []  # none for time_axis 0 in this example
    T = tf.shape(time_tensor)[0]
    C = tf.shape(time_tensor)[1]
    H = tf.shape(time_tensor)[2]
    W = tf.shape(time_tensor)[3]
    inner_dims = tf.shape(time_tensor)[4:]
    inner_prod = tf.cond(tf.equal(tf.size(inner_dims), 0),
                         lambda: tf.constant(1, dtype=tf.int32),
                         lambda: tf.reduce_prod(inner_dims))
    new_batch = T  # since no outer_dims in this test.
    new_W = W * inner_prod
    reshaped_tensor = tf.reshape(time_tensor, tf.stack([new_batch, C, H, new_W]))

    # Use the same mechanism as in the function to recover representative indices.
    num_indices = tf.shape(tf.constant(index_list, dtype=tf.int32))[0]
    ones_batch = tf.ones([new_batch], dtype=tf.int32)
    batch_idx = tf.cast(tf.math.cumsum(ones_batch) - 1, tf.int32)
    batch_idx_exp = tf.reshape(batch_idx, [-1, 1])
    batch_idx_tiled = tf.tile(batch_idx_exp, [1, num_indices])
    
    spatial_indices = tf.constant(index_list, dtype=tf.int32)
    spatial_indices_exp = tf.reshape(spatial_indices, [1, num_indices, 3])
    spatial_indices_tiled = tf.tile(spatial_indices_exp, [new_batch, 1, 1])
    c_val = spatial_indices_tiled[..., 0]
    h_val = spatial_indices_tiled[..., 1]
    w_orig = spatial_indices_tiled[..., 2]
    block_start = w_orig * inner_prod
    full_indices = tf.stack([batch_idx_tiled, c_val, h_val, block_start], axis=-1)
    full_indices_reshaped = tf.reshape(full_indices, [-1, 4])
    
    orig_values = tf.gather_nd(reshaped_tensor, full_indices_reshaped)
    orig_values = tf.reshape(orig_values, [new_batch, num_indices])
    
    # For the binary tensor, extract the corresponding updated block values.
    # We choose the first column in each updated block.
    binary_tensor_flat = tf.reshape(updated_tensor, tf.shape(time_tensor))
    # To inspect the update only at the spatial locations (ignoring the inner dims), we extract the same representative element.
    # (We could also work on the reshaped binary_tensor; here we reassemble the representative sampling.)
    binary_reshaped = tf.reshape(updated_tensor, tf.stack([new_batch, C, H, new_W]))
    updated_values = tf.gather_nd(binary_reshaped, full_indices_reshaped)
    updated_values = tf.reshape(updated_values, [new_batch, num_indices])
    
    tf.print("Representative original values (per time step):")
    tf.print(orig_values)
    tf.print("\nRepresentative updated binary values (0 means condition met, 1 means condition NOT met):")
    tf.print(updated_values)
    
    # For example, display one sample (time index 0) of the complete binary tensor in the reshaped form.
    tf.print("\nBinary tensor for batch (time) index 0:")
    tf.print(binary_reshaped[0])

