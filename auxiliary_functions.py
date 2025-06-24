import tensorflow as tf
import pandas as pd
import tensorflow as tf
import pandas as pd

###############################################################################
# Training Statistics Tensor Description:
#   INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate', ...}
#   KEYS:  {'min', 'max', 'mean', 'std', 'count'}
###############################################################################

###############################################################################
# 1. Analytical Derivative of the Normalization Function
###############################################################################
@tf.function(jit_compile=True)
def normfunc_derivative(training_stats=None, norm_config=None, stat_idx=0, compute=False, dtype=tf.float32):
    """
    Analytically computes the derivative of the normalization function.
    
    Parameters:
        training_stats: A tensor of shape [N, 5], where each row is [min, max, mean, std, count].
        norm_config: A dictionary with keys:
                     - "Norm_Limits": tuple (norm_min, norm_max)
                     - "Input_Normalization": string indicating normalization type.
        stat_idx: Index selecting which row of training_stats to use.
        compute: Boolean flag; if False, returns a default derivative (ones).
        dtype: TensorFlow data type (default is tf.float32).
        
    Returns:
        A scalar tensor representing the derivative. If training_stats or norm_config is None,
        returns ones.
    """
    # Early exit if configuration is missing.
    if training_stats is None or norm_config is None:
        return tf.ones((), dtype=dtype)

    norm_min, norm_max = norm_config["Norm_Limits"]

    def _lnk_linear_scaling():
        ts = training_stats[stat_idx]
        lin_scale_no_log = (norm_max - norm_min) / (ts[1] - ts[0])
        lin_scale_log = (norm_max - norm_min) / tf.math.log(ts[1] / ts[0])
        cond = tf.logical_and(tf.math.not_equal(stat_idx, 5), tf.math.not_equal(stat_idx, 6))
        return tf.cond(cond, lambda: lin_scale_no_log, lambda: lin_scale_log)

    def _linear_scaling():
        ts = training_stats[stat_idx]
        return (norm_max - norm_min) / (ts[1] - ts[0])
    
    def _z_score():
        ts = training_stats[stat_idx]
        return 1 / ts[3]

    derivative = tf.cond(
        tf.math.equal(compute, True),
        lambda: tf.cond(
            tf.math.equal(norm_config["Input_Normalization"], "linear-scaling"),
            lambda: _linear_scaling(),
            lambda: tf.cond(
                tf.math.equal(norm_config["Input_Normalization"], "lnk-linear-scaling"),
                lambda: _lnk_linear_scaling(),
                lambda: _z_score()
            )
        ),
        lambda: tf.ones((), dtype=dtype)
    )

    derivative = tf.where(
        tf.logical_or(tf.math.is_nan(derivative), tf.math.is_inf(derivative)),
        tf.zeros_like(derivative),
        derivative
    )
    return derivative


###############################################################################
# 2. Finite-Difference Derivative
###############################################################################
@tf.function
def finite_difference_derivative(x, func, diff_type='central_difference', grid_spacing=0.01):
    """
    Computes the derivative of func at x using a finite-difference scheme.
    
    Parameters:
        x: The input tensor (the point where the derivative is computed).
        func: A callable that accepts a tensor and returns a tensor output.
        diff_type: String specifying the finite-difference type ('central_difference' is default).
        grid_spacing: A scalar representing the spacing for the finite-difference computation.
        
    Returns:
        A tensor representing the derivative. Any NaN or Inf values are replaced with zeros.
    """
    def central_difference():
        return (tf.stack(func(x + grid_spacing), axis=0) -
                tf.stack(func(x - grid_spacing), axis=0)) / (2 * grid_spacing)
    
    def forward_difference():
        return (tf.stack(func(x + grid_spacing), axis=0) -
                tf.stack(func(x), axis=0)) / grid_spacing

    derivative = (central_difference() if diff_type == 'central_difference'
                  else forward_difference())
    
    derivative = tf.where(
        tf.logical_or(tf.math.is_nan(derivative), tf.math.is_inf(derivative)),
        tf.zeros_like(derivative),
        derivative
    )
    return derivative


###############################################################################
# 3. Normalization of Data Based on Training Statistics
###############################################################################
@tf.function(jit_compile=True)
def normalize(nonorm_input, training_stats=None, norm_config=None, stat_idx=0, compute=False, dtype=tf.float32):
    """
    Normalizes an input tensor using training statistics.
    
    Parameters:
        nonorm_input: The raw input tensor to be normalized.
        training_stats: Tensor of shape [N, 5] with each row as [min, max, mean, std, count].
        norm_config: Dictionary containing:
                     - "Norm_Limits": tuple (norm_min, norm_max)
                     - "Input_Normalization": string indicating the normalization type.
        stat_idx: Index selecting the row in training_stats.
        compute: Boolean flag; if False, returns nonorm_input unchanged.
        dtype: TensorFlow data type.
        
    Returns:
        The normalized tensor. If training_stats or norm_config is None, returns nonorm_input unchanged.
    """
    if training_stats is None or norm_config is None:
        return nonorm_input

    norm_min, norm_max = norm_config["Norm_Limits"]
    nonorm_input = tf.convert_to_tensor(nonorm_input, dtype=dtype, name='nonorm_input')

    def _lnk_linear_scaling():
        ts = training_stats[stat_idx]
        lin_scale_no_log = (((nonorm_input - ts[0]) / (ts[1] - ts[0])) *
                            (norm_max - norm_min)) + norm_min
        lin_scale_log = ((tf.math.log(nonorm_input / ts[0]) /
                          tf.math.log(ts[1] / ts[0])) * (norm_max - norm_min)) + norm_min
        cond = tf.logical_and(tf.math.not_equal(stat_idx, 5), tf.math.not_equal(stat_idx, 6))
        return tf.cond(cond, lambda: lin_scale_no_log, lambda: lin_scale_log)
    
    def _linear_scaling():
        ts = training_stats[stat_idx]
        return (((nonorm_input - ts[0]) / (ts[1] - ts[0])) *
                (norm_max - norm_min)) + norm_min
    
    def _z_score():
        ts = training_stats[stat_idx]
        return (nonorm_input - ts[2]) / ts[3]

    norm_func = tf.where(
        tf.math.equal(norm_config["Input_Normalization"], "lnk-linear-scaling"),
        _lnk_linear_scaling(), _linear_scaling()
    )
    norm_value = tf.where(tf.math.equal(compute, True), norm_func, nonorm_input)
    
    norm_value = tf.where(
        tf.logical_or(tf.math.is_nan(norm_value), tf.math.is_inf(norm_value)),
        tf.zeros_like(norm_value),
        norm_value
    )
    return norm_value


###############################################################################
# 4. Unnormalization of Data Using Training Statistics
###############################################################################
@tf.function
def nonormalize(norm_input, training_stats=None, norm_config=None, stat_idx=0, compute=False, dtype=tf.float32):
    """
    Reverses the normalization on a tensor using training statistics.
    
    Parameters:
        norm_input: The normalized tensor to be unnormalized.
        training_stats: Tensor of shape [N, 5] with each row as [min, max, mean, std, count].
        norm_config: Dictionary containing:
                     - "Norm_Limits": tuple (norm_min, norm_max)
                     - "Input_Normalization": string indicating normalization type.
        stat_idx: Index for selecting the row in training_stats.
        compute: Boolean flag; if False, returns norm_input unchanged.
        dtype: TensorFlow data type.
        
    Returns:
        The unnormalized tensor. If training_stats or norm_config is None, returns norm_input.
    """
    if training_stats is None or norm_config is None:
        return norm_input

    norm_min, norm_max = norm_config["Norm_Limits"]

    def _lnk_linear_scaling():
        ts = training_stats[stat_idx]
        lin_scale_no_log = (ts[1] - ts[0]) * ((norm_input - norm_min) / (norm_max - norm_min)) + ts[0]
        lin_scale_log = tf.math.exp(tf.math.log(ts[1] / ts[0]) *
                                    ((norm_input - norm_min) / (norm_max - norm_min)) +
                                    tf.math.log(ts[0]))
        cond = tf.logical_and(tf.math.not_equal(stat_idx, 5), tf.math.not_equal(stat_idx, 6))
        return tf.cond(cond, lambda: lin_scale_no_log, lambda: lin_scale_log)
    
    def _linear_scaling():
        ts = training_stats[stat_idx]
        return (ts[1] - ts[0]) * ((norm_input - norm_min) / (norm_max - norm_min)) + ts[0]
    
    def _z_score():
        ts = training_stats[stat_idx]
        return norm_input * ts[3] + ts[2]
    
    nonorm_func = tf.where(
        tf.math.equal(norm_config["Input_Normalization"], "lnk-linear-scaling"),
        _lnk_linear_scaling(), _linear_scaling()
    )
    nonorm_value = tf.where(tf.math.equal(compute, True), nonorm_func, norm_input)
    
    nonorm_value = tf.where(
        tf.logical_or(tf.math.is_nan(nonorm_value), tf.math.is_inf(nonorm_value)),
        tf.zeros_like(nonorm_value),
        nonorm_value
    )
    return nonorm_value


###############################################################################
# 5. Normalized Difference Computation
###############################################################################
@tf.function(jit_compile=True)
def normalize_diff(diff, training_stats=None, norm_config=None, stat_idx=0, compute=False, x0=3., dtype=tf.float32):
    """
    Computes a normalized difference.
    
    Parameters:
        diff: The input difference tensor.
        training_stats: Tensor of shape [N, 5] for training statistics.
        norm_config: Dictionary containing:
                     - "Norm_Limits": tuple (norm_min, norm_max)
                     - "Input_Normalization": string indicating the normalization type.
        stat_idx: Index selecting the row in training_stats.
        compute: Boolean flag; if False, returns diff unchanged.
        x0: A constant used in logarithmic scaling of differences.
        dtype: TensorFlow data type.
        
    Returns:
        The normalized difference tensor. If training_stats or norm_config is None, returns diff.
    """
    if training_stats is None or norm_config is None:
        return diff

    norm_min, norm_max = norm_config["Norm_Limits"]
    diff = tf.convert_to_tensor(diff, dtype=dtype, name='diff')

    def _lnk_linear_scaling():
        ts = training_stats[stat_idx]
        scale_factor_no_log = tf.convert_to_tensor((norm_max - norm_min) / (ts[1] - ts[0]), dtype=dtype)
        scale_factor_log = tf.convert_to_tensor((norm_max - norm_min) / tf.math.log(ts[1] / ts[0]), dtype=dtype)
        lin_scale_no_log = scale_factor_no_log * diff
        lin_scale_log = scale_factor_log * tf.math.log((x0 + diff) / x0)
        cond = tf.logical_and(tf.math.not_equal(stat_idx, 5), tf.math.not_equal(stat_idx, 6))
        return tf.cond(cond, lambda: lin_scale_no_log, lambda: lin_scale_log)
    
    def _linear_scaling():
        ts = training_stats[stat_idx]
        return tf.convert_to_tensor((norm_max - norm_min) / (ts[1] - ts[0]), dtype=dtype) * diff
    
    def _z_score():
        ts = training_stats[stat_idx]
        return tf.convert_to_tensor(1 / ts[3], dtype=dtype) * diff

    norm_func = tf.where(
        tf.math.equal(norm_config["Input_Normalization"], "lnk-linear-scaling"),
        _lnk_linear_scaling(), _linear_scaling()
    )
    norm_value = tf.where(tf.math.equal(compute, True), norm_func, diff)
    
    norm_value = tf.where(
        tf.logical_or(tf.math.is_nan(norm_value), tf.math.is_inf(norm_value)),
        tf.zeros_like(norm_value),
        norm_value
    )
    return norm_value

class DataSummary:
    """
    DataSummary

    A utility class to manage and query statistical data/summaries (e.g., training data, PVT experimental data) for training data,
    including both features (`x`) and labels (`y`).

    Supports cases where `ts_f` or `ts_l` can be None and stores statistics as
    a float tensor (default `tf.float32`) for easy integration with TensorFlow.

    Features:
    - Stores all statistics in a single [N, 5] tensor with specified dtype (default `tf.float32`).
    - Lookup by variable name or index.
    - Reverse lookup from index to variable name.
    - Handles missing feature or label stats gracefully.
    """
    def __init__(self, ts_f: pd.DataFrame = None, ts_l: pd.DataFrame = None, dtype=tf.float32):
        """
        Initializes the TrainingDataSummary object.

        Parameters:
        - ts_f: DataFrame with feature statistics (optional).
        - ts_l: DataFrame with label statistics (optional).
        - dtype: TensorFlow dtype for the statistics (default: tf.float32).
        """
        # Set the dtype to the user-provided value (default: tf.float32)
        self.dtype = dtype
        
        # Combine both dataframes if they exist
        if ts_f is not None and ts_l is not None:
            # Concatenate ts_f and ts_l vertically (along the 0th axis)
            combined_df = pd.concat([ts_f, ts_l], axis=0, ignore_index=True)
        elif ts_f is not None:
            combined_df = ts_f
        elif ts_l is not None:
            combined_df = ts_l
        else:
            combined_df = pd.DataFrame()  # Empty DataFrame if both are None

        # Convert the combined DataFrame into a TensorFlow tensor
        self.statistics = tf.convert_to_tensor(combined_df, dtype=self.dtype)

        # Initialize keys for x and y from provided DataFrames        
        # "x" keys are the same for both ts_f and ts_l
        self.x_keys = [s.lower() for s in list(combined_df.keys())]            # Case insensitive  

        # "y" keys are a combination of both ts_f and ts_l
        self.y_keys = [s.lower() for s in (
                        (list(ts_f.index) if ts_f is not None else []) +
                        (list(ts_l.index) if ts_l is not None else [])
                       )]

        # Create reverse lookup dictionaries for both "x" and "y"
        self._x_lookup = {key: idx for idx, key in enumerate(self.x_keys)}
        self._y_lookup = {
            key: idx 
            for idx, key in enumerate(self.y_keys)
        }

        # Reverse lookup dictionary (index ➝ key)
        self._reverse_lookup = {
            idx: key for key, idx in {**self._x_lookup, **self._y_lookup}.items()
        }

    def lookup(self, key: str):
        """Lookup by variable name (feature or label) and return its stats."""
        if key in self._x_lookup:
            idx = self._x_lookup[key]
            value = self.statistics[:,idx]
        elif key in self._y_lookup:
            idx = self._y_lookup[key]
            value = self.statistics[idx]
        else:
            raise KeyError(f"Key '{key}' not found.")
        return value

    def by_index(self, idx: int):
        """Directly access stats by row index."""
        if idx < 0 or idx >= self.statistics.shape[0]:
            raise IndexError(f"Index {idx} out of range.")
        return self.statistics[idx]

    def get_key(self, idx: int):
        """Reverse lookup: get variable name from index."""
        return self._reverse_lookup.get(idx, None)

    def keys(self):
        """Return feature and label keys."""
        return {
            "x": self.x_keys,
            "y": self.y_keys
        }

    def all_stats(self):
        """Return the full statistics tensor."""
        return self.statistics

    
# Testing the trainingsummary class
# Assuming ts_f and ts_l are your DataFrames
# Data as a dictionary
data1 = {
    'min': [37.179485, 37.179485, 11040.000000, 0.000000, 0.200000, 0.891495, 0.089149],
    'max': [2862.820557, 2862.820557, 11040.000000, 380.000000, 0.200000, 7.900850, 0.790085],
    'mean': [1449.999878, 1450.000366, 11040.000000, 160.812561, 0.199999, 2.979298, 0.297930],
    'std': [836.882751, 836.882751, 0.000000, 123.770752, 0.000000, 0.980748, 0.098075],
    'count': [3942432.0, 3942432.0, 3942432.0, 3942432.0, 3942432.0, 3942432.0, 3942432.0]
}

data2 = {
    'min': [0.0, 0.0, 0.0],
    'max': [0.0, 0.0, 2000.0],
    'mean': [0.000000, 0.000000, 3.218825],
    'std': [0.000000, 0.000000, 69.410934],
    'count': [3942432.0, 3942432.0, 3942432.0]
}
# Index names (coordinates and time)
index1 = ['x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz']

# Index names for the rows (pressure, gsat, grate)
index2 = ['pressure', 'gsat', 'grate']
# Create DataFrame
ts_f = pd.DataFrame(data1, index=index1)
ts_l = pd.DataFrame(data2, index=index2)

summary = DataSummary(ts_f, ts_l)

# Lookup stats for a feature or label
permx_stats = summary.lookup("permx")  # [min, max, mean, std, count]
print("permx stats:", permx_stats)

# Get stats by index
pressure_stats = summary.by_index(5)  # [min, max, mean, std, count]
print("pressure stats:", pressure_stats)

# Get feature and label keys
print("x keys:", summary.keys()["x"])
print("y keys:", summary.keys()["y"])

# Get the full tensor of stats
print("All stats tensor:", summary.all_stats())

@tf.function
def scatter_y(target_shape, index_list, y, dtype=tf.float32):
    """
    Scatter values into a tensor of shape `target_shape` using the indices from `index_list`.
    
    - If y is a single scalar value, that value is applied for every index in index_list.
    - If y is a list or vector, its elements are used one‐to‐one with index_list.
    - All indices not in index_list will have 0.
    - Finally, the output is expanded by adding an outer dimension.
    
    This function avoids explicit Python if statements by relying on broadcasting and tf.where.
    
    Args:
        target_shape: A tuple indicating the shape of the base tensor (e.g., (1, 39, 39)).
                      In our case the first dimension is the channel index.
        index_list: List of tuples for indices (e.g., [(0,15,15), (0,10,10), ...]).
        y: A scalar or a list of values. If a scalar, it is broadcast to all indices.
        dtype: The type of the output tensor (default tf.float32).
        
    Returns:
        A tensor of shape (1, ...) where ... is target_shape. Values are inserted at the
        provided indices and all other positions remain 0.
    """
    # Convert the index_list into a constant tensor.
    indices = tf.constant(index_list, dtype=tf.int32)  # shape: (num_indices, 3)
    
    # Convert y to a tensor and reshape it to a 1-D vector.
    y_tensor = tf.reshape(tf.convert_to_tensor(y, dtype=dtype), [-1])
    
    # Determine the number of indices.
    num_indices = tf.shape(indices)[0]
    
    # Create a ones tensor of shape (num_indices,). Multiplying by this forces
    # any scalar in y_tensor (shape [1]) to broadcast, or leaves a vector unchanged.
    ones_tensor = tf.ones([num_indices], dtype=y_tensor.dtype)
    
    # Multiply. If y_tensor is scalar (shape [1]), the result will be a vector of length num_indices.
    # If y_tensor is already a vector of length num_indices, the multiplication is elementwise.
    # (tf.where is used here to illustrate using tensor-based selection, though broadcasting suffices.)
    updates_candidate = ones_tensor * y_tensor
    # Use tf.where to ensure that if the size of y_tensor is 1, we pick that scalar for all positions.
    # This formulation avoids an explicit if-statement.
    updates = tf.where(tf.equal(tf.size(y_tensor), 1),
                       ones_tensor * y_tensor[0],
                       updates_candidate)
    # Use tf.scatter_nd to place the updates into a base tensor of zeros.
    scattered = tf.scatter_nd(indices, updates, target_shape)
    
    # Expand the outer dimension (e.g., if scattered is (1, 39, 39), this makes it (1, 1, 39, 39)).
    # Adjust the axis as desired.
    result = tf.expand_dims(scattered, axis=0)
    return result

# Preprocess the range_conditions in Python.
# For any index whose conditions list is empty, substitute a dummy condition that is never satisfied.
def preprocess_conditions(range_conditions):
    """
    Preprocess the range_conditions.
    
    If an index has a condition list that is either empty or only contains empty lists,
    then substitute a dummy condition (here, [1001.0, 0.0]) that is always false.
    
    Also, for any nonempty condition that is itself an empty list, substitute the dummy.
    Finally, pad each index's list of conditions so that all indices have the same number of conditions.
    
    Args:
      range_conditions: List of conditions for each spatial index. For example:
          [
              [[],],            # For index (0, 15, 15)
              [[],],            # For index (0, 10, 10)
              [[],],            # For index (0, 20, 20)
              [[],]             # For index (0, 30, 30)
          ]
    Returns:
      A padded list of conditions where each inner element is a valid [start, stop] pair.
    """
    dummy = [1001.0, 0.0]  # This condition is impossible to satisfy.
    new_range_conditions = []
    for cond in range_conditions:
        # Check if the condition list itself is empty, or if every condition in it is empty.
        if not cond or all(len(c) == 0 for c in cond):
            new_range_conditions.append([dummy])
        else:
            # For nonempty conditions: if an individual condition is empty, replace it with dummy.
            new_cond = []
            for c in cond:
                if not c:  # if the condition is empty
                    new_cond.append(dummy)
                else:
                    new_cond.append(c)
            new_range_conditions.append(new_cond)
    
    # Determine the maximum number of conditions among all indices.
    max_conditions = max(len(cond) for cond in new_range_conditions)
    # Pad each index's condition list to have the same length.
    padded_conditions = []
    for cond in new_range_conditions:
        pad_count = max_conditions - len(cond)
        padded_conditions.append(cond + [dummy] * pad_count)
    
    return padded_conditions

# Multi-Dimensional Tensors
def conn_shutins_factors(time_tensor, index_list, range_conditions, time_axis=0):
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
    spatial_indices = tf.constant(index_list, dtype=tf.int32)
    num_indices = tf.shape(spatial_indices)[0]  # number of indices
    range_conditions_tensor = tf.constant(range_conditions, dtype=tf.float32)  
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


def main():
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
    
    index_list_1 = [
        (0, 15, 1),   # A central-ish location.
        (0, 10, 1),   # Top left area.
        (0, 20, 2),   # Another center location.
        (0, 30, 0)    # Bottom right area.
    ]
    # Define range_conditions corresponding to each index.
    range_conditions = [
        [[50, 200], [300, 400]],    # For index (0, 15, 15)
        [[200, 500], [600, 800]],    # For index (0, 10, 10)
        [[0, 100], [500, 700]],      # For index (0, 20, 20)
        [[400, 800], [900, 1000]]     # For index (0, 30, 30)
    ]
    
    # Compute the updated binary tensor.
    # Try with time_axis=0 (as in our example) ...
    updated_tensor = conn_shutins_factors(time_tensor, index_list, range_conditions, time_axis=0)
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

if __name__ == '__main__':
    main()
