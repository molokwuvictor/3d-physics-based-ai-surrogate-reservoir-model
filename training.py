import numpy as np
import tensorflow as tf
import logging
from collections import Counter
from typing import List, Union, Dict, Sequence, Optional

from packaging import version  # more robust than distutils
try:
   import tensorflow_addons as tfa
   tfa_installed = True
except ImportError:
   tfa_installed = False




class BatchGenerator:
    """
    Generic batch generator for features and labels from a list of feature-label pairs.

    Input is a list of pairs, where each pair consists of:
    - features: a single array with shape (d0, d1, ...), e.g., np.random.rand(n0, n1, c, h, w)
    - labels: either an array or a dictionary with arrays, e.g., {'a': array, 'b': array}

    You may specify which axes to collapse (e.g., [0, 1] by default).
    If `collapse_axes` is None or empty, no collapsing is performed.

    You may also specify the axis used for batching (after flattening).
    By default, batching is performed along axis 0 for features and labels.

    If the input `pairs` is empty, initializes an empty BatchGenerator with zero samples.

    Parameters:
    - stack_labels: bool, default False. If True and labels are dictionaries, returns y as a stacked tensor
      with shape (num_keys, batch_size, *label_dims). Requires all label arrays to have the same shape after flattening.

    Returns:
    - x: shape (batch_size, *feature_dims)
    - y: if labels are arrays => shape (batch_size, *label_dims)
         if labels are dictionaries and stack_labels=False => dictionary with keys corresponding to label keys,
           each with shape (batch_size, *label_dims)
         if labels are dictionaries and stack_labels=True => tensor with shape (num_keys, batch_size, *label_dims),
           where num_keys is the number of label keys, and all label arrays must have the same shape after flattening
    """
    def __init__(
        self,
        pairs: List[tuple[np.ndarray, Union[np.ndarray, Dict[str, np.ndarray]]]],
        batch_size: int,
        collapse_axes: Optional[Sequence[int]] = (0, 1),
        batch_axis: int = 0,
        shuffle: bool = True,
        stack_labels: bool = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collapse_axes = list(collapse_axes) if collapse_axes else []
        self.batch_axis = batch_axis
        self.stack_labels = stack_labels

        # Validate that pairs is a list
        if not isinstance(pairs, list):
            raise ValueError("Input 'pairs' must be a list of feature-label tuples")

        # Handle empty pairs case
        if not pairs:
            self.x_all = np.array([])  # Empty array for features
            self.y_all = np.array([])  # Empty array for labels
            self.indices = np.array([])  # Empty indices
            self.N = 0  # No samples
            self.is_dict = False  # Default to non-dict labels
            self.label_keys = []  # No label keys
            return

        # Validate label consistency and determine label type
        self.is_dict = isinstance(pairs[0][1], dict)
        if self.is_dict:
            self.label_keys = list(pairs[0][1].keys())
            # Ensure all pairs have consistent label dictionary keys
            for _, labels in pairs[1:]:
                if not isinstance(labels, dict) or set(labels.keys()) != set(self.label_keys):
                    raise ValueError("All label dictionaries must have the same keys across pairs")

        # Process features: flatten each feature array and concatenate across pairs
        x_all_list = [self._maybe_flatten(features) for features, _ in pairs]
        self.x_all = np.concatenate(x_all_list, axis=0)  # Concatenate along batch axis (axis 0)

        # Process labels
        if self.is_dict:
            # For dictionary labels, concatenate flattened arrays for each key across pairs
            self.y_all = {k: np.concatenate([self._maybe_flatten(labels[k]) for _, labels in pairs], axis=0) for k in self.label_keys}
            # If stack_labels is True, ensure all label arrays have the same shape after flattening
            if self.stack_labels:
                shapes = [self.y_all[k].shape[1:] for k in self.label_keys]
                if not all(s == shapes[0] for s in shapes):
                    raise ValueError("All label arrays must have the same shape after flattening when stack_labels=True")
        else:
            # For array labels, flatten each label and concatenate across pairs
            self.y_all = np.concatenate([self._maybe_flatten(labels) for _, labels in pairs], axis=0)

        # Set up indices for batching
        self.N = self.x_all.shape[0]  # Total number of samples after flattening
        self.indices = np.arange(self.N)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """Return the number of batches."""
        return int(np.ceil(self.N / self.batch_size))
    
    def __getitem__(self, idx: Union[int, tf.Tensor]):
        """Retrieve a batch by index, supporting both integer and tensor indices."""
        if self.N == 0:
            return tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.float32)  # Return empty tensors for empty generator

        # Convert idx to tensor if it’s an integer

        idx = tf.convert_to_tensor(idx, dtype=tf.int32) if isinstance(idx, int) else idx
        batch_size = tf.convert_to_tensor(self.batch_size, dtype=tf.int32)
        N = tf.convert_to_tensor(self.N, dtype=tf.int32)

        # Compute start and end indices using tensor operations
        start = idx * batch_size
        end = tf.minimum((idx + 1) * batch_size, N)
        batch_inds = tf.gather(self.indices, tf.range(start, end))

        # Convert x_all and y_all to tensors
        x_all = tf.convert_to_tensor(self.x_all, dtype=tf.float32)
        x_batch = tf.gather(x_all, batch_inds, axis=self.batch_axis)

        if self.is_dict:
            y_all = {k: tf.convert_to_tensor(self.y_all[k], dtype=tf.float32) for k in self.label_keys}
            y_batch = {k: tf.gather(y_all[k], batch_inds, axis=self.batch_axis) for k in self.label_keys}
            if self.stack_labels:
                y_batch = tf.stack([y_batch[k] for k in self.label_keys], axis=0)
        else:
            y_all = tf.convert_to_tensor(self.y_all, dtype=tf.float32)
            y_batch = tf.gather(y_all, batch_inds, axis=self.batch_axis)

        # Log shapes for debugging
        # tf.print("BatchGenerator.__getitem__: x_batch shape: ", tf.shape(x_batch), " y_batch shape: ", 
        #          tf.shape(y_batch) if not self.is_dict else {k: tf.shape(y_batch[k]) for k in self.label_keys})

        return x_batch, y_batch

    def on_epoch_end(self):
        """Shuffle indices at the end of an epoch if shuffle is enabled."""
        if self.shuffle and self.N > 0:
            np.random.shuffle(self.indices)

    def _maybe_flatten(self, arr: np.ndarray, flatten_order: str = 'F', shuffle: bool = False, seed: int = 42) -> np.ndarray:
        """
        Flatten specified axes of an array if collapse_axes is set, supporting both Fortran and C-style orderings,
        with optional deterministic stratified shuffling via Latin Hypercube Sampling.
    
        Parameters:
        - arr: np.ndarray, the input array to reshape.
        - flatten_order: str, either 'C' (row-major) or 'F' (column-major) style flattening.
        - shuffle: bool, if True, performs stratified shuffling along the new flattened axis.
        - seed: int, seed for reproducible shuffling using a stateless PRNG.
    
        Returns:
        - A flattened and optionally shuffled view of the array.
        """
        if not self.collapse_axes:
            return arr  # No flattening requested — return original array unchanged.
    
        # Normalize and sort axes: convert negative indices and enforce order for reshaping.
        axes = sorted([a if a >= 0 else arr.ndim + a for a in self.collapse_axes])
        shape = list(arr.shape)
    
        if flatten_order.upper() == 'C':
            # C-style flattening: treat flattened axis as row-major (last axis changes fastest).
            prod = 1
            for a in reversed(axes):
                prod *= shape.pop(a)  # Multiply out the dimensions to be collapsed (from last to first).
    
            first = axes[0]  # Insert flattened dimension at the position of the first collapsed axis.
            new_shape = shape[:first] + [prod] + shape[first:]
    
            # Reshape to new flattened shape.
            flat = np.reshape(arr, new_shape)
    
            if first != 0:
                # Move flattened axis to the front if it’s not already first.
                flat = np.moveaxis(flat, first, 0)
    
        elif flatten_order.upper() == 'F':
            # Fortran-style flattening: column-major logic (first axis varies fastest).
            collapse_shape = [arr.shape[a] for a in axes]  # Sizes of axes to collapse.
            other_axes = [i for i in range(arr.ndim) if i not in axes]  # All other axes stay in place.
    
            # Reorder: move non-collapsed axes before collapsed ones.
            permuted_axes = other_axes + axes
            arr_perm = np.transpose(arr, permuted_axes)  # Transpose to bring collapse_axes to the end.
    
            # Compute reshaped dimensions with flattened collapse region.
            new_shape = [arr.shape[i] for i in other_axes] + [np.prod(collapse_shape)]
            flat = np.reshape(arr_perm, new_shape, order='F')  # Reshape in Fortran order.
    
            # Move flattened axis back to its logical place if not at the front.
            first = axes[0]
            flat_axis_index = len(new_shape) - 1
            if first != flat_axis_index:
                flat = np.moveaxis(flat, flat_axis_index, first)
    
        else:
            raise ValueError("flatten_order must be either 'C' or 'F'")
    
        if shuffle:
            # Deterministic stratified shuffling using Latin Hypercube Sampling (LHS) principles.
            n = flat.shape[0]  # Size of flattened axis.
    
            rng = np.random.default_rng(seed)  # Stateless RNG seeded for reproducibility.
    
            # Partition the axis into n bins, one per sample, ensuring spread across the range.
            bins = np.linspace(0, n, n + 1, dtype=int)
    
            # Choose one sample per bin to preserve stratification.
            lhs_indices = np.array([
                rng.integers(low=bins[i], high=bins[i + 1]) for i in range(n)
            ], dtype=int)  # <--- enforce integer dtype for valid indexing
    
            # Randomize index order to finalize shuffling.
            rng.shuffle(lhs_indices)
    
            # Apply shuffled indices along the flattened axis (assumed to be axis 0).
            flat = flat[lhs_indices]
    
        return flat

# Example usage and testing
if __name__ == "__main__":
    n0, n1, c, h, w = 140, 52, 1, 39, 39

    # Define two feature-label pairs
    features1 = np.random.rand(n0, n1, c, h, w)
    labels1 = {'a': np.random.rand(n0, n1, c, h, w),
               'b': np.random.rand(n0, n1, c, h, w)}
    features2 = np.random.rand(n0, n1, c, h, w)
    labels2 = {'a': np.random.rand(n0, n1, c, h, w),
               'b': np.random.rand(n0, n1, c, h, w)}
    pairs = [(features1, labels1), (features2, labels2)]

    # Test with non-empty pairs and stack_labels=False (default)
    gen = BatchGenerator(pairs, batch_size=64)
    x, y = gen[0]
    if isinstance(y, dict):
        print('Default collapse, stack_labels=False: x', x.shape, 'y', {k: v.shape for k, v in y.items()})
    else:
        print('Default collapse, stack_labels=False: x', x.shape, 'y', y.shape)
    # Expected: x (64, 1, 39, 39), y {'a': (64, 1, 39, 39), 'b': (64, 1, 39, 39)}

    # Test with stack_labels=True
    gen_stacked = BatchGenerator(pairs, batch_size=64, stack_labels=True)
    x, y = gen_stacked[0]
    print(f'Stacked labels: x, {x.shape}, y, {y.shape}')
    # Expected: x (64, 1, 39, 39), y (2, 64, 1, 39, 39)

    # Test with empty pairs
    empty_gen = BatchGenerator([], batch_size=64)
    print('Empty generator length:', len(empty_gen))  # Expected: 0
    x, y = empty_gen[0]
    print('Empty batch: x', x.shape, 'y', y.shape)
    # Expected: x (0,), y (0,)
   
    # # No collapse
    # gen_no = BatchGenerator(pairs, batch_size=64, collapse_axes=[])
    # x, y = gen_no[0]
    # print('No collapse:', x.shape, y.shape)  # Expected: (64, 52, 1, 39, 39), (2, 64, 52, 1, 39, 39)

    # # Collapse only axis 0
    # gen0 = BatchGenerator(pairs, batch_size=64, collapse_axes=[0])
    # x, y = gen0[0]
    # print('Collapse axis0:', x.shape, y.shape)  # Expected: (64, 52, 1, 39, 39), (2, 64, 52, 1, 39, 39)

    # # Iterate over batches to complete an epoch
    # gen_custom = BatchGenerator(pairs, batch_size=64, collapse_axes=[0, 1], batch_axis=0)
    # for xb, yb in gen_custom:
    #     pass
    print('Epoch complete')

def build_optimizer_from_config(config):
    """
    Build a Keras optimizer from a configuration dict.

    NOTE: Due to TensorFlow/Keras limitations, weight_decay must be a float (constant) value.
    Passing a schedule (e.g., ExponentialDecay) as weight_decay will raise a ValueError.
    If exponential decay for weight_decay is requested, a clear error is raised.
    """
    opt_type = config['type'].lower()
    
    # Extract learning rate and weight decay base values
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    
    # Check for exponential decay configuration using the new nested structure
    decay_config = config.get('exponential_decay', {})
    decay_enabled = decay_config.get('enabled', False)
    staircase = decay_config.get('staircase', False)
    
    if decay_enabled:
        # Extract learning rate decay parameters
        lr_decay_config = decay_config.get('learning_rate', {})
        lr_decay_enabled = lr_decay_config.get('enabled', False)
        decay_steps = lr_decay_config.get('decay_steps', 100)
        decay_rate = lr_decay_config.get('decay_rate', 0.96)
        
        # Apply exponential decay to learning rate if enabled
        if lr_decay_enabled:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=staircase
            )
            # Update the config to use our scheduled learning rate
            config['learning_rate'] = learning_rate
        
        # Apply exponential decay to weight decay if enabled (only for AdamW)
        if opt_type == 'adamw':
            wd_decay_config = decay_config.get('weight_decay', {})
            wd_decay_enabled = wd_decay_config.get('enabled', False)
            wd_decay_rate = wd_decay_config.get('decay_rate', 0.98)
            
            if wd_decay_enabled:
                if tfa_installed and version.parse(tf.__version__) < version.parse("2.10.1"):
                    weight_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                       initial_learning_rate=weight_decay,  # Reusing the ExponentialDecay scheduler
                       decay_steps=decay_steps,             # Using same decay steps as learning rate
                       decay_rate=wd_decay_rate,            # But potentially different decay rate
                       staircase=staircase                  # Same staircase behavior
                    )
                # Update the config to use our scheduled weight decay
                config['weight_decay'] = weight_decay

                    
    # Common kwargs across most optimizers
    common_args = {k: v for k, v in config.items() if k in {
        'learning_rate', 'beta_1', 'beta_2', 'epsilon', 'weight_decay'
    }}

    if opt_type == 'adamw':
        if tfa_installed and version.parse(tf.__version__) < version.parse("2.10.1"):
            print("TensorFlow version is below 2.10.1. Activating fallback logic.")
            return tfa.optimizers.AdamW(**common_args)
        else:
            print("TensorFlow version is 2.10.1 or above. Proceeding as normal.")
            return tf.keras.optimizers.AdamW(**common_args)
    
    elif opt_type == 'adam':
        # Adam doesn't support weight_decay
        common_args.pop('weight_decay', None)
        return tf.keras.optimizers.Adam(**common_args)
    
    elif opt_type == 'adabelief':
        try:
            from adabelief_tf import AdaBeliefOptimizer
        except ImportError:
            raise ImportError("AdaBelief requires `adabelief-tf`. Install via: pip install adabelief-tf")

        return AdaBeliefOptimizer(**common_args)
    
    else:
        raise ValueError(f"Unsupported optimizer type: {config['type']}")

# Validate loss key when physics-mode fraction is zero.
def validate_loss_keys(
    train_ds: Union[tf.data.Dataset, list],
    loss_keys: Union[Dict[str, List], List],
    general_config: dict
) -> None:
    """Validate loss keys against y_batch in non-physics mode."""
    if general_config.get('physics_mode_fraction', 1.0) != 0:
        return  # Skip validation if not in non-physics mode

    if len(train_ds) == 0:
        raise ValueError("Training data is empty. Loss keys cannot be inferred.")

    # Get the first batch
    if isinstance(train_ds, tf.data.Dataset):
        x_batch, y_batch = next(iter(train_ds.take(1)))
    else:
        x_batch, y_batch = train_ds[0]

    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    
    # Convert y_batch to tensor(s)
    if hasattr(train_ds, 'is_dict') and train_ds.is_dict and hasattr(train_ds, 'label_keys'):
        y_batch = {k: tf.convert_to_tensor(y_batch[k], dtype=tf.float32) for k in train_ds.label_keys}
    else:
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

    # Count loss keys
    if isinstance(loss_keys, dict):
        for v in loss_keys.values():
            if not hasattr(v, '__len__'):
                raise ValueError(f"Loss keys values must be iterable, got {v}")
        n_loss_keys = sum(len(v) for v in loss_keys.values())
    else:
        n_loss_keys = len(loss_keys)

    # Count y_batch keys or dimensions
    n_y_keys = len(y_batch) if isinstance(y_batch, dict) else (
        y_batch.shape[0] if hasattr(y_batch, 'shape') and len(y_batch.shape) > 0 else 1
    )

    assert n_loss_keys == n_y_keys, (
        f"Mismatch between number of loss keys ({n_loss_keys}) and number of y_batch keys ({n_y_keys}) in non-physics mode."
    )

# Build model map and training loop
def build_model_map(input_shape, optimizer_model_names_map=None, fluid_type=None):
    """
    Build all required models using input shape derived from training data, including well_rate_bhp and (for GC) saturation model.
    Returns model_map.
    optimizer_model_names_map is used to build the models
    Example: optimizer_model_names_map = {
                                          'pressure': 'encoder_decoder',
                                          'time_step': 'residual_network',
                                          'fluid_property': 'pvt_model',
                                          'well_rate_bhp': 'well_rate_bhp_model',
                                          'saturation': 'saturation_model',
                                          }
    """
    # Get input shape from the shape of the training data
    # Given the training data shape is (K, T, D, H, W, C), the shape is (None, D, H, W, C)
    # K and T dimensions are the permeability-time dimensions obtained from weaving which is 
    # combined to a single dimension and split during batching

    input_shape = (None, *input_shape[2:])
    logging.info(f"Input shape inferred from training data: {input_shape}")

    # Always build main models
    
    main_model = build_encoder_decoder_with_hard(input_shape=input_shape)
    #main_model = build_residual_network_with_hard(input_shape=input_shape)
    time_step_model = build_residual_network_without_hard(input_shape=input_shape)
    pvt_model = build_pvt_model_without_hard(main_model)
    well_rate_bhp_model = WellRatesPressure()

    # Determine fluid type
    if fluid_type is None:
        fluid_type = DEFAULT_GENERAL_CONFIG.get('fluid_type', 'DG')

    # Build saturation model for GC
    saturation_model = None
    if fluid_type == 'GC':
        saturation_model = build_encoder_decoder_with_hard(input_shape=input_shape, name="saturation_model")

    # Build model map
    model_map = {
        'pressure': main_model,
        'time_step': time_step_model,
        'pvt_model': pvt_model,
        'well_rate_bhp_model': well_rate_bhp_model,
    }
    if fluid_type == 'GC' and saturation_model is not None:
        model_map['saturation_model'] = saturation_model

    # Log model summaries
    logging.info("\n" + "="*50)
    logging.info("MODEL ARCHITECTURE SUMMARY")
    logging.info("="*50)
    for name, model in model_map.items():
        logging.info(f"Model: {name}")

    return model_map

def watch_losses_and_log_variables(
    epoch,
    avg_losses_train,
    loss_keys,
    optimizer_keys,
    optimizer_trainable_models,
    log_variables_callback,
    loss_min_max,
    model_variables_history
):
    """
    Helper method to track and log model trainable variables and losses during watched epochs.

    Records model trainable variables and average training losses for the epoch, updating
    min/max loss values for later normalization. Calls the provided callback to log variables
    and stores the variables and losses in the history for best model selection.

    Args:
        epoch: Current epoch number (0-based).
        avg_losses_train: Dictionary of average training losses for the epoch.
        loss_keys: Dictionary of loss keys for each phase (e.g., {'gas': ['dom_g', 'ibc_g', ...]}).
        optimizer_keys: List of keys for trainable models.
        optimizer_trainable_models: List of trainable models corresponding to optimizer_keys.
        log_variables_callback: Callback function to log model trainable variables.
        loss_min_max: Dictionary tracking min/max loss values for normalization.
        model_variables_history: List to store history of model variables and losses.

    Returns:
        None (updates loss_min_max and model_variables_history in place).
    """
    model_variables = {}
    for i, key in enumerate(optimizer_keys):
        model = optimizer_trainable_models[i]
        model_variables[key] = [v.numpy() for v in model.trainable_variables]
    log_variables_callback(epoch, model_variables, sum(sum(avg_losses_train[phase].values()) for phase in loss_keys))
    # Update min/max loss values for normalization
    for phase in loss_keys:
        for key in loss_keys[phase]:
            loss_val = avg_losses_train[phase][key]
            loss_min_max[phase][key]['min'] = min(loss_min_max[phase][key]['min'], loss_val)
            loss_min_max[phase][key]['max'] = max(loss_min_max[phase][key]['max'], loss_val)
    # Store variables and losses for best model selection
    model_variables_history.append({
        'epoch': epoch + 1,
        'variables': model_variables,
        'losses': {phase: {key: avg_losses_train[phase][key] for key in loss_keys[phase]} for phase in loss_keys}
    })

def train_combined_models_unified(
    train_groups,
    val_groups,
    test_groups=None,
    model_map=None,
    optimizer_model_names_map=None,
    training_batch_size=None,
    testing_batch_size=None,
    epochs=5,
    callbacks=None,
    custom_loss_fn=None,
    verbose=1,
    general_config=None,
    validate_loss_keys=None,
    print_total_loss_only={'train':False, 'val':True},
    log_variables_callback=None,
    log_epoch_percentage=0.2
):
    """
    Unified method to build, configure, and train a multi-model architecture with multi-optimizer and custom loss logic.

    This function merges the setup and training logic from train_all_models_combined and train_combined_model_with_mapping.
    It builds all required models, configures optimizers, prepares datasets, and performs the training loop with detailed 
    logging, metric tracking, and callback support. Uses pre-computed gradients from custom_loss_fn. Losses are averaged
    across steps for epoch-level logging. In pure physics mode, total_val_loss is set to 0. Includes optional callback
    for logging model trainable variables at a specified percentage of epochs. After all epochs, losses from watched epochs
    are normalized per loss index to [0, 1], summed to compute total normalized loss per epoch, and the model variables
    from the epoch with the lowest total normalized loss are used to update the models and returned.

    Args:
        train_groups: Training data groups (list/tuple of (inputs, labels)).
        val_groups: Validation data groups (list/tuple of (inputs, labels)).
        test_groups: Test data groups (optional, same format as train_groups).
        model_map: Pre-built models map (optional).
        optimizer_model_names_map: Custom mapping of optimizer keys to model logical names (optional).
        training_batch_size: Batch size for training (defaults to value from DEFAULT_GENERAL_CONFIG if None).
        testing_batch_size: Batch size for validation/testing (defaults to value from DEFAULT_GENERAL_CONFIG if None).
        epochs: Number of training epochs.
        callbacks: Optional list of callbacks.
        custom_loss_fn: Optional custom loss function (if None, will be constructed based on fluid type).
        verbose: Verbosity level (0: silent, 1: progress).
        general_config: Optional general configuration dictionary (if None, will use DEFAULT_GENERAL_CONFIG).
        validate_loss_keys: Optional function to validate loss keys (if None, will use default validation).
        print_total_loss_only: Dictionary. If True, print only total loss (average of weighted losses); if False, print all loss components.
        log_variables_callback: Optional callback function to log model trainable variables (called at specified epoch percentage).
        log_epoch_percentage: Float, percentage of epochs at which to log trainable variables (default: 0.2 for last 20%).
    Returns:
        tuple: (model_map, history, best_model_variables) where model_map is a dictionary of trained models updated
               with the variables from the epoch with the lowest total normalized loss, history is a dictionary of
               training metrics for plotting, and best_model_variables is a dictionary of the selected model trainable
               variables.
    """

    # 1. Build model map and optimizer mapping
    if general_config is None:
        general_config = DEFAULT_GENERAL_CONFIG
        
    fluid_type = general_config['fluid_type']
    if model_map is None:
        model_map = build_model_map(train_groups, optimizer_model_names_map=optimizer_model_names_map, fluid_type=fluid_type)
        if model_map is None:
            logging.error("Model map could not be built. Exiting training pipeline.")
            return None, None, None
    logging.info(f"Built models with optimizer mapping: {optimizer_model_names_map}")

    if optimizer_model_names_map is None:
        optimizer_model_names_map = get_optimizer_model_mapping(fluid_type=fluid_type)
    logging.info(f"Using optimizer model mapping: {optimizer_model_names_map}")

    # 2. Print label shapes for debugging
    def print_label_shapes(labels, prefix="Label"): 
        if isinstance(labels, dict):
            for k, v in labels.items():
                print(f"{prefix} key '{k}': shape {np.shape(v)}")
        else:
            print(f"{prefix} shape: {np.shape(labels)}")
    [print_label_shapes(train_groups[i][1], prefix="Train label") for i in range(len(train_groups))]
    [print_label_shapes(val_groups[i][1], prefix="Val label") for i in range(len(val_groups))]
    if test_groups is not None:
        [print_label_shapes(test_groups[i][1], prefix="Test label") for i in range(len(test_groups))]

    # 3. Batch size configuration
    if training_batch_size is None:
        training_batch_size = general_config['training_batch_size']
    if testing_batch_size is None:
        testing_batch_size = general_config['testing_batch_size']
    logging.info(f"Using batch sizes: training={training_batch_size}, testing={testing_batch_size}, epochs={epochs}")

    # 4. Prepare datasets
    train_ds = BatchGenerator(train_groups, batch_size=training_batch_size)
    val_ds = BatchGenerator(val_groups, batch_size=testing_batch_size)
    
    # 5. Build optimizer-trainable variable lists for each logical key
    optimizer_model_map = custom_loss_fn.optimizer_model_map             # {'pressure': pressure_optimizer, 'time_step': time_step_optimizer, 'saturation': saturation_model_optimizer}
    optimizer_trainable_models = custom_loss_fn.trainable_models
    optimizer_keys = custom_loss_fn.trainable_models_keys
    # Add the different optimizer for each key using the get_optimizer_config    
    
    # 6. Get the loss keys from the loss function and asserts it matches the keys in the dataset
    # This is a dictionary of loss lists to be computed and displayed during training
    # e.g.,for DG = {'gas': ['dom_g', 'ibc_g',..., 'tde_g']}, and for GC = {'gas': ['dom_g', 'ibc_g',..., 'tde_g'], 'oil': ['dom_o', 'ibc_o', ..., 'tde_o']}
    loss_keys = custom_loss_fn.loss_keys

    # Validate loss keys using a small batch or dummy input when non-physics-based
    if validate_loss_keys:
        validate_loss_keys(train_ds, loss_keys, general_config)

    cbks = callbacks or []
    history = {
        'train': {phase: {key: [] for key in keys} for phase, keys in loss_keys.items()},
        'val': {phase: {key: [] for key in keys} for phase, keys in loss_keys.items()},
        'epoch_times': [],
        'total_train_loss': [],
        'total_val_loss': [],
    }
    # Initialize storage for tracking model variables and losses
    model_variables_history = []
    # Initialize storage for min/max loss values during watched epochs for normalization
    loss_min_max = {phase: {key: {'min': float('inf'), 'max': float('-inf')} for key in keys} for phase, keys in loss_keys.items()}
    total_training_start = time.time()
    
    # Calculate epochs to log variables (last log_epoch_percentage of epochs)
    log_start_epoch = max(0, int(epochs * (1.0 - log_epoch_percentage)))
    
    # 7. Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"{'-'*60}")
        train_losses = {phase: {key: [] for key in keys} for phase, keys in loss_keys.items()}
        if len(train_ds) == 0:
            if verbose:
                print("No training data available. Skipping epoch.")
            continue
        for step in range(len(train_ds)):
            x_batch, y_batch = train_ds[step]
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            if train_ds.is_dict:
                y_batch = {k: tf.convert_to_tensor(y_batch[k], dtype=tf.float32) for k in train_ds.label_keys}
            else:
                y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
    
            # Compute losses and use pre-computed gradients from custom_loss_fn
            loss_outputs = custom_loss_fn.pinn_batch_sse_grad(x_batch, y_batch)
    
            if custom_loss_fn.physics_mode_fraction >= 1.0:  # Pure physics mode
                if fluid_type == 'DG':
                    wmse, wmse_grad, wsse, error_count, y_model = loss_outputs
                    loss_dict = {phase: {key: wmse[0][i].numpy() for i, key in enumerate(loss_keys[phase])} 
                                 for phase in loss_keys}
                    total_loss = tf.reduce_sum(wmse[0]).numpy()
                    if np.any(np.array(error_count) == 0):
                        logging.warning(f"Zero error count detected in DG physics mode, step {step+1}")
                    if np.all(np.array(wmse[0]) == 0):
                        logging.warning(f"All wmse values are zero in DG physics mode, step {step+1}")
                else:  # GC
                    wmse_g_o, wmse_grad, wsse_g_o, error_count_g_o, y_model = loss_outputs
                    loss_dict = {
                        'gas': {key: wmse_g_o[0][i].numpy() for i, key in enumerate(loss_keys['gas'])},
                        'oil': {key: wmse_g_o[1][i].numpy() for i, key in enumerate(loss_keys['oil'])}
                    }
                    total_loss = tf.reduce_sum(wmse_g_o[0] + wmse_g_o[1]).numpy()
                    if np.any(np.array(error_count_g_o) == 0):
                        logging.warning(f"Zero error count detected in GC physics mode, step {step+1}")
                    if np.all(np.array(wmse_g_o[0] + wmse_g_o[1]) == 0):
                        logging.warning(f"All wmse values are zero in GC physics mode, step {step+1}")
            else:  # Non-physics mode
                td_wmse, wmse_grad, td_wsse, error_count, y_model = loss_outputs
                if fluid_type == 'DG':
                    loss_dict = {'gas': {key: td_wmse[0].numpy() for key in loss_keys['gas']}}
                    total_loss = tf.reduce_sum(td_wmse).numpy()
                else:  # GC
                    loss_dict = {
                        'gas': {key: td_wmse[0].numpy() for key in loss_keys['gas']},
                        'oil': {key: td_wmse[1].numpy() for key in loss_keys['oil']}
                    }
                    total_loss = tf.reduce_sum(td_wmse).numpy()
                if np.any(np.array(error_count) == 0):
                    logging.warning(f"Zero error count detected in non-physics mode, step {step+1}")
                if np.all(np.array(td_wmse) == 0):
                    logging.warning(f"All td_wmse values are zero in non-physics mode, step {step+1}")

            # Apply pre-computed gradients
            for i, key in enumerate(optimizer_keys):
                model = optimizer_trainable_models[i]
                if len(model.trainable_variables) > 0:
                    grads = wmse_grad[i]
                    if any(grad is None for grad in grads):
                        logging.warning(f"Some gradients for {key} are None. Check loss calculation or model connectivity.")
                        logging.info(f"Module: {key}, Variables: {len(model.trainable_variables)}, Total loss: {total_loss}")
                    optimizer_model_map[key].optimizer.apply_gradients(
                        zip(grads, model.trainable_variables)
                    )
    
            # Store losses for the step
            for phase in loss_keys:
                for key in loss_keys[phase]:
                    train_losses[phase][key].append(loss_dict[phase][key])
    
            # Print losses
            if verbose:
                if print_total_loss_only['train']:
                    print(f"Step {step+1}/{len(train_ds)} - Total Loss: {total_loss:.4f}", end='\r')
                else:
                    print_loss = []
                    for phase in loss_keys:
                        for key in loss_keys[phase]:
                            loss_val = loss_dict[phase][key]
                            if abs(loss_val) < 1e-4 and loss_val != 0:
                                formatted_loss = f"{loss_val:.4e}"
                            else:
                                formatted_loss = f"{loss_val:.4f}"
                            print_loss.append(f"{phase}_{key}: {formatted_loss}")
                    print(f"Step {step+1}/{len(train_ds)} - {' - '.join(print_loss)}", end='\r')
    
        # Ensure newline after step loop to avoid overlap
        if verbose:
            print()
    
        # Compute average losses for the epoch (for logging, not optimization)
        if train_losses:
            # Average losses across all steps
            avg_losses_train = {phase: {key: np.mean(train_losses[phase][key]) 
                                       for key in train_losses[phase]} 
                               for phase in loss_keys}
            epoch_time_ms = (time.time() - epoch_start_time) * 1000
            history['epoch_times'].append(epoch_time_ms)
            for phase in loss_keys:
                for key in loss_keys[phase]:
                    history['train'][phase][key].append(float(avg_losses_train[phase][key]))
            total_train_loss = sum(sum(avg_losses_train[phase].values()) for phase in loss_keys)
            history['total_train_loss'].append(float(total_train_loss))
            if verbose:
                if print_total_loss_only['train']:
                    print(f"Training: Total Loss: {total_train_loss:.4f} - time: {epoch_time_ms:.0f} ms")
                else:
                    loss_str = []
                    for phase in loss_keys:
                        for key in loss_keys[phase]:
                            loss_val = avg_losses_train[phase][key]
                            if abs(loss_val) < 1e-4 and loss_val != 0:
                                formatted_loss = f"{loss_val:.4e}"
                            else:
                                formatted_loss = f"{loss_val:.4f}"
                            loss_str.append(f"{phase}_{key}: {formatted_loss}")
                    print(f"Training: {' - '.join(loss_str)} - time: {epoch_time_ms:.0f} ms")
    
        # Log model trainable variables and losses if in the specified epoch range
        if epoch >= log_start_epoch and log_variables_callback is not None:
            watch_losses_and_log_variables(
                epoch,
                avg_losses_train,
                loss_keys,
                optimizer_keys,
                optimizer_trainable_models,
                log_variables_callback,
                loss_min_max,
                model_variables_history
            )
    
        # Validation loop
        val_losses = {phase: {key: [] for key in keys} for phase, keys in loss_keys.items()}
        total_val_loss_sum = 0.0  # Accumulate for averaging in non-physics mode
        if len(val_ds) > 0:
            for step in range(len(val_ds)):
                x_batch, y_batch = val_ds[step]
                x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                if val_ds.is_dict:
                    y_batch = {k: tf.convert_to_tensor(y_batch[k], dtype=tf.float32) for k in val_ds.label_keys}
                else:
                    y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
                loss_outputs = custom_loss_fn.pinn_batch_sse_grad(x_batch, y_batch)
                if custom_loss_fn.physics_mode_fraction >= 1.0:  # Pure physics mode
                    total_val_loss = 0.0  # Set to 0 as required
                    if fluid_type == 'DG':
                        wmse, _, wsse, error_count, y_model = loss_outputs
                        loss_dict = {phase: {key: wmse[0][i].numpy() for i, key in enumerate(loss_keys[phase])} 
                                     for phase in loss_keys}
                        if np.any(np.array(error_count) == 0):
                            logging.warning(f"Zero error count in validation, DG physics mode, step {step+1}")
                        if np.all(np.array(wmse[0]) == 0):
                            logging.warning(f"All wmse values are zero in validation, DG physics mode, step {step+1}")
                    else:  # GC
                        wmse_g_o, _, wsse_g_o, error_count_g_o, y_model = loss_outputs
                        loss_dict = {
                            'gas': {key: wmse_g_o[0][i].numpy() for i, key in enumerate(loss_keys['gas'])},
                            'oil': {key: wmse_g_o[1][i].numpy() for i, key in enumerate(loss_keys['oil'])}
                        }
                        if np.any(np.array(error_count_g_o) == 0):
                            logging.warning(f"Zero error count in validation, GC physics mode, step {step+1}")
                        if np.all(np.array(wmse_g_o[0] + wmse_g_o[1]) == 0):
                            logging.warning(f"All wmse values are zero in validation, GC physics mode, step {step+1}")
                    logging.info(f"Pure physics mode: total_val_loss set to {total_val_loss} for validation step {step+1}")
                else:  # Non-physics mode
                    td_wmse, _, td_wsse, error_count, y_model = loss_outputs
                    if fluid_type == 'DG':
                        loss_dict = {'gas': {key: td_wmse[0].numpy() for key in loss_keys['gas']}}
                        total_val_loss = tf.reduce_sum(td_wmse).numpy()
                    else:  # GC
                        loss_dict = {
                            'gas': {key: td_wmse[0].numpy() for key in loss_keys['gas']},
                            'oil': {key: td_wmse[1].numpy() for key in loss_keys['oil']}
                        }
                        total_val_loss = tf.reduce_sum(td_wmse).numpy()
                    total_val_loss_sum += total_val_loss
                    if np.any(np.array(error_count) == 0):
                        logging.warning(f"Zero error count in validation, non-physics mode, step {step+1}")
                    if np.all(np.array(td_wmse) == 0):
                        logging.warning(f"All td_wmse values are zero in validation, non-physics mode, step {step+1}")
    
                for phase in loss_keys:
                    for key in loss_keys[phase]:
                        val_losses[phase][key].append(loss_dict[phase][key])
    
        # Compute average validation losses (for logging)
        if val_losses:
            # Average losses across all steps
            avg_losses_val = {phase: {key: np.mean(val_losses[phase][key]) 
                                     for key in loss_keys[phase]} 
                             for phase in loss_keys}
            if custom_loss_fn.physics_mode_fraction >= 1.0:
                history['total_val_loss'].append(0.0)  # Set to 0 in pure physics mode
            else:
                # Average total_val_loss across steps
                history['total_val_loss'].append(float(total_val_loss_sum / len(val_ds)) if len(val_ds) > 0 else 0.0)
            for phase in loss_keys:
                for key in loss_keys[phase]:
                    history['val'][phase][key].append(float(avg_losses_val[phase][key]))
            if verbose:
                if print_total_loss_only['val']:
                    print(f"Validation: Total Loss: {history['total_val_loss'][-1]:.4f}")
                else:
                    loss_str = []
                    for phase in loss_keys:
                        for key in loss_keys[phase]:
                            loss_val = avg_losses_val[phase][key]
                            if abs(loss_val) < 1e-4 and loss_val != 0:
                                formatted_loss = f"{loss_val:.4e}"
                            else:
                                formatted_loss = f"{loss_val:.4f}"
                            loss_str.append(f"{phase}_{key}: {formatted_loss}")
                    print(f"Validation: {' - '.join(loss_str)} - Total: {history['total_val_loss'][-1]:.4f}")
            # Ensure newline after validation to separate from summary
            if verbose:
                print()
    
        # Summary
        if verbose:
            print()  # Extra newline for clear separation
            if print_total_loss_only['train']:
                print(f"Epoch {epoch+1} summary - Total: {total_train_loss:.4f} (val: {history['total_val_loss'][-1]:.4f})")
            else:
                loss_table = []
                for phase in loss_keys:
                    for key in loss_keys[phase]:
                        val_mean = avg_losses_val[phase][key] if val_losses else 0.0
                        train_loss_val = avg_losses_train[phase][key]
                        val_loss_val = val_mean
                        if abs(train_loss_val) < 1e-4 and train_loss_val != 0:
                            formatted_train = f"{train_loss_val:.4e}"
                        else:
                            formatted_train = f"{train_loss_val:.4f}"
                        if abs(val_loss_val) < 1e-4 and val_loss_val != 0:
                            formatted_val = f"{val_loss_val:.4e}"
                        else:
                            formatted_val = f"{val_loss_val:.4f}"
                        loss_table.append(f"{phase}_{key}: {formatted_train} (val: {formatted_val})")
                print(f"Epoch {epoch+1} summary - {' | '.join(loss_table)}")
    
        for cbk in cbks:
            cbk.on_epoch_end(epoch)
        train_ds.on_epoch_end()
    
    # Select best model variables by normalizing watched losses and finding the lowest total
    best_model_variables = None
    if model_variables_history:
        # Normalize losses across watched epochs
        normalized_losses = []
        for record in model_variables_history:
            total_normalized_loss = 0.0
            for phase in loss_keys:
                for key in loss_keys[phase]:
                    loss_val = record['losses'][phase][key]
                    min_val = loss_min_max[phase][key]['min']
                    max_val = loss_min_max[phase][key]['max']
                    # Avoid division by zero in normalization
                    if max_val > min_val:
                        normalized_loss = (loss_val - min_val) / (max_val - min_val)
                    else:
                        normalized_loss = 0.0 if loss_val == min_val else 1.0
                    total_normalized_loss += normalized_loss
            normalized_losses.append(total_normalized_loss)
        
        # Find epoch with lowest total normalized loss
        best_epoch_idx = np.argmin(normalized_losses)
        best_total_normalized_loss = normalized_losses[best_epoch_idx]
        best_model_variables = model_variables_history[best_epoch_idx]['variables']
        best_epoch = model_variables_history[best_epoch_idx]['epoch']
        
        # Update models in model_map with best variables
        for key in best_model_variables:
            if key in optimizer_keys:
                model_idx = optimizer_keys.index(key)
                model = optimizer_trainable_models[model_idx]
                for var, best_var in zip(model.trainable_variables, best_model_variables[key]):
                    var.assign(best_var)
        logging.info(f"Updated models with variables from epoch {best_epoch} with lowest total normalized loss: {best_total_normalized_loss:.4f}")
    else:
        logging.info("No model variables were logged during training.")
    
    total_training_time = time.time() - total_training_start
    if verbose:
        print(f"\n{'='*60}")
        print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time*1000:.0f} ms)")
        print(f"{'='*60}")
    
    # Log the final loss values
    if history and 'total_train_loss' in history and len(history['total_train_loss']) > 0:
        final_train_loss = history['total_train_loss'][-1]
        logging.info(f"Final total training loss: {final_train_loss:.4f}")
        logging.info("Final training loss components:")
        for phase in history['train']:
            for key in history['train'][phase]:
                if len(history['train'][phase][key]) > 0:
                    logging.info(f"  {phase}_{key}: {history['train'][phase][key][-1]:.4f}")

    return model_map, history, best_model_variables

