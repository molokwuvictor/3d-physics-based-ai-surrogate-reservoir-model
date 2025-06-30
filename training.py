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