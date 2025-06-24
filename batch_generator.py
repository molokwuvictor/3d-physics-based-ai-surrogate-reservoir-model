import numpy as np
import tensorflow as tf
import logging
from collections import Counter
from typing import List, Union, Dict, Sequence, Optional
try:
    import tensorflow_addons as tfa
except ImportError:
    tfa = None

class BatchGenerator:
    """
    Generic batch generator for features and labels from a list of feature-label pairs.

    Input is a list of pairs, where each pair consists of:
    - features: a single array with shape (d0, d1, ...), e.g., np.random.rand(n0, n1, c, h, w)
    - labels: either an array or a dictionary with arrays, e.g., {'a': array, 'b': array}

    You may specify which axes to collapse (e.g., [0, 1] by default).
    If `collapse_axes` is None or empty, no collapsing is performed.

    You may also specify the axis used for batching (after flattening).
    By default, batching is performed along axis 0 for features and axis 0 or 1 for labels based on type.

    If the input `pairs` is empty, initializes an empty BatchGenerator with zero samples.

    Returns:
      x: shape (batch_size, *feature_dims)
      y: if labels array => (batch_size, *label_dims)
         if labels dict  => (k, batch_size, *label_dims) where k = number of keys
    """
    def __init__(
        self,
        pairs: List[tuple[np.ndarray, Union[np.ndarray, Dict[str, np.ndarray]]]],
        batch_size: int,
        collapse_axes: Optional[Sequence[int]] = (0, 1),
        batch_axis: int = 0,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collapse_axes = list(collapse_axes) if collapse_axes else []
        self.batch_axis = batch_axis

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
            # For dictionary labels, collect flattened arrays for each key across pairs
            y_all_dict = {k: [self._maybe_flatten(labels[k]) for _, labels in pairs] for k in self.label_keys}
            # Concatenate arrays for each key across pairs, then stack
            y_all_stacked = [np.concatenate(y_all_dict[k], axis=0) for k in self.label_keys]
            self.y_all = np.stack(y_all_stacked, axis=0)  # Shape: (num_keys, total_samples, *dims)
        else:
            # For array labels, flatten each label and concatenate across pairs
            y_all_list = [self._maybe_flatten(labels) for _, labels in pairs]
            self.y_all = np.concatenate(y_all_list, axis=0)  # Shape: (total_samples, *dims)

        # Set up indices for batching
        self.N = self.x_all.shape[0]  # Total number of samples after flattening
        self.indices = np.arange(self.N)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """Return the number of batches."""
        return int(np.ceil(self.N / self.batch_size))

    def __getitem__(self, idx: int):
        """Retrieve a batch by index."""
        if self.N == 0:
            return np.array([]), np.array([])  # Return empty arrays for empty generator

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.N)
        batch_inds = self.indices[start:end]

        x_batch = np.take(self.x_all, batch_inds, axis=self.batch_axis)  # Batch axis is 0 for x
        if self.is_dict:
            y_batch = np.take(self.y_all, batch_inds, axis=1)  # Batch axis is 1 for dict labels
        else:
            y_batch = np.take(self.y_all, batch_inds, axis=self.batch_axis)  # Batch axis is 0

        return x_batch, y_batch

    def on_epoch_end(self):
        """Shuffle indices at the end of an epoch if shuffle is enabled."""
        if self.shuffle and self.N > 0:
            np.random.shuffle(self.indices)

    def _maybe_flatten(self, arr: np.ndarray) -> np.ndarray:
        """Flatten specified axes of an array if collapse_axes is set."""
        if not self.collapse_axes:
            return arr

        axes = sorted([a if a >= 0 else arr.ndim + a for a in self.collapse_axes])
        shape = list(arr.shape)
        prod = 1
        for a in reversed(axes):
            prod *= shape.pop(a)
        first = axes[0]
        new_shape = shape[:first] + [prod] + shape[first:]
        flat = np.reshape(arr, new_shape)  # Changed from tf.reshape to np.reshape
        if first != 0:
            flat = np.moveaxis(flat, first, 0)
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

    # Test with non-empty pairs
    gen = BatchGenerator(pairs, batch_size=64)
    x, y = gen[0]
    print('Default collapse: x', x.shape, 'y', y.shape)  # Expected: (64, 1, 39, 39), (2, 64, 1, 39, 39)

    # Test with empty pairs
    empty_gen = BatchGenerator([], batch_size=64)
    print('Empty generator length:', len(empty_gen))  # Expected: 0
    x, y = empty_gen[0]
    print('Empty batch: x', x.shape, 'y', y.shape)  # Expected: (0,), (0,)
    
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