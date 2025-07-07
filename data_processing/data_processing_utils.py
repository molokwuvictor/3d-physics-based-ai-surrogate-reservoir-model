# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 02:05:34 2025

@author: User
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import json
from typing import List, Tuple, Union, Dict, Any, Optional

def reshape_and_save_dict(A, filetype='df', filename='x', save_dir=None):
    """
    Reshape dictionary values to 1D tensors, convert to DataFrame, and save.

    Args:
        A (dict): Dictionary with string keys and tensor-convertible values.
        filetype (str): 'df' to save as Pickle (.df), 'csv' to save as CSV (.csv).
        filename (str): Name of the output file (without extension).
        save_dir (str or None): Directory to save the file. Defaults to current directory.
    """
    reshaped_dict = {}

    for key, value in A.items():
        tensor = tf.convert_to_tensor(value)
        flat_tensor = tf.reshape(tensor, [-1])
        reshaped_dict[key] = flat_tensor.numpy()

    max_len = max(len(v) for v in reshaped_dict.values())
    padded_dict = {
        k: list(v) + [None] * (max_len - len(v))
        for k, v in reshaped_dict.items()
    }

    df = pd.DataFrame(padded_dict)

    # Handle save directory
    save_dir = save_dir or os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    # File path construction
    ext = 'df' if filetype == 'df' else 'csv'
    path = os.path.join(save_dir, f"{filename}.{ext}")

    # Save
    if filetype == 'csv':
        df.to_csv(path, index=False)
    elif filetype == 'df':
        df.to_pickle(path)
    else:
        raise ValueError("Invalid filetype. Use 'df' or 'csv'.")

# Save as x.csv in current directory
# reshape_and_save_dict(poly_coeff_Bu_spl, filetype='df',filename='pvt_data')

# Save as custom_name.df in a specific folder
#reshape_and_save_dict(A, filetype='df', filename='custom_name', save_dir='/path/to/save')

def load_dataframe(filename='x', filetype='df', load_dir=None):
    """
    Load a DataFrame from a specified or current working directory.

    Args:
        filename (str): Name of the file (without extension).
        filetype (str): 'df' for Pickle (.df), 'csv' for CSV (.csv).
        load_dir (str or None): Directory to load the file from. Defaults to current directory.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # Set directory
    load_dir = load_dir or os.getcwd()
    
    # Determine file extension
    ext = 'df' if filetype == 'df' else 'csv'
    
    # Build full file path
    path = os.path.join(load_dir, f"{filename}.{ext}")

    # Load file based on filetype
    if filetype == 'csv':
        return pd.read_csv(path)
    elif filetype == 'df':
        return pd.read_pickle(path)
    else:
        raise ValueError("Invalid filetype. Use 'df' or 'csv'.")

def weave_tensors(tensor_list, target_trailing_shape=None, flip_innermost_index=True, flatten_first_axes=False, merge_consecutive_singleton_dims=True):
    """
    Weave an arbitrary number of tensors by interleaving their leading values along new axes.
    
    Each tensor in the tensor_list is assumed to have shape:
       (N_i, *tail_i)
    where the trailing dimensions (tail_i) are intended to be broadcast to a common target
    spatial shape. If a given tensor's trailing dimensions do not match target_trailing_shape,
    then it will be tiled accordingly (assuming its trailing dims are 1 where needed).
    
    Parameters:
    -----------
    tensor_list : list of np.ndarray
        A list of tensors to be woven. Their leading dimensions are the ones that get woven.
    target_trailing_shape : tuple, optional
        The target shape for the trailing (non-leading) dimensions. If None, the trailing shape 
        of the first tensor in tensor_list is used.
    flip_innermost_index: bool, default True
        Tensor with the same shape, but with the innermost dimension indices reversed
        (i.e., from [0, 1, ..., d-1] to [d-1, d-2, ..., 0]).
    flatten_first_axes : bool, default False
        If True, the woven tensor’s first d axes (one per input tensor) will be flattened into one.
    merge_consecutive_singleton_dims : bool, default True
        If True, consecutive singleton dimensions in the woven tensor will be merged into one.
    
    Returns:
    --------
    woven : np.ndarray
        The woven tensor with shape:
          (N_1, N_2, ..., N_d, *target_trailing_shape, d)
        or if flattened: ((N_1*N_2*...*N_d), *target_trailing_shape, d)
    """
    d = len(tensor_list)
    if d == 0:
        raise ValueError("tensor_list must contain at least one tensor.")
    
    # Determine target trailing shape:
    if target_trailing_shape is None:
        # Use trailing dims of first tensor.
        target_trailing_shape = tensor_list[0].shape[1:]
    
    # Process each tensor so that it has shape: (N_1, N_2, ... N_d, *target_trailing_shape)
    processed_tensors = []
    # First, record the sizes from each tensor's leading dimension.
    leading_sizes = [tensor.shape[0] for tensor in tensor_list]
    
    for i, tensor in enumerate(tensor_list):
        # Get original shape components:
        orig_shape = tensor.shape
        N_i = orig_shape[0]
        tail = orig_shape[1:]
        
        # If tail has lower rank than target_trailing_shape, prepend singleton dimensions.
        if len(tail) < len(target_trailing_shape):
            tail = (1,) * (len(target_trailing_shape) - len(tail)) + tail
            tensor = tensor.reshape((N_i,) + tail)
        
        # Now, for each dimension in tail, if it does not match target_trailing_shape,
        # then it must be 1 and we will tile it.
        tile_factors_tail = []
        new_tail = list(tail)
        for j, (dim_size, target_dim) in enumerate(zip(new_tail, target_trailing_shape)):
            if dim_size == target_dim:
                tile_factors_tail.append(1)
            elif dim_size == 1:
                tile_factors_tail.append(target_dim)
            else:
                raise ValueError(f"Tensor {i} trailing dimension {j} (size {dim_size}) "
                                 f"cannot be broadcast to target dimension {target_dim}.")
        
        # Tile the tensor over its trailing dimensions if needed.
        tensor_tiled = np.tile(tensor, (1, *tile_factors_tail))
        # At this point, tensor_tiled has shape: (N_i, *target_trailing_shape)
        
        # Now, we want to “weave” tensor i with the others.
        # We want to add d (number of tensors) axes in front of the trailing dims so that each tensor's
        # leading axis appears in a different location.
        # For tensor i, we add singleton axes before the trailing dimensions such that it has shape:
        # (1,)*i + (N_i,) + (1,)*(d - i - 1) + target_trailing_shape.
        new_shape = ( (1,)*i ) + (N_i,) + ( (1,)*(d - i - 1) ) + target_trailing_shape
        tensor_expanded = tensor_tiled.reshape(new_shape)
        
        # Next, we tile along the new axes to broadcast to the full leading dimensions.
        # For each leading axis position j (0 <= j < d), if j == i, factor=1, otherwise factor=leading_sizes[j].
        tile_factors = [leading_sizes[j] if j != i else 1 for j in range(d)]
        # For the trailing dimensions, we do not tile further.
        tile_factors = tuple(tile_factors + [1]*len(target_trailing_shape))
        
        tensor_broadcasted = np.tile(tensor_expanded, tile_factors)
        # Now tensor_broadcasted shape should be: (N_1, N_2, ..., N_d, *target_trailing_shape)
        processed_tensors.append(tensor_broadcasted)
    
    # Stack all processed tensors along a new last axis.
    # The stacked tensor has shape: (N_1, N_2, ..., N_d, *target_trailing_shape, d)
    woven = np.stack(processed_tensors, axis=-1)
    
    if flatten_first_axes:
        # Flatten the first d dimensions into one.
        new_leading = np.prod(leading_sizes)
        woven = woven.reshape((new_leading,) + woven.shape[d:-1] + (woven.shape[-1],))
    
    # Optional: merge_consecutive_singleton_dims=True, collapse any run of 1’s into a single 1
    if merge_consecutive_singleton_dims:
        def _collapse_runs_of_ones(arr: np.ndarray) -> np.ndarray:
            """
            Given a numpy array arr, collapse every maximal consecutive run of dimensions == 1
            into exactly one dimension of size 1.
            E.g. shape (4, 1, 1, 3, 1, 1, 1, 5) → (4, 1, 3, 1, 5).
            """
            old_shape = arr.shape
            new_shape = []
            seen_one = False
            for dim in old_shape:
                if dim == 1:
                    if not seen_one:
                        # first '1' in a run → keep one
                        new_shape.append(1)
                        seen_one = True
                    else:
                        # skip this dimension (it’s a consecutive 1)
                        continue
                else:
                    new_shape.append(dim)
                    seen_one = False

            return arr.reshape(tuple(new_shape))

        woven = _collapse_runs_of_ones(woven) 
        
    # Innermost dimension indices reversed to conform to SRM architecture - [..., [z, y, x, t, k]]
    if flip_innermost_index:
        # Flip the innermost index
        woven = tf.reverse(woven, axis=[-1])
    return woven

def create_positional_grids(D, N, indexing='ij', transpose_order=None):
    """
    Create a list of coordinate arrays representing the midpoints of each grid cell,
    with an option to transpose the dimensions of each coordinate array.

    Parameters:
      D (list or array-like): Spatial distances along each dimension, e.g., [X, Y, Z, ...].
      N (list or array-like): Number of grid cells per dimension, e.g., [Nx, Ny, Nz, ...].
      indexing (str): 'ij' for matrix indexing or 'xy' for Cartesian indexing.
      transpose_order (list or None): A list specifying the desired order of dimensions
                                      for each coordinate array. For example, [2, 1, 0]
                                      would reorder dimensions from [0, 1, 2] to [2, 1, 0].
                                      If None, no transposition is performed.

    Returns:
      grids (list of np.ndarray): A list where each element is an ndarray representing
                                  the coordinate values along one dimension, possibly
                                  transposed as specified.
    """
    # Validate inputs
    if len(D) != len(N):
        raise ValueError("The length of D and N must be the same, representing each spatial dimension.")

    # Calculate the midpoints for each dimension
    positions = []
    for d, n in zip(D, N):
        delta = d / n  # grid cell width
        # Create an array of midpoints: (0.5*delta, 1.5*delta, ..., (n-0.5)*delta)
        pos = (np.arange(n) + 0.5) * delta
        positions.append(pos)

    # Create a meshgrid from the list of midpoints with specified indexing
    grids = np.meshgrid(*positions, indexing=indexing)

    # If a transpose order is specified, permute the dimensions accordingly
    if transpose_order is not None:
        grids = [np.transpose(grid, axes=transpose_order) for grid in grids]

    return grids

# Test with example inputs
if __name__ == "__main__":
    # Example spatial dimensions and number of grids per dimension:
    D = [10.0, 20.0, 30.0]  # e.g., X=10, Y=20, Z=30
    N = [2, 4, 3]           # e.g., 2 grids in X, 4 in Y, 3 in Z

    # Create the positional grids without transposition
    grids_default = create_positional_grids(D, N)
    print("Default positional grids shapes:")
    for i, grid in enumerate(grids_default):
        print(f"Dimension {i} grid shape: {grid.shape}")

    # Create the positional grids with transposition (e.g., swapping first and third dimensions)
    transpose_order = [2, 1, 0]
    grids_transposed = create_positional_grids(D, N, transpose_order=transpose_order)
    print("\nTransposed positional grids shapes:")
    for i, grid in enumerate(grids_transposed):
        print(f"Dimension {i} grid shape: {grid.shape}")


# Test with example inputs
if __name__ == "__main__":
    # Example spatial dimensions and number of grids per dimension:
    D = [10.0, 20.0, 30.0]  # e.g., X=10, Y=20, Z=30
    N = [2, 4, 3]          # e.g., 2 grids in X, 4 in Y, 3 in Z
    
    tensor = create_positional_grids(D, N)
    print("Positional tensor shape:", np.shape(tensor))  # Expected shape: [3, 2, 4, 3]
    print("Positional tensor values:")
    print(tensor)

    # ------------------------------------------------------------------------------
    # Example with simple numbers:
    # Let tensor A have shape (2, 3, 4) and tensor B have shape (3, 1)
    # We'll assume tensor A is like spatial data (with trailing shape (3,4)) and tensor B
    # is auxiliary data that should broadcast to (3,4).
    A = np.array([
        [[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12]],
        
        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]
    ])
    # A.shape = (2, 3, 4)
    B = np.array([
        [100],
        [200],
        [300]
    ])
    # B.shape = (3, 1) -> trailing dims (1,) which will be tiled to (3,4) if we set target_trailing_shape=(3,4)

    # We want to weave these two tensors (here d = 2) so that the final tensor has shape:
    # (2, 3, 4, 2) with the leading dimensions woven.
    woven = weave_tensors([A, B], target_trailing_shape=(3,4),flatten_first_axes=True)
    
    print("Woven tensor shape:", woven.shape)
    print("Woven tensor:\n", woven)

def align_and_trim_pair_lists(
    alist: List[tf.Tensor],
    blist: List[Union[tf.Tensor, Dict[Any, tf.Tensor]]],
    dims: List[int],
    trim_target: str = "both"  # one of "a", "b", "both"
) -> Tuple[List[tf.Tensor], List[Union[tf.Tensor, Dict[Any, tf.Tensor]]]]:
    """
    Align and optionally trim two parallel lists of tensors/dicts:

      - For each index i, compare a = alist[i] and b_item = blist[i].
      - For each axis in `dims`, compute target = min(a.shape[axis], b.shape[axis]).
      - Depending on `trim_target`, slice off the end of that axis in a, b, or both.

    Parameters
    ----------
    alist : List[tf.Tensor]
        Reference list of tensors.
    blist : List[Union[tf.Tensor, Dict[Any, tf.Tensor]]]
        List of tensors or dicts of tensors to align/trim alongside `alist`.
    dims : List[int]
        Axes to align (0-based).
    trim_target : str
        Which side(s) to trim: `"a"`, `"b"`, or `"both"`.

    Returns
    -------
    Tuple of two lists:
        - trimmed_alist: List[tf.Tensor]
        - trimmed_blist: List[Union[tf.Tensor, Dict[Any, tf.Tensor]]]
    """
    if len(alist) != len(blist):
        raise ValueError(f"alist and blist must be same length: {len(alist)} vs {len(blist)}")
    if trim_target not in {"a", "b", "both"}:
        raise ValueError(f"trim_target must be 'a', 'b', or 'both'; got {trim_target!r}")

    def _trim_tensor_to(x: tf.Tensor, axis: int, target: int) -> tf.Tensor:
        slicer = [slice(None)] * len(x.shape)
        slicer[axis] = slice(0, target)
        return x[tuple(slicer)]

    trimmed_alist: List[tf.Tensor] = []
    trimmed_blist: List[Union[tf.Tensor, Dict[Any, tf.Tensor]]] = []

    for idx, (a, b_item) in enumerate(zip(alist, blist)):
        # Validate dims
        rank_a = len(a.shape)
        for axis in dims:
            if axis < 0 or axis >= rank_a:
                raise ValueError(f"Axis {axis} out of range for alist[{idx}] (ndim={rank_a})")

        # Collect all shapes along dims: first a, then b_item's tensors
        axis_targets = {}
        for axis in dims:
            # a's length
            len_a = int(a.shape[axis])
            # b_item's lengths
            if isinstance(b_item, tf.Tensor):
                lens_b = [int(b_item.shape[axis])]
            else:
                lens_b = [int(v.shape[axis]) for v in b_item.values()]

            # min across all
            axis_targets[axis] = min([len_a] + lens_b)

        # Trim a
        a_trim = a
        for axis, tgt in axis_targets.items():
            if a_trim.shape[axis] > tgt:
                a_trim = _trim_tensor_to(a_trim, axis, tgt)

        # Trim b_item
        def _trim_b(x: tf.Tensor) -> tf.Tensor:
            y = x
            for axis, tgt in axis_targets.items():
                if y.shape[axis] > tgt:
                    y = _trim_tensor_to(y, axis, tgt)
            return y

        if isinstance(b_item, tf.Tensor):
            b_trim = _trim_b(b_item)
        else:
            b_trim = {k: _trim_b(v) for k, v in b_item.items()}

        trimmed_alist.append(a_trim)
        trimmed_blist.append(b_trim)

    return trimmed_alist, trimmed_blist

def split_tensor_sequence(
    tensor_groups: List[Union[tf.Tensor, np.ndarray,
                              Dict[str, Union[tf.Tensor, np.ndarray]],
                              List[Union[tf.Tensor, np.ndarray]]]],
    split_ratios: Union[
        Tuple[float, float, float],
        Dict[int, Tuple[float, float, float]]
    ],
    split_axes: Union[int, List[Any]],
    seed: int = 42,
    merge_consecutive_singleton_dims: bool = False
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Splits heterogeneous Tensors/arrays into (train, val, test) by slicing each axis
    in sequence (space–time split).

    Parameters
    ----------
    tensor_groups
        List of elements, each one of:
          • tf.Tensor or np.ndarray,
          • dict[str, Tensor/ndarray],
          • list[Tensor/ndarray].
    split_ratios
        - A single (train, val, test) tuple → apply to every axis.
        - A dict mapping axis index → (train, val, test) tuple.
    split_axes
        - int or list of ints / nested lists/dicts matching tensor_groups,
          specifying axis or axes to slice.
    seed
        (unused) placeholder for future shuffling.
    merge_consecutive_singleton_dims
        If True, collapse runs of size-1 dims after splitting.

    Returns
    -------
    train_groups, val_groups, test_groups
        Three lists mirroring the structure of tensor_groups.
    """

    # Step 1: Normalize and broadcast split_axes
    if isinstance(split_axes, int) or (
        isinstance(split_axes, list) and all(isinstance(a, int) for a in split_axes)
    ):
        normalized_axes = []
        for grp in tensor_groups:
            if isinstance(grp, (tf.Tensor, np.ndarray)):
                normalized_axes.append(split_axes)
            elif isinstance(grp, dict):
                normalized_axes.append({k: split_axes for k in grp})
            elif isinstance(grp, list):
                normalized_axes.append([split_axes] * len(grp))
            else:
                raise ValueError(f"Unsupported group type: {type(grp)}")
        split_axes = normalized_axes
    else:
        if len(split_axes) != len(tensor_groups):
            raise ValueError("`split_axes` must match `tensor_groups` length.")

    # Step 2: Flatten groups into (array, axes) with metadata
    structures: List[Tuple[List[Tuple[Any, Any]], type, Any]] = []
    for grp, axes in zip(tensor_groups, split_axes):
        if isinstance(grp, (tf.Tensor, np.ndarray)):
            structures.append(([(grp, axes)], type(grp), None))
        elif isinstance(grp, dict):
            flat = [(grp[k], axes[k]) for k in grp]
            structures.append((flat, dict, list(grp.keys())))
        elif isinstance(grp, list):
            flat = list(zip(grp, axes))
            structures.append((flat, list, None))
        else:
            raise ValueError(f"Unsupported group type: {type(grp)}")

    # Step 3: Sequential slicing helper
    def gather_along_axes(arr, axes, part: str, split_ratios):
        """
        Slice `arr` along each axis in `axes` for the given `part`.
        """
        axes_list = [axes] if isinstance(axes, int) else axes
        out = arr
        for ax in axes_list:
            length = int(arr.shape[ax])  # Use original shape
            if isinstance(split_ratios, dict):
                if ax not in split_ratios:
                    raise KeyError(f"No split_ratios provided for axis {ax}")
                r0, r1, r2 = split_ratios[ax]
            else:
                r0, r1, r2 = split_ratios
            if length == 1:
                if part == "train":
                    idx = np.array([0], dtype=np.int32)
                else:
                    idx = np.array([], dtype=np.int32)
            else:
                total = r0 + r1 + r2
                if total <= 0:
                    raise ValueError(f"Ratios for axis {ax} sum to zero.")
                t_end = int(length * (r0 / total))
                v_end = t_end + int(length * (r1 / total))
                base = np.arange(length, dtype=np.int32)
                if part == "train":
                    idx = base[:t_end]
                elif part == "val":
                    idx = base[t_end:v_end]
                elif part == "test":
                    idx = base[v_end:]
                else:
                    raise ValueError(f"Unknown part: {part}")
            if isinstance(out, tf.Tensor):
                out = tf.gather(out, idx, axis=ax)
            else:
                out = np.take(out, idx, axis=ax)
        return out

    # Step 4: Split & reconstruct
    train_out, val_out, test_out = [], [], []
    for flat, typ, keys in structures:
        t_parts = [gather_along_axes(a, ax, "train", split_ratios) for a, ax in flat]
        v_parts = [gather_along_axes(a, ax, "val", split_ratios) for a, ax in flat]
        s_parts = [gather_along_axes(a, ax, "test", split_ratios) for a, ax in flat]

        if typ in (tf.Tensor, np.ndarray):
            train_out.append(t_parts[0])
            val_out.append(v_parts[0])
            test_out.append(s_parts[0])
        elif typ is dict:
            train_out.append({k: p for k, p in zip(keys, t_parts)})
            val_out.append({k: p for k, p in zip(keys, v_parts)})
            test_out.append({k: p for k, p in zip(keys, s_parts)})
        else:  # list
            train_out.append(list(t_parts))
            val_out.append(list(v_parts))
            test_out.append(list(s_parts))

    # Step 5: Optional collapse of singleton dims
    if merge_consecutive_singleton_dims:
        def squeeze_runs(x):
            shp = x.shape if isinstance(x, np.ndarray) else x.shape.as_list()
            new_shape, seen = [], False
            for d in shp:
                if d == 1:
                    if not seen:
                        new_shape.append(1); seen = True
                else:
                    new_shape.append(d); seen = False
            return (np.reshape(x, new_shape)
                    if isinstance(x, np.ndarray)
                    else tf.reshape(x, new_shape))

        def apply_all(group):
            out = []
            for x in group:
                if isinstance(x, (tf.Tensor, np.ndarray)):
                    out.append(squeeze_runs(x))
                elif isinstance(x, dict):
                    out.append({k: squeeze_runs(v) for k, v in x.items()})
                else:
                    out.append([squeeze_runs(v) for v in x])
            return out

        train_out = apply_all(train_out)
        val_out   = apply_all(val_out)
        test_out  = apply_all(test_out)

    return train_out, val_out, test_out


def print_group_shapes(name: str, group_list: List[Any]):
    """
    Utility to print the structure and shapes of each group element.
    """
    print(f"\n{name}:")
    for i, grp in enumerate(group_list):
        if isinstance(grp, (np.ndarray, tf.Tensor)):
            print(f"  Group {i}: array, shape = {grp.shape}")
        elif isinstance(grp, dict):
            shapes = {k: v.shape for k, v in grp.items()}
            print(f"  Group {i}: dict, shapes = {shapes}")
        elif isinstance(grp, list):
            shapes = [x.shape for x in grp]
            print(f"  Group {i}: list, shapes = {shapes}")
        else:
            print(f"  Group {i}: unexpected type {type(grp)}")

if __name__ == "__main__":
    # Demo: space–time split with per-axis dict ratios

    # Build groups with different ranks
    t1 = tf.random.uniform((200,120,39,39,5))  # Rank 5
    d1 = {"a": np.random.rand(4,5), "b": np.random.rand(4,5)}  # Rank 2
    l1 = [tf.random.uniform((4,5)), np.random.rand(4,5)]  # Rank 2

    groups = [t1, d1, l1]

    # Specify multi-axis splits per group
    axes = [
        [0,1],
        {"a":[0,1], "b":[0,1]},
        [[0,1], [0,1]]
    ]

    # Per-axis ratio dict: rows vs. cols
    ratios = {
        0: (0.7, 0., 0.3),   # Split along axis 0
        1: (0.7, 0., 0.3)    # Split along axis 1
    }

    # ratios = (0.7, 0., 0.3)
    train, val, test = split_tensor_sequence(
        groups,
        split_ratios=ratios,
        split_axes=axes
    )

    # Print results with helper
    print_group_shapes("TRAIN", train)
    print_group_shapes("VAL",   val)
    print_group_shapes("TEST",  test)

def slice_statistics(
    data: Union[
        tf.Tensor,
        np.ndarray,
        Dict[str, Union[tf.Tensor, np.ndarray]]
    ],
    slice_keys: Optional[List[str]] = None,
    dim: int = -1
) -> Dict[str, Dict[str, Union[float, Tuple[int, ...]]]]:
    """
    Compute summary statistics (mean, std, min, max, shape) either:
      • Per-slice along axis `dim` for a single Tensor/np.ndarray.
      • Whole-array per-key for dict inputs (no slicing).

    When `data` is a tensor/array:
      - You may supply `slice_keys`, e.g. ["a","b","c"].  
      - If `slice_keys` has more entries than slices, it is trimmed.  
      - If it has fewer, the remainder are named "feature_{i}".  

    Args:
        data:        A tf.Tensor, np.ndarray, or a dict mapping str→(Tensor|ndarray).
        slice_keys:  Optional list of names for each slice (only used for non‑dict inputs).
        dim:         Axis along which to slice for single-array inputs (default −1, last axis).

    Returns:
        stats_out:
          - If `data` is array/tensor:
              { slice_key → {mean, std, min, max, shape} }
          - If `data` is dict:
              { original_key → {mean, std, min, max, shape} }
    """

    def _per_slice_stats(arr: np.ndarray) -> Dict[str, Dict[str, Union[float, Tuple[int, ...]]]]:
        """
        Compute stats per slice of a NumPy array along `dim`.
        Respects partial slice_keys by trimming or auto‑filling.
        """
        n_slices = arr.shape[dim]

        # 1) Start with provided keys (trim to n_slices)
        if slice_keys:
            keys = list(slice_keys[:n_slices])
        else:
            keys = []

        # 2) Fill any missing entries with defaults
        if len(keys) < n_slices:
            keys += [f"feature_{i}" for i in range(len(keys), n_slices)]

        stats: Dict[str, Dict[str, Union[float, Tuple[int, ...]]]] = {}
        for i, key in enumerate(keys):
            # take the i-th slice along axis=dim
            sl = np.take(arr, indices=i, axis=dim)
            stats[key] = {
                "min":  float(np.min(sl)),
                "max":  float(np.max(sl)),
                "mean": float(np.mean(sl)),
                "std":  float(np.std(sl)),
                "shape": sl.shape
            }
        return stats

    def _whole_stats(arr: np.ndarray) -> Dict[str, Union[float, Tuple[int, ...]]]:
        """
        Compute stats over the entire array (no slicing).
        Used for dict inputs.
        """
        return {
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "shape": arr.shape
        }

    # — If dict: compute whole-array stats per key ——
    if isinstance(data, dict):
        output: Dict[str, Dict[str, Union[float, Tuple[int, ...]]]] = {}
        for key, val in data.items():
            # Convert tf.Tensor → np.ndarray, or accept np.ndarray directly
            arr = val.numpy() if isinstance(val, tf.Tensor) else val
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Value at key '{key}' must be a tf.Tensor or np.ndarray")
            output[key] = _whole_stats(arr)
        return output

    # — Otherwise: handle single tensor/array with per-slice stats ——
    arr = data.numpy() if isinstance(data, tf.Tensor) else data
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a tf.Tensor, np.ndarray, or dict thereof")
    return _per_slice_stats(arr)

@tf.function
def slice_tensor(tensor, indices, dim=-1):
    """
    Extracts a slice from a specified dimension of a tensor.
    The operation is differentiable and compatible with graph mode.

    Args:
        tensor (tf.Tensor): Input tensor of any shape (e.g., (None, 1, 39, 39, 5)).
        indices (tf.Tensor or list): Indices to slice from the specified dimension.
                                   Can be a single index (e.g., [0]) or multiple (e.g., [0, 1]).
        dim (int): Dimension to slice (default -1 for innermost dimension).
                   Negative indices are supported (e.g., -1 is last dimension).

    Returns:
        tf.Tensor: Sliced tensor with the specified dimension reduced to len(indices).

    Example:
        tensor = tf.random.normal([32, 1, 39, 39, 5])
        # Slice innermost dimension (axis -1) at index 0
        sliced = slice_tensor(tensor, [0])  # Shape: (32, 1, 39, 39, 1)
        # Slice dimension 2 at indices [0, 1]
        sliced = slice_tensor(tensor, [0, 1], dim=2)  # Shape: (32, 1, 2, 39, 5)
    """
    # Convert indices to tensor
    indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    
    # Get tensor rank
    rank = tf.rank(tensor)
    
    # Normalize dim to non-negative
    dim = tf.where(dim < 0, dim + rank, dim)
    
    # Validate dim
    tf.debugging.assert_less(dim, rank, message="Dimension index out of range")
    
    # Get shape for validation
    shape = tf.shape(tensor)
    
    # Validate indices are within dimension size
    dim_size = shape[dim]
    tf.debugging.assert_less(tf.reduce_max(indices), dim_size,
                           message="Indices out of range for specified dimension")
    tf.debugging.assert_greater_equal(tf.reduce_min(indices), 0,
                                    message="Negative indices not supported")
    
    # Slice the specified dimension using tf.gather
    sliced = tf.gather(tensor, indices, axis=dim)
    
    return sliced

# Example usage
if __name__ == "__main__":
    # Create a sample tensor with shape (32, 1, 39, 39, 5)
    batch_size = 32
    sample_tensor = tf.random.normal([batch_size, 1, 39, 39, 5], dtype=tf.float32)
    
    # Slice innermost dimension (axis -1, default) at index 0
    sliced_innermost = slice_tensor(sample_tensor, [0])
    print("Sliced innermost dimension (index 0) shape:", sliced_innermost.shape)
    
    # Slice innermost dimension at indices [0, 1]
    sliced_innermost_range = slice_tensor(sample_tensor, [0, 1])
    print("Sliced innermost dimension (indices [0, 1]) shape:", sliced_innermost_range.shape)
    
    # Slice dimension 2 (height) at index 0
    sliced_dim2 = slice_tensor(sample_tensor, [0], dim=2)
    print("Sliced dimension 2 (index 0) shape:", sliced_dim2.shape)
    
    # Slice dimension 2 at indices [0, 1]
    sliced_dim2_range = slice_tensor(sample_tensor, [0, 1], dim=2)
    print("Sliced dimension 2 (indices [0, 1]) shape:", sliced_dim2_range.shape)
    
    # Verify differentiability with gradient computation
    with tf.GradientTape() as tape:
        tape.watch(sample_tensor)
        sliced = slice_tensor(sample_tensor, [0], dim=-1)
        loss = tf.reduce_sum(sliced)  # Dummy loss
    grad = tape.gradient(loss, sample_tensor)
    print("Gradient shape:", grad.shape)  # Should be (32, 1, 39, 39, 5)
    
class DataSummary:
    """
    DataSummary

    A utility to manage and query statistical summaries for numeric features/records.

    Supports inputs:
      - pandas DataFrame
      - dict of sequences → DataFrame
      - (dict, index_list) tuple → DataFrame
      - JSON file path (outer keys→rows, inner dicts→columns)
      - dict of dicts (JSON-like)

    Combines all into one DataFrame, stores stats in `self.statistics` (shape [R, F]),
    preserves per-record `shape` metadata (`self.shapes`), and provides case-insensitive
    lookups (`lookup`, `by_index`, `get_key`, `keys`, `all_stats`, `get_shape`).

    Includes `normalize()` for pure TF graph-mode normalization, delegating index-map
    construction to `create_statistics_index_full` for robust handling of scalar,
    vector, and 2×K mapping indices.
    """

    def __init__(
        self,
        data_list: List[Union[pd.DataFrame, Dict[str, Any], Tuple[Dict, List[str]], str]],
        dtype: tf.DType = tf.float32
    ):
        self.dtype = dtype
        self.shapes: Dict[str, List[int]] = {}

        dfs: List[pd.DataFrame] = []
        for item in data_list:
            if isinstance(item, pd.DataFrame):
                df = item.copy()
            elif isinstance(item, str) and os.path.isfile(item) and item.lower().endswith('.json'):
                with open(item, 'r') as f:
                    data_dict = json.load(f)
                df = pd.DataFrame.from_dict(data_dict, orient='index')
            elif isinstance(item, dict) and all(isinstance(v, dict) for v in item.values()):
                df = pd.DataFrame.from_dict(item, orient='index')
            elif isinstance(item, tuple) and isinstance(item[0], dict) and isinstance(item[1], list):
                data_dict, index_list = item
                df = pd.DataFrame(data_dict, index=index_list)
            elif isinstance(item, dict):
                df = pd.DataFrame(item)
            else:
                raise TypeError(
                    "Each element must be DataFrame, JSON path, nested dict, dict, or (dict,index_list) tuple."
                )

            if 'shape' in df.columns:
                for idx_key, shp in df['shape'].items():
                    self.shapes[str(idx_key).lower()] = shp
                df = df.drop(columns=['shape'])

            dfs.append(df)

        combined = pd.concat(dfs, axis=0, ignore_index=False) if dfs else pd.DataFrame()
        self.statistics = tf.convert_to_tensor(combined.values, dtype=self.dtype)

        self.x_keys = [col.lower() for col in combined.columns]
        self.y_keys = [str(idx).lower() for idx in combined.index]
        self._x_lookup = {k: i for i, k in enumerate(self.x_keys)}
        self._y_lookup = {k: i for i, k in enumerate(self.y_keys)}
        self._reverse_lookup = {
            **{i: k for k, i in self._x_lookup.items()},
            **{i: k for k, i in self._y_lookup.items()}
        }

    def lookup(self, key: str) -> tf.Tensor:
        lk = key.lower()
        if lk in self._x_lookup:
            return self.statistics[:, self._x_lookup[lk]]
        if lk in self._y_lookup:
            return self.statistics[self._y_lookup[lk], :]
        raise KeyError(f"Key '{key}' not found.")

    def by_index(self, idx: int) -> tf.Tensor:
        if not (0 <= idx < tf.shape(self.statistics)[0]):
            raise IndexError(f"Index {idx} out of range.")
        return self.statistics[idx, :]

    def get_key(self, idx: int) -> Union[str, None]:
        return self._reverse_lookup.get(idx)

    def keys(self) -> Dict[str, List[str]]:
        return {'x': self.x_keys, 'y': self.y_keys}

    def all_stats(self) -> tf.Tensor:
        return self.statistics

    def get_shape(self, key: str) -> List[int]:
        lk = key.lower()
        if lk in self.shapes:
            return self.shapes[lk]
        raise KeyError(f"Shape for key '{key}' not found.")
    
    def get_key_index(self, key: str) -> int:
        lk = key.lower()  # Convert search value to lowercase
        matches = []

        data_dict = {'x': self.x_keys, 'y': self.y_keys}
        # Iterate through each key-value pair
        for key, value_list in data_dict.items():
            # Check each value in the list
            for idx, value in enumerate(value_list):
                if value.lower() == lk:
                    matches.append((key, idx))

        # Return results at first instance of occurence and only the index
        if matches:
            return matches[0][1]
        else:
            return f"No matches found for value '{lk}'"

    def create_statistics_index_full(
        self,
        nonorm_input: tf.Tensor,
        statistics_index: Union[int, tf.Tensor],
        normalization_dimension: int
    ) -> tf.Tensor:
        """
        Builds a full-map tensor of the same shape as `nonorm_input`, mapping each element
        along `normalization_dimension` to a statistics row index.

        - Scalar index → filled tensor.
        - 1D vector of length D → reshaped and broadcast.
        - 2×K mapping tensor → scatter into length-D then reshape and broadcast.
        """
        nonorm_input = tf.convert_to_tensor(nonorm_input)
        in_shape = tf.shape(nonorm_input)
        idx = tf.convert_to_tensor(statistics_index, dtype=tf.int32)
        ndims = tf.rank(nonorm_input)
        norm_dim = tf.where(normalization_dimension < 0,
                            normalization_dimension + ndims,
                            normalization_dimension)
        D = in_shape[norm_dim]

        def _build_full_from_mapping(map_idx):
            positions = map_idx[0]
            rows = map_idx[1]
            row_map = tf.fill([D], -1)
            indices = tf.expand_dims(positions, 1)
            row_map = tf.tensor_scatter_nd_update(row_map, indices, rows)
            # broadcast shape
            broadcast_shape = tf.ones([ndims], dtype=tf.int32)
            broadcast_shape = tf.tensor_scatter_nd_update(
                broadcast_shape, [[norm_dim]], [D]
            )
            row_map_rs = tf.reshape(row_map, broadcast_shape)
            
            # Ensure map_idx is 2×K
            map_shape = tf.shape(map_idx)
            tf.debugging.assert_equal(map_shape[0], 2,
                                      message="Mapping must have shape [2, K]")
            return tf.broadcast_to(row_map_rs, in_shape)
        return _build_full_from_mapping(idx)

    # Check scalar mapping
    def _build_scalar_mapping(self, idx, in_shape, norm_dim):
        D = in_shape[norm_dim]
        rank_idx = tf.rank(idx)
        positions = tf.cast(
            tf.linspace(0.0, tf.cast(D - 1, tf.float32), tf.cast(D, tf.int32)),
            tf.int32
        )
        rows = tf.repeat(idx, D)
        map_idx = tf.cond(
            tf.equal(rank_idx, 0),
            lambda: tf.stack([positions, rows], axis=0),
            lambda: idx
        )
        return map_idx
    
    @tf.function(jit_compile=False)
    def normalize(
        self,
        nonorm_input: tf.Tensor,
        norm_config: Dict[str, Any] = {'normalization_limits': (0.0, 1.0), 'feature_normalization_method': 'lnk-linear-scaling'},
        statistics_index: Union[int, tf.Tensor] =  tf.constant([[0, 2], [0, 2]], dtype=tf.int32),
        compute: bool = False,
        normalization_dimension: int = -1,
        dtype: tf.DType = tf.float32
    ) -> tf.Tensor:
        """
        Apply per-slice normalization to `nonorm_input` using stored statistics.

        statistics_index: can be either:
          - A scalar `tf.Tensor` (e.g. `0`): use that row for ALL slices.
          - A full ND tensor matching `nonorm_input.shape[:normalization_dimension] + nonorm_input.shape[normalization_dimension+1:]`:
            maps each element to its stats row (>=0 → normalize; <0 → skip normalization).
          - A 2×D mapping tensor where D = number of slices along the normalization dimension:
            first row = slice positions (0 ≤ s < D),
            second row = corresponding stats row indices.
            Any slice position not listed is left unnormalized.

        Behavior:
          1. If `compute` is False, returns `nonorm_input`.
          2. Build a full `statistics_index_full` tensor of shape matching `nonorm_input` slices:
             - For scalar index: `statistics_index_full = tf.fill([...], scalar)`
             - For full-map: `statistics_index_full = statistics_index`
             - For 2×D map: scatter to vector length D, then broadcast.
          3. Gather with `tf.gather(self.statistics, statistics_index_full) → ts`, shape = input_shape + [5].
          4. Compute three schemes:
             - lnk-linear-scaling (special log branch for row indices 5 or 6),
             - linear-scaling,
             - z-score.
          5. Select based on `norm_config['feature_normalization_method']`.
          6. Mask: apply normalization only where `statistics_index_full >= 0`.
          7. Replace NaN or Inf with zeros.

        Returns same shape as `nonorm_input`.
        """
        if not compute:
            return nonorm_input

        nonorm_input = tf.convert_to_tensor(nonorm_input, dtype=dtype)
        # Build full index map
        stats_idx_full = self.create_statistics_index_full(
            nonorm_input, statistics_index, normalization_dimension
        )
        # Gather per-element stats [min,max,mean,std,count]

        ts = tf.gather(self.statistics, stats_idx_full)
        norm_min, norm_max = norm_config['normalization_limits']

        # Normalization functions
        def _lnk():
            no_log = (((nonorm_input - ts[...,0])/(ts[...,1]-ts[...,0]))*(norm_max-norm_min))+norm_min
            log    = ((tf.math.log(nonorm_input/ts[...,0])/
                       tf.math.log(ts[...,1]/ts[...,0]))*(norm_max-norm_min))+norm_min
            cond   = tf.logical_and(
                tf.not_equal(stats_idx_full, 4),
                tf.not_equal(stats_idx_full, 5)
            )
            return tf.where(cond, no_log, log)

        def _lin():
            return (((nonorm_input - ts[...,0])/(ts[...,1]-ts[...,0]))*(norm_max-norm_min))+norm_min

        def _z():
            return (nonorm_input - ts[...,2]) / ts[...,3]

        base = tf.where(
            tf.equal(norm_config['feature_normalization_method'], 'lnk-linear-scaling'),
            _lnk(), _lin()
        )
        normed = tf.where(
            tf.equal(norm_config['feature_normalization_method'], 'z-score'),
            _z(), base
        )

        # Apply mask: only index>=0
        out = tf.where(tf.greater_equal(stats_idx_full, 0), normed, nonorm_input)
        # Clean NaN/Inf
        return tf.where(
            tf.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)),
            tf.zeros_like(out), out
        )
    
    @tf.function(jit_compile=False)
    def nonormalize(
        self,
        norm_input: tf.Tensor,
        norm_config: Dict[str, Any],
        statistics_index: Union[int, tf.Tensor] =  tf.constant([[0, 2], [0, 2]], dtype=tf.int32),
        compute: bool = False,
        nonormalization_dimension: int = -1,
        dtype: tf.DType = tf.float32
    ) -> tf.Tensor:
        """
        Reverses normalization using stored statistics.

        Parameters:
          - norm_input: normalized tensor
          - norm_config: {'normalization_limits':(min,max), 'feature_normalization_method':str}
          - statistics_index: row index or full map
          - compute: if False, returns norm_input
          - nonormalization_dimension: dimension for which denormalization is performed
          - dtype: tf dtype
        """
        if not compute:
            return norm_input
       
        nonorm_input = tf.convert_to_tensor(norm_input, dtype=dtype)
        # Build full index map
        stats_idx_full = self.create_statistics_index_full(
            nonorm_input, statistics_index, nonormalization_dimension
        )
        # Gather per-element stats [min,max,mean,std,count]

        ts = tf.gather(self.statistics, stats_idx_full)
        norm_min, norm_max = norm_config['normalization_limits']

        def _lnk_rev():
            no_log = (ts[...,1] - ts[...,0]) * ((norm_input - norm_min) / (norm_max - norm_min)) + ts[...,0]
            log   = tf.math.exp(
                tf.math.log(ts[...,1] / ts[...,0]) * ((norm_input - norm_min) / (norm_max - norm_min)) + tf.math.log(ts[...,0])
            )
            cond = tf.logical_and(
                tf.not_equal(stats_idx_full, 4), tf.not_equal(stats_idx_full, 5)            # Permeability indices
            )
            return tf.where(cond, no_log, log)

        def _lin_rev():
            return (ts[...,1] - ts[...,0]) * ((norm_input - norm_min) / (norm_max - norm_min)) + ts[...,0]

        def _z_rev():
            return norm_input * ts[...,3] + ts[...,2]

        base = tf.where(
            tf.equal(norm_config['feature_normalization_method'], 'lnk-linear-scaling'),
            _lnk_rev(), _lin_rev()
        )
        reved = tf.where(
            tf.equal(norm_config['feature_normalization_method'], 'z-score'), _z_rev(), base
        )
        out = tf.where(tf.greater_equal(stats_idx_full, 0), reved, norm_input)
        return tf.where(
            tf.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)),
            tf.zeros_like(out), out
        )
    
    @tf.function(jit_compile=False)
    def normalize_diff(
        self,
        diff: tf.Tensor,
        norm_config: Dict[str, Any],
        statistics_index: Union[int, tf.Tensor] = tf.constant([[0, 2], [0, 2]], dtype=tf.int32),
        compute: bool = False,
        x0: float = 3.0,
        nonormalization_dimension: int = -1,
        dtype: tf.DType = tf.float32
    ) -> tf.Tensor:
        """
        Computes a normalized difference using stored statistics.

        Parameters:
            diff: The input difference tensor.
            norm_config: {'normalization_limits':(min,max), 'feature_normalization_method':str}.
            statistics_index: Scalar, full map, or 2×K mapping tensor (default 2×2 [[0,2],[0,2]]).
            compute: If False, returns input diff unchanged.
            x0: Constant for logarithmic difference scaling.
            nonormalization_dimension: dimension for which denormalization is performed
            dtype: TF dtype.

        Returns:
            The normalized difference tensor.
        """
        if not compute:
            return diff

        diff = tf.convert_to_tensor(diff, dtype=dtype)
        # Build full index map
        stats_idx_full = self.create_statistics_index_full(
            diff, statistics_index, nonormalization_dimension
        )
        # Gather per-element stats [min,max,mean,std,count]

        ts = tf.gather(self.statistics, stats_idx_full)
        norm_min, norm_max = norm_config['normalization_limits']

        def _lnk_lin_scaling():
            scale_no = (norm_max - norm_min) / (ts[...,1] - ts[...,0])
            scale_log = (norm_max - norm_min) / tf.math.log(ts[...,1] / ts[...,0])
            no_log = scale_no * diff
            logv  = scale_log * tf.math.log((x0 + diff) / x0)
            cond = tf.logical_and(tf.not_equal(stats_idx_full, 4), tf.not_equal(stats_idx_full, 5))
            return tf.where(cond, no_log, logv)

        def _linear_scaling():
            return (norm_max - norm_min) / (ts[...,1] - ts[...,0]) * diff

        def _z_score():
            return 1.0 / ts[...,3] * diff

        base = tf.where(tf.equal(norm_config['feature_normalization_method'], 'lnk-linear-scaling'), _lnk_lin_scaling(), _linear_scaling())
        normed = tf.where(tf.equal(norm_config['feature_normalization_method'], 'z-score'), _z_score(), base)
        out = tf.where(tf.greater_equal(stats_idx_full, 0), normed, diff)
        return tf.where(tf.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)), tf.zeros_like(out), out)
    
    
# Example usage
if __name__ == '__main__':
    summary = DataSummary([{
        'permx': {'min':0.26,'max':24.03,'mean':2.96,'std':1.43,"shape": [200,52,1,39,39]},
        'time': {'min':0.0,'max':255.0,'mean':127.5,'std':75.0,"shape": [200,52,1,39,39]},
        'PRESSURE':{'min': 1387.1, 'max': 5000.0, 'mean': 3654.7, 'std': 804.4, 'shape': [12,52,1,39,39]}
    }])

    inp = tf.constant([[1.,2.,1.,4.],[5.,6.,5.,8.]])  # Shape [2,4]
    stats_idx_scalar = tf.constant(1, dtype=tf.int32)
    stats_idx_map = tf.constant([[0,1,2,3],[0,0,0,2]], dtype=tf.int32)  # norm dimension index - normalization index (perm, time, etc.)

    cfg = {'normalization_limits':(0.,1.), 'feature_normalization_method':'linear-scaling'}
    # out_scalar = summary.normalize(inp, cfg, stats_idx_scalar, True)
    out_map    = summary.normalize(inp, cfg, stats_idx_map, True)
    inp_map = summary.nonormalize(out_map, cfg, stats_idx_map, True)
    diff = out_map - inp_map
    norm_diff = summary.normalize_diff(diff, cfg, stats_idx_map, True)
    # print("Scalar-normalized:\n", out_scalar.numpy())
    print("Map-normalized:\n", out_map.numpy())
    print("Map-nonormalized:\n", inp_map.numpy())
    print("Map-norm_difference:\n", norm_diff.numpy())

@tf.function
def l1_normalize_excluding_index(tensor, axis, exclude_index):
    # Resolve negative axis
    axis = tf.where(axis < 0, tf.rank(tensor) + axis, axis)

    # Get tensor shape
    shape = tf.shape(tensor)

    # Create a 1D mask for the axis to exclude
    indices = tf.range(shape[axis])
    axis_mask = tf.not_equal(indices, exclude_index)

    # Create a broadcastable shape: [1, 1, ..., axis_dim, ..., 1]
    rank = tf.rank(tensor)
    mask_shape = tf.ones([rank], dtype=tf.int32)
    mask_shape = tf.tensor_scatter_nd_update(mask_shape, [[axis]], [shape[axis]])
    axis_mask = tf.reshape(axis_mask, mask_shape)

    # Broadcast mask to match tensor shape
    full_mask = tf.broadcast_to(axis_mask, shape)

    # Apply mask: zero-out excluded index
    masked_tensor = tf.where(full_mask, tensor, tf.zeros_like(tensor))

    # Compute L1 norm excluding the masked-out index
    l1_norms = tf.reduce_sum(tf.abs(masked_tensor), axis=axis, keepdims=True)

    # Normalize the included values
    normalized = tf.math.divide_no_nan(masked_tensor, l1_norms)

    # Restore original (excluded) values
    output_tensor = tf.where(full_mask, normalized, tensor)
    return output_tensor


# Test tensor of shape [2, 8]
weight_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 10.0],
                             [2.0, 4.0, 6.0, 8.0, 0.0, 0.0, 0.0, 5.0]])

# Exclude last index along axis 1
exclude_axis = -1
exclude_index = tf.shape(weight_tensor)[exclude_axis] - 1

# Apply normalization
l1_normalized = l1_normalize_excluding_index(weight_tensor, exclude_axis, exclude_index)

# Display result
tf.print("Original tensor:\n", weight_tensor)
tf.print("\nL1-normalized tensor (excluding last index):\n", l1_normalized)

