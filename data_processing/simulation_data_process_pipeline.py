
#!/usr/bin/env python3
"""
Unified Processing Pipeline
Description:
    This module provides a unified processing pipeline for two types of file processing:

    1. Simulation File Processing Pipeline:
       - Processes simulation files (e.g., .RSM, .FINRST, .FUNRST, etc.) using specialized parsing routines.
       - Supports both tabular (.RSM) and continuous (.INIT, .UNRST) file formats.
       - Offers parallel processing, reshaping options, and the ability to save combined results.
       - This pipeline runs first.

    2. Array Processing Pipeline:
       - Processes array files with extension .npz (default) or .json.
       - Searches for specified dictionary keys in the file.
       - Performs slicing along a user-specified dimension (default: axis 1, the 2nd outermost) and reshapes (merges) user-specified dimensions.
       - This pipeline is optionally activated.

Usage Example:
    The configuration dictionary below is used to control all settings. Simply adjust the parameters as needed and run the script.

    To run the pipeline:
        python unified_pipeline.py

Configuration Dictionary Parameters:
    "simulation_pipeline": {
        "enabled": True,                    # (bool) Run simulation pipeline first. Default: True.
        "input_folder": "./sim_files",        # (str) Directory containing simulation files. Default: "./sim_files".
        "output_folder": "./sim_files/Output",# (str) Directory to save processed simulation results. Default: "./sim_files/Output".
        "file_vectors": {                     # (dict) Mapping of file extensions to target vectors.
            ".FINIT": ["PERMX", "PERMZ", "PORO"],
            ".FUNRST": ["PRESSURE", "SOIL", "SGAS"],
            ".RSM": [["TIME"], ["WOPR", "15 15 1"], "WGPR", "WWPR", "WBHP"]
        },
        "parallel": True,                   # (bool) Enable parallel processing. Default: True.
        "max_workers": 4,                   # (int) Maximum number of parallel workers. Default: 4.
        "shape": (29, 29, 1),               # (tuple) Optional shape to reshape parsed arrays. Default: (1, 29, 29).
        "save_results": True,               # (bool) Save combined simulation results. Default: True.
        "combine": True,                    # (bool) Save as a single combined file. Default: True.
        "flatten": True,                    # (bool) Remove top-level extension keys when saving. Default: True.
        "stack_realizations": True,         # (bool) Stack data across realizations. Default: True.
        "combined_filename": "combined_results.npz"  # (str) Filename for combined simulation results. Default: "combined_results.npz"
    }
    "array_pipeline": {
        "enabled": True,                    # (bool) Option to run the array processing pipeline. Default: True.
        "directory": "./array_data",          # (str) Directory for array files. Default: "./array_data".
        "ext": ".npz",                      # (str) File extension to process (.npz or .json). Default: ".npz".
        "file": None,                       # (str or None) Specific file name to load, if desired. Default: None.
        "keys": ["PRESSURE", "SGAS"],       # (list) Dictionary keys to extract. Default: ["PRESSURE", "SGAS"].
        "exclusions": ["PERMX", "PERMY", "PERMZ", "PORO"],  # (list) Keys to exclude from processing. Default: ["PERMX", "PERMY", "PERMZ", "PORO"].
        "slice_dim": 1,                     # (int) Axis on which to perform slicing. Default: 1.
        "slices": [0, 1, 2],                # (list or None) Slice indices along the specified axis (None means full dimension). Default: [0, 1, 2].
        "reshape_dims": (0, 1)              # (tuple) Axes to merge during reshaping. Default: (0, 1).
    }

Run:
    Adjust the configuration below as desired and run the script.
"""

import os
import glob
import json
import numpy as np
import concurrent.futures
from typing import List, Dict, Union, Optional, Tuple, Any
import logging, sys

# Configure logging to ensure it works with Spyder's IPython console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Explicitly use stdout
)

# Custom logger for numerical data (can be enabled/disabled)
log_numerical = logging.getLogger("numerical")
log_numerical.setLevel(logging.WARNING)  # Default: suppress numerical values
# To enable numerical logging, set: log_numerical.setLevel(logging.INFO)

# ============================================================================
# Helper Functions for Simulation File Processing (Tabular & Continuous)
# ============================================================================

def is_float(s: str) -> bool:
    """Return True if the string s can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_mostly_numbers(line: str, threshold: float = 0.7) -> bool:
    """Return True if most tokens (after tab splitting) in the line can be converted to floats."""
    tokens = [tok.strip() for tok in line.split("\t") if tok.strip() != ""]
    if not tokens:
        return False
    num_numeric = sum(1 for t in tokens if is_float(t))
    return num_numeric / len(tokens) >= threshold

def merge_header_lines(header_lines: List[str]) -> List[str]:
    """
    Merge multiple header lines (tab-delimited) into fixed columns.
    The first header line defines the number of columns.
    """
    first_tokens = [token.strip() for token in header_lines[0].split("\t")]
    ncols = len(first_tokens)
    columns = first_tokens.copy()
    for hl in header_lines[1:]:
        tokens = [token.strip() for token in hl.split("\t")]
        if len(tokens) < ncols:
            tokens.extend([""] * (ncols - len(tokens)))
        elif len(tokens) > ncols:
            tokens = tokens[:ncols]
        for i in range(ncols):
            if tokens[i]:
                columns[i] += " " + tokens[i]
    return [col.strip() for col in columns]

def convert_target_spec(input_spec: Union[Dict[str, List[str]], List[Union[str, List[str]]]]
                       ) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """
    Convert the target_spec to a unified dictionary format.
    Simple targets become key => [string] and compound targets are stored as nested dictionaries.
    """
    target_dict = {}
    if isinstance(input_spec, dict):
        target_dict = input_spec
    elif isinstance(input_spec, list):
        for item in input_spec:
            if isinstance(item, list):
                if len(item) < 2:
                    target_dict[item[0]] = [item[0]]
                else:
                    main_key = item[0]
                    sub_key = ' '.join(item[1:]).strip()
                    if main_key in target_dict:
                        if isinstance(target_dict[main_key], dict):
                            target_dict[main_key][sub_key] = item
                        else:
                            target_dict[main_key] = {sub_key: item}
                    else:
                        target_dict[main_key] = {sub_key: item}
            elif isinstance(item, str):
                target_dict[item] = [item]
    return target_dict

def parse_tabular_file_from_string(
    data_str: str,
    target_spec: Union[Dict[str, List[str]], List[Union[str, List[str]]]],
    dtype: np.dtype = np.float32
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Parse tabular data from a string with segmented tables.
    Header lines are merged and normalized. Data columns matching the target spec are extracted.

    Parameters:
        data_str (str): Input string containing tabular data.
        target_spec (Union[Dict, List]): Specification of target columns to extract.
        dtype (np.dtype): Data type for created NumPy arrays (default: np.float32).

    Returns:
        Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]: Parsed data.
    """
    target_dict = convert_target_spec(target_spec)
    result = {}
    for key, val in target_dict.items():
        if isinstance(val, dict):
            result[key] = {}
            for sub_key in val.keys():
                result[key][sub_key] = []
        else:
            result[key] = []
    
    lines = data_str.split("\n")
    lines = [line.lstrip("\t").rstrip() for line in lines]
    n_lines = len(lines)
    i = 0
    while i < n_lines:
        while i < n_lines and (not lines[i].strip() or lines[i].strip().upper().startswith("SUMMARY")):
            i += 1
        if i >= n_lines:
            break
        header_block = []
        while i < n_lines and lines[i].strip() and not is_mostly_numbers(lines[i]):
            if not lines[i].strip().upper().startswith("SUMMARY"):
                header_block.append(lines[i].strip())
            i += 1
        if not header_block:
            continue
        merged_headers = merge_header_lines(header_block)
        normalized_headers = [' '.join(col.split()) for col in merged_headers]
        key_col_map = {}
        for main_key, spec in target_dict.items():
            if isinstance(spec, dict):
                key_col_map[main_key] = {}
                for sub_key, phrases in spec.items():
                    norm_phrases = [' '.join(phrase.split()) for phrase in phrases]
                    for col_idx, col_text in enumerate(normalized_headers):
                        if all(phrase in col_text for phrase in norm_phrases):
                            key_col_map[main_key][sub_key] = col_idx
                            break
            else:
                norm_phrases = [' '.join(phrase.split()) for phrase in spec]
                for col_idx, col_text in enumerate(normalized_headers):
                    if all(phrase in col_text for phrase in norm_phrases):
                        key_col_map[main_key] = col_idx
                        break
        if not key_col_map or all((isinstance(val, dict) and not val) or (not isinstance(val, dict) and val is None)
                                  for val in key_col_map.values()):
            while i < n_lines and lines[i].strip():
                i += 1
            continue
        while i < n_lines and not lines[i].strip():
            i += 1
        while i < n_lines and lines[i].strip() and is_mostly_numbers(lines[i]):
            tokens = [token.strip() for token in lines[i].split("\t")]
            for main_key, mapping in key_col_map.items():
                if isinstance(mapping, dict):
                    for sub_key, col_idx in mapping.items():
                        if col_idx < len(tokens) and tokens[col_idx]:
                            try:
                                result[main_key][sub_key].append(float(tokens[col_idx]))
                            except ValueError:
                                result[main_key][sub_key].append(np.nan)
                else:
                    col_idx = mapping
                    if col_idx < len(tokens) and tokens[col_idx]:
                        try:
                            result[main_key].append(float(tokens[col_idx]))
                        except ValueError:
                            result[main_key].append(np.nan)
            i += 1
        while i < n_lines and not lines[i].strip():
            i += 1

    for main_key, value in result.items():
        if isinstance(value, dict):
            for sub_key in value:
                result[main_key][sub_key] = np.array(value[sub_key], dtype=dtype) if value[sub_key] else None
        else:
            result[main_key] = np.array(value, dtype=dtype) if value else None

    return result

def parse_continuous_file(
    file_content: str,
    target_keys: List[str],
    dtype: np.dtype = np.float32
) -> Dict[str, List[np.ndarray]]:
    """
    Parse continuous-format simulation file content (e.g., .INIT, .UNRST).

    Parameters:
        file_content (str): Content of the continuous file.
        target_keys (List[str]): Keys to extract from the file.
        dtype (np.dtype): Data type for created NumPy arrays (default: np.float32).

    Returns:
        Dict[str, List[np.ndarray]]: Parsed data as lists of arrays.
    """
    data = {key: [] for key in target_keys}
    lines = file_content.splitlines()
    current_key = None
    current_block = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("'"):
            if current_key in target_keys and current_block:
                data[current_key].append(np.array(current_block, dtype=dtype))
            parts = stripped.split("'")
            if len(parts) > 1:
                current_key = parts[1].strip()
            else:
                current_key = None
            current_block = []
        elif stripped == "":
            if current_key in target_keys and current_block:
                data[current_key].append(np.array(current_block, dtype=dtype))
            current_key = None
            current_block = []
        else:
            if current_key in target_keys:
                try:
                    numbers = [float(x) for x in stripped.split()]
                    current_block.extend(numbers)
                except Exception:
                    pass
    if current_key in target_keys and current_block:
        data[current_key].append(np.array(current_block, dtype=dtype))
    return data

# ============================================================================
# Array Processing Pipeline for .npz / .json Files
# ============================================================================

def load_file(file_path: str) -> dict:
    """
    Load a file (.json or .npz) and return its contents as a dictionary.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    elif file_path.endswith('.npz'):
        with np.load(file_path, allow_pickle=True) as data:
            if "results" not in data:
                raise KeyError(f"File {file_path} does not contain a 'results' key")
            data = data["results"].item() if np.ndim(data["results"]) == 0 else data["results"]
        return data
    else:
        raise ValueError("Unsupported file extension. Provide a .json or .npz file.")

def search_directory(directory: str, file_extension: str, file_name: str = None) -> str:
    """
    Recursively search the given directory for a file with the specified extension.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                if file_name is None or file == file_name:
                    return os.path.join(root, file)
    return None

def process_array(
    array: list,
    slices: list = None,
    slice_dim: int = 1,
    reshape_dims: tuple = (0, 1),
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Process an array: convert to NumPy, slice along the specified dimension, and reshape by merging axes.

    Parameters:
        array (list): Input array to process.
        slices (list, optional): Indices to slice along slice_dim.
        slice_dim (int): Axis to slice along (default: 1).
        reshape_dims (tuple): Axes to merge during reshaping (default: (0, 1)).
        dtype (np.dtype): Data type for the NumPy array (default: np.float32).

    Returns:
        np.ndarray: Processed array.
    """
    np_array = np.array(array, dtype=dtype)
    if slices is not None and len(slices) > 0:
        np_array = np.take(np_array, indices=slices, axis=slice_dim)
    shape = np_array.shape
    if reshape_dims:
        axes = sorted(reshape_dims)
        combined = 1
        for ax in axes:
            combined *= shape[ax]
        new_shape = []
        merged_inserted = False
        for i in range(len(shape)):
            if i in axes:
                if not merged_inserted:
                    new_shape.append(combined)
                    merged_inserted = True
            else:
                new_shape.append(shape[i])
        np_array = np_array.reshape(new_shape)
    return np_array

def process_file_data(
    file_path: str,
    keys: list = ['PRESSURE', 'SGAS'],
    exclusions: list = ['PERMX', 'PERMY', 'PERMZ', 'PORO'],
    slices: list = None,
    slice_dim: int = 1,
    reshape_dims: tuple = (0, 1),
    dtype: np.dtype = np.float32
) -> dict:
    """
    Process arrays in a .npz or .json file by extracting specified keys.

    Parameters:
        file_path (str): Path to the input file.
        keys (list): Keys to extract (default: ['PRESSURE', 'SGAS']).
        exclusions (list): Keys to exclude (default: ['PERMX', 'PERMY', 'PERMZ', 'PORO']).
        slices (list, optional): Indices to slice along slice_dim.
        slice_dim (int): Axis to slice along (default: 1).
        reshape_dims (tuple): Axes to merge during reshaping (default: (0, 1)).
        dtype (np.dtype): Data type for processed arrays (default: np.float32).

    Returns:
        dict: Processed arrays keyed by extracted keys.
    """
    data = load_file(file_path)
    processed_arrays = {}
    for key in keys:
        if key in data:
            array = data[key]
            if key not in exclusions:
                processed_array = process_array(array, slices=slices, slice_dim=slice_dim, reshape_dims=reshape_dims, dtype=dtype)
                processed_arrays[key] = processed_array
            else:
                logging.info(f"Key '{key}' is in the exclusion list. Skipping.")
        else:
            logging.info(f"Key '{key}' not found in file. Skipping.")
    return processed_arrays

def run_array_pipeline(
    config: dict,
) -> dict:
    """
    Execute the array processing pipeline using the provided configuration.

    Parameters:
        config (dict): Configuration dictionary for the array pipeline.
        includes dtype (np.dtype): Data type for processed arrays (default: np.float32).

    Returns:
        dict: Processed arrays.
    """
    directory = config.get("directory")
    ext = config.get("ext", ".npz")
    file_name = config.get("file", None)
    keys = config.get("keys", ['PRESSURE', 'SGAS'])
    exclusions = config.get("exclusions", ['PERMX', 'PERMY', 'PERMZ', 'PORO'])
    slice_dim = config.get("slice_dim", 1)
    slices = config.get("slices", None)
    reshape_dims = tuple(config.get("reshape_dims", (0, 1)))
    file_path = search_directory(directory, ext, file_name)
    dtype = config.get("dtype")
    if not file_path:
        raise FileNotFoundError("No file found matching criteria.")
    processed_arrays = process_file_data(file_path, keys=keys, exclusions=exclusions,
                                        slices=slices, slice_dim=slice_dim, reshape_dims=reshape_dims, dtype=dtype)
    if not processed_arrays:
        raise ValueError("No arrays processed.")
    return processed_arrays

# ============================================================================
# Simulation File Processing Pipeline
# ============================================================================

import warnings
import math

def reshape_array(
    arr: np.ndarray,
    shape: tuple,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Quickly reshape an array to (-1, *reversed(shape)), with x-axis fastest (F-order).
    1. If exact reshape is possible, do it.
    2. If extra elements, trim outer blocks via a fast flat slice.
    3. If too few elements for one block, collapse the two innermost reversed dims into
       the largest square (c × c) that fits.

    Parameters:
        arr (np.ndarray or None): Input array; if None, returns None.
        shape (tuple): Desired last-k dims (d1, ..., dk).
        dtype (np.dtype): Data type for the output array (default: np.float32).

    Returns:
        np.ndarray or None: Reshaped view or new view; or None if input None.

    Raises:
        ValueError: If reshape/fallback cannot produce any valid shape.
    """
    if arr is None:
        return None

    arr = arr.astype(dtype) if arr.dtype != dtype else arr
    flat = arr.reshape(-1)
    total = flat.size
    rev = tuple(reversed(shape))
    needed = math.prod(rev)

    if total >= needed:
        num_blocks, rem = divmod(total, needed)
        if rem == 0:
            return flat.reshape((num_blocks,) + rev)
        trimmed = flat[: num_blocks * needed]
        warnings.warn(
            f"Trimmed array from {total} to {trimmed.size} elements to fit shape ((-1),{rev})",
            UserWarning
        )
        return trimmed.reshape((num_blocks,) + rev)

    prefix = rev[:-2]
    prefix_prod = math.prod(prefix) if prefix else 1
    slots = total // prefix_prod
    c = math.isqrt(slots)
    while c > 0 and slots % (c * c) != 0:
        c -= 1
    if c == 0:
        raise ValueError(
            f"Cannot fallback reshape: {total} elements too few for any square under prefix={prefix}"
        )

    fallback_shape = (-1,) + prefix + (c, c)
    warnings.warn(
        f"Fallback: forcing reshape to square innermost dims {fallback_shape} "
        f"(original target was ((-1),{rev}))",
        UserWarning
    )
    return flat.reshape(fallback_shape)

def process_file_sim(
    file_path: str,
    file_vectors: Dict[str, Union[List[str], List[Union[str, List[str]]]]],
    shape: Optional[Tuple[int, ...]] = None,
    dtype: np.dtype = np.float32
) -> Dict[str, Dict]:
    """
    Process a single simulation file based on its extension.
    For .RSM files, use tabular parsing; otherwise, use continuous parsing.

    Parameters:
        file_path (str): Path to the simulation file.
        file_vectors (Dict): Mapping of file extensions to target vectors.
        shape (tuple, optional): Shape to reshape continuous file arrays.
        dtype (np.dtype): Data type for parsed arrays (default: np.float32).

    Returns:
        Dict[str, Dict]: Parsed data keyed by extension.
    """
    ext = os.path.splitext(file_path)[1].upper()
    with open(file_path, "r") as f:
        content = f.read()
    targets = file_vectors.get(ext, None)
    if targets is None:
        return {}
    if ext == ".RSM":
        parsed = parse_tabular_file_from_string(content, targets, dtype=dtype)
    else:
        parsed = parse_continuous_file(content, targets, dtype=dtype)
        if shape is not None:
            for key, arr_list in parsed.items():
                parsed[key] = reshape_array(np.array(arr_list), shape, dtype=dtype)
    return {ext: parsed}

def process_files_in_directory(
    directory: str,
    file_vectors: Dict[str, Union[List[str], List[Union[str, List[str]]]]],
    parallel: bool = True,
    max_workers: int = None,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: np.dtype = np.float32
) -> Dict[str, Dict[str, Dict]]:
    """
    Process all simulation files in the specified directory using the provided file_vectors mapping.

    Parameters:
        directory (str): Directory containing simulation files.
        file_vectors (Dict): Mapping of file extensions to target vectors.
        parallel (bool): Enable parallel processing (default: True).
        max_workers (int, optional): Maximum number of parallel workers.
        shape (tuple, optional): Shape to reshape continuous file arrays.
        dtype (np.dtype): Data type for parsed arrays (default: np.float32).

    Returns:
        Dict[str, Dict[str, Dict]]: Processed results.
    """
    all_files = []
    for ext in file_vectors.keys():
        files = glob.glob(os.path.join(directory, "*" + ext))
        all_files.extend(files)
    
    combined_results = {ext: {} for ext in file_vectors.keys()}
    index_counters = {ext: 0 for ext in file_vectors.keys()}

    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file_sim, fp, file_vectors, shape, dtype): fp for fp in all_files}
            for future in concurrent.futures.as_completed(futures):
                fp = futures[future]
                try:
                    res = future.result()
                    for ext, data in res.items():
                        index_str = str(index_counters[ext])
                        combined_results[ext][index_str] = data
                        index_counters[ext] += 1
                except Exception as e:
                    logging.error(f"Error processing {fp}: {e}")
    else:
        for fp in all_files:
            res = process_file_sim(fp, file_vectors, shape, dtype)
            for ext, data in res.items():
                index_str = str(index_counters[ext])
                combined_results[ext][index_str] = data
                index_counters[ext] += 1
    
    return combined_results

def _flatten_loaded_results(loaded: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the results dictionary by removing top-level extension keys.
    """
    flat_dict = {}
    for ext_group in loaded.values():
        if isinstance(ext_group, list):
            for parsed in ext_group:
                if not isinstance(parsed, dict):
                    continue
                for key, value in parsed.items():
                    if isinstance(value, dict):
                        if key not in flat_dict:
                            flat_dict[key] = {}
                        flat_dict[key].update(value)
                    else:
                        flat_dict[key] = value
        elif isinstance(ext_group, dict):
            for key, value in ext_group.items():
                if isinstance(value, dict):
                    if key not in flat_dict:
                        flat_dict[key] = {}
                    flat_dict[key].update(value)
                else:
                    flat_dict[key] = value
    return flat_dict

def _stack_realizations_in_dict(d: Union[List[Dict], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stack data across multiple realizations.
    """
    if isinstance(d, list):
        if not d:
            return {}
        if len(d) == 1 and isinstance(d[0], dict):
            return _stack_realizations_in_dict(d[0])
        all_keys = set()
        for realization in d:
            if not isinstance(realization, dict):
                raise ValueError("List elements must be dictionaries")
            all_keys.update(realization.keys())
        stacked_result = {}
        for key in all_keys:
            values = [r[key] for r in d if key in r]
            if len(values) != len(d):
                raise ValueError(f"Key '{key}' missing in some realizations")
            if all(isinstance(v, np.ndarray) for v in values):
                stacked_result[key] = np.stack(values, axis=0)
            elif all(isinstance(v, dict) for v in values):
                sub_keys = set().union(*(v.keys() for v in values))
                stacked_sub = {}
                for sk in sub_keys:
                    sub_values = [v[sk] for v in values if sk in v]
                    if len(sub_values) != len(values):
                        raise ValueError(f"Sub-key '{sk}' missing in some realizations for '{key}'")
                    stacked_sub[sk] = np.stack(sub_values, axis=0)
                stacked_result[key] = stacked_sub
            else:
                raise ValueError(f"Cannot stack mixed types for key '{key}'")
        return stacked_result
    elif isinstance(d, dict):
        numeric_keys = all(k.isdigit() for k in d.keys())
        if numeric_keys:
            sorted_keys = sorted(d.keys(), key=int)
            if all(isinstance(d[k], dict) for k in sorted_keys):
                all_fields = set().union(*(d[k].keys() for k in sorted_keys))
                stacked_result = {}
                for field in all_fields:
                    field_arrays = [d[k][field] for k in sorted_keys if field in d[k]]
                    if len(field_arrays) != len(sorted_keys):
                        raise ValueError(f"Field '{field}' missing in some realizations")
                    stacked_result[field] = np.stack(field_arrays, axis=0)
                return stacked_result
            else:
                return np.stack([d[k] for k in sorted_keys], axis=0)
        else:
            return {k: _stack_realizations_in_dict(v) if isinstance(v, (list, dict)) else v
                    for k, v in d.items()}
    else:
        raise ValueError("Input must be a list or dictionary")

def save_results(
    results: Dict[str, Any],
    output_dir: str,
    combine: bool = True,
    flatten: bool = True,
    stack_realizations: bool = True,
    combined_filename: str = "combined_results.npz"
) -> None:
    """
    Save processed simulation file results either as one combined file or as separate files.
    
    After flattening the results, this function computes summary statistics (mean, std, min, max, and shape)
    for each key. The statistics preserve the nested structure where applicable, and are saved as a 
    'summary.json' file in the same output directory.
    
    Parameters:
        results (Dict[str, Any]): The combined results dictionary.
        output_dir (str): Directory where output files will be saved.
        combine (bool): If True, save the entire results in one file; otherwise, save separate files.
        flatten (bool): If True, flatten the results by removing top-level extension keys.
        stack_realizations (bool): If True, stack data across realizations.
        combined_filename (str): Filename for the combined results file when combine=True.
    """
    if stack_realizations:
        for ext in list(results.keys()):
            results[ext] = _stack_realizations_in_dict(results[ext])
    
    if flatten:
        results = _flatten_loaded_results(results)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if combine:
        out_path = os.path.join(output_dir, combined_filename)
        np.savez_compressed(out_path, results=results)
        logging.info(f"Saved combined results to {out_path}")
    else:
        for key, data in results.items():
            out_path = os.path.join(output_dir, f"{key}.npz")
            np.savez_compressed(out_path, results=data)
            logging.info(f"Saved results for {key} to {out_path}")

    def compute_statistics(data: Any) -> Any:
        if isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.number):
                return {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "shape": list(data.shape)
                }
            else:
                return {"type": "non-numerical", "shape": list(data.shape)}
        elif isinstance(data, dict):
            stats = {}
            for key, value in data.items():
                stats[key] = compute_statistics(value)
            return stats
        else:
            return None
    
    summary_stats = compute_statistics(results)
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=4)
    logging.info(f"Saved summary statistics to {summary_file}")

def load_results(
    directory: str,
    combine: bool = True,
    combined_filename: str = "combined_results.npz",
    flatten: bool = False,
    stack_realizations: bool = False
) -> Dict[str, Any]:
    """
    Load simulation file results from saved .npz file(s).
    """
    import glob
    if combine:
        combined_path = os.path.join(directory, combined_filename)
        if not os.path.exists(combined_path):
            raise FileNotFoundError(f"Combined results file not found: {combined_path}")
        with np.load(combined_path, allow_pickle=True) as data:
            if "results" not in data:
                raise KeyError(f"File {combined_path} does not contain a 'results' key")
            loaded_results = data["results"].item() if np.ndim(data["results"]) == 0 else data["results"]
        logging.info(f"Loaded combined results from {combined_path}")
    else:
        npz_files = glob.glob(os.path.join(directory, "*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in directory: {directory}")
        loaded_results = {}
        for npz_file in npz_files:
            base_name = os.path.splitext(os.path.basename(npz_file))[0]
            with np.load(npz_file, allow_pickle=True) as data:
                if "results" not in data:
                    raise KeyError(f"File {npz_file} does not contain a 'results' key")
                file_data = data["results"].item() if np.ndim(data["results"]) == 0 else data["results"]
                loaded_results[base_name] = file_data
            logging.info(f"Loaded results from {npz_file}")
    
    if stack_realizations:
        for ext in list(loaded_results.keys()):
            loaded_results[ext] = _stack_realizations_in_dict(loaded_results[ext])
    
    if flatten:
        loaded_results = _flatten_loaded_results(loaded_results)
    
    return loaded_results

# ============================================================================
# Unified Pipeline Runner Using a Single Configuration Dictionary
# ============================================================================

def check_required_extensions(input_folder, required_extensions):
    """
    Check if the input_folder contains at least one file for each required extension.
    If any extension is missing, print a message and exit the simulation pipeline.
    """
    import os
    found = {ext: False for ext in required_extensions}
    for file in os.listdir(input_folder):
        for ext in required_extensions:
            if file.endswith(ext):
                found[ext] = True
    missing = [ext for ext, exists in found.items() if not exists]
    if missing:
        print(f"Simulation pipeline exited: No file(s) with extension(s) {missing} found in {input_folder}.")
        return False
    return True

def run_pipeline_from_config(config: dict) -> None:
    """
    Run the unified processing pipeline using the provided configuration dictionary.
    The simulation pipeline runs first, followed by the array pipeline if enabled.
    """
    if "simulation_pipeline" in config and config["simulation_pipeline"].get("enabled", False):
        logging.info("Running Simulation File Processing Pipeline...")
        sim_conf = config["simulation_pipeline"]
        input_folder = sim_conf.get("input_folder")
        file_vectors = sim_conf.get("file_vectors")
        required_extensions = list(file_vectors.keys())
        if not check_required_extensions(input_folder, required_extensions):
            return None
        parallel = sim_conf.get("parallel", True)
        max_workers = sim_conf.get("max_workers", None)
        shape = sim_conf.get("shape", None)
        combined_results = process_files_in_directory(input_folder, file_vectors, parallel, max_workers, shape, dtype=np.float32)
        if sim_conf.get("save_results", False):
            output_folder = sim_conf.get("output_folder")
            combine_flag = sim_conf.get("combine", True)
            flatten_flag = sim_conf.get("flatten", True)
            stack_flag = sim_conf.get("stack_realizations", True)
            combined_filename = sim_conf.get("combined_filename", "combined_results.npz")
            save_results(combined_results, output_folder, combine_flag, flatten_flag, stack_flag, combined_filename)
        logging.info("Simulation Processing Completed. Combined Results available.")
        log_numerical.info(f"Combined Results: {combined_results}")

    if "array_pipeline" in config and config["array_pipeline"].get("enabled", False):
        logging.info("Running Array Processing Pipeline...")
        array_result = run_array_pipeline(config["array_pipeline"])
        logging.info("Array Processing Completed. Result available.")
        log_numerical.info(f"Array Processing Result: {array_result}")
        return array_result

# ============================================================================
# Main Function: Direct Configuration Assignment and Pipeline Execution
# ============================================================================

if __name__ == "__main__":
    # Unified configuration dictionary with direct assignments
    sim_folder = r'C:\Users\User\Documents\PHD_HW_Machine_Learning\ML_Cases_2025\Main_Library\New Methods\KL_Realizations\Simulations_39x39x1_R200_a75670de'
    unified_config = {
        "simulation_pipeline": {
            "enabled": True,
            "input_folder": sim_folder,
            "output_folder": sim_folder + r'\Output',
            "file_vectors": {
                ".FINIT": ["PERMX", "PERMZ", "PORO"],
                ".FUNRST": ["PRESSURE", "SOIL", "SGAS"],
                ".RSM": [["TIME"], ["WOPR", "15 15 1"], "WGPR", "WWPR", "WBHP"]
            },
            "parallel": False,
            "max_workers": 4,
            "shape": (39, 39, 1),
            "save_results": True,
            "combine": True,
            "flatten": True,
            "stack_realizations": True,
            "combined_filename": "combined_results.npz"
        },
        "array_pipeline": {
            "enabled": True,
            "directory": sim_folder + r'\Output',
            "ext": ".npz",
            "file": None,
            "keys": ["PRESSURE", "SGAS"],
            "exclusions": ["PERMX", "PERMY", "PERMZ", "PORO"],
            "slice_dim": 1,
            "slices": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                       100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
                       180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255,
                       260, 265, 270, 274, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330,
                       335, 340, 345, 350, 355, 360, 365],
            "reshape_dims": (0,),
            "dtype": np.float32
        }
    }
    
    # Run the unified pipeline
    array_result = run_pipeline_from_config(unified_config)