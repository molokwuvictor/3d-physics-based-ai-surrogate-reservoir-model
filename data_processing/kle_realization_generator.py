# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 23:56:31 2025

@author: Victor PhD
"""

"""
KL Realizations Generator

This script generates a set of 3D reservoir property realizations using the Karhunen-Loève expansion.
Each realization represents a 3D permeability field modeled as log-normal distribution.
The script saves each realization as a separate .dat file and the grid information for reference.
"""

import numpy as np
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import hashlib
import warnings

# Add the main library path to sys.path
main_library_path = os.path.abspath(os.path.dirname(__file__))
if main_library_path not in sys.path:
    sys.path.insert(0, main_library_path)

# Import default configurations upfront
from default_configurations import WORKING_DIRECTORY, DEFAULT_RESERVOIR_CONFIG, DEFAULT_GENERAL_CONFIG, DEFAULT_WELLS_CONFIG, DEFAULT_SCAL_CONFIG, get_configuration


# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types to make them JSON serializable"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Import the KL expansion module
from KL_expansion import generate_kl_log_normal_real_params_3D

# Configuration class for KL expansion
class KLConfig:
    """
    Configuration settings for KL expansion and realization generation.
    """
    def __init__(self, 
                 number_of_realizations=10,
                 Nx=30, Ny=30, Nz=10,
                 Lx=100.0, Ly=50.0, Lz=20.0,
                 mean=3.0, std=1.0,
                 correlation_length_factor=0.2,
                 energy_threshold=0.95,
                 seed=2000,
                 reverse_order=False,
                 output_keyword="PERMX",
                 comment_prefix="--",
                 add_comments=True,
                 dtype=np.float32):
        """
        Initialize KL expansion configuration.
        
        Parameters:
        -----------
        number_of_realizations : int
            Number of realizations to generate
        Nx, Ny, Nz : int
            Number of grid points in the x, y, and z directions
        Lx, Ly, Lz : float
            Physical dimensions along x, y, and z directions (length, width, depth)
        mean : float
            Mean permeability (physical)
        std : float
            Standard deviation of permeability (physical)
        correlation_length_factor : float
            Correlation length factor for the exponential covariance kernel
        energy_threshold : float
            Fraction of the total energy (variance) to capture
        seed : int
            Seed for reproducibility
        reverse_order : bool
            If True, perform a final transpose to output fields in (Nz,Ny,Nx) ordering
        output_keyword : str
            Keyword to include in .dat files (e.g., "PERMX", "PERMY")
        comment_prefix : str
            Prefix for comment lines (e.g., "--")
        add_comments : bool
            Whether to include comments in the output files
        dtype : np.dtype
            Data type for generated arrays (default: np.float32)
        """
        self.number_of_realizations = number_of_realizations
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx  # Length of reservoir (x-direction)
        self.Ly = Ly  # Width of reservoir (y-direction)
        self.Lz = Lz  # Depth of reservoir (z-direction)
        self.mean = mean
        self.std = std
        self.correlation_length_factor = correlation_length_factor
        self.energy_threshold = energy_threshold
        self.seed = seed
        self.reverse_order = reverse_order
        self.output_keyword = output_keyword
        self.comment_prefix = comment_prefix
        self.add_comments = add_comments
        self.dtype = dtype   

    def to_dict(self):
        """Convert configuration to dictionary for serialization"""
        config_dict = self.__dict__.copy()
        # Convert dtype to string for JSON serialization
        if isinstance(config_dict.get('dtype'), (np.generic, np.dtype, type)):
            config_dict['dtype'] = str(config_dict['dtype'])
        return config_dict

    @staticmethod
    def save_full_config_json(path, general_config, reservoir_config, wells_config=None, scal_config=None, get_configuration=None, kle_config=None):
        """
        Save the full configuration (all config dicts and KLConfig if provided) as JSON for debugging.
        """
        config_str, _ = generate_full_config_hash(general_config, reservoir_config, wells_config, scal_config, get_configuration)
        with open(path, 'w') as f:
            json.dump(json.loads(config_str), f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        print(f"Saved full configuration for debugging to: {path}")

    def __str__(self):
        """String representation of the configuration"""
        return (f"KL Configuration: {self.number_of_realizations} realizations, "
                f"Grid: {self.Nx}x{self.Ny}x{self.Nz}, "
                f"Domain: {self.Lx}x{self.Ly}x{self.Lz}, "
                f"Mean: {self.mean}, Std: {self.std}, "
                f"Corr Length: {self.correlation_length_factor}, "
                f"Energy: {self.energy_threshold}, "
                f"Dtype: {self.dtype}")

def save_grid_information(grid, out_dir, config):
    """
    Save grid coordinates to files.
    
    Parameters:
    -----------
    grid : tuple
        Tuple of grid arrays (X, Y, Z)
    out_dir : Path
        Output directory
    config : KLConfig
        Configuration object with dtype setting
    """
    X, Y, Z = grid
    
    # Save grid coordinates with specified dtype
    np.save(out_dir / "grid_X.npy", X.astype(config.dtype))
    np.save(out_dir / "grid_Y.npy", Y.astype(config.dtype))
    np.save(out_dir / "grid_Z.npy", Z.astype(config.dtype))
    
    # Save a combined grid file for easy reference
    grid_info = {
        'shape': X.shape,
        'x_range': [float(X.min()), float(X.max())],
        'y_range': [float(Y.min()), float(Y.max())],
        'z_range': [float(Z.min()), float(Z.max())],
    }
    
    # Save grid information as JSON
    with open(out_dir / "grid_info.json", 'w') as f:
        json.dump(grid_info, f, indent=4, cls=NumpyEncoder)
    
    print(f"Saved grid information to: {out_dir}")

def save_realization_to_dat(realization, index, out_dir, config):
    """
    Save a single realization to a .dat file in unformatted ASCII with required format:
    1. Comments (preceded by prefix like "--")
    2. KEYWORD (e.g., "PERMX")
    3. Data values
    4. End with "/"
    
    Parameters:
    -----------
    realization : ndarray
        3D array containing the realization data
    index : int
        Realization index
    out_dir : Path
        Output directory
    config : KLConfig
        Configuration object with output settings
    """
    # Format index with leading zeros
    index_str = f"{index:04d}"
    
    # Create the filename with the format: KEYWORD_xxx.dat
    filename = f"{config.output_keyword}_{index_str}.dat"
    
    # Get the full path
    filepath = out_dir / filename
    
    # Write the file in unformatted ASCII format
    with open(filepath, 'w') as f:
        if config.add_comments:
            f.write(f"{config.comment_prefix} REALIZATION: {index}\n")
            f.write(f"{config.comment_prefix} GRID: {config.Nx}x{config.Ny}x{config.Nz}\n")
            f.write(f"{config.comment_prefix} PHYSICAL SIZE: {config.Lx}x{config.Ly}x{config.Lz}\n")
            f.write(f"{config.comment_prefix} MEAN: {config.mean}\n")
            f.write(f"{config.comment_prefix} STD: {config.std}\n")
            f.write(f"{config.comment_prefix} CORRELATION LENGTH FACTOR: {config.correlation_length_factor}\n")
            f.write(f"{config.comment_prefix} ORDER: {'(Z,Y,X)' if config.reverse_order else '(X,Y,Z)'}\n")
            f.write(f"{config.comment_prefix}\n")
        
        f.write(f"{config.output_keyword}\n")
        
        # Flatten the 3D array and write each value on a new line
        for val in realization.flatten():
            f.write(f"{val}\n")
        
        f.write("/\n")
    
    print(f"Saved realization {index} to: {filename}")

def save_all_realizations(permeability_fields, out_dir, config, save_compressed=False):
    """
    Save all realizations as a single numpy array for easy loading.
    
    Parameters:
    -----------
    permeability_fields : ndarray
        Array containing all realizations
    out_dir : Path
        Output directory
    config : KLConfig
        Configuration object with dtype setting
    save_compressed : bool, optional
        Whether to save a compressed version of the realizations (default: False)
    """
    # Save full array of realizations with specified dtype
    np.save(out_dir / "realizations_all.npy", permeability_fields.astype(config.dtype))
    
    # Optionally save a compressed version for potential space savings
    if save_compressed:
        np.savez_compressed(out_dir / "realizations_all.npz", realizations=permeability_fields.astype(config.dtype))
        print(f"Saved all {permeability_fields.shape[0]} realizations as numpy arrays (standard and compressed)")
    else:
        print(f"Saved all {permeability_fields.shape[0]} realizations as numpy arrays")

def create_dat_files_subfolders(out_dir, general_config, reservoir_config, wells_config=None, scal_config=None, get_configuration=None, indices=None):
    """
    Create subfolders for .dat files with unique identifiers for train/val/test splits.
    
    Parameters:
    -----------
    out_dir : Path
        Main output directory
    config : KLConfig
        Configuration object
    indices : dict, optional
        Dictionary containing indices for train/val/test splits
        If None, only creates a single dat_files folder without splits
    
    Returns:
    --------
    dict
        Dictionary with paths to the created .dat files subfolders
    """
    _, short_hash = generate_full_config_hash(general_config, reservoir_config, wells_config, scal_config, get_configuration)[:16]
    
    def make_static_dynamic(parent_dir):
        static_dir = parent_dir / "static"
        dynamic_dir = parent_dir / "dynamic"
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(dynamic_dir, exist_ok=True)
        return static_dir, dynamic_dir

    # Create main dat_files folder for all realizations if needed
    if indices is None:
        dat_folder_name = f"dat_files_{short_hash}"
        dat_dir = out_dir / dat_folder_name
        static_dir, dynamic_dir = make_static_dynamic(dat_dir)
        print(f"Created subfolder for all .dat files: {static_dir}")
        return {"all": static_dir, "all_dynamic": dynamic_dir}
    
    # Create train/val/test dat files folders
    train_folder_name = f"dat_files_train_{short_hash}"
    val_folder_name = f"dat_files_val_{short_hash}"
    test_folder_name = f"dat_files_test_{short_hash}"
    
    train_dir = out_dir / train_folder_name
    val_dir = out_dir / val_folder_name
    test_dir = out_dir / test_folder_name
    
    train_static, train_dynamic = make_static_dynamic(train_dir)
    val_static, val_dynamic = make_static_dynamic(val_dir)
    test_static, test_dynamic = make_static_dynamic(test_dir)
    
    print(f"Created subfolders for split .dat files:")  
    print(f"  - Train: {train_static}")
    print(f"  - Validation: {val_static}")
    print(f"  - Test: {test_static}")
    
    return {
        "train": train_static,
        "val": val_static,
        "test": test_static,
        "train_dynamic": train_dynamic,
        "val_dynamic": val_dynamic,
        "test_dynamic": test_dynamic
    }

def split_realizations(permeability_fields, out_dir, config, general_config, save_compressed=False):
    """
    Split realizations into train, validation, and test sets according to configuration settings.
    Creates numpy arrays in parent directory and prepares indices for .dat file organization.
    
    Parameters:
    -----------
    permeability_fields : ndarray
        Array containing all realizations
    out_dir : Path
        Output directory where split folders will be created
    config : KLConfig
        Configuration object with dtype setting
    general_config : dict
        General configuration dictionary containing split settings
    save_compressed : bool, optional
        Whether to save a compressed version of the realizations (default: False)
        
    Returns:
    --------
    dict
        Dictionary with train_indices, val_indices, and test_indices for .dat file organization
    """
    # Note: We no longer create Train/Validation/Test folders for numpy arrays
    # as they will now be saved in the parent folder
    
    # Get split parameters
    split_method = general_config.get('split_sampling_method')
    split_ratios = general_config.get('split_ratio')
    seed = general_config.get('seed')
    split_axes = general_config.get('split_axis')

    # Check all required split parameters are in config
    missing = []
    if split_method is None:
        missing.append('split_sampling_method')
    if split_ratios is None:
        missing.append('split_ratio')
    if seed is None:
        missing.append('seed')
    if split_axes is None:
        missing.append('split_axis')
    if missing:
        raise ValueError(f"Missing required split configuration(s) in general_config: {missing}")

    if not isinstance(split_axes, list):
        split_axes = [split_axes]
    if not split_axes:
        raise ValueError("split_axis in general_config must not be empty")
    split_axis = split_axes[0]
    if split_axis not in split_ratios:
        raise ValueError(f"split_ratio in general_config does not contain entry for axis {split_axis}")
    train_ratio, val_ratio, test_ratio = split_ratios[split_axis]
    
    # Ensure ratios sum to 1
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        # Normalize ratios
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    n_samples = permeability_fields.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val  # Ensure all samples are used
    
    print(f"Splitting {n_samples} realizations: {n_train} train, {n_val} validation, {n_test} test")
    print(f"Using split method: {split_method} with seed: {seed}")
    
    # Create indices for splitting
    indices = np.arange(n_samples)
    
    if split_method.lower() == 'random':
        # Use seed for reproducibility
        rng = np.random.RandomState(seed)  # Use RandomState for more controlled seeding
        rng.shuffle(indices)
        print(f"Using random sampling with seed: {seed} for reproducible splits")
    
    # Split indices
    train_indices = np.sort(indices[:n_train])
    val_indices = np.sort(indices[n_train:n_train+n_val])
    test_indices = np.sort(indices[n_train+n_val:])
    
    # Split and save data
    train_fields = permeability_fields[train_indices]
    val_fields = permeability_fields[val_indices]
    test_fields = permeability_fields[test_indices]
    
    # Save split datasets in parent directory
    np.save(out_dir / "realizations_train.npy", train_fields.astype(config.dtype))
    np.save(out_dir / "realizations_val.npy", val_fields.astype(config.dtype))
    np.save(out_dir / "realizations_test.npy", test_fields.astype(config.dtype))
    
    # Optionally save compressed versions in parent directory
    if save_compressed:
        np.savez_compressed(out_dir / "realizations_train.npz", realizations=train_fields.astype(config.dtype))
        np.savez_compressed(out_dir / "realizations_val.npz", realizations=val_fields.astype(config.dtype))
        np.savez_compressed(out_dir / "realizations_test.npz", realizations=test_fields.astype(config.dtype))
        print(f"Saved split datasets with compression in parent directory")
    else:
        print(f"Saved split datasets in parent directory")
    
    # Save indices for reproducibility
    np.save(out_dir / "indices_train.npy", train_indices)
    np.save(out_dir / "indices_val.npy", val_indices)
    np.save(out_dir / "indices_test.npy", test_indices)
    
    # Save split info to summary
    split_info = {
        "method": split_method,
        "ratios": {
            "train": float(train_ratio),  # Convert to regular float for JSON serialization
            "validation": float(val_ratio),
            "test": float(test_ratio)
        },
        "counts": {
            "train": int(n_train),
            "validation": int(n_val),
            "test": int(n_test)
        },
        "seed": general_config.get('seed'),
        "indices": {
            "train": train_indices.tolist(),  # Store indices for .dat file organization
            "validation": val_indices.tolist(),
            "test": test_indices.tolist()
        }
    }
    
    with open(out_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=4)
    
    # Return indices for .dat file organization
    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }

def flatten_dict(nested_dict, prefix=''):
    """
    Recursively flatten a nested dictionary.
    
    Parameters:
    -----------
    nested_dict : dict
        Nested dictionary to be flattened
    prefix : str, optional
        Prefix for the flattened keys (default: '')
    
    Returns:
    --------
    dict
        Flattened dictionary
    """
    if nested_dict is None:
        return {}
    
    flattened_dict = {}
    for key, value in nested_dict.items():
        new_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key + '_'))
        elif isinstance(value, set):
            flattened_dict[new_key] = sorted(list(value))
        elif isinstance(value, (np.ndarray,)):
            flattened_dict[new_key] = value.tolist()
        elif isinstance(value, (np.generic,)):
            flattened_dict[new_key] = value.item()
        elif isinstance(value, type):
            flattened_dict[new_key] = str(value)
        elif isinstance(value, (np.dtype,)):
            flattened_dict[new_key] = str(value)
        elif hasattr(value, "to_dict"):
            flattened_dict[new_key] = value.to_dict()
        elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            # Use class name for custom objects for stable hashing
            flattened_dict[new_key] = value.__class__.__name__
        else:
            flattened_dict[new_key] = value
    return flattened_dict

def generate_full_config_hash(general_config, reservoir_config, wells_config=None, scal_config=None, get_configuration=None):
    """
    Generate a hash from all configuration settings for folder naming.
    
    Parameters:
    -----------
    general_config : dict
        General configuration dictionary
    reservoir_config : dict
        Reservoir configuration dictionary
    wells_config : dict, optional
        Wells configuration dictionary
    
    Returns:
    --------
    str
        Hash string based on configuration
    """
    # Create a combined configuration dictionary
    config_dict = {}
    
    # Add timestep settings from general config
    time_keys = [
        'srm_start_time', 'srm_end_time', 'cfd_start_time', 'cfd_end_time',
        'srm_timestep', 'cfd_timestep', 'maximum_srm_timestep', 'minimum_srm_timestep',
        'maximum_cfd_timestep', 'minimum_cfd_timestep', 'seed', 'split_axis', 'split_ratio',
        'split_sampling_method', 'physics_mode_fraction', 'fluid_type', 'pvt_fitting_method'
        ]
    for key in time_keys:
        if key in general_config:
            config_dict[key] = general_config[key]
    
    # Flatten general config and add all keys
    # flattened_general = flatten_dict(general_config, prefix="general_")
    # config_dict.update(flattened_general)
    
    # Flatten and add full reservoir config (including 'realizations')
    flattened_reservoir = flatten_dict(reservoir_config, prefix="reservoir_")
    config_dict.update(flattened_reservoir)
    
    # Flatten and add wells config if available
    if wells_config:
        flattened_wells = flatten_dict(wells_config, prefix="wells_")
        config_dict.update(flattened_wells)
    
    # Get and flatten PVT config using getter to respect fluid_type
    if get_configuration is not None:
        pvt_config = get_configuration('pvt_layer', fluid_type=general_config.get('fluid_type', 'DG'), fitting_method=general_config.get('pvt_fitting_method', 'spline'))
        flattened_pvt = flatten_dict(pvt_config, prefix="pvt_")
        config_dict.update(flattened_pvt)
    
    # Flatten and add SCAL config
    flattened_scal = flatten_dict(scal_config, prefix="scal_")
    config_dict.update(flattened_scal)
    
    # Create hash
    config_str = json.dumps(config_dict, sort_keys=True)
    return config_str, hashlib.md5(config_str.encode()).hexdigest()[:16]

def create_output_directory(config, base_dir, general_config, reservoir_config, wells_config=None, scal_config=None, get_configuration=None, overwrite=False):
    """
    Create output directory structure according to specified naming convention.
    
    Parameters:
    -----------
    config : KLConfig
        KLE configuration object
    base_dir : str
        Base directory path
    general_config : dict
        General configuration dictionary
    reservoir_config : dict
        Reservoir configuration dictionary
    wells_config : dict, optional
        Wells configuration dictionary
    overwrite : bool, optional
        Whether to overwrite (delete) existing directory (default: False)
    
    Returns:
    --------
    Path
        Path to the created output directory
    """
    # Create main directory
    main_dir = Path(base_dir) / "Static and Dynamic Properties"
    os.makedirs(main_dir, exist_ok=True)
    
    # Generate hash from all configurations
    config_str, config_hash = generate_full_config_hash(general_config, reservoir_config, wells_config, scal_config, get_configuration)
    
    # Create directory name with grid size and hash
    dir_name = f"KLE_{config.Nx}x{config.Ny}x{config.Nz}_R{config.number_of_realizations}_{config_hash}"
    out_dir = main_dir / dir_name
    
    # Check if directory exists and should be overwritten
    if os.path.exists(out_dir):
        if overwrite:
            print(f"Removing existing directory: {out_dir}")
            import shutil
            shutil.rmtree(out_dir)
        else:
            print(f"Directory already exists: {out_dir}")
            print(f"Using existing directory. Set overwrite=True to recreate.")
            return out_dir
    
    # Create directory
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory: {out_dir}")
    
    return out_dir

def generate_and_save_realizations(config, working_dir=None, general_config=None, reservoir_config=None, wells_config=None, scal_config=None, get_configuration=None, save_compressed=False, overwrite=True):
    """
    Generate KL expansion realizations and save them to files.
    
    Parameters:
    -----------
    config : KLConfig
        Configuration object
    general_config : dict
        General configuration dictionary
    reservoir_config : dict
        Reservoir configuration dictionary
    wells_config : dict
        Wells configuration dictionary
    save_compressed : bool, optional
        Whether to save compressed versions of data (default: False)
    overwrite : bool, optional
        Whether to overwrite existing directory (default: True)
    
    Returns:
    --------
    out_dir : Path
        Path to the output directory
    """
    # Get the working directory from general_config
    base_dir = working_dir if working_dir else os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory with new structure (overwrite if it exists)
    out_dir = create_output_directory(config, base_dir, general_config, reservoir_config, wells_config, scal_config, get_configuration, overwrite=overwrite)
    
    print(f"Generating {config.number_of_realizations} KL realizations with configuration:")
    print(config)
    
    permeability_fields, num_modes, grid = generate_kl_log_normal_real_params_3D(
        n_realizations=config.number_of_realizations,
        Nx=config.Nx, Ny=config.Ny, Nz=config.Nz,
        Lx=config.Lx, Ly=config.Ly, Lz=config.Lz,
        real_mean=config.mean, real_std=config.std,
        corr_length_fac=config.correlation_length_factor,
        energy_threshold=config.energy_threshold,
        seed=config.seed,
        reverse_order=config.reverse_order,
        dtype=config.dtype
    )
    
    # Save grid information
    save_grid_information(grid, out_dir, config)
    
    # Save all realizations as a single numpy array
    save_all_realizations(permeability_fields, out_dir, config, save_compressed)
    
    # Split realizations into train/validation/test if general_config is provided
    indices = None
    if general_config:
        indices = split_realizations(permeability_fields, out_dir, config, general_config, save_compressed)
    
    # Create subfolders for .dat files based on indices
    dat_dirs = create_dat_files_subfolders(out_dir, general_config, reservoir_config, wells_config, scal_config, get_configuration, indices)
    
    # Save individual realizations as .dat files
    if indices:
        # Split .dat files according to indices
        train_indices = indices["train_indices"]
        val_indices = indices["val_indices"]
        test_indices = indices["test_indices"]
        
        print(f"Saving .dat files to split folders (maintaining original indices)...")
        
        # Save train .dat files
        print(f"Saving {len(train_indices)} training realizations as individual .dat files...")
        for idx in train_indices:
            save_realization_to_dat(permeability_fields[idx], idx+1, dat_dirs["train"], config)
            
        # Save validation .dat files
        print(f"Saving {len(val_indices)} validation realizations as individual .dat files...")
        for idx in val_indices:
            save_realization_to_dat(permeability_fields[idx], idx+1, dat_dirs["val"], config)
            
        # Save test .dat files
        print(f"Saving {len(test_indices)} test realizations as individual .dat files...")
        for idx in test_indices:
            save_realization_to_dat(permeability_fields[idx], idx+1, dat_dirs["test"], config)
    else:
        # Save all .dat files to a single folder (no splitting)
        print(f"Saving {config.number_of_realizations} realizations as individual .dat files...")
        for i in range(config.number_of_realizations):
            save_realization_to_dat(permeability_fields[i], i+1, dat_dirs["all"], config)
    
    config_file = out_dir / "full_reservoir_config.json"
    KLConfig.save_full_config_json(config_file, general_config, reservoir_config, wells_config, scal_config, get_configuration, kle_config=config)
    
    def save_summary_info(fields, config, split_name, out_dir, num_modes=None):
        summary = {
            'n_realizations': fields.shape[0],
            'grid_dimensions': {
                'Nx': config.Nx,
                'Ny': config.Ny,
                'Nz': config.Nz
            },
            'physical_dimensions': {
                'Lx': config.Lx,
                'Ly': config.Ly,
                'Lz': config.Lz
            },
            'statistical_parameters': {
                'mean': float(np.mean(fields)),
                'std': float(np.std(fields)),
                'min': float(np.min(fields)),
                'max': float(np.max(fields)),
                'target_mean': getattr(config, 'mean', None),
                'target_std': getattr(config, 'std', None)
            },
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'correlation_length_factor': getattr(config, 'correlation_length_factor', None),
                'energy_threshold': getattr(config, 'energy_threshold', None),
                'modes_used': int(num_modes) if num_modes is not None else None,
                'seed': config.seed,
                'reverse_order': config.reverse_order,
                'dtype': str(config.dtype),
                'split': split_name
            }
        }
        file_path = out_dir / f"summary_{split_name}.json"
        with open(file_path, 'w') as f:
            json.dump(summary, f, cls=NumpyEncoder, indent=4)
        print(f"Saved summary_{split_name} information to {file_path}")
        return file_path

    # Save summary info for 'all' and 'train' by default
    save_summary_info(permeability_fields, config, 'all', out_dir, num_modes=num_modes)
    if 'train_indices' in locals() and train_indices is not None:
        save_summary_info(permeability_fields[train_indices], config, 'train', out_dir, num_modes=num_modes)

    # Optionally, you can add calls for 'val' and 'test' splits if desired
    # if 'val_indices' in locals() and val_indices is not None:
    #     save_summary_info(permeability_fields[val_indices], config, 'val', out_dir, num_modes=num_modes)
    # if 'test_indices' in locals() and test_indices is not None:
    #     save_summary_info(permeability_fields[test_indices], config, 'test', out_dir, num_modes=num_modes)
    
    return out_dir

def main():
    """Main function to run the KL realization generator"""
    # Try to get the configuration from default_configurations
    try:
        # Get working directory
        working_dir = WORKING_DIRECTORY

        # Get general configuration
        general_config = DEFAULT_GENERAL_CONFIG
        
        # Get reservoir configuration
        reservoir_config = DEFAULT_RESERVOIR_CONFIG
        
        # Check if wells configuration exists
        wells_config = DEFAULT_WELLS_CONFIG 

        # scal configuration
        scal_config = DEFAULT_SCAL_CONFIG
        
        # Use reservoir dimensions from configuration
        Lx = reservoir_config.get('length')
        Ly = reservoir_config.get('width')
        Lz = reservoir_config.get('thickness')
        
        if None in (Lx, Ly, Lz):
            raise ValueError("Reservoir dimensions (length, width, thickness) must be specified in DEFAULT_RESERVOIR_CONFIG")
        
        # Get grid dimensions
        Nx = reservoir_config.get('Nx')
        Ny = reservoir_config.get('Ny')
        Nz = reservoir_config.get('Nz')
        
        if None in (Nx, Ny, Nz):
            raise ValueError("Grid dimensions (Nx, Ny, Nz) must be specified in DEFAULT_RESERVOIR_CONFIG")
        
        # Get permeability realization settings
        perm_real_config = reservoir_config.get('realizations', {}).get('permx', {})
        
        number_of_realizations = perm_real_config.get('number')
        mean = perm_real_config.get('mean')
        std = perm_real_config.get('std')
        correlation_length_factor = perm_real_config.get('correlation_length_factor')
        energy_threshold = perm_real_config.get('energy_threshold')
        reverse_order = perm_real_config.get('reverse_order')
        
        # Get seed from DEFAULT_GENERAL_CONFIG if available, otherwise from permx config
        seed = perm_real_config.get('seed')
        if seed is None and DEFAULT_GENERAL_CONFIG and 'seed' in DEFAULT_GENERAL_CONFIG:
            seed = DEFAULT_GENERAL_CONFIG.get('seed')
            print(f"Using global seed from DEFAULT_GENERAL_CONFIG: {seed}")
        
        if seed is None:
            seed = 2000
            print(f"No seed found, using default seed: {seed}")
            
        # Get dtype from DEFAULT_GENERAL_CONFIG if available, otherwise set to np.float32
        dtype=DEFAULT_GENERAL_CONFIG.get('dtype')
        if dtype is None:
            dtype = np.float32
                
        # Check if any required value is missing
        required_values = {
            'number of realizations': number_of_realizations,
            'mean permeability': mean,
            'standard deviation': std,
            'correlation length factor': correlation_length_factor,
            'energy threshold': energy_threshold
        }
        
        missing_values = [k for k, v in required_values.items() if v is None]
        if missing_values:
            raise ValueError(f"Missing required values in DEFAULT_RESERVOIR_CONFIG: {', '.join(missing_values)}")
        
        print("Successfully loaded configuration from default_configurations.py")
        
    except (ImportError, ModuleNotFoundError) as e:
        print(f"ERROR: Could not import DEFAULT_RESERVOIR_CONFIG from default_configurations.py")
        print(f"Exception: {str(e)}")
        print("This script requires DEFAULT_RESERVOIR_CONFIG to be defined in default_configurations.py")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: Invalid configuration in DEFAULT_RESERVOIR_CONFIG: {str(e)}")
        print("Please check the structure of DEFAULT_RESERVOIR_CONFIG in default_configurations.py")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error loading configuration: {str(e)}")
        sys.exit(1)
        
    # Create configuration for KL expansion with values from default_configurations
    config = KLConfig(
        number_of_realizations=number_of_realizations,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        mean=mean,
        std=std,
        correlation_length_factor=correlation_length_factor,
        energy_threshold=energy_threshold,
        seed=seed,                                  # Use seed from configuration
        reverse_order=reverse_order,
        output_keyword="PERMX",                     # Default keyword for permeability in x-direction
        comment_prefix="--",                        # Default comment prefix
        add_comments=True,                          # Include comments by default
        dtype=dtype
    )
    
    # Generate and save realizations with the new structure
    # Get the save_compressed setting from general config with default of False
    save_compressed = DEFAULT_GENERAL_CONFIG.get('save_compressed', False)
    
    # Default to overwrite=True to ensure clean generation
    overwrite = True
    
    output_dir = generate_and_save_realizations(
        config,
        working_dir=working_dir,
        general_config=general_config,
        reservoir_config=reservoir_config,
        wells_config=wells_config,
        scal_config=scal_config,
        get_configuration=get_configuration,
        save_compressed=save_compressed,
        overwrite=overwrite
    )
    
    print("=" * 80)
    print(f"KL Realizations Generation Complete")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()