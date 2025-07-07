#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Victor Molokwu PhD
# Date: 07/07/2025 13:05
# 
"""
Data Processing Package for SRM

This package contains modules for processing simulation and reservoir model data.
"""

# Import from data_processing_utils.py
from .data_processing_utils import (
    DataSummary,
    weave_tensors,
    create_positional_grids,
    align_and_trim_pair_lists,
    split_tensor_sequence,
    slice_statistics,
    slice_tensor,
    reshape_and_save_dict,
    load_dataframe,
    l1_normalize_excluding_index,
    print_group_shapes
)

# Import from srm_data_processing.py
from .srm_data_processing import (
    SRMDataProcessor,
    load_srm_data,
    validate_split_ratios
)

# Import from kl_expansion.py
from .kl_expansion import (
    generate_kl_log_normal_real_params_3D,
    plot_realizations_3D,
    plot_model_3d_grid
)

# Import from kle_realization_generator.py
from .kle_realization_generator import (
    KLConfig,
    NumpyEncoder,
    generate_full_config_hash,
    generate_and_save_realizations,
    create_output_directory,
    split_realizations
)

# Import from simulation_data_process_pipeline.py
from .simulation_data_process_pipeline import (
    run_pipeline_from_config,
    save_results,
    load_results,
    run_array_pipeline,
    process_files_in_directory
)

__all__ = [
    # Data Processing Utils
    'DataSummary', 'weave_tensors', 'create_positional_grids', 'align_and_trim_pair_lists',
    'split_tensor_sequence', 'slice_statistics', 'slice_tensor', 'reshape_and_save_dict',
    'load_dataframe', 'l1_normalize_excluding_index', 'print_group_shapes',
    
    # SRM Data Processing
    'SRMDataProcessor', 'load_srm_data', 'validate_split_ratios',
    
    # KL Expansion
    'generate_kl_log_normal_real_params_3D', 'plot_realizations_3D', 'plot_model_3d_grid',
    
    # KLE Realization Generator
    'KLConfig', 'NumpyEncoder', 'generate_full_config_hash', 'generate_and_save_realizations',
    'create_output_directory', 'split_realizations',
    
    # Simulation Data Process Pipeline
    'run_pipeline_from_config', 'save_results', 'load_results', 'run_array_pipeline',
    'process_files_in_directory'
]
