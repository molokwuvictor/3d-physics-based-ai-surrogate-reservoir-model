#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Processing Package for SRM

This package contains modules for processing simulation and reservoir model data.
"""

# Import key functions and classes to make them available at the package level
from .srm_data_processing import SRMDataProcessor, create_positional_grids, DataSummary, weave_tensors, align_and_trim_pair_lists, split_tensor_sequence, slice_statistics

# Define what symbols to export when using 'from data_processing import *'
__all__ = [
    'SRMDataProcessor',
    'create_positional_grids',
    'DataSummary', 
    'weave_tensors',
    'align_and_trim_pair_lists',
    'split_tensor_sequence',
    'slice_statistics'
]
