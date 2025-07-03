"""
Default configurations for various neural network models and layers.
This file contains standard configurations that can be imported and used
throughout the codebase to maintain consistency.
"""

import tensorflow as tf
import os
import sys
import numpy as np

# Define working directory
WORKING_DIRECTORY = "C:/Users/User/Documents/PHD_HW_Machine_Learning/ML_Cases_2025/Main_Library/New Methods"

# Try to import data_processing for spline data
try:
    sys.path.append(WORKING_DIRECTORY)
    import data_processing as dp
except ImportError:
    print("Warning: data_processing module not found. Spline fitting will not be available.")
    dp = None

# Default general settings
DEFAULT_GENERAL_CONFIG = {
    'save_compressed': False,  # Whether to save compressed versions of numpy arrays
    'load_compressed': False,   # Whether to load compressed versions of numpy arrays
    'seed': 2000,
    'dtype': np.float32,
    # Batch sizes for training and testing
    'training_batch_size': 32,
    'testing_batch_size': 64,
    'unit_target_shape': (1, 1, 39, 39, 1),                     # Input to the SRM (already transposed)
    
    # Time step settings
    'srm_start_time': 0.,
    'srm_end_time': 365.,
    'cfd_start_time': 0.,
    'cfd_end_time': 540.,
    'srm_timestep': 5.,
    'cfd_timestep': 1.,
    'maximum_srm_timestep': 10.,
    'minimum_srm_timestep': 0.1,
    'maximum_cfd_timestep': 1.,
    'minimum_cfd_timestep': 1.,

    # Data normalization settings
    # Possible methods: 'z-score', 'linear-scaling', 'lnk-linear-scaling'
    'data_normalization': {
        'feature_normalization_method': 'lnk-linear-scaling',  # Default normalization method
        'normalization_limits': [-1.0, 1.0],  # [a, b] for linear scaling methods
        'save_stats': True  # Whether to save normalization statistics
    },
    # Added for tensor splitting
    'split_keys': ['train', 'val', 'test'],
    'split_axis': [0,1],             # single or multi-axis to split indices
    'split_ratio': {0:(0.3, 0., 0.7),1:(0.7, 0., 0.3)},  # split ratios corresponding to the permeability time multi axis - split keys (train, val, test)
    'split_sampling_method':'random',                    # Samping method for the split - sequential or random
    # Option for physics or non-physics mode
    'physics_mode_fraction':1.,       # Physics mode fraction of 1 is pure physics - no training data snapshot is used
    
    # Fluid type
    'fluid_type': 'GC',                 # Dry Gas (DG) or Gas Condensate (GC)
    'above_dew_point': True,            # Whether to use a simple hard enforcement (above dew point conditions) or a deep learning based hard enforcement (below dew point conditions)
    'pvt_fitting_method': 'spline',     # 'spline' or 'polynomial'

    # Define default weights for gas and oil phases
    'default_weights': {
            'gas': {
                'dom': 1.0,  # Gas domain loss
                'ibc': 1.0,  # Gas inner boundary condition
                'obc': 0.0,  # Gas outer boundary condition
                'ic':  0.0,  # Gas initial condition
                'td': 0.0,   # Gas data loss (pressure)
                'mbc': 1.0,  # Gas material balance
                'cmbc': 0.0, # Gas cumulative material balance
                'tde': 1.0,  # Gas time discretization error
            },
            'oil': {
                'dom': 1.0,  # Oil domain loss
                'ibc': 1.0,  # Oil inner boundary condition
                'obc': 0.0,  # Oil outer boundary condition
                'ic':  0.0,  # Oil initial condition
                'td': 0.0,   # Oil data loss (Sg|So)
                'mbc': 1.0,  # Oil material balance
                'cmbc': 0.0, # Oil cumulative material balance
                'tde': 1.0,  # Oil time discretization error
            }
        },
        
    # SRM units
    'srm_units': 'field',
    
}

# Default configuration for reservoir model
DEFAULT_RESERVOIR_CONFIG = {
    'porosity': 0.2,
    'permx': 3.0,
    #'permy': 3.0,
    'horizontal_anisotropy': 1.0,
    'vertical_anisotropy':1.0,
    'depth': 11000.0,
    'length': 2900.0,
    'width': 2900.0,
    'thickness': 80.0,
    'Nx': 39,
    'Ny': 39,
    'Nz': 1,
    'initialization': {
    'Pi': 5000,
    'Pa': 1000},
    'realizations': {
        'permx': {
            'number': 200,
            'mean': 3.0,
            'std': 1.5, 
            'method': 'KLE',
            'correlation_length_factor': 0.2,
            'energy_threshold': 0.95,
            'seed': None,
            'reverse_order': True,
            'conditional_values': {
                (29, 29, 0): 2.0, 
                (29, 9, 0): 1.5, 
                (9, 9, 0): 1.0, 
                (9, 29, 0): 0.5
            }
        }, 
        'poro':{None}
    }
}

# Default configuration for wells, including shut-in periods
# Shutin_days: [[start_time, end_time]], where start_time is the time at which the well is shut-in and end_time is the time at which the well is shut-in
# if start_time must be less than end_time for a shut-in period to be valid. if start_time is greater than end_time, it implies no shut-in period
DEFAULT_WELLS_CONFIG = {
    'connections': [
        {'name': 'P1', 'i': 29, 'j': 29, 'k': 0, 'type': 'producer', 'control': 'ORAT', 'value': 500.0, 'minimum_bhp': 4100.0, 'wellbore_radius': 0.09525, 'completion_ratio':0.5, 'shutin_days': [[1000.0, 0.0]]},
        {'name': 'P2', 'i': 29, 'j': 9, 'k': 0, 'type': 'producer', 'control': 'ORAT', 'value': 1000.0, 'minimum_bhp': 4100.0,'wellbore_radius': 0.09525, 'completion_ratio':0.5, 'shutin_days': [[1000.0, 0.0]]},        # 10000
        {'name': 'P3', 'i': 9, 'j': 9, 'k': 0, 'type': 'producer', 'control': 'ORAT', 'value': 500.0, 'minimum_bhp': 4100.0,'wellbore_radius': 0.09525, 'completion_ratio':0.5, 'shutin_days': [[1000.0, 0.0]]},          # 1500
        {'name': 'P4', 'i': 9, 'j': 29, 'k': 0, 'type': 'producer', 'control': 'ORAT', 'value': 1000.0, 'minimum_bhp': 4100.0,'wellbore_radius': 0.09525, 'completion_ratio':0.5, 'shutin_days': [[1000.0, 0.0]]},        # 2000
        {'name': 'I1', 'i': 19, 'j': 19, 'k': 0, 'type': 'injector', 'control': 'ORAT', 'value': 0., 'minimum_bhp': 4100.0,'wellbore_radius': 0.09525, 'completion_ratio':0.5, 'shutin_days': [[1000.0, 0.0]]}
    ],
}

# Default configuration for Encoder-Decoder model
DEFAULT_ENCODER_DECODER_CONFIG = {
    'depth': 4,
    'width': {'Bottom_Size': 32, 'Growth_Rate': 1.5},
    'spatial_dims': 2,  # Default is 2D
    'temporal': False,
    'output_filters': 1,  # Number of output filters/channels in the final layer
    'residual_params': {
        'Kernel_Size': 3,
        'Kernel_Init': 'glorot_normal',
        'Activation_Func': 'swish',
        'Out_Activation_Func': None,
        'Dropout': {'Add': False, 'Rate': 0.2, 'Layer': [1, 0, 0, 0]},
        'Skip_Connections': {'Add': True, 'Layers': [1, 1, 1, 1]},
        'Decoder_Filter_Fac': 1.0,
        'Latent_Layer': {'Flatten': False, 'Depth': 1, 'Width': 128, 'Activation': None},
        'Extra_Conv_Layers': {'Count': 2},
        'Extra_Dec_Conv_Layers': {'Count': 2}
    }
}

# Default configuration for 3D Encoder-Decoder model
DEFAULT_ENCODER_DECODER_3D_CONFIG = {
    'depth': 4,
    'width': {'Bottom_Size': 32, 'Growth_Rate': 1.5},
    'spatial_dims': 3,  # Using 3D spatial dimensions
    'temporal': False,
    'output_filters': 1,  # Number of output filters/channels in the final layer
    'residual_params': {
        'Kernel_Size': 3,
        'Kernel_Init': 'glorot_normal',
        'Activation_Func': 'swish',
        'Out_Activation_Func': None,
        'Dropout': {'Add': False, 'Rate': 0.2, 'Layer': [1, 0, 0, 0]},
        'Skip_Connections': {'Add': True, 'Layers': [1, 1, 1, 1]},
        'Decoder_Filter_Fac': 1.0,
        'Latent_Layer': {'Flatten': False, 'Depth': 1, 'Width': 128, 'Activation': None},
        'Extra_Conv_Layers': {'Count': 2},
        'Extra_Dec_Conv_Layers': {'Count': 2}
    }
}

# Default configuration for Residual Network
DEFAULT_RESIDUAL_NETWORK_CONFIG = {
    'num_blocks': 4,
    'filters': 32,
    'kernel_size': 3,
    'hidden_activation': tf.nn.swish,
    'output_activation': None,
    'output_filters':1,
    'kernel_initializer':"glorot_normal",
    'network_type': 'cnn',  # Use CNN layers by default or 'dense'
    'use_batch_norm': False,
    'dropout_rate': 0.0,
    'output_distribution': True,
    'number_of_output_bins': 50,                        # Used to model timesteps as a discrete categorical distribution
}

# Default configuration for Hard Layer
DEFAULT_HARD_LAYER_CONFIG = {
    'norm_limits': [-1, 1],
    'init_value': 1.0,
    'kernel_activation': None,
    'input_activation': None,
    'kernel_exponent_config': {
        'initial_value': 0.5,
        'trainable': True,
        'min_value': 0.1,
        'max_value': 0.99
    },
    'use_rbf': False,  # Default is no RBF
    'regularization': 0.001,
    'rectifier': None
}

# Default configuration for input slicing
DEFAULT_INPUT_SLICE_CONFIG = {
    'encoder_decoder': slice(None),  # Use all channels for encoder-decoder
    'residual_network': slice(None),  # Use all channels for residual network
    'hard_layer': {
        'time': slice(-2, -1),      # Second-to-last channel for time
        'property': slice(-1, None)  # Last channel for property
    }
}

# Default configuration for PVT Layer - Dry Gas (DG)
DEFAULT_PVT_DG_CONFIG = {
    'fluid_type': 'DG',              # Dry Gas fluid type
    'fitting_method': 'polynomial',  # Default fitting method is polynomial
    'polynomial_config': {
        'invBg': [1.0, 0.1, 0.01],  # Default polynomial coefficients for inverse gas formation volume factor
        'invug': [0.5, 0.05, 0.005]  # Default polynomial coefficients for inverse gas viscosity
    },
    'spline_order': 2,
    'regularization_weight': 0.001,
    'min_input_threshold': 14.7,      # Minimum pressure threshold (default 14.7 psi)
    'max_input_threshold': 10000.0    # Maximum pressure threshold (default 10000 psi)
}

# Default configuration for PVT Layer - Gas Condensate (GC)
DEFAULT_PVT_GC_CONFIG = {
    'fluid_type': 'GC',              # Gas Condensate fluid type
    'fitting_method': 'polynomial',  # Default fitting method is polynomial
    'polynomial_config': {
        'invBg': [1.0, 0.1, 0.01],
        'invBo': [1.2, 0.12, 0.012],
        'invug': [0.5, 0.05, 0.005],
        'invuo': [0.6, 0.06, 0.006],
        'Rs': [0.7, 0.07, 0.007],
        'Rv': [0.8, 0.08, 0.008],
        'Vro': [0.9, 0.09, 0.009]
    },
    'spline_order': 2,
    'regularization_weight': 0.001,
    'min_input_threshold': 14.7,      # Minimum pressure threshold (default 14.7 psi)
    'max_input_threshold': 10000.0,   # Maximum pressure threshold (default 10000 psi)
    'dew_point':4048.4,
}

# Default SCAL configuration
DEFAULT_SCAL_CONFIG = {
    'end_points': {'kro_Somax': 0.90, 'krg_Sorg': 0.80, 'krg_Swmin': 0.90, 'Swmin': 0.22, 'Sorg': 0.2, 'Sgc': 0.05, 'Socr':0.2, 'So_max':0.28},
    'corey_exponents': {'nog': 3., 'ng': 6., 'nw': 2.},
    'blocking_factor':{'number_of_intervals':5, 'number_of_iterations':5}
}

# Default configuration for PVT Layer (general)
DEFAULT_PVT_LAYER_CONFIG = DEFAULT_PVT_DG_CONFIG.copy()

# Default configuration for PVT Module (combining Hard Layer and PVT Layer)
DEFAULT_PVT_MODULE_CONFIG = {
    'use_hard_layer': True,
    'hard_layer_config': DEFAULT_HARD_LAYER_CONFIG.copy(),
    'pvt_layer_config': DEFAULT_PVT_LAYER_CONFIG.copy(),
    'input_slice_config': DEFAULT_INPUT_SLICE_CONFIG.copy()
}

# Default configuration for simulation data processing pipeline
DEFAULT_SIMDATA_PROCESS_CONFIG = {
    "simulation_pipeline": {
        "enabled": True,                    # Run simulation pipeline? (Default: True)
        "parallel": False,                   # Enable parallel processing (Default: True)
        "max_workers": 4,                   # Maximum number of parallel workers (Default: 4)
        "save_results": True,               # Save simulation results to file? (Default: True)
        "combine": True,                    # Save as one combined file? (Default: True)
        "flatten": True,                    # Flatten results by removing top-level extension keys (Default: True)
        "stack_realizations": True,         # Stack data across realizations (Default: True)
        "combined_filename": "combined_results.npz",  # Filename for combined simulation results
        "file_vectors": {                   # Mapping of file extensions to target vectors
            ".FINIT": ["PERMX", "PERMZ", "PORO"],
            ".FUNRST": ["PRESSURE", "SOIL", "SGAS"],
            ".RSM": [["TIME"], ["WOPR", "15 15 1"], "WGPR", "WWPR", "WBHP"]
        },
        "shape": (39, 39, 1)                # Optional shape to reshape continuous file arrays
    },
    "array_pipeline": {
        "enabled": True,                    # Enable array processing pipeline? (Default: True)
        "ext": ".npz",                      # File extension to process (.npz or .json)
        "file": None,                       # Specific file name to process (Default: None)
        "keys": ["PRESSURE", "SGAS"],       # Dictionary keys to extract from the file
        "exclusions": ["PERMX", "PERMY", "PERMZ", "PORO"],  # Keys to exclude from processing
        "slice_dim": 1,                     # Axis on which to perform slicing (Default: 1)
        "reshape_dims": (0,),               # Axes to merge during reshaping (Default: (0, 1))
        "dtype": DEFAULT_GENERAL_CONFIG.get("dtype"),  # Use the DEFAULT_GENERAL_CONFIG dtype setting
    }
}

# === Optimizer settings for network modules (keyed by logical task) ===
DEFAULT_OPTIMIZER_CONFIGS = {
    'pressure': {
        'type': 'adamw',
        'learning_rate': 0.005,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'weight_decay': 0.00005,
        'trainable': True,
        'exponential_decay': {
            'enabled': True,              # Master switch for all decay
            'learning_rate': {
                'enabled': True,          # Enable learning rate decay
                'decay_steps': 25,       # Steps between decay applications
                'decay_rate': 0.90,       # Multiplicative factor (0.96 = 96% of previous value)
            },
            'weight_decay': {
                'enabled': True,          # Enable weight decay parameter decay (AdamW only)
                'decay_rate': 0.90,       # Decay rate for weight decay parameter
            },
            'staircase': False           # If True, decay occurs in discrete steps
        }
    },
    'time_step': {
        'type': 'adam',
         'learning_rate': 0.0001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'weight_decay': 0.00001,
        'trainable': True,
        'exponential_decay': {
            'enabled': True,              # Master switch for all decay
            'learning_rate': {
                'enabled': True,          # Enable learning rate decay
                'decay_steps': 25,       # Steps between decay applications
                'decay_rate': 0.90,       # Multiplicative factor (0.96 = 96% of previous value)
            },
            'weight_decay': {
                'enabled': False,         # Disable weight decay parameter decay
                'decay_rate': 0.90,       # Decay rate for weight decay parameter
            },
            'staircase': False           # If True, decay occurs in discrete steps
        }
    },
    'fluid_property': {
        'type': 'adamw',
        'learning_rate': 0.0005,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'weight_decay': 0.0005,
        'trainable': False,
        'exponential_decay': {
            'enabled': False,             # Master switch for all decay (disabled for this model)
            'learning_rate': {
                'enabled': False,          # Disable learning rate decay
                'decay_steps': 100,       # Steps between decay applications
                'decay_rate': 0.96,       # Multiplicative factor
            },
            'weight_decay': {
                'enabled': False,          # Disable weight decay parameter decay
                'decay_rate': 0.98,       # Decay rate for weight decay parameter
            },
            'staircase': False           # If True, decay occurs in discrete steps
        }
    },
    'well_rate_bhp': {
        'type': 'adamw',
        'learning_rate': 0.0005,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'weight_decay': 0.0005,
        'trainable': False,    # Well rate/BHP model is not trainable
        'exponential_decay': {
            'enabled': False,             # Master switch for all decay (disabled for this model)
            'learning_rate': {
                'enabled': False,          # Disable learning rate decay
                'decay_steps': 100,       # Steps between decay applications
                'decay_rate': 0.96,       # Multiplicative factor
            },
            'weight_decay': {
                'enabled': False,          # Disable weight decay parameter decay
                'decay_rate': 0.98,       # Decay rate for weight decay parameter
            },
            'staircase': False           # If True, decay occurs in discrete steps
        }
    },
    'saturation': {
        'type': 'adamw',
        'learning_rate': 0.0005,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'weight_decay': 0.0005,
        'trainable': True,     # Only for GC
        'exponential_decay': {
            'enabled': True,              # Master switch for all decay
            'learning_rate': {
                'enabled': True,          # Enable learning rate decay
                'decay_steps': 100,       # Steps between decay applications
                'decay_rate': 0.96,       # Multiplicative factor
            },
            'weight_decay': {
                'enabled': False,         # Disable weight decay parameter decay
                'decay_rate': 0.98,       # Decay rate for weight decay parameter
            },
            'staircase': False           # If True, decay occurs in discrete steps
        }
    }
}


# === Optimizer-to-model mapping (user can edit this) ===
DEFAULT_OPTIMIZER_MODEL_MAPPING_DG = {
    'pressure': 'encoder_decoder',
    'time_step': 'residual_network',
    'fluid_property': 'pvt_model',
    'well_rate_bhp': 'well_rate_bhp_model',
}

DEFAULT_OPTIMIZER_MODEL_MAPPING_GC = {
    'pressure': 'encoder_decoder',
    'time_step': 'residual_network',
    'fluid_property': 'pvt_model',
    'well_rate_bhp': 'well_rate_bhp_model',
    'saturation': 'saturation_model',
}

def get_optimizer_model_mapping(fluid_type=None):
    """
    Returns the optimizer-to-model mapping based on the fluid type.
    If fluid_type is 'GC', includes the saturation model.
    """
    if fluid_type is None:
        fluid_type = DEFAULT_GENERAL_CONFIG.get('fluid_type', 'DG')
    if fluid_type == 'GC':
        return DEFAULT_OPTIMIZER_MODEL_MAPPING_GC.copy()
    else:
        return DEFAULT_OPTIMIZER_MODEL_MAPPING_DG.copy()


# Default unit conversion constants
DEFAULT_CONVERSION_CONSTANTS = {
    'field':{'C': 0.001127, 'D': 5.6145833334}
    }

# Getter functions for the artificial neural network architectures
def get_optimizer_config(name):
    return DEFAULT_OPTIMIZER_CONFIGS.get(name, None)

def get_conversion_constants(name):
    return DEFAULT_CONVERSION_CONSTANTS.get(name, None)

def get_configuration(config_type, input_shape=None, use_rbf=False, fluid_type=None, fitting_method=None):
    """
    Get the appropriate configuration based on the requested type and input parameters.
    
    Args:
        config_type (str): Type of configuration to retrieve 
                          ('encoder_decoder', 'residual', 'hard_layer', 'input_slice', 'pvt_layer', 'pvt_module')
        input_shape (tuple, optional): Shape of the input data, used to determine spatial dimensions
        use_rbf (bool, optional): Whether to use RBF in hard layer configuration
        fluid_type (str, optional): Fluid type for PVT layer ('DG' or 'GC')
        fitting_method (str, optional): Fitting method for PVT layer ('polynomial' or 'spline')
        
    Returns:
        dict: Configuration dictionary with appropriate settings
    """
    if config_type.lower() == 'encoder_decoder':
        # Determine if we should use 3D configuration
        if input_shape and len(input_shape) >= 4 and input_shape[-3] > 1:               # Only used when D>1
            config = DEFAULT_ENCODER_DECODER_3D_CONFIG.copy()
        else:
            config = DEFAULT_ENCODER_DECODER_CONFIG.copy()
        return config
    
    elif config_type.lower() == 'residual':
        config = DEFAULT_RESIDUAL_NETWORK_CONFIG.copy()
        return config
    
    elif config_type.lower() == 'hard_layer':
        config = DEFAULT_HARD_LAYER_CONFIG.copy()
        
        # Update for RBF if requested
        # if use_rbf:
        #     config.update(DEFAULT_HARD_LAYER_RBF_CONFIG)
            
        return config
    
    elif config_type.lower() == 'input_slice':
        config = DEFAULT_INPUT_SLICE_CONFIG.copy()
        return config
    
    elif config_type.lower() == 'pvt_layer':
        # Get the base configuration based on fluid type
        if fluid_type and fluid_type.upper() == 'GC':
            config = DEFAULT_PVT_GC_CONFIG.copy()
        else:
            config = DEFAULT_PVT_DG_CONFIG.copy()
            
        # Update fitting method if provided
        if fitting_method:
            config['fitting_method'] = fitting_method.lower()
            
            # If spline method is requested, load spline data
            if fitting_method.lower() == 'spline':
                spline_config = load_spline_data()
                if spline_config:
                    config['spline_config'] = spline_config
                else:
                    # Fall back to polynomial if spline data loading failed
                    print("Falling back to polynomial fitting due to missing spline data")
                    config['fitting_method'] = 'polynomial'
                    
        return config
    
    elif config_type.lower() == 'pvt_module':
        config = DEFAULT_PVT_MODULE_CONFIG.copy()
        
        # Update PVT layer configuration based on fluid type and fitting method
        pvt_config = get_configuration('pvt_layer', fluid_type=fluid_type, fitting_method=fitting_method)
        config['pvt_layer_config'] = pvt_config
        
        # Update Hard Layer configuration
        hard_config = get_configuration('hard_layer', use_rbf=use_rbf)
        config['hard_layer_config'] = hard_config
        
        return config
    
    else:
        raise ValueError(f"Unknown configuration type: {config_type}. " 
                         f"Valid types: encoder_decoder, residual, hard_layer, input_slice, pvt_layer, pvt_module")

# Function to load spline data
def load_spline_data():
    """
    Load spline data for PVT calculations.
    
    Returns:
        spline_config: DataSummary object containing spline data or None if loading fails
    """
    if dp is None:
        print("Warning: data_processing module not available, cannot load spline data")
        return None
    
    try:
        # Explicitly set the path to the New Methods directory
        new_methods_dir = os.path.dirname(os.path.abspath(__file__))
        _pvt_data = dp.load_dataframe(filename='pvt_data', filetype='df', load_dir=new_methods_dir)
        spline_config = dp.DataSummary(data_list=[_pvt_data,], dtype=DEFAULT_GENERAL_CONFIG['dtype'] )
        return spline_config
    except Exception as e:
        print(f"Warning: Failed to load PVT data for spline fitting: {str(e)}")
        return None
