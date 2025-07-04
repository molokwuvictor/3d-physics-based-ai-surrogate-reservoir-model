"""
Network Architecture Case Study

This file builds and tests three different model architectures:
1. Encoder-decoder WITH Hard Layer (no RBF)
2. Residual Neural Network WITHOUT Hard Layer 
3. PVT model WITHOUT Hard Layer (using output from model 1)

The first two models share the same input shape, and the third model 
uses the output from the first model as its input.
"""
import os
import logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 

# Suppress Python warnings from TensorFlow
tf.get_logger().setLevel(logging.ERROR)

# Must be done before any TF GPU ops or model loading
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

import numpy as np
import sys
import pickle
import time
from srm_data_processing import SRMDataProcessor

from training import BatchGenerator, build_optimizer_from_config, validate_loss_keys
from default_configurations import get_optimizer_config, get_configuration, get_optimizer_model_mapping, WORKING_DIRECTORY, DEFAULT_GENERAL_CONFIG, DEFAULT_RESERVOIR_CONFIG
from physics_loss_Subclassed import PhysicsLoss
from well_rate_bhp_Subclassed import WellRatesPressure

# Import EncoderDecoderSubclassed module to disable debug shape printing
import EncoderDecoderSubclassed
from complete_pvt_module import PVTModuleWithHardLayer
from complete_trainable_module import CompleteTrainableModule
from plot_functions_tf import ModelPlotter

# Add the working directory to path to import model classes
sys.path.append(WORKING_DIRECTORY)

# Configure logging to show INFO level by default
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable debug shape printing in EncoderDecoderSubclassed module
EncoderDecoderSubclassed.DEBUG_SHAPES = False

def build_encoder_decoder_with_hard(input_shape=(None, 1, 39, 39, 3), name="encoder_decoder_with_hard"):
    """
    Build an encoder-decoder network with hard layer.
    
    Args:
        input_shape: Shape of the input tensor (None, T, H, W, C) where T is temporal dimension
        name: Name for the model
        
    Returns:
        Keras model with encoder-decoder and hard layer
    """
    logging.info(f"\n{'='*50}\n=== Building Model 1: {name} ===\n{'='*50}")
    
    # Get configurations - explicitly set spatial_dims to 2 and temporal to True for 2D input with temporal dimension
    encoder_decoder_config = get_configuration('encoder_decoder')
    encoder_decoder_config['spatial_dims'] = 2  # Explicitly set to 2D to use temporal dimension processing
    encoder_decoder_config['temporal'] = True   # Enable temporal dimension processing
    encoder_decoder_config['residual_params']['Extra_Conv_Layers']['Count'] = 2
    encoder_decoder_config['residual_params']['Extra_Dec_Conv_Layers']['Count'] = 2
    encoder_decoder_config['residual_params']['Latent_Layer']['Depth'] = 1
    encoder_decoder_config['residual_params']['Latent_Layer']['Activation'] = None
    encoder_decoder_config['residual_params']['Out_Activation_Func'] = None
    encoder_decoder_config['residual_params']['Skip_Connections']={'Add': False, 'Layers': [1, 1, 1, 1]}
    # Set the number of output filters in the final layer
    # This controls the innermost dimension of the encoder-decoder output
    # encoder_decoder_config['output_filters'] = 1  # Explicitly set to 1 output filter
    def scaled_tanh_with_xtanhx(x, min_val = 0.1, max_val = 10, steepness=1.0):
        # Calculate lisht = x * tanh(x)
        lisht = x * tf.math.tanh(x)
        # Apply steepness scaling inside tanh as (1 - tanh(steepness * lisht))
        return (max_val - min_val) * (tf.math.tanh(steepness * lisht)) + min_val  
    
    hard_layer_config = get_configuration('hard_layer', use_rbf=False)
    hard_layer_config['init_value'] = DEFAULT_RESERVOIR_CONFIG['initialization']['Pi']
    hard_layer_config['kernel_activation'] = None# lambda x: scaled_tanh_with_xtanhx(x,max_val = 1)
    hard_layer_config['kernel_exponent_config']['initial_value'] = 0.5,
    hard_layer_config['kernel_exponent_config']['min_value'] = 0.1
    hard_layer_config['kernel_exponent_config']['max_value'] = 1

    input_slice_config = get_configuration('input_slice')
    
    # Ensure input slice config is correctly set for time and property dimensions
    # Time data is in the second-to-last channel, property data is in the last channel
    if not input_slice_config:
        input_slice_config = {
            'time': slice(-2, -1),  # Second-to-last channel contains time data
            'property': slice(-1, None)  # Last channel contains property data
        }
    
    # Create the complete trainable module
    complete_module = CompleteTrainableModule(
        network_type='encoder_decoder',
        encoder_decoder_config=encoder_decoder_config,
        use_hard_layer=True,
        hard_layer_config=hard_layer_config,
        input_slice_config=input_slice_config
    )
    
    # Create Keras model
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    outputs = complete_module(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    # Print summary
    model.summary()
    
    # Print parameters
    network_params = np.sum([np.prod(v.shape) for v in complete_module.main_network.trainable_weights])
    hard_layer_params = np.sum([np.prod(v.shape) for v in complete_module.hard_layer.trainable_weights])
    
    logging.info(f"Encoder-Decoder trainable parameters: {network_params}")
    logging.info(f"Hard Layer trainable parameters: {hard_layer_params}")
    logging.info(f"Total trainable parameters: {network_params + hard_layer_params}")
    
    return model

def build_residual_network_with_hard(input_shape=(None, 1, 39, 39, 3), name="residual_network_only"):
    """
    Build a residual neural network without hard layer.
    
    Args:
        input_shape: Shape of the input tensor (None, T, H, W, C) where T is temporal dimension
        name: Name for the model
        
    Returns:
        Keras model with residual network only
    """
    logging.info(f"\n{'='*50}\n=== Building Model 2: {name} ===\n{'='*50}")
    
    # Get configurations
    residual_network_config = get_configuration('residual')
    input_slice_config = get_configuration('input_slice')
    
    # Update configuration for temporal data if input shape has 5 dimensions
    residual_network_config['temporal'] = False
    residual_network_config['output_distribution']=False
        
    hard_layer_config = get_configuration('hard_layer', use_rbf=False)
    hard_layer_config['init_value'] = DEFAULT_RESERVOIR_CONFIG['initialization']['Pi']
    # Ensure input slice config is correctly set for time and property dimensions
    if not input_slice_config:
        input_slice_config = {
            'time': slice(-2, -1),  # Second-to-last channel contains time data
            'property': slice(-1, None)  # Last channel contains property data
        }
    
    # Create the complete trainable module
    complete_module = CompleteTrainableModule(
        network_type='residual',
        residual_network_config=residual_network_config,
        use_hard_layer=True,
        hard_layer_config=hard_layer_config,
        input_slice_config=input_slice_config
    )
    
    # Create Keras model - need to handle temporal dimension before passing to residual network
    # This will be managed in the CompleteTrainableModule's call method
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    outputs = complete_module(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    # Print summary
    model.summary()
    
    # Print parameters
    network_params = np.sum([np.prod(v.shape) for v in complete_module.main_network.trainable_weights])
    logging.info(f"Residual Network trainable parameters: {network_params}")
    logging.info(f"Total trainable parameters: {network_params}")
    
    return model

def build_residual_network_without_hard(input_shape=(None, 1, 39, 39, 3), name="residual_network_only"):
    """
    Build a residual neural network without hard layer.
    
    Args:
        input_shape: Shape of the input tensor (None, T, H, W, C) where T is temporal dimension
        name: Name for the model
        
    Returns:
        Keras model with residual network only
    """
    logging.info(f"\n{'='*50}\n=== Building Model 2: {name} ===\n{'='*50}")
    
    # Get configurations
    residual_network_config = get_configuration('residual')
    input_slice_config = get_configuration('input_slice')
    
    # Update configuration for temporal data if input shape has 5 dimensions
    residual_network_config['network_type'] = 'cnn'
    residual_network_config['number_of_output_bins'] = 50
    residual_network_config['temporal'] = True
    residual_network_config['output_distribution'] = False
    max_lim = DEFAULT_GENERAL_CONFIG['maximum_srm_timestep']

    def scaled_tanh_with_xtanhx(x, min_val = 0.1, max_val = 10, steepness=1.0):
        # Calculate lisht = x * tanh(x)
        lisht = x * tf.math.tanh(x)
        # Apply steepness scaling inside tanh as (1 - tanh(steepness * lisht))
        return (max_val - min_val) * (tf.math.tanh(steepness * lisht)) + min_val  
    
    # Sinusoidal-LiSHT activation: oscillates between min_val and max_val
    def sinusoidal_lisht(x, min_val=0.1, max_val=10.0, frequency=0.5):
        A = (max_val - min_val) / 2
        C = (max_val + min_val) / 2
        lisht = x * tf.math.tanh(x)
        return A * tf.math.cos(frequency * lisht) + C
    
    # Smoothly bounded function between min_val and max_val using sigmoid
    def bounded_delta_t(x, min_val=0.1, max_val=10.0):
        return min_val + (max_val - min_val) * tf.sigmoid(x)
    
    # Standard LiSHT activation with scaling
    def lisht(x, a=1.0, b=1.0):
        return a * x * tf.tanh(b * x)
    
    # Exponential soft clip above max_val
    def exp_upper_clip(y, max_val=10.0, rate=5.0):
        return tf.where(
            y <= max_val,
            y,
            max_val + (y - max_val) * tf.exp(-rate * (y - max_val))
        )
    
    # LiSHT activation with exponential upper clipping
    def lisht_exp_clip(x, a=1.0, b=1.0, max_val=10.0, rate=5.0):
        y = lisht(x, a, b)
        return exp_upper_clip(y, max_val=max_val, rate=rate)

    residual_network_config['output_activation'] = lambda x: scaled_tanh_with_xtanhx(x, max_val = 10)
    
    # Ensure input slice config is correctly set for time and property dimensions
    if not input_slice_config:
        input_slice_config = {
            'time': slice(-2, -1),  # Second-to-last channel contains time data
            'property': slice(-1, None)  # Last channel contains property data
        }
    
    # Create the complete trainable module
    complete_module = CompleteTrainableModule(
        network_type='residual',
        residual_network_config=residual_network_config,
        use_hard_layer=False,
        input_slice_config=input_slice_config
    )
    
    # Create Keras model - need to handle temporal dimension before passing to residual network
    # This will be managed in the CompleteTrainableModule's call method
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    outputs = complete_module(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    # Print summary
    model.summary()
    
    # Print parameters
    network_params = np.sum([np.prod(v.shape) for v in complete_module.main_network.trainable_weights])
    logging.info(f"Residual Network trainable parameters: {network_params}")
    logging.info(f"Total trainable parameters: {network_params}")
    
    return model

def build_pvt_model_without_hard(input_model, name="pvt_model_only"):
    """
    Build a PVT model without hard layer that uses output from another model.
    
    Args:
        input_model: Model whose output will serve as input to this model
        name: Name for the model
        
    Returns:
        Keras model with PVT layer only
    """
    logging.info(f"\n{'='*50}\n=== Building Model 3: {name} ===\n{'='*50}")
    
    # Get the output shape from the input model
    output_shape = input_model.output_shape
    logging.info(f"Input model output shape: {output_shape}")
    
    # Get configurations - use spline method as requested
    # The get_configuration function now handles loading spline data internally
    logging.info("Configuring PVT Layer with spline fitting method")
    pvt_layer_config = get_configuration('pvt_layer', fluid_type='DG', fitting_method='spline')
    pvt_layer_config['spline_order']=1
    # pvt_layer_config['min_input_threshold'] = -30000

    # Create the PVT module without hard layer
    pvt_module = PVTModuleWithHardLayer(
        use_hard_layer=False,
        pvt_layer_config=pvt_layer_config
    )
    
    # Create a model that directly uses the output from model 1
    inputs = tf.keras.layers.Input(shape=output_shape[1:])
    
    # Apply the PVT module directly to the input
    outputs = pvt_module(inputs)
    
        #
    # PVT Model Output Shape Explanation:
    #   (2, n_properties, batch_size, time, depth, height, width, 1)
    #   |      |             |           |      |      |   |
    #   |      |             |           |      |      |   +-- Channel (singleton, usually 1)
    #   |      |             |           |      |      +------ Width (spatial grid)
    #   |      |             |           |      +------------- Height (spatial grid)
    #   |      |             |           +-------------------- Depth (singleton for 2D, >1 for 3D)
    #   |      |             |      
    #   |      |             +------------------------------- Batch size/Time steps (if temporal, else singleton)
    #   |      +--------------------------------------------- Number of fluid properties (e.g., 2 for DG: invBg, invug)
    #   +---------------------------------------------------- 0: value, 1: derivative
    #
    # For example, for output (2, 2, 2, 1, 39, 39, 1):
    #   - 0: value/derivative selector (0: value, 1: derivative)
    #   - 1: fluid property index (2: invBg, invug)
    #   - 2: batch size/time steps (2 or more)
    #   - 3: depth (1 for 2D)
    #   - 4: height (39)
    #   - 5: width (39)
    #   - 6: channel (1)
    
    # Create Keras model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    # Print summary
    model.summary()
    
    # Print parameters
    pvt_params = np.sum([np.prod(v.shape) for v in pvt_module.pvt_layer.trainable_weights])
    logging.info(f"PVT Layer trainable parameters: {pvt_params}")
    logging.info(f"Total trainable parameters: {pvt_params}")
    
    return model

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

# The unified training method replaces both train_all_models_combined and train_combined_model_with_mapping.
# All training logic is now in train_combined_models_unified.
# Any previous references to the old functions should be updated to use train_combined_models_unified.
       
if __name__ == "__main__":   
    logging.info("\n" + "="*50)
    logging.info("CASE STUDY: THREE NETWORK ARCHITECTURES")
    logging.info("="*50)
    
    # Get batch sizes from default configuration
    training_batch_size = 32#DEFAULT_GENERAL_CONFIG['training_batch_size']
    testing_batch_size = DEFAULT_GENERAL_CONFIG['testing_batch_size']
    fluid_type = DEFAULT_GENERAL_CONFIG['fluid_type']
    processor = SRMDataProcessor(base_dir=WORKING_DIRECTORY)
    
    # Getting training, validation, and test data groups...
    # Each group is a list of tuple of (stacked features, targets -- 
    # if exists (only in non-physics-based models) is usually unstacked and exist as dictionary)
    logging.info("\nGetting training, validation, and test data groups...")

    training_data_path, train_groups, val_groups, test_groups, pred_groups = processor.get_or_generate_training_data(train_silent=False)
    logging.info(f"Training data path: {training_data_path}")

    # Build model map
    model_map = build_model_map(train_groups[0][0].shape, fluid_type=fluid_type)

    # Get optimizer names => model names map
    optimizer_model_names_map = get_optimizer_model_mapping(fluid_type)

    # Prepare PhysicsLoss and loss function
    main_model = model_map['pressure']
    pvt_model = model_map['pvt_model']
    time_step_model = model_map['time_step']
    well_rate_bhp_model = model_map['well_rate_bhp_model']
    saturation_model = model_map['saturation_model'] if fluid_type == 'GC' else None

    loss_fn = PhysicsLoss(
        main_model=main_model,
        pvt_model=pvt_model,
        time_step_model=time_step_model,
        well_rate_bhp_model=well_rate_bhp_model,
        saturation_model=(saturation_model if fluid_type == 'GC' else None),
        optimizer_model_names_map=optimizer_model_names_map
    )
    
    # Define a log variable callback
    def my_log_callback(epoch, model_variables, total_loss):
        logging.info(f"Epoch {epoch + 1}: Total Loss = {total_loss:.4f}")
        for key, variables in model_variables.items():
            logging.info(f"Model {key}: {len(variables)} trainable variables")
    # Optionally save variables to disk or perform other logging
    # Train all models with the unified approach, passing batch sizes explicitly
    model_map, history, best_model_variables = train_combined_models_unified(
        train_groups,
        val_groups,
        test_groups=test_groups,
        model_map=model_map,
        optimizer_model_names_map=optimizer_model_names_map,
        training_batch_size=training_batch_size,
        testing_batch_size=testing_batch_size,
        epochs=5,
        custom_loss_fn=loss_fn,
        log_variables_callback=my_log_callback,
        log_epoch_percentage=0.2
    )

    # Print model information
    for model_name, model in model_map.items():
        if not hasattr(model, 'summary'):
            logging.warning(f"Skipping model '{model_name}' — no summary method.")
            continue
    
        logging.info(f"\nModel: {model_name}")
        model.summary(print_fn=logging.info)
    
        try:
            network_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            logging.info(f"Total trainable parameters: {network_params}")
        except Exception as e:
            logging.warning(f"Could not calculate trainable parameters for '{model_name}': {e}")


    # Save history data for plotting
    try:
        history_path = os.path.join(os.path.dirname(training_data_path), 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        logging.info(f"\nSaved training history to: {history_path}")
        logging.info("You can use this file to plot learning curves with:")
        logging.info("  import pickle")
        logging.info("  with open('training_history.pkl', 'rb') as f:")
        logging.info("      history = pickle.load(f)")
        logging.info("  # Plot total loss curve:")
        logging.info("  plt.plot(history['total_train_loss'])")
    except Exception as e:
        logging.warning(f"Could not save training history: {e}")
    
    # --- Batch shape check: print first batch shapes for train_ds ---
    logging.info("\nChecking batch shapes for train_ds:")
    train_ds = BatchGenerator(train_groups, batch_size=DEFAULT_GENERAL_CONFIG['training_batch_size'])
    if isinstance(train_ds[0], tuple):
        x, y = train_ds[0]
        logging.info(f"Features batch shape: {getattr(x, 'shape', None)}; Labels batch shape: {getattr(y, 'shape', None)}")
    else:
        logging.info(f"Batch shape: {getattr(train_ds[0], 'shape', None)}")
    logging.info("\n" + "="*50)
    logging.info("TRAINING COMPLETE")
    logging.info("="*50)
    
    # Plot graphs
    test_ds = BatchGenerator(test_groups, batch_size=training_batch_size)
    plotter = ModelPlotter(
        model_map=model_map,
        test_pairs=test_groups,    
    )
    # 2D image plot at time‐points [0, 5, 10, 15, 20]:
    # plotter.set_font_settings(font_size=16.0, font_type='Times New Roman')
    # plotter.plot_images(
    #     key='PRESSURE',
    #     figsize_per = (3.5, 3.5),
    #     a_indices=[0, 5, 10, 15, 20,],
    #     b_indices=[0, 5, 10, 15, 20,],
    #     suptitle='Pred vs True Images (a)'
    # )
    plotter.set_unit_labels(x_unit_label='s', y_unit_label='m/s')
    plotter.set_font_settings(font_size=10.0, font_type='Times New Roman')
    plotter.plot_line(
        key='PRESSURE',
        b_indices=None,
        a_indices=[0, 1],  # Limit to two a indices to reduce output
        avg=False,
        indices=[(0, 29, 29), (0, 9, 29)],
        superimpose_indices=True,
        figsize=(8, 4),
        title='Model (b) vs True – Superimposed Points'
    )
