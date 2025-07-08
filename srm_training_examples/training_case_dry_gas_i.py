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
import sys
import logging

# Add parent directory to path
project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_directory not in sys.path:
    sys.path.insert(0, project_directory)

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

from data_processing import SRMDataProcessor
from training import BatchGenerator, build_optimizer_from_config, validate_loss_keys
from default_configurations import get_optimizer_config, get_configuration, get_optimizer_model_mapping, WORKING_DIRECTORY, DEFAULT_GENERAL_CONFIG, DEFAULT_RESERVOIR_CONFIG
from physics_loss_Subclassed import PhysicsLoss
from well_rate_bhp_Subclassed import WellRatesPressure

from complete_pvt_module import PVTModuleWithHardLayer
from complete_trainable_module import CompleteTrainableModule
from training import train_combined_models_unified
from plot_functions_tf import ModelPlotter

# Add the working directory to path to import model classes
sys.path.append(WORKING_DIRECTORY)

# Configure logging to show INFO level by default
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import EncoderDecoderSubclassed module to disable debug shape printing
import EncoderDecoderSubclassed
EncoderDecoderSubclassed.DEBUG_SHAPES = False

# Build the different models
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
    
    # Optionally reset configurations 
    encoder_decoder_config = get_configuration('encoder_decoder')
    encoder_decoder_config['spatial_dims'] = 2  # Explicitly set to 2D to use temporal dimension processing
    encoder_decoder_config['temporal'] = True   # Enable temporal dimension processing
    encoder_decoder_config['residual_params']['Extra_Conv_Layers']['Count'] = 2
    encoder_decoder_config['residual_params']['Extra_Dec_Conv_Layers']['Count'] = 2
    encoder_decoder_config['residual_params']['Latent_Layer']['Depth'] = 1
    encoder_decoder_config['residual_params']['Latent_Layer']['Activation'] = None
    encoder_decoder_config['residual_params']['Out_Activation_Func'] = None
    encoder_decoder_config['residual_params']['Skip_Connections']={'Add': False, 'Layers': [1, 1, 1, 1]}
    
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

    residual_network_config['output_activation'] = lambda x: scaled_tanh_with_xtanhx(x, max_val = max_lim)
    
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
    
    # Create Keras model using CompleteTrainableModule's call method
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
    pvt_layer_config = get_configuration('pvt_layer', fluid_type=DEFAULT_GENERAL_CONFIG['fluid_type'], fitting_method='spline')
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
