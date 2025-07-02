"""
Complete Trainable Module integrating neural networks with optional Hard Layer.

This module provides a framework for combining different neural network architectures
(EncoderDecoderModel or ResidualNetworkLayer) with an optional HardLayer that 
enforces constraints and applies mathematical transformations.
"""

import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.layers import Layer

# Import the required components
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
from EncoderDecoderSubclassed import EncoderDecoderModel
from Hard_Layer_Subclassed import HardLayer
from residual_neural_network_subclassed import ResidualNetworkLayer
from default_configurations import get_configuration

class CompleteTrainableModule(tf.keras.layers.Layer):
    """
    Complete trainable module that integrates neural networks with an optional HardLayer.
    
    This module can combine either:
    1. An encoder-decoder network, or
    2. A residual neural network
    with an optional hard layer. The architecture can be configured based on user preference.
    
    When hard_enforcement_only=True, only the hard enforcement layer will be used
    without any deep learning layers (encoder-decoder or residual network).
    """
    def __init__(self, 
                 network_type='encoder_decoder',  # 'encoder_decoder' or 'residual'
                 encoder_decoder_config=None,
                 residual_network_config=None,
                 use_hard_layer=True,
                 hard_layer_config=None,
                 input_slice_config=None,
                 hard_enforcement_only=False,  # New param: When True, only use hard layer with no DL networks
                 **kwargs):
        """
        Initialize the complete trainable module.
        
        Args:
            network_type: Type of neural network to use ('encoder_decoder' or 'residual')
            encoder_decoder_config: Configuration for the EncoderDecoderModel (if network_type is 'encoder_decoder')
            residual_network_config: Configuration for the ResidualNetworkLayer (if network_type is 'residual')
            use_hard_layer: Whether to use the hard layer or bypass it
            hard_layer_config: Configuration for the HardLayer (if use_hard_layer is True)
            input_slice_config: Configuration for how to slice the input data
            **kwargs: Additional keyword arguments for the Layer
        """
        super(CompleteTrainableModule, self).__init__(**kwargs)
        
        # Store configuration
        self.network_type = network_type.lower()
        self.encoder_decoder_config = encoder_decoder_config or {}
        self.residual_network_config = residual_network_config or {}
        self.hard_layer_config = hard_layer_config or {}
        self.use_hard_layer = use_hard_layer
        self.hard_enforcement_only = hard_enforcement_only
        
        # Configure how inputs should be sliced for each component
        self.input_slice_config = input_slice_config or get_configuration('input_slice')
        
        # Validate network type
        if self.network_type not in ['encoder_decoder', 'residual']:
            raise ValueError(f"Invalid network_type: {network_type}. Use 'encoder_decoder' or 'residual'.")
        
    def build(self, input_shape):
        """
        Build the layer components based on input shape.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # Initialize the neural network based on type (skip if hard_enforcement_only is True)
        if not self.hard_enforcement_only:
            if self.network_type == 'encoder_decoder':
                # Create encoder-decoder with default parameters
                default_ed_config = get_configuration('encoder_decoder', input_shape)
                
                # Update with user-provided configuration
                if self.encoder_decoder_config:
                    default_ed_config.update(self.encoder_decoder_config)
                
                # Initialize the encoder-decoder
                self.main_network = EncoderDecoderModel(**default_ed_config)
                # Store the network type for parameter tracking
                self.main_network_type = 'encoder_decoder'
                
            else:  # residual network
                # Create residual network with default parameters
                default_res_config = get_configuration('residual')
                
                # Update with user-provided configuration
                if self.residual_network_config:
                    default_res_config.update(self.residual_network_config)
                
                # Initialize the residual network
                self.main_network = ResidualNetworkLayer(**default_res_config)
                # Store the network type for parameter tracking
                self.main_network_type = 'residual'
        else:
            # For hard_enforcement_only mode, no neural network is created
            self.main_network = None
            self.main_network_type = 'none'
        
        # Initialize and build the hard layer if enabled or if using hard_enforcement_only mode
        if self.use_hard_layer or self.hard_enforcement_only:
            # Create default hard layer configuration
            default_hard_config = get_configuration('hard_layer', use_rbf=False)
            
            # Update with user-provided configuration
            if self.hard_layer_config:
                default_hard_config.update(self.hard_layer_config)
            
            # Initialize the hard layer
            self.hard_layer = HardLayer(**default_hard_config)
            
            # Build hard layer with input shape
            hard_layer_input_shape = [
                input_shape,  # Shape of input variables
                input_shape   # Shape for features (will be replaced in call)
            ]
            
            # Build hard layer
            self.hard_layer.build(hard_layer_input_shape)
            
        # Create trainable weights groups for separate optimization
        self.main_network_trainable = [] if not self.hard_enforcement_only else None
        self.hard_layer_trainable = []
        
        # Call the parent's build method
        super(CompleteTrainableModule, self).build(input_shape)
    
    def call(self, inputs, training=False):
        """
        Forward pass through the module.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Module output (either main network output or hard layer output)
        """
        # If using hard_enforcement_only mode, skip the main network processing
        if self.hard_enforcement_only:
            network_output = tf.reduce_mean(inputs[...,-2:], axis=-1, keepdims=True)  # Pass a reduced inner dimension of inputs directly to hard layer
        else:
            # Process input through the main network
            network_output = self.main_network(inputs, training=training)
            
            # If hard layer is not enabled, return main network output
            if not self.use_hard_layer:
                return network_output
            
        # Otherwise, process through hard layer
        # Extract variables for hard layer
        if isinstance(inputs, list):
            # If inputs is already a list, extract the components
            hard_layer_inputs = inputs
        else:
            # Extract slices from the input using the new slice format
            slice_config = self.input_slice_config['hard_layer']
            # Use the time and property slices directly
            time = inputs[..., slice_config['time']]
            property_val = inputs[..., slice_config['property']]
            hard_layer_inputs = [time, property_val]
        
        # Process through hard layer
        output = self.hard_layer([hard_layer_inputs, network_output], training=training)
        
        return output
    
    def get_config(self):
        """
        Get the configuration of the layer.
        
        Returns:
            Dictionary of layer configuration
        """
        config = super(CompleteTrainableModule, self).get_config()
        config.update({
            'network_type': self.network_type,
            'encoder_decoder_config': self.encoder_decoder_config,
            'residual_network_config': self.residual_network_config,
            'use_hard_layer': self.use_hard_layer,
            'hard_layer_config': self.hard_layer_config,
            'input_slice_config': self.input_slice_config,
            'hard_enforcement_only': self.hard_enforcement_only
        })
        return config


def test_combined_model(
    input_shape=(None, 39, 39, 3),
    network_type='encoder_decoder',  # 'encoder_decoder' or 'residual'
    use_hard_layer=True,
    encoder_decoder_config=None,
    residual_network_config=None,
    hard_layer_config=None,
    use_rbf=False
):
    """
    Test the combined model with different network architectures and optional hard layer.
    
    Args:
        input_shape: Input shape for testing
        network_type: Type of network to use ('encoder_decoder' or 'residual')
        use_hard_layer: Whether to use the hard layer
        encoder_decoder_config: Configuration for the EncoderDecoderModel (if network_type is 'encoder_decoder')
        residual_network_config: Configuration for the ResidualNetworkLayer (if network_type is 'residual')
        hard_layer_config: Configuration for the hard layer
        use_rbf: Whether to use RBF in the hard layer
        
    Returns:
        Test model with the complete trainable module
    """
    import numpy as np
    
    # Display test configuration
    print(f"=== Testing Complete Module with {network_type.title()} ===")
    if use_hard_layer:
        print(f"    (Hard Layer {'enabled with RBF' if use_rbf else 'enabled'})")
    else:
        print(f"    (Hard Layer disabled)")
    
    # Determine if input is 2D or 3D based on shape
    # For 3D input: (batch, depth, height, width, channels)
    is_3d_input = len(input_shape) > 4
    print(f"Input type: {'3D' if is_3d_input else '2D'}")
    print(f"Input shape: {input_shape}")
    
    # Get configurations from default settings
    if network_type == 'encoder_decoder':
        if encoder_decoder_config is None:
            if is_3d_input:
                # Get 3D encoder-decoder config
                from default_configurations import DEFAULT_ENCODER_DECODER_3D_CONFIG
                encoder_decoder_config = DEFAULT_ENCODER_DECODER_3D_CONFIG.copy()
            else:
                # Get 2D encoder-decoder config
                encoder_decoder_config = get_configuration('encoder_decoder', input_shape).copy()
        else:
            # If config is provided, ensure it's a copy to avoid modifying the original
            encoder_decoder_config = encoder_decoder_config.copy()
            
        # Explicitly set spatial_dims based on input shape to ensure consistency
        if is_3d_input:
            encoder_decoder_config['spatial_dims'] = 3
        else:
            encoder_decoder_config['spatial_dims'] = 2
            
        print(f"Encoder-Decoder spatial dimensions: {encoder_decoder_config['spatial_dims']}")
        
    if network_type == 'residual' and residual_network_config is None:
        residual_network_config = get_configuration('residual')
        
    if hard_layer_config is None:
        hard_layer_config = get_configuration('hard_layer', use_rbf=use_rbf)
    
    # Get input slice configuration from defaults
    input_slice_config = get_configuration('input_slice')
    
    # Create the complete module as a layer
    complete_module = CompleteTrainableModule(
        network_type=network_type,
        encoder_decoder_config=encoder_decoder_config,
        residual_network_config=residual_network_config,
        use_hard_layer=use_hard_layer,
        hard_layer_config=hard_layer_config,
        input_slice_config=input_slice_config
    )
    
    # Create a Keras model with the module
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    outputs = complete_module(inputs)
    test_model = tf.keras.Model(inputs, outputs)
    
    # Print model summary
    test_model.summary()
    
    # Test with random data
    batch_size = 2
    test_input = tf.random.normal((batch_size,) + input_shape[1:])
    
    # Run inference
    output = test_model(test_input)
    
    # Print shapes
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print number of trainable variables
    network_params = np.sum([np.prod(v.get_shape()) for v in complete_module.main_network.trainable_weights])
    print(f"{network_type.title()} trainable parameters: {network_params}")
    
    if use_hard_layer:
        hard_layer_params = np.sum([np.prod(v.get_shape()) for v in complete_module.hard_layer.trainable_weights])
        print(f"Hard Layer trainable parameters: {hard_layer_params}")
        print(f"Total trainable parameters: {network_params + hard_layer_params}")
    else:
        print(f"Total trainable parameters: {network_params}")
    
    return test_model

if __name__ == "__main__":
    # Test with Encoder-Decoder and Hard Layer
    print("\n=== Testing with Encoder-Decoder and Hard Layer ===")
    model_ed_with_hard = test_combined_model(
        input_shape=(None, 39, 39, 3), 
        network_type='encoder_decoder',
        use_hard_layer=True, 
        use_rbf=False
    )
    
    # Test with Encoder-Decoder Only
    print("\n=== Testing with Encoder-Decoder Only ===")
    model_ed_only = test_combined_model(
        input_shape=(None, 39, 39, 3),
        network_type='encoder_decoder',
        use_hard_layer=False
    )
    
    # Test with Residual Network and Hard Layer
    print("\n=== Testing with Residual Network and Hard Layer ===")
    model_res_with_hard = test_combined_model(
        input_shape=(None, 39, 39, 3),
        network_type='residual',
        use_hard_layer=True,
        use_rbf=False
    )
    
    # Test with Residual Network Only
    print("\n=== Testing with Residual Network Only ===")
    model_res_only = test_combined_model(
        input_shape=(None, 39, 39, 3),
        network_type='residual',
        use_hard_layer=False
    )
    
    # Test with 3D Input using Encoder-Decoder (conditional based on configuration availability)
    try:
        print("\n=== Testing with 3D Input using Encoder-Decoder ===")
        # Import the default 3D configuration
        try:
            from default_configurations import DEFAULT_ENCODER_DECODER_3D_CONFIG
            has_3d_config = True
        except (ImportError, AttributeError):
            print("Warning: DEFAULT_ENCODER_DECODER_3D_CONFIG not found in default_configurations.")
            print("Creating a basic 3D configuration for testing.")
            has_3d_config = False
        
        if has_3d_config:
            # Use the predefined 3D configuration
            model_3d = test_combined_model(
                input_shape=(None, 16, 16, 16, 3), 
                network_type='encoder_decoder',
                use_hard_layer=True,
                encoder_decoder_config=DEFAULT_ENCODER_DECODER_3D_CONFIG,
                use_rbf=False
            )
        else:
            # Create a basic 3D encoder-decoder configuration
            basic_3d_config = {
                'is_3d': True,
                'filters': [32, 64, 128, 256],
                'kernel_size': 3,
                'pool_size': 2,
                'dropout_rate': 0.2,
                'conv_activation': 'relu',
                'use_batch_norm': True
            }
            
            model_3d = test_combined_model(
                input_shape=(None, 16, 16, 16, 3), 
                network_type='encoder_decoder',
                use_hard_layer=True,
                encoder_decoder_config=basic_3d_config,
                use_rbf=False
            )
    except Exception as e:
        print(f"Error in 3D test case: {str(e)}")
        print("Skipping 3D testing due to errors.")
