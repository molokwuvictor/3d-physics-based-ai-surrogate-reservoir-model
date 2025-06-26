"""
Encoder-Decoder Network Using Layer Subclassing

This module implements an encoder-decoder network architecture using TensorFlow's
layer subclassing approach. It supports:
- 2D and 3D spatial dimensions
- Temporal dimensions
- Non-power of two input dimensions
- Skip connections
- Customizable to specific network width and depth
"""

import tensorflow as tf
import numpy as np
import logging

# Set default logging level for this module
_LOGGER = logging.getLogger(__name__)

# Debug flag - set to False to disable shape debug messages
DEBUG_SHAPES = False

# Global flag to control skip connection logging
SKIP_CONNECTION_LOGGING = False

# Helper function to conditionally log debug information
def debug_print(message, *args):
    if DEBUG_SHAPES:
        tf.print(message % args)
        _LOGGER.debug(message, *args)

# Helper function to log skip connection information
def log_skip_connection(message, *args):
    if SKIP_CONNECTION_LOGGING:
        tf.print(message % args)
    _LOGGER.debug(message, *args)

def network_width_list(depth, width, ngens, growth_rate=0.5, growth_type='smooth', network_type='plain'):
    """
    Compute filter sizes for network layers.
    
    Args:
        depth: Number of layers
        width: Base width (number of filters) for the network
        ngens: Number of generations of filter growth
        growth_rate: Rate at which filter counts grow between generations
        growth_type: Type of growth pattern ('smooth' or 'step')
        network_type: Network architecture type ('plain' or 'resnet')
        
    Returns:
        List of filter counts for each layer
    """
    def create_even(num):
        return int(np.ceil(num / 2.) * 2)
    
    if ngens == 0:
        ngens = 1
    
    no_per_gen = depth // ngens
    rem_gen = depth % ngens
    new_list = []
    
    for i in range(ngens):
        if network_type == 'plain':
            gen = [growth_rate**i] * (no_per_gen + (rem_gen if i == ngens-1 else 0))
        else:  # resnet-like
            gen = [growth_rate**i] + [0]*(no_per_gen-1 + (rem_gen if i == ngens-1 else 0))
        new_list += gen
    
    new_list = [create_even(width * x) for x in new_list]
    return new_list


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer with support for 2D, 3D, and temporal dimensions.
    """
    def __init__(self, depth, width, spatial_dims=2, temporal=False, residual_params=None, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.depth = depth
        self.width_config = width
        self.spatial_dims = spatial_dims
        self.temporal = temporal
        self.residual_params = residual_params or {}
        
        # Initialize layer collections
        self.conv_layers = []
        self.padding_layers = []
        self.activation_layers = []
        self.dropout_layers = []
        self.skip_connections = {}
        self.skip_connections_proj = {}
        
        # Extra conv layers after main encoder
        self.extra_conv_layers = []
        self.extra_act_layers = []
        
        # Number of extra conv layers (default: 0 = deactivated)
        self.num_extra_layers = self.residual_params.get('Extra_Conv_Layers', {}).get('Count', 0)
        
        # Track spatial dimensions for proper upsampling
        self.spatial_dims_at_layers = []

    def build(self, input_shape):
        # Store the input shape for reference
        self.input_spatial_shape = input_shape[1:-1] if not self.temporal else input_shape[2:-1]
        
        # Determine appropriate layer types based on spatial dimensions
        spatial_conv = tf.keras.layers.Conv2D if self.spatial_dims == 2 else tf.keras.layers.Conv3D
        spatial_pad = tf.keras.layers.ZeroPadding2D if self.spatial_dims == 2 else tf.keras.layers.ZeroPadding3D
        
        # Wrap with TimeDistributed if temporal
        if self.temporal:
            base_conv = spatial_conv
            base_pad = spatial_pad
            spatial_conv = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(base_conv(*args, **kwargs))
            spatial_pad = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(base_pad(*args, **kwargs))

        # Generate filter sizes for each layer
        self.filter_list = network_width_list(
            depth=self.depth,
            width=self.width_config['Bottom_Size'],
            ngens=self.depth,
            growth_rate=self.width_config['Growth_Rate'],
            growth_type='smooth',
            network_type='plain'
        )

        # Create layers
        current_shape = list(self.input_spatial_shape)
        self.spatial_dims_at_layers.append(tuple(current_shape))

        for i in range(self.depth):
            kernel_size = self.residual_params['Kernel_Size']
            
            if i == 0:
                # First layer
                conv = spatial_conv(
                    filters=self.filter_list[i],
                    kernel_size=kernel_size,
                    strides=1,
                    padding='valid',  # Changed to 'valid'
                    kernel_initializer=self.residual_params['Kernel_Init'],
                    name=f'{self.name}_CNV_ENC_{i+1}'
                )
                pad = None
            else:
                # Subsequent layers with padding and downsampling
                pad_size = (1,1) if self.spatial_dims == 2 else (1,1,1)
                pad = spatial_pad(padding=pad_size, name=f'{self.name}_PAD_ENC_{i+1}')
                
                # Adjust kernel size for middle layers
                kernel_size = kernel_size + 2 if i < self.depth-1 else kernel_size
                
                # Calculate new dimensions after convolution
                for j in range(len(current_shape)):
                    # Add padding
                    current_shape[j] += 2  # From padding of (1,1)
                    # Apply strided convolution
                    current_shape[j] = (current_shape[j] - kernel_size) // 2 + 1
                
                conv = spatial_conv(
                    filters=self.filter_list[i],
                    kernel_size=kernel_size,
                    strides=2,
                    padding='valid',
                    kernel_initializer=self.residual_params['Kernel_Init'],
                    kernel_regularizer=self.residual_params.get('Kernel_Regu', {}).get('Hidden_Layer', None),
                    name=f'{self.name}_CNV_ENC_{i+1}'
                )
                
                self.spatial_dims_at_layers.append(tuple(current_shape))
            
            # Activation layer
            activation = tf.keras.layers.Activation(
                self.residual_params['Activation_Func'],
                name=f'{self.name}_ACT_ENC_{i+1}'
            )
            
            # Optional dropout
            dropout = None
            if self.residual_params.get('Dropout', {}).get('Add', False) in [True, 'encoder']:
                dropout = tf.keras.layers.Dropout(
                    self.residual_params['Dropout']['Rate']
                )

            # Store layers
            self.padding_layers.append(pad)
            self.conv_layers.append(conv)
            self.activation_layers.append(activation)
            self.dropout_layers.append(dropout)

            # More robust skip connection handling to prevent index errors
            use_skip = False
            skip_connect = self.residual_params.get('Skip_Connections', {})
            if skip_connect.get('Add', False):
                # Get skip layers list, ensure it's a flat list
                skip_layers = skip_connect.get('Layers', [])
                # Convert to list if it's nested
                if skip_layers and isinstance(skip_layers[0], list):
                    skip_layers = skip_layers[0]
                
                # Check if this layer should have a skip connection
                if i < len(skip_layers) and skip_layers[i] not in [None, 0]:
                    use_skip = True
            
            # Store skip connection if enabled
            if use_skip:
                skip_key = f'skip_{i+1}'
                self.skip_connections_proj[skip_key] = tf.keras.layers.Dense(
                    units=self.filter_list[i],
                    activation=None,
                    kernel_initializer=self.residual_params['Kernel_Init'],
                    name=f'{self.name}_SKIP_PROJ_DENSE_{i+1}'
                )

        # Add extra convolution layers with stride=1, padding=same, and initial kernel size
        # Only if num_extra_layers > 0
        if self.num_extra_layers > 0:
            for i in range(self.num_extra_layers):
                extra_conv = spatial_conv(
                    filters=self.filter_list[-1],  # Use same filters as last encoder layer
                    kernel_size=self.residual_params['Kernel_Size'],  # Use initial kernel size
                    strides=1,
                    padding='same',  # Use 'same' padding to maintain dimensions
                    kernel_initializer=self.residual_params['Kernel_Init'],
                    kernel_regularizer=self.residual_params.get('Kernel_Regu', {}).get('Hidden_Layer', None),
                    name=f'{self.name}_EXTRA_CNV_{i+1}'
                )
                
                extra_act = tf.keras.layers.Activation(
                    self.residual_params['Activation_Func'],
                    name=f'{self.name}_EXTRA_ACT_{i+1}'
                )
                
                self.extra_conv_layers.append(extra_conv)
                self.extra_act_layers.append(extra_act)
        
        super(EncoderLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        
        for i in range(self.depth):
            # Apply padding if present
            if self.padding_layers[i] is not None:
                x = self.padding_layers[i](x)
            
            # Apply convolution
            x = self.conv_layers[i](x)
            
            # More robust skip connection handling to prevent index errors
            use_skip = False
            skip_connect = self.residual_params.get('Skip_Connections', {})
            if skip_connect.get('Add', False):
                # Get skip layers list, ensure it's a flat list
                skip_layers = skip_connect.get('Layers', [])
                # Convert to list if it's nested
                if skip_layers and isinstance(skip_layers[0], list):
                    skip_layers = skip_layers[0]
                
                # Check if this layer should have a skip connection
                if i < len(skip_layers) and skip_layers[i] not in [None, 0]:
                    use_skip = True
            
            # Store skip connection if enabled
            if use_skip:
                self.skip_connections[i+1] = x
            
            # Apply activation
            x = self.activation_layers[i](x)
            
            # Apply dropout if enabled for this layer
            dropout_layer = self.dropout_layers[i]
            if dropout_layer and self.residual_params.get('Dropout', {}).get('Layer', [])[i] == 1:
                x = dropout_layer(x, training=training)

        # Apply n extra convolution layers if configured
        if self.num_extra_layers > 0:
            for i, (extra_conv, extra_act) in enumerate(zip(self.extra_conv_layers, self.extra_act_layers)):
                x = extra_conv(x)
                x = extra_act(x)
            debug_print(f"Extra Conv Layers = {i+1} - Shape: %s", tf.shape(x))

        return x


class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder layer with support for 2D, 3D, and temporal dimensions.
    """
    def __init__(self, depth, width, spatial_dims=2, temporal=False, residual_params=None, encoder_skip_proj=None, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.depth = depth
        self.width_config = width
        self.spatial_dims = spatial_dims
        self.temporal = temporal
        self.residual_params = residual_params or {}
        self.encoder_skip_proj = encoder_skip_proj or {}
        
        # Initialize layer collections
        self.deconv_layers = []
        self.activation_layers = []
        self.dropout_layers = []
        self.skip_projection_layers = {}
        self.skip_connection_add = {}
        self.skip_connection_time_distributed = {}
        
        # Extra conv layers after resizing
        self.extra_conv_layers = []
        self.extra_act_layers = []
        
        # Number of extra conv layers (default: 0 = deactivated)
        self.num_extra_layers = self.residual_params.get('Extra_Dec_Conv_Layers', {}).get('Count', 0)

    def build(self, input_shape):
        # Get spatial dimensions from encoder to ensure matching dimensions
        if hasattr(self, 'encoder_dims'):
            encoder_input_shape = self.encoder_dims[0]
            encoder_layer_dims = self.encoder_dims[1]
        else:
            # Default if not provided
            encoder_input_shape = None
            encoder_layer_dims = None
        
        # Determine appropriate layer types based on spatial dimensions
        spatial_conv_t = tf.keras.layers.Conv2DTranspose if self.spatial_dims == 2 else tf.keras.layers.Conv3DTranspose
        spatial_conv = tf.keras.layers.Conv2D if self.spatial_dims == 2 else tf.keras.layers.Conv3D
        spatial_dense = tf.keras.layers.Dense
        
        # Wrap with TimeDistributed if temporal
        if self.temporal:
            base_conv_t = spatial_conv_t
            base_conv = spatial_conv
            spatial_conv_t = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(base_conv_t(*args, **kwargs))
            spatial_conv = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(base_conv(*args, **kwargs))

        # Generate filter sizes for each layer
        self.filter_list = network_width_list(
            depth=self.depth,
            width=self.width_config['Bottom_Size'],
            ngens=self.depth,
            growth_rate=self.width_config['Growth_Rate'],
            growth_type='smooth',
            network_type='plain'
        )

        # Create layers
        for i in range(self.depth):
            # Use reversed filter list for decoder
            filters = int(self.filter_list[self.depth - i - 1] * self.residual_params.get('Decoder_Filter_Fac', 1))
            
            kernel_size = self.residual_params['Kernel_Size']
            if i < self.depth - 1:
                kernel_size = kernel_size  # Retain same kernel size 
            
            # Set output padding if needed to ensure dimensions match
            output_padding = None
            
            # Skip convolution entirely for i==0 as requested
            if i == 0:
                # For first layer, don't create a convolution layer
                deconv = None
            else:
                # Transposed convolution for upsampling (layers 1 to depth-1)
                # Use 'valid' padding for transposed convolutions
                deconv = spatial_conv_t(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=2,
                    padding='valid',  # Changed to 'valid' as requested
                    kernel_initializer=self.residual_params['Kernel_Init'],
                    kernel_regularizer=self.residual_params.get('Kernel_Regu', {}).get('Hidden_Layer', None),
                    name=f'{self.name}_CNVT_DEC_{self.depth-i}',
                    output_padding=output_padding
                )
            
            # Activation layer
            activation = tf.keras.layers.Activation(
                self.residual_params['Activation_Func'],
                name=f'{self.name}_ACT_DEC_{self.depth-i}'
            )
            
            # Optional dropout
            dropout = None
            if self.residual_params.get('Dropout', {}).get('Add', False) in [True, 'decoder']:
                dropout = tf.keras.layers.Dropout(
                    self.residual_params['Dropout']['Rate']
                )
            
            self.deconv_layers.append(deconv)
            self.activation_layers.append(activation)
            self.dropout_layers.append(dropout)
        
        # Starting deconvolution dense layer
        self.dense_starting_layer = tf.keras.layers.Dense(
            units=self.filter_list[self.depth - 1],
            activation=self.residual_params['Activation_Func'],
            kernel_initializer=self.residual_params['Kernel_Init'],
            name=f'{self.name}_DENSE_STARTING'
        )

        if self.temporal:
            self.dense_starting_layer = tf.keras.layers.TimeDistributed(self.dense_starting_layer)
        
        # Addition layers for the skip connection
        self.skip_connection_add = {}
        if self.encoder_skip_proj:
            for skip_key in self.encoder_skip_proj:
                add_layer = tf.keras.layers.Add(name=f'{self.name}_SKIP_ADD_{skip_key}')
                self.skip_connection_add[skip_key] = add_layer

        # Add TimeDistributed layers for skip connection projections if temporal
        if self.temporal and self.encoder_skip_proj:
            for skip_key in self.encoder_skip_proj:
                time_distributed = tf.keras.layers.TimeDistributed(
                    self.encoder_skip_proj[skip_key],
                    name=f'{self.name}_SKIP_TIME_DIST_{skip_key}'
                )
                self.encoder_skip_proj[skip_key] = time_distributed
        
        # Add extra convolution layers with stride=1, padding='same', and initial kernel size
        # These will be applied after resizing and before the dense layer
        if self.num_extra_layers > 0:
            for i in range(self.num_extra_layers):
                extra_conv = spatial_conv(
                    filters=self.filter_list[0],  # Use filters from first decoder layer
                    kernel_size=self.residual_params['Kernel_Size'],  # Use initial kernel size
                    strides=1,
                    padding='same',  # Use 'same' padding to maintain dimensions
                    kernel_initializer=self.residual_params['Kernel_Init'],
                    kernel_regularizer=self.residual_params.get('Kernel_Regu', {}).get('Hidden_Layer', None),
                    name=f'{self.name}_EXTRA_CNV_{i+1}'
                )
                
                extra_act = tf.keras.layers.Activation(
                    self.residual_params['Activation_Func'],
                    name=f'{self.name}_EXTRA_ACT_{i+1}'
                )
                
                self.extra_conv_layers.append(extra_conv)
                self.extra_act_layers.append(extra_act)
        
        # Dense Layer prior to output
        self.dense_layer = spatial_dense(
            int(self.filter_list[0] * self.residual_params.get('Decoder_Filter_Fac', 1)), activation = self.residual_params['Activation_Func'],
            kernel_initializer=self.residual_params['Kernel_Init'],
            kernel_regularizer=self.residual_params.get('Kernel_Regu', {}).get('Output_Layer', None),
            name=f'{self.name}_FINAL_DENSE'
            )
        
        # Final output convolution to match input channels
        input_channels = self._input_channels if hasattr(self, '_input_channels') else 3
        self.final_conv = spatial_conv(
            filters=input_channels,  
            kernel_size=1,  
            padding='same',
            kernel_initializer=self.residual_params['Kernel_Init'],
            kernel_regularizer=self.residual_params.get('Kernel_Regu', {}).get('Output_Layer', None),
            name=f'{self.name}_FINAL_CONV'
        )
        
        self.final_activation = tf.keras.layers.Activation(
            self.residual_params['Out_Activation_Func'],
            name=f'{self.name}_FINAL_ACT'
        )
        
        super(DecoderLayer, self).build(input_shape)

    def pad_skip_connection(self, skip, target, name_prefix="skip_pad"):
        """Properly pad a skip connection to match target dimensions."""
        # For temporal data, we need to handle the time dimension separately
        if self.temporal:
            # Get shapes excluding batch dimension and channels
            skip_shape = skip.shape[1:-1]  
            target_shape = target.shape[1:-1]
            
            # Only pad spatial dimensions (not time)
            time_dim = skip_shape[0]
            skip_spatial = skip_shape[1:]
            target_spatial = target_shape[1:]
            
            # Calculate spatial padding
            spatial_pad = []
            for s, t in zip(skip_spatial, target_spatial):
                diff = t - s
                pad_before = diff // 2
                pad_after = diff - pad_before
                spatial_pad.extend([pad_before, pad_after])
            
            # Apply padding to spatial dimensions only
            if self.spatial_dims == 2:
                pad_layer = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.ZeroPadding2D(padding=((spatial_pad[0], spatial_pad[1]), 
                                                          (spatial_pad[2], spatial_pad[3]))),
                    name=f"{name_prefix}_spatial"
                )
            else:  # 3D
                pad_layer = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.ZeroPadding3D(padding=((spatial_pad[0], spatial_pad[1]), 
                                                          (spatial_pad[2], spatial_pad[3]), 
                                                          (spatial_pad[4], spatial_pad[5]))),
                    name=f"{name_prefix}_spatial"
                )
            
            padded = pad_layer(skip)
            
            # Now handle time dimension if needed
            time_diff = target_shape[0] - time_dim
            if time_diff > 0:
                time_pad_before = time_diff // 2
                time_pad_after = time_diff - time_pad_before
                
                # Create new tensor with time padding
                padded_shape = padded.shape.as_list()
                batch_size = tf.shape(padded)[0]
                output_shape = [batch_size, target_shape[0]] + padded_shape[2:]
                
                # Create zeros tensor for padding
                zeros_shape = [batch_size, time_pad_before] + padded_shape[2:]
                zeros_shape_after = [batch_size, time_pad_after] + padded_shape[2:]
                
                # Only pad if needed
                if time_pad_before > 0:
                    zeros_before = tf.zeros(zeros_shape, dtype=padded.dtype)
                    if time_pad_after > 0:
                        zeros_after = tf.zeros(zeros_shape_after, dtype=padded.dtype)
                        padded = tf.concat([zeros_before, padded, zeros_after], axis=1)
                    else:
                        padded = tf.concat([zeros_before, padded], axis=1)
                elif time_pad_after > 0:
                    zeros_after = tf.zeros(zeros_shape_after, dtype=padded.dtype)
                    padded = tf.concat([padded, zeros_after], axis=1)
            
            return padded
        
        else:  # Non-temporal case - simpler padding
            # Get shapes excluding batch and channel dimensions
            skip_shape = skip.shape[1:-1]
            target_shape = target.shape[1:-1]
            
            # Calculate padding for each dimension
            pads = []
            for s, t in zip(skip_shape, target_shape):
                diff = t - s
                pad_before = diff // 2
                pad_after = diff - pad_before
                pads.extend([pad_before, pad_after])
            
            # Create padding layer based on spatial dimensions
            if self.spatial_dims == 2:
                pad_layer = tf.keras.layers.ZeroPadding2D(
                    padding=((pads[0], pads[1]), (pads[2], pads[3])),
                    name=name_prefix
                )
            else:  # 3D
                pad_layer = tf.keras.layers.ZeroPadding3D(
                    padding=((pads[0], pads[1]), (pads[2], pads[3]), (pads[4], pads[5])),
                    name=name_prefix
                )
            
            return pad_layer(skip)

    def call(self, inputs, skip_connections=None, training=False):
        x = inputs

        for i in range(self.depth):
            # Apply transposed convolution only for i>0 as requested

            if i==0:
                # Add a starting dense unactivated layer if skip connections are to be applied to the innermost index
                use_skip = False
                skip_connect = self.residual_params['Skip_Connections']
                if skip_connect['Add']:
                    layers = skip_connect['Layers']
                    if layers and isinstance(layers[0], list):
                        layers = layers[0]
                    if layers and layers[-1] == 1:
                        use_skip = True
                if use_skip:
                    x = self.dense_starting_layer(x)
            if i > 0:
                x = self.deconv_layers[i](x)

            # Apply skip connection if available 
            if skip_connections and (self.depth - i) in skip_connections:
                skip = skip_connections[self.depth - i]
                
                # Log detailed skip connection information
                log_skip_connection("\n=== Skip Connection Details (Layer %d) ===", i)
                log_skip_connection("Decoder layer input shape: %s", x.shape)
                log_skip_connection("Skip connection original shape: %s", skip.shape)
                
                # Pad skip connection to match spatial dimensions
                skip_name = f'{self.name}_SKIP_PAD_{self.depth-i}'
                skip_before_pad = skip  # Store for logging
                skip_padded = self.pad_skip_connection(skip, x, skip_name)
                
                # Check if padding was applied by comparing shapes
                was_padded = any(s1 != s2 for s1, s2 in zip(skip_before_pad.shape, skip_padded.shape))
                
                log_skip_connection("Skip connection after padding: %s (Padding applied: %s)", skip_padded.shape, was_padded)
                
                # Project channels if needed
                if skip_padded.shape[-1] != x.shape[-1]:
                    log_skip_connection("Channel projection needed: %d → %d", skip_padded.shape[-1], x.shape[-1])
                    
                    # Apply projection using correct key format
                    level_key = self.depth - i
                    skip_key = f'skip_{level_key}'
                    
                    if skip_key in self.encoder_skip_proj:
                        log_skip_connection("Using encoder projection layer for %s", skip_key)
                        skip_padded = self.encoder_skip_proj[skip_key](skip_padded)
                        
                        # Apply TimeDistributed if needed
                        if self.temporal and skip_key in self.skip_connection_time_distributed:
                            skip_padded = self.skip_connection_time_distributed[skip_key](skip_padded)
                    else:
                        log_skip_connection("WARNING: No projection layer found for %s", skip_key)
                    
                    log_skip_connection("Skip connection after projection: %s", skip_padded.shape)
                
                # Add skip connection
                log_skip_connection("Final shapes before addition - Decoder: %s, Skip: %s", x.shape, skip_padded.shape)
                level_key = self.depth - i
                skip_key = f'skip_{level_key}'
                
                if skip_key in self.skip_connection_add:
                    x = self.skip_connection_add[skip_key]([x, skip_padded])
                    
                log_skip_connection("=== End Skip Connection Details ===\n")

            # Apply activation
            x = self.activation_layers[i](x)
            
            # Apply dropout if enabled
            dropout_layer = self.dropout_layers[i]
            if dropout_layer and self.residual_params.get('Dropout', {}).get('Layer', [])[self.depth - i - 1] == 1:
                x = dropout_layer(x, training=training)

        # IMPORTANT: Match dimensions with input shape BEFORE the final convolution
        if hasattr(self, '_target_shape') and self._target_shape is not None:
            target_shape = self._target_shape[1:-1] if not self.temporal else self._target_shape[2:-1]
            current_shape = x.shape[1:-1] if not self.temporal else x.shape[2:-1]
            
            # Check if dimensions need to be adjusted
            if target_shape != current_shape:
                debug_print("Before final conv - Input shape:", target_shape, "Current shape:", current_shape)
                
                # Use resize operation for 2D data
                if self.spatial_dims == 2:
                    resize_layer = tf.keras.layers.Resizing(
                        height=target_shape[0],
                        width=target_shape[1],
                        interpolation='bilinear'
                    )
                    
                    if self.temporal:
                        # For temporal data, apply resizing to each time step
                        batch_size = tf.shape(x)[0]
                        time_steps = x.shape[1]
                        channels = x.shape[-1]
                        
                        # Reshape to combine batch and time
                        x_reshaped = tf.reshape(x, [-1, current_shape[0], current_shape[1], channels])
                        
                        # Apply resize
                        x_resized = resize_layer(x_reshaped)
                        
                        # Reshape back
                        x = tf.reshape(x_resized, [batch_size, time_steps, target_shape[0], target_shape[1], channels])
                    else:
                        x = resize_layer(x)
                
                # Use padding/cropping approach for 3D data
                else:
                    debug_print("Resizing 3D output to match input dimensions...")
                    
                    # Get target and current dimensions
                    depth, height, width = target_shape
                    curr_depth, curr_height, curr_width = current_shape
                    channels = x.shape[-1]
                    
                    # Start with resizing height and width first
                    if curr_height != height or curr_width != width:
                        # Create a custom lambda to apply resizing to inner dimensions
                        def resize_inner_dims(x_inner, h, w):
                            # Reshape to combine batch and depth
                            batch_size = tf.shape(x_inner)[0]
                            depth_size = x_inner.shape[1] 
                            # Reshape to (batch*depth, h, w, channels)
                            reshaped = tf.reshape(x_inner, [-1, curr_height, curr_width, channels])
                            # Apply 2D resizing
                            resized = tf.image.resize(reshaped, [h, w])
                            # Reshape back
                            return tf.reshape(resized, [batch_size, depth_size, h, w, channels])
                        
                        # Apply the resizing
                        x = resize_inner_dims(x, height, width)
                    
                    # Now handle depth dimension - use padding or cropping
                    if curr_depth != depth:
                        if curr_depth > depth:
                            # Need to crop
                            diff = curr_depth - depth
                            start = diff // 2
                            x = x[:, start:start+depth, :, :, :]
                        else:
                            # Need to pad
                            diff = depth - curr_depth
                            pad_before = diff // 2
                            pad_after = diff - pad_before
                            
                            # Create padding configuration
                            paddings = [[0, 0], [pad_before, pad_after], [0, 0], [0, 0], [0, 0]]
                            x = tf.pad(x, paddings)
                
                debug_print("After resize - Shape:", tf.shape(x))
        
        # Apply extra convolution layers if configured
        if self.num_extra_layers > 0:
            for i, (extra_conv, extra_act) in enumerate(zip(self.extra_conv_layers, self.extra_act_layers)):
                x = extra_conv(x)
                x = extra_act(x)
            debug_print(f"Decoder Extra Conv Layers: {i+1} - Shape: %s", tf.shape(x))
                
        # Add a dense layer before final convolution
        x = self.dense_layer(x)
        
        # Final convolution and activation to match input shape
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x


class EncoderDecoderModel(tf.keras.Model):
    """
    Full encoder-decoder model with support for:
    - 2D and 3D spatial dimensions
    - Temporal dimension
    - Latent space processing
    - Skip connections
    
    For 2D: Input shape is (None, H, W, C)
    For 3D: Input shape is (None, D, H, W, C)
    With temporal: (None, T, H, W, C) or (None, T, D, H, W, C)
    """
    def __init__(self, depth, width, spatial_dims=2, temporal=False, 
                 residual_params=None, output_filters=1, **kwargs):
        super(EncoderDecoderModel, self).__init__(**kwargs)
        
        self.depth = depth
        self.width = width
        self.spatial_dims = spatial_dims
        self.temporal = temporal
        self.residual_params = residual_params or {}
        self.output_filters = output_filters  # Number of output filters/channels
        self._input_dims = None  # Store input dimensions for reference
        
        # Initialize encoder
        self.encoder = EncoderLayer(
            depth=depth,
            width=width,
            spatial_dims=spatial_dims,
            temporal=temporal,
            residual_params=residual_params,
            name='Encoder'
        )
        
        # Initialize latent processing layers if needed
        self.flatten_latent = self.residual_params['Latent_Layer']['Flatten']
        self.latent_depth = self.residual_params['Latent_Layer']['Depth']
        
        if self.flatten_latent:
            self.flatten_layer = tf.keras.layers.Flatten(name='Latent_Flatten')
            self.latent_dense = None  # Will be created in build()
            self.reshape_layer = None  # Will be created in build()
        
        # Initialize decoder
        self.decoder = DecoderLayer(
            depth=depth,
            width=width,
            spatial_dims=spatial_dims,
            temporal=temporal,
            residual_params=residual_params,
            # Pass encoder's skip connection projection layers to decoder
            encoder_skip_proj=None,  # Will be set after encoder is built
            name='Decoder'
        )

    def build(self, input_shape):
        # Store input shape for reference
        self._input_dims = input_shape
        
        # Pass input shape info to decoder for proper output shape
        self.decoder._target_shape = input_shape
        self.decoder._input_channels = input_shape[-1]
        
        # Build encoder with a dummy tensor to get encoder output shape
        dummy_input = tf.keras.layers.Input(shape=input_shape[1:])
        encoder_output = self.encoder(dummy_input)
        encoder_output_shape = encoder_output.shape[1:]  # Exclude batch dimension
        
        # After encoder is built, get spatial dimensions at each layer
        if hasattr(self.encoder, 'spatial_dims_at_layers'):
            self.decoder.encoder_dims = (input_shape, self.encoder.spatial_dims_at_layers)
            
        # Pass encoder's skip connection projection layers to decoder
        if hasattr(self.encoder, 'skip_connections_proj'):
            self.decoder.encoder_skip_proj = self.encoder.skip_connections_proj
            print(f"Passed {len(self.encoder.skip_connections_proj)} skip connection projection layers from encoder to decoder")
            
        # Pre-create the output projection layer if needed
        # Need to consider both spatial dimensions and temporal flag
        # For temporal data, we might be dealing with tensor of shape [batch, time, height, width, channels]
        # or [batch, time, depth, height, width, channels]
        
        if self.temporal:
            # For temporal data, use TimeDistributed wrapper around the appropriate spatial conv
            if self.spatial_dims == 2:
                # TimeDistributed wrapper around Conv2D for [batch, time, height, width, channels]
                conv_cls = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(*args, **kwargs),
                    name=kwargs.get('name', None)
                )
                kernel_size = (1, 1)
            else:  # 3D
                # TimeDistributed wrapper around Conv3D for [batch, time, depth, height, width, channels]
                conv_cls = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv3D(*args, **kwargs),
                    name=kwargs.get('name', None)
                )
                kernel_size = (1, 1, 1)
        else:
            # Standard non-temporal case
            if self.spatial_dims == 2:
                conv_cls = tf.keras.layers.Conv2D
                kernel_size = (1, 1)
            else:  # 3D
                conv_cls = tf.keras.layers.Conv3D
                kernel_size = (1, 1, 1)
        
        # Create the output projection layer
        self.output_projection = conv_cls(
            filters=self.output_filters,
            kernel_size=kernel_size,
            padding='same',
            activation=self.residual_params.get('Out_Proj_Activation_Func', None),
            kernel_initializer=self.residual_params.get('Kernel_Init', 'glorot_uniform'),
            name='output_proj'
        )
        
        # Create latent processing layers if needed
        if self.flatten_latent:
            # Calculate total elements in the encoder output shape
            total_elements = np.prod(encoder_output_shape)
            
            # Use the latent width from parameters or default to the total elements
            latent_width = self.residual_params.get('Latent_Layer', {}).get('Width', int(total_elements))
            
            # Create the latent dense layer
            self.latent_dense = tf.keras.layers.Dense(
                units=latent_width,
                activation=self.residual_params['Latent_Layer']['Activation'],
                kernel_initializer=self.residual_params['Kernel_Init'],
                name='Latent_Dense'
            )
            
            # Create the reshape layer that matches the latent width back to spatial dimensions
            # If latent_width doesn't match total_elements exactly, adjust the channel dimension
            if latent_width == total_elements:
                # Simple case: reshape to original dimensions
                self.reshape_layer = tf.keras.layers.Reshape(
                    target_shape=encoder_output_shape,
                    name='Latent_Reshape'
                )
            else:
                # Complex case: adjust the channel dimension to accommodate the latent width
                spatial_elements = np.prod(encoder_output_shape[:-1])  # All except channels
                
                # Check if latent width is sufficient
                if latent_width < spatial_elements:
                    print(f"Warning: Latent width {latent_width} is too small for spatial elements {spatial_elements}")
                    print(f"Increasing latent width to {spatial_elements}")
                    latent_width = int(spatial_elements)
                    
                    # Recreate the latent dense with adjusted width
                    self.latent_dense = tf.keras.layers.Dense(
                        units=latent_width,
                        activation=self.residual_params['Latent_Layer']['Activation'],
                        kernel_initializer=self.residual_params['Kernel_Init'],
                        name='Latent_Dense'
                    )
                
                # Calculate channels based on latent width and spatial elements
                channels = latent_width // spatial_elements
                
                if channels * spatial_elements != latent_width:
                    print(f"Warning: Latent width {latent_width} is not divisible by spatial elements {spatial_elements}")
                    # Adjust channels to ensure divisibility
                    adjusted_width = channels * spatial_elements
                    print(f"Adjusting latent width to {adjusted_width}")
                    
                    # Recreate the latent dense with adjusted width
                    self.latent_dense = tf.keras.layers.Dense(
                        units=adjusted_width,
                        activation=self.residual_params['Latent_Layer']['Activation'],
                        kernel_initializer=self.residual_params['Kernel_Init'],
                        name='Latent_Dense'
                    )
                
                # Create reshape layer with adjusted channel dimension
                new_shape = tuple(encoder_output_shape[:-1]) + (channels,)
                self.reshape_layer = tf.keras.layers.Reshape(
                    target_shape=new_shape,
                    name='Latent_Reshape'
                )
        else:
            # Use the latent width from parameters or default to the total elements
            latent_width = self.residual_params['Latent_Layer']['Width']
            
            # Create the latent dense layer
            if self.latent_depth > 0:
                self.latent_dense = []
                for i in range(self.latent_depth):
                    self.latent_dense.append(tf.keras.layers.Dense(
                        units=latent_width,
                        activation=self.residual_params['Latent_Layer']['Activation'],
                        kernel_initializer=self.residual_params['Kernel_Init'],
                        name='Latent_Dense'
                    ))
        # Now let TensorFlow finish the build process
        super(EncoderDecoderModel, self).build(input_shape)

    def call(self, inputs, training=False):
        # Encoder pass - encoder now stores skip connections internally
        encoded = self.encoder(inputs, training=training)
        
        # Get skip connections from encoder's attribute
        skip_connections = self.encoder.skip_connections if hasattr(self.encoder, 'skip_connections') else {}
        
        # Latent processing if enabled
        if self.flatten_latent:
            # Store original shape for reshaping
            orig_shape = encoded.shape[1:]
            
            # Flatten and apply dense layer
            x = self.flatten_layer(encoded)
            x = self.latent_dense(x)
            
            # Reshape back to spatial dimensions
            encoded = self.reshape_layer(x)
        else:
            if self.latent_depth > 0:
                for i in range(self.latent_depth):
                    encoded = self.latent_dense[i](encoded)

        
        # Decoder pass with input shape info for matching output dimensions
        decoded = self.decoder(encoded, skip_connections, training=training)
        
        # If the final layer doesn't have the correct number of filters, apply the 1x1 conv to adjust
        if decoded.shape[-1] != self.output_filters:
            # Apply the output projection
            decoded = self.output_projection(decoded)
        
        return decoded

# Example usage and test function
def test_encoder_decoder(input_shape=(None, 64, 64, 3), spatial_dims=2, temporal=False, 
                      extra_enc_layers=2, extra_dec_layers=2):
    """
    Test the encoder-decoder model with given parameters.
    
    Args:
        input_shape: Input shape for the model
        spatial_dims: Number of spatial dimensions (2 or 3)
        temporal: Whether to include temporal dimension
        extra_enc_layers: Number of extra conv layers in encoder (0 to deactivate)
        extra_dec_layers: Number of extra conv layers in decoder (0 to deactivate)
    """
    # Default residual parameters
    residual_params = {
        'Kernel_Size': 3,
        'Kernel_Init': 'he_normal',
        'Activation_Func': 'relu',
        'Out_Activation_Func': 'sigmoid',
        'Dropout': {
            'Add': True,
            'Rate': 0.2,
            'Layer': [1, 0, 0, 0]  # For depth=4
        },
        'Skip_Connections': {
            'Add': True,  
            'Layers': [1, 1, 1, 1]  
        },
        'Decoder_Filter_Fac': 1.0,
        'Latent_Layer': {
            'Flatten': False,
            'Depth': 1,
            'Width': 128,
            'Activation': 'relu'
        },
        'Extra_Conv_Layers': {
            'Count': extra_enc_layers  # Configurable number of extra encoder layers
        },
        'Extra_Dec_Conv_Layers': {
            'Count': extra_dec_layers  # Configurable number of extra decoder layers
        }
    }
    
    width = {
        'Bottom_Size': 32,
        'Growth_Rate': 1.5
    }
    
    depth = 4
    
    # Create model
    model = EncoderDecoderModel(
        depth=depth,
        width=width,
        spatial_dims=spatial_dims,
        temporal=temporal,
        residual_params=residual_params
    )
    
    # Print model summary
    model.build(input_shape=(2,) + input_shape[1:])
    print(f"Input spatial shape: {input_shape[1:-1] if not temporal else input_shape[2:-1]}")
    print(f"Output spatial shape: {input_shape[1:-1] if not temporal else input_shape[2:-1]}")
    
    # Create a model wrapper for better visualization
    i = tf.keras.layers.Input(shape=input_shape[1:])
    o = model(i)
    wrapped_model = tf.keras.Model(inputs=i, outputs=o)
    wrapped_model.summary()
    
    # Create dummy input for testing
    if spatial_dims == 2:
        if not temporal:
            # 2D model
            input_data = tf.random.normal((2,) + input_shape[1:])
        else:
            # 2D with temporal
            time_dim = input_shape[1]
            input_data = tf.random.normal((2, time_dim) + input_shape[2:])
    else:
        if not temporal:
            # 3D model
            input_data = tf.random.normal((2,) + input_shape[1:])
        else:
            # 3D with temporal
            time_dim = input_shape[1]
            input_data = tf.random.normal((2, time_dim) + input_shape[2:])
    
    # Run model
    tf.keras.backend.clear_session()
    _ = model(input_data)
    
    # Print input/output shapes
    # print(f"Input shape: {input_data.shape}")
    # print(f"Output shape: {model.call(input_data).shape}")
    
    return model

def test_skip_connection_logging(enable_logging=None):
    """Run a test case with detailed skip connection logging enabled"""
    global SKIP_CONNECTION_LOGGING
    
    # Store original value to restore later
    original_logging_value = SKIP_CONNECTION_LOGGING
    
    # Only change logging state if explicitly requested
    if enable_logging is not None:
        SKIP_CONNECTION_LOGGING = enable_logging
        
    if SKIP_CONNECTION_LOGGING:
        print("\n==== SKIP CONNECTION SHAPE ANALYSIS ====\n")
        print("Running test with skip connection logging enabled...")
    else:
        print("\n==== RUNNING MODEL TEST WITHOUT SKIP CONNECTION LOGGING ====\n")
    
    # Use a simple 2D model for clearer output
    model = test_encoder_decoder(
        input_shape=(None, 64, 64, 3),
        spatial_dims=2,
        temporal=False,
        extra_enc_layers=1,
        extra_dec_layers=1
    )
    
    # Create a simple test input
    input_data = tf.random.normal((1, 64, 64, 3))
    
    # Run model once to log skip connections
    print("\nRunning forward pass to log skip connection details:")
    output = model(input_data)
    print(f"Input shape: {input_data.shape}, Output shape: {output.shape}")
    
    if SKIP_CONNECTION_LOGGING:
        print("\nSkip connection logging complete")
        print("\nTo disable this logging in future tests, set SKIP_CONNECTION_LOGGING = False")
    
    # Restore original logging state
    SKIP_CONNECTION_LOGGING = original_logging_value
    return model

if __name__ == "__main__":
    # Run test without forcing the logging state - will respect global SKIP_CONNECTION_LOGGING setting
    test_skip_connection_logging(enable_logging=None)
    
    # Test 2D model with extra convolution layers
    print("\n=== Testing 2D Model with Extra Conv Layers ===")
    model_2d = test_encoder_decoder(
        input_shape=(None, 39, 39, 3),
        spatial_dims=2,
        temporal=False,
        extra_enc_layers=2,
        extra_dec_layers=2
    )
    
    # Test 2D model without extra convolution layers
    print("\n=== Testing 2D Model without Extra Conv Layers ===")
    model_2d_no_extra = test_encoder_decoder(
        input_shape=(None, 39, 39, 3),
        spatial_dims=2,
        temporal=False,
        extra_enc_layers=0,
        extra_dec_layers=0
    )
    
    # Test 3D model
    print("\n=== Testing 3D Model ===")
    model_3d = test_encoder_decoder(
        input_shape=(None, 16, 39, 39, 3),
        spatial_dims=3,
        temporal=False,
        extra_enc_layers=2,
        extra_dec_layers=2
    )
    
    # Test 2D model with temporal dimension
    print("\n=== Testing 2D Model with Temporal Dimension ===")
    model_temporal = test_encoder_decoder(
        input_shape=(None, 1, 39, 39, 3),
        spatial_dims=2,
        temporal=True,
        extra_enc_layers=2,
        extra_dec_layers=2
    )

# from tensorflow.keras.utils import plot_model
# plot_model(model_temporal, to_file='model_temporal.png', show_shapes=True)