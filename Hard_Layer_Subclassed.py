"""
Hard_Layer implementation using TensorFlow layer subclassing.

This module provides a layer that enforces initial conditions and performs
mathematical transformations on the input data.
"""

import tensorflow as tf
import numpy as np
import os
import sys

# Import custom modules and configurations
from default_configurations import get_configuration, WORKING_DIRECTORY, DEFAULT_GENERAL_CONFIG, DEFAULT_RESERVOIR_CONFIG
try:
    current_dir = os.path.dirname(WORKING_DIRECTORY)
    if current_dir not in sys.path:
        sys.path.append(current_dir)
except ImportError:
    raise ImportError("Critical error: default_configurations module not found. Cannot continue execution.")

class HardLayer(tf.keras.layers.Layer):
    """
    A layer that enforces the initial condition of the pressure solution.
    
    This layer takes slices of input dimensions and performs mathematical transformations
    using trainable parameters. It can be combined with the encoder-decoder model to form
    a single model with different optimizers.
    """
    def __init__(self,
                 norm_limits=[-1, 1],               # Normalization limits for inputs
                 init_value=None,                    # Initial value for hard enforcement
                 dim_slices=None,                    # Dictionary of slices for dimensions
                 nonormalize_func=None,              # Function to unnormalize data
                 kernel_exponent_config={           # Configuration for trainable exponent
                     'initial_value': 0.5,
                     'trainable': True,
                     'min_value': 0.01,
                     'max_value': 0.99
                 },
                 use_rbf=False,                      # Whether to use RBF
                 rbf_config=None,                    # RBF configuration
                 kernel_activation=None,             # Activation for kernel computation
                 input_activation=None,              # Activation for input
                 rectifier=None,                     # Rectifier for gas condensate fluid above dew point
                 regularization=0.0,                 # Regularization parameter
                 pvt_config=None,                    # PVT configuration including fluid_type and parameters
                 name='hard_layer',
                 **kwargs):
        """
        Initialize the HardLayer.
        
        Args:
            norm_limits: List of normalization limits for input dimensions [[min1, max1], [min2, max2], ...]
            init_value: Initial value for hard enforcement (e.g., initial pressure)
            dim_slices: Dictionary of slices for dimensions {'time': slice(a,b), 'property': slice(c,d)}
            nonormalize_func: Function to unnormalize data (defaults to identity function)
            kernel_exponent_config: Configuration for trainable exponent
            use_rbf: Whether to use radial basis function
            rbf_config: Configuration for RBF (if use_rbf is True)
            kernel_activation: Activation function(s) for kernel calculations
            input_activation: Activation function for input
            regularization: Regularization parameter
            pvt_config: PVT configuration including fluid_type and parameters
            name: Name for the layer
        """
        super(HardLayer, self).__init__(name=name, **kwargs)
        
        # Store configuration
        self.norm_limits = norm_limits
        self.init_value = init_value
        self.regularization = regularization
        
        # Set up dimension slices with defaults
        self.dim_slices = dim_slices or {
            'time': slice(-2, -1),     # Default: second-to-last channel for time
            'property': slice(-1, None)  # Default: last channel for property
        }
        
        # For backward compatibility
        self.inner_dim_slice = None
        
        self.nonormalize_func = nonormalize_func or (lambda x, *args, **kwargs: x)  # Default to identity
        
        # Kernel exponent configuration
        self.kernel_exponent_config = kernel_exponent_config
        
        # RBF configuration
        self.use_rbf = use_rbf
        self.rbf_config = rbf_config or {
            'output_dim': 25,
            'activation': 'sigmoid'
        }
        
        # Activation functions
        if isinstance(kernel_activation, list):
            self.kernel_activation = kernel_activation
        else:
            self.kernel_activation = [kernel_activation]
        
        self.input_activation = input_activation
        
        # Initialize kernel parameters for radial basis function
        self._kernel_initializer = tf.keras.initializers.get('glorot_normal')

        # Rectifier configuration
        self.rectifier = rectifier

        # Get fluid type from default config
        self.fluid_type = DEFAULT_GENERAL_CONFIG['fluid_type']
        if self.fluid_type == 'GC':
            breakpoint()
            self.pmax = DEFAULT_RESERVOIR_CONFIG['initialization']['Pi']
            self.pmin = DEFAULT_RESERVOIR_CONFIG['initialization']['Pa']           
        
        # PVT configuration
        # If pvt_config is None, get it from default configurations
        if pvt_config is None:
            # Get PVT configuration for this fluid type
            self.pvt_config = get_configuration('pvt_layer', fluid_type=self.fluid_type, fitting_method='spline')
        else:
            self.pvt_config = pvt_config

        # Extract key PVT (dew point) properties
        self.pdew = self.pvt_config['dew_point'] if self.fluid_type == 'GC' else None

    def build(self, input_shape):
        """
        Build the layer with the given input shape.
        
        Args:
            input_shape: Shape of the input tensor(s)
        """

        # Initialize kernel exponent as a trainable variable with constraints
        self.kernel_exponent = self.add_weight(
            shape=(*input_shape[0][1:-1],1),
            initializer=tf.constant_initializer(value=self.kernel_exponent_config.get('initial_value', 0.5)),
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=self.kernel_exponent_config.get('min_value', 0.01),
                max_value=self.kernel_exponent_config.get('max_value', 0.99),
                rate=1.0,
                axis=0
            ),
            trainable=self.kernel_exponent_config.get('trainable', False),
            name='kernel_exponent'
        )
        
        # Set up kernel activations
        if self.kernel_activation[0] in [None, '']:
            self.kernel_activation[0] = lambda x: x
        # if self.kernel_activation[1] in [None, '']:
        #     self.kernel_activation[1] = lambda x: x
        
        # Set up input activation
        if (self.input_activation is None) or (hasattr(self.input_activation, 'lower')):
            self.input_activation = lambda x: x
        
        # Set up rectifier - default to a relu activation function
        if self.fluid_type == 'GC' and self.rectifier is None:
            self.rectifier = tf.nn.relu
        
        # Initialize RBF layers if needed
        if self.use_rbf:
            activation_fn = None
            if self.rbf_config.get('activation') == 'sigmoid':
                activation_fn = tf.nn.sigmoid
            elif self.rbf_config.get('activation') == 'relu':
                activation_fn = tf.nn.relu
            elif self.rbf_config.get('activation') == 'tanh':
                activation_fn = tf.nn.tanh
            
            # Replace TFA's SpectralNormalization with a custom solution
            # We'll use Keras's kernel constraint for L2 normalization
            self.rbf_dense = tf.keras.layers.Dense(
                1, 
                activation=activation_fn,
                kernel_initializer=self._kernel_initializer,
                kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
                name=f"{self.name}_rbf_dense"
            )
        
        super(HardLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Apply the layer transformation to the inputs.
        
        Args:
            inputs: List of [input_variables, output_from_encoder_decoder]
            
        Returns:
            Transformed output tensor
        """
        # Extract inputs
        # Expecting inputs[0] to be a tensor with shape [..., inner_dim]
        # and inputs[1] to be the output from encoder-decoder
        input_vars = inputs[0]
        p = inputs[1]
        
        # Apply slice to inner dimension
        if isinstance(input_vars, list):
            # Inputs provided as a list of tensors
            tn1 = input_vars[0]  # Time
            tn2 = input_vars[1]  # Permeability or other property
        else:
            # Extract from tensor using explicit dimension slices
            time_slice = self.dim_slices['time']
            property_slice = self.dim_slices['property']
            # Extract time and property using provided slices
            tn1 = input_vars[..., time_slice]       # Time
            tn2 = input_vars[..., property_slice]   # Property

        # Unnormalize time if needed
        t1 = self.nonormalize_func(tn1, stat_idx=3, compute=True)
        
        # Apply time normalization
        alpha_t = ((t1 - self.norm_limits[0]) / (self.norm_limits[1] - self.norm_limits[0]))

        # Add a rectifying function to the time-changing variable for gas condensate fluid above dew point
        if self.fluid_type == 'GC' and self.rectifier is not None:
            alpha_p = self.rectifier((p-self.pdew)/(self.pmin-self.pdew))
        else:
            alpha_p = 1.0

        # Apply exponentiation with trainable parameter
        alpha = alpha_p * alpha_t ** self.kernel_activation[-1](self.kernel_exponent)

        # Apply property modification if using RBF
        if self.use_rbf:
            # Process property through RBF
            property_factor = self.rbf_dense(tn2)
            # Modify alpha based on property factor
            alpha = alpha * property_factor
        
        # Apply input activation to p
        p_activated = self.input_activation(p)
        
        # Compute final output based on alpha and p
        # output = (1 - alpha) * self.init_value - alpha * p_activated
        output = self.init_value - alpha * p_activated          # Stablity achieved
        
        return output
    
    def get_config(self):
        """
        Get configuration for this layer.
        
        Returns:
            Dictionary with layer configuration
        """
        config = super(HardLayer, self).get_config()
        config.update({
            'norm_limits': self.norm_limits,
            'init_value': self.init_value,
            'dim_slices': self.dim_slices,
            'kernel_exponent_config': self.kernel_exponent_config,
            'use_rbf': self.use_rbf,
            'rbf_config': self.rbf_config,
            'regularization': self.regularization,
            'pvt_config': self.pvt_config,
            'rectifier': self.rectifier,
            # Don't include fluid_type, pmax, pmin, pdew as they're derived from pvt_config or defaults
        })
        return config
