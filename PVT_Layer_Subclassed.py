"""
PVT Layer using TensorFlow Layer Subclassing.

This module provides a PVT (Pressure-Volume-Temperature) Layer implementation
using TensorFlow's layer subclassing approach. The layer computes fluid properties
and their derivatives based on either polynomial fitting or polyharmonic spline
interpolation.
"""

import tensorflow as tf
import numpy as np
import sys
import os
import data_processing as dp

# Import the PolyharmonicSplineInterpolationLayer from polyhm_splines.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
from polyhm_splines import PolyharmonicSplineInterpolationLayer

class PVTLayer(tf.keras.layers.Layer):
    """
    PVT Layer for computing fluid properties and their derivatives.
    
    This layer computes fluid properties (e.g., invBg, invug) and their derivatives
    using either polynomial fitting or polyharmonic spline interpolation. The layer
    supports both dry gas (DG) and gas condensate (GC) fluid types.
    """
    def __init__(self,
                 fluid_type='DG',           # 'DG' (Dry Gas) or 'GC' (Gas Condensate)
                 fitting_method='spline',   # 'polynomial' or 'spline'
                 polynomial_config=None,    # Dictionary of polynomial coefficients
                 spline_config=None,        # Configuration class with lookup method
                 spline_order=2,            # Order for polyharmonic splines
                 regularization_weight=0.0, # Regularization weight for splines
                 min_input_threshold=14.7,   # Minimum pressure threshold (default 14.7 psi)
                 max_input_threshold=10000.0,  # Maximum pressure threshold (default 10000 psi)
                 name='pvt_layer',
                 **kwargs):
        """
        Initialize the PVT Layer.
        
        Args:
            fluid_type: Type of fluid - 'DG' (Dry Gas) or 'GC' (Gas Condensate)
            fitting_method: Method to fit data - 'polynomial' or 'spline'
            polynomial_config: Configuration for polynomial fitting
                {property_name: [coefficients]}
                e.g., {'invBg': [a0, a1, a2], 'invug': [b0, b1, b2], ...}
            spline_config: Configuration class with lookup method to get training data
                Should have methods like: lookup('p'), lookup('invBg'), etc.
            spline_order: Order for polyharmonic splines (default: 2)
            regularization_weight: Regularization weight for splines (default: 0.0)
            name: Name for the layer
            **kwargs: Additional arguments for the Layer base class
        """
        super(PVTLayer, self).__init__(name=name, **kwargs)
        
        # Store configuration
        self.fluid_type = fluid_type.upper()
        self.fitting_method = fitting_method.lower()
        self.spline_order = spline_order
        self.regularization_weight = regularization_weight
        self.min_input_threshold = min_input_threshold  # Store the minimum pressure threshold
        self.max_input_threshold = max_input_threshold  # Store the maximum pressure threshold
        
        # Configure fluid properties based on fluid type
        if self.fluid_type == 'DG':  # Dry Gas (single phase)
            self.properties = ['invBg', 'invug',]
        elif self.fluid_type == 'GC':  # Gas Condensate (two phase)
            self.properties = ['invBg', 'invBo', 'invug', 'invuo', 'Rs', 'Rv', 'Vro']
        else:
            raise ValueError(f"Unknown fluid type: {fluid_type}. Use 'DG' or 'GC'.")
            
        # Store fitting configuration
        if self.fitting_method == 'polynomial':
            if polynomial_config is None:
                raise ValueError("polynomial_config must be provided when fitting_method is 'polynomial'")
            self.polynomial_config = polynomial_config
            
            # Validate that all required properties have polynomial coefficients
            for prop in self.properties:
                if prop not in self.polynomial_config:
                    raise ValueError(f"Polynomial coefficients missing for property: {prop}")
                    
        elif self.fitting_method == 'spline':
            if spline_config is None:
                raise ValueError("spline_config must be provided when fitting_method is 'spline'")
            self.spline_config = spline_config
            
            # Verify that the spline_config has a lookup method
            if not hasattr(self.spline_config, 'lookup'):
                raise ValueError("spline_config must have a 'lookup' method to retrieve training data")
                
            # Create a dictionary to store spline layers
            self.spline_layers = {}
        else:
            raise ValueError(f"Unknown fitting method: {fitting_method}. Use 'polynomial' or 'spline'")
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Shape of the input tensor
        """
        if self.fitting_method == 'polynomial':
            # Store polynomial coefficients as variables for each property
            for prop in self.properties:
                setattr(self, f"{prop}_coeffs", self.add_weight(
                    name=f"{prop}_coefficients",
                    shape=(len(self.polynomial_config[prop]),),
                    initializer=tf.keras.initializers.Constant(self.polynomial_config[prop]),
                    trainable=True
                ))
                
        elif self.fitting_method == 'spline':
            # Get pressure (train_points) from spline_config
            try:
                train_points = self.spline_config.lookup('pre')
            except Exception as e:
                raise ValueError(f"Failed to get pressure values from spline_config.lookup('pre'): {e}")
            
            # Create spline interpolation layers for each property
            for prop in self.properties:
                try:
                    # Get training values for this property
                    train_values = self.spline_config.lookup(prop)
                    
                    # Create spline layer
                    self.spline_layers[prop] = PolyharmonicSplineInterpolationLayer(
                        train_points=train_points,
                        train_values=train_values,
                        order=self.spline_order,
                        regularization_weight=self.regularization_weight,
                        name=f"{prop}_spline"
                    )
                    
                except Exception as e:
                    raise ValueError(f"Failed to create spline layer for property {prop}: {e}")
        
        # Call parent's build method
        super(PVTLayer, self).build(input_shape)
        
    def call(self, inputs, training=False):
        """
        Forward pass of the PVT layer.
        
        Args:
            inputs: Input tensor - Pressure values
            training: Whether the call is in training mode or inference mode
            
        Returns:
            Output tensor with shape [2, n_properties, batch_size, *spatial_dims, 1]
            The first dimension represents [value, derivative] respectively
        """
        # Get the input shape
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        spatial_dims = input_shape[1:-1]  # All dims except batch and channel
        
        # Apply pressure validation - ensure inputs are within the valid range [min_threshold, max_threshold]
        # First, ensure non-negative and above min threshold
        inputs_safe = tf.maximum(inputs, self.min_input_threshold)
        # Then ensure below max threshold
        inputs_safe = tf.minimum(inputs_safe, self.max_input_threshold)
       
        # Number of properties
        n_properties = len(self.properties)
        
        # Initialize lists to store values and derivatives
        values = []
        derivatives = []
        
        if self.fitting_method == 'polynomial':
            # Compute values and derivatives for each property using polynomial fitting
            for prop in self.properties:
                # Get coefficients for this property
                coeffs = getattr(self, f"{prop}_coeffs")
                
                # Evaluate polynomial and its derivative with validated inputs
                value, deriv = self.evaluate_polynomial(inputs_safe, coeffs)
                
                # Append to lists
                values.append(value)
                derivatives.append(deriv)
                
        elif self.fitting_method == 'spline':
            # Compute values and derivatives for each property using spline interpolation
            for prop in self.properties:
                # Get spline layer for this property
                spline_layer = self.spline_layers[prop]
                
                # Evaluate using the spline layer with validated inputs
                with tf.GradientTape() as tape:
                    tape.watch(inputs_safe)
                    value = spline_layer(inputs_safe)
                
                # Compute the derivative
                deriv = tape.gradient(value, inputs_safe)
                
                # Append to lists
                values.append(value)
                derivatives.append(deriv)
        
        # Stack values and derivatives along a new first dimension
        # Each will have shape [n_properties, batch_size, *spatial_dims, 1]
        values_stacked = tf.stack(values, axis=0)
        derivatives_stacked = tf.stack(derivatives, axis=0)
        
        # Stack values and derivatives for output
        # Output shape: [2, n_properties, batch_size, *spatial_dims, 1]
        output = tf.stack([values_stacked, derivatives_stacked], axis=0)
        
        return output
    
    def evaluate_polynomial(self, x, coefficients):
        """
        Evaluate a polynomial with the given coefficients at points x.
        
        Args:
            x: Input tensor with shape [batch_size, ...spatial_dims, 1]
            coefficients: Polynomial coefficients [a0, a1, a2, ...]
            
        Returns:
            Tuple of (values, derivatives) with same shape as x
        """
        # Remember the original shape for later reshaping
        orig_shape = tf.shape(x)
        
        # Flatten x to a 1D tensor
        x_flat = tf.reshape(x, [-1])
        
        # Convert coefficients to tensor to ensure we can get its length and index it
        coeffs_tensor = tf.convert_to_tensor(coefficients)
        n_coeffs = tf.shape(coeffs_tensor)[0]
        
        # Function to evaluate polynomial in TensorFlow
        def poly_eval(x_val, coeffs):
            result = tf.zeros_like(x_val)
            for i in range(n_coeffs):
                # Cast i to float32 to match the type of x_val
                i_float = tf.cast(i, tf.float32)
                result += coeffs[i] * tf.pow(x_val, i_float)
            return result
        
        # Function to evaluate polynomial derivative in TensorFlow
        def poly_deriv(x_val, coeffs):
            result = tf.zeros_like(x_val)
            for i in range(1, n_coeffs):
                # Cast both i and (i-1) to float32 for the power operation
                i_float = tf.cast(i, tf.float32)
                i_minus_1_float = tf.cast(i-1, tf.float32)
                result += i_float * coeffs[i] * tf.pow(x_val, i_minus_1_float)
            return result
        
        # Evaluate polynomial and derivative
        value_flat = poly_eval(x_flat, coeffs_tensor)
        deriv_flat = poly_deriv(x_flat, coeffs_tensor)

        # Reshape back to original shape
        value = tf.reshape(value_flat, orig_shape)
        deriv = tf.reshape(deriv_flat, orig_shape)
        
        return value, deriv
    
    def get_config(self):
        """
        Get the configuration of the layer.
        
        Returns:
            Dictionary of layer configuration
        """
        config = super(PVTLayer, self).get_config()
        config.update({
            'fluid_type': self.fluid_type,
            'fitting_method': self.fitting_method,
            'spline_order': self.spline_order,
            'regularization_weight': self.regularization_weight,
            'min_input_threshold': self.min_input_threshold,
            'max_input_threshold': self.max_input_threshold
        })
        # Note: polynomial_config and spline_config are not included as they may not be serializable
        return config


def create_test_pvt_dataframe(fluid_type='DG', num_samples=20):
    """
    Create a test PVT data frame for testing purposes.
    
    Args:
        fluid_type: Type of fluid ('DG' or 'GC')
        num_samples: Number of data points to generate
        
    Returns:
        DataFrame with PVT data
    """
    import numpy as np
    import pandas as pd
    
    # Generate pressure values (10 to 10000)
    pressure_values = np.linspace(10, 10000, num_samples)
    
    # Create a dictionary to store data
    data = {'pre': pressure_values}
    
    # Define property ranges based on fluid type
    if fluid_type.upper() == 'DG':
        # DG (Dry Gas) properties
        property_ranges = {
            'invBg': (0.005, 2.5),   # Inverse gas formation volume factor
            'invug': (50, 10)        # Inverse gas viscosity
        }
    else:  # GC (Gas Condensate)
        # GC properties
        property_ranges = {
            'invBg': (0.005, 2.5),   # Inverse gas formation volume factor
            'invBo': (0.9, 0.3),     # Inverse oil formation volume factor
            'invug': (50, 10),       # Inverse gas viscosity
            'invuo': (0.1, 1),       # Inverse oil viscosity
            'Rs': (0, 5.5),          # Solution gas-oil ratio
            'Rv': (0.01, 0.09),      # Volatilized oil-gas ratio
            'Vro': (0.0002, 0.1)     # Volume ratio
        }
    
    # Generate data for each property
    for prop, (min_val, max_val) in property_ranges.items():
        # Create values with some non-linearity to simulate real PVT data
        # Use logarithmic spacing for some properties to better simulate real behavior
        if prop in ['invBg', 'invug', 'Vro']:
            # These properties often have logarithmic behavior
            log_min = np.log10(min_val) if min_val > 0 else -3
            log_max = np.log10(max_val)
            log_values = np.linspace(log_min, log_max, num_samples)
            prop_values = 10 ** log_values
        else:
            # Linear behavior for other properties
            prop_values = np.linspace(min_val, max_val, num_samples)
            
            # Add some slight non-linearity
            x = np.linspace(0, 1, num_samples)
            nonlinearity = 0.1 * np.sin(3 * np.pi * x)
            prop_values = prop_values + (max_val - min_val) * nonlinearity
        
        data[prop] = prop_values
    
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    
    return df


def test_pvt_layer():
    """
    Test the PVT layer with different configurations.
    """
    print("\n=== Testing PVT Layer ===")
    
    # Import data processing utilities for test
    try:
        import data_processing as dp
    except ImportError:
        print("Warning: data_processing module not found.")
        print("The spline fitting method requires the data_processing module to be available.")
        print("Please ensure it's properly installed and importable before using spline fitting.")
        dp = None
    
    # Import configuration utilities
    try:
        from default_configurations import get_configuration, load_spline_data
    except ImportError:
        print("Warning: default_configurations module not found. Using hardcoded configurations.")
        get_configuration = None
        load_spline_data = None
    
    # Test data
    batch_size = 2
    spatial_dims = (10, 10)
    time_value = 0.5
    property_value = 1.0
    
    # Create test input data
    # Shape: [batch_size, *spatial_dims, channels]
    # Channels are: features (optional), time, property
    input_data = tf.random.normal((batch_size,) + spatial_dims + (1,))  # Features
    time_channel = tf.ones((batch_size,) + spatial_dims + (1,)) * time_value  # Time
    property_channel = tf.ones((batch_size,) + spatial_dims + (1,)) * property_value  # Property
    
    input_data = tf.concat([input_data, time_channel, property_channel], axis=-1)
    print("\n--- Testing Polynomial Fitting for DG ---")
    
    # Get DG polynomial configuration from central configuration system
    if get_configuration:
        dg_config = get_configuration('pvt_layer', fluid_type='DG', fitting_method='polynomial')
    else:
        # Fallback to hardcoded configuration
        dg_config = {
            'fluid_type': 'DG',
            'fitting_method': 'polynomial',
            'polynomial_config': {
                'invBg': [1.0, 0.1, 0.01],
                'invug': [0.5, 0.05, 0.005]
            }
        }
    
    # Create DG layer with polynomial fitting
    pvt_poly_dg = PVTLayer(
        fluid_type=dg_config['fluid_type'],
        fitting_method=dg_config['fitting_method'],
        polynomial_config=dg_config['polynomial_config'],
        name='pvt_poly_dg'
    )
    
    # Create input tensor
    input_tensor = tf.keras.layers.Input(shape=(None, None, 3))
    
    # Apply PVT layer
    output_tensor_dg = pvt_poly_dg(input_tensor)
    poly_dg_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor_dg)
    
    # Forward pass
    poly_dg_output = poly_dg_model(input_data)
    print(f"Polynomial DG output shape: {poly_dg_output.shape}")
    # Expected shape: [2, 2, batch_size, *spatial_dims, 1]
    
    print("\n--- Testing Polynomial Fitting for GC ---")
    
    # Get GC polynomial configuration from central configuration system
    if get_configuration:
        gc_config = get_configuration('pvt_layer', fluid_type='GC', fitting_method='polynomial')
    else:
        # Fallback to hardcoded configuration
        gc_config = {
            'fluid_type': 'GC',
            'fitting_method': 'polynomial',
            'polynomial_config': {
                'invBg': [1.0, 0.1, 0.01],
                'invBo': [1.2, 0.12, 0.012],
                'invug': [0.5, 0.05, 0.005],
                'invuo': [0.6, 0.06, 0.006],
                'Rs': [0.7, 0.07, 0.007],
                'Rv': [0.8, 0.08, 0.008],
                'Vro': [0.9, 0.09, 0.009]
            }
        }
    
    # Create GC layer with polynomial fitting
    pvt_poly_gc = PVTLayer(
        fluid_type=gc_config['fluid_type'],
        fitting_method=gc_config['fitting_method'],
        polynomial_config=gc_config['polynomial_config'],
        name='pvt_poly_gc'
    )
    
    # Apply PVT layer
    output_tensor_gc = pvt_poly_gc(input_tensor)
    poly_gc_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor_gc)
    
    # Forward pass
    poly_gc_output = poly_gc_model(input_data)
    print(f"Polynomial GC output shape: {poly_gc_output.shape}")
    # Expected shape: [2, 7, batch_size, *spatial_dims, 1]
    
    print("\n--- Testing Spline Fitting for DG ---")
    
    # Try to get spline data through different methods in the following order:
    # 1. From default_configurations.load_spline_data()
    # 2. Direct loading from data file
    fluid_type = 'DG'  # Default to DG for testing
    spline_config = None
    
    # Method 1: Try loading from default_configurations
    if load_spline_data:
        try:
            print("Attempting to load spline data from default_configurations...")
            spline_config = load_spline_data()
            print("Successfully loaded spline data from default_configurations")
        except Exception as e:
            print(f"Warning: Failed to load spline data from default_configurations: {str(e)}")
    
    # Method 2: Try direct loading from data file
    if spline_config is None and dp:
        try:
            print("Attempting to load spline data directly from file...")
            _pvt_data = dp.load_dataframe(filename='pvt_data', filetype='df', load_dir=None)
            spline_config = dp.DataSummary(ts_f=_pvt_data, ts_l=None)
            print("Successfully loaded spline data directly from file")
        except Exception as e:
            print(f"Warning: Failed to load PVT data directly from file: {str(e)}")
            try:
                # Create simple PVT data from dictionary if loading failed
                print("Creating simple PVT data for testing...")
                import numpy as np
                import pandas as pd
                
                # Number of data points
                n_points = 20
                
                # Define property keys and ranges based on fluid type
                if fluid_type.upper() == 'DG':
                    properties = {
                        'pre': (10, 10000),       # Pressure
                        'invBg': (0.005, 2.5),    # Inverse gas formation volume factor
                        'invug': (50, 10)         # Inverse gas viscosity
                    }
                else:  # GC fluid
                    properties = {
                        'pre': (10, 10000),       # Pressure
                        'invBg': (0.005, 2.5),    # Inverse gas formation volume factor
                        'invBo': (0.9, 0.3),      # Inverse oil formation volume factor
                        'invug': (50, 10),        # Inverse gas viscosity
                        'invuo': (0.1, 1),        # Inverse oil viscosity
                        'Rs': (0, 5.5),           # Solution gas-oil ratio
                        'Rv': (0.01, 0.09),       # Volatile oil ratio
                        'Vro': (0.0002, 0.1)      # Volume ratio
                    }
                
                # Generate data dictionary
                data_dict = {}
                
                # Generate pressure values (evenly spaced in log space)
                pressures = np.linspace(properties['pre'][0], properties['pre'][1], n_points)
                data_dict['pre'] = pressures
                
                # Generate property values (with realistic trends based on pressure)
                for prop, (min_val, max_val) in properties.items():
                    if prop == 'pre':
                        continue  # Already handled
                        
                    # Create property values with a realistic trend based on pressure
                    # Some properties increase with pressure, others decrease
                    if prop in ['invBg', 'invBo', 'Rs', 'Vro']:
                        # Properties that increase with pressure
                        fraction = (pressures - properties['pre'][0]) / (properties['pre'][1] - properties['pre'][0])
                        data_dict[prop] = min_val + fraction * (max_val - min_val)
                    else:
                        # Properties that decrease with pressure
                        fraction = (pressures - properties['pre'][0]) / (properties['pre'][1] - properties['pre'][0])
                        data_dict[prop] = max_val - fraction * (max_val - min_val)
                
                # Create pandas DataFrame
                _pvt_data = pd.DataFrame(data_dict)
                print(f"Created test PVT data with {len(_pvt_data)} samples for {fluid_type} fluid")
                print(_pvt_data.head())
                
                # Create DataSummary from the generated data
                spline_config = dp.DataSummary(ts_f=_pvt_data, ts_l=None)
                print("Successfully created spline configuration from generated test data")
                
            except Exception as e:
                print(f"Error creating test PVT data: {str(e)}")
                print("Spline fitting requires a valid data source. Please create a 'pvt_data.df' file")
                print("or use the 'data_processing' module to load data from another source.")
                print("Skipping spline test - returning only polynomial models")
                return poly_dg_model, poly_gc_model, None
    else:
        if spline_config is None:
            print("The spline fitting method requires either:")
            print("1. Properly configured default_configurations.load_spline_data() function")
            print("2. Access to the data_processing module with PVT data available")
            print("Neither condition was met. Skipping spline test.")
            return poly_dg_model, poly_gc_model, None
    
    # Create DG layer with spline fitting
    pvt_spline_dg = PVTLayer(
        fluid_type='DG',
        fitting_method='spline',
        spline_config=spline_config,
        spline_order=2,
        regularization_weight=0.001,
        name='pvt_spline_dg'
    )
    
    # Apply PVT layer
    output_tensor_spline_dg = pvt_spline_dg(input_tensor)
    spline_dg_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor_spline_dg)
    
    # Forward pass - only if spline_config is available
    try:
        spline_dg_output = spline_dg_model(input_data)
        print(f"Spline DG output shape: {spline_dg_output.shape}")
        # Expected shape: [2, 2, batch_size, *spatial_dims, 1]
    except Exception as e:
        print(f"Error during spline forward pass: {str(e)}")
        spline_dg_model = None
    
    return poly_dg_model, poly_gc_model, spline_dg_model

if __name__ == "__main__":
    poly_dg_model, poly_gc_model, spline_dg_model = test_pvt_layer()
