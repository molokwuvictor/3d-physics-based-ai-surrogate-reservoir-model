"""
A PVT Module that integrates a Hard Layer and a PVT Layer.

This module provides a framework for integrating the PVT layer with hard enforcment
constraints in surrogate reservoir modeling. It uses composition to instantiate both 
a HardLayer and a PVTLayer during initialization. This resulting module provides a 
drop-in replacement for a standalone PVTLayer while incorporating additional 
enforcement constraints.
"""

import tensorflow as tf
import numpy as np
import os
import sys
import data_processing as dp
from default_configurations import get_configuration

from Hard_Layer_Subclassed import HardLayer
from PVT_Layer_Subclassed import PVTLayer

# Add directory to path to import model classes or use working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Do not load PVT data here - will load on-demand when needed
class PVTModuleWithHardLayer(tf.keras.layers.Layer):
    """
    Complete module that integrates HardLayer and PVTLayer.
    
    This module combines:
    1. A hard layer for enforcing physical constraints (optional)
    2. A PVT layer for computing fluid properties and their derivatives
    """
    def __init__(self, 
                 use_hard_layer=True,
                 hard_layer_config=None,
                 input_slice_config=None,
                 pvt_layer_config=None,
                 name='pvt_module_with_hard_layer',
                 **kwargs):
        """
        Initialize the PVT module with optional Hard layer.
        
        Args:
            use_hard_layer: Whether to use the hard layer or bypass it
            hard_layer_config: Configuration for the HardLayer
            input_slice_config: Configuration for how to slice the input data
            pvt_layer_config: Configuration for the PVTLayer
            name: Name for the layer
            **kwargs: Additional keyword arguments for the Layer
        """
        super(PVTModuleWithHardLayer, self).__init__(name=name, **kwargs)
        
        # Store configuration
        self.hard_layer_config = hard_layer_config or {}
        self.pvt_layer_config = pvt_layer_config or {}
        self.use_hard_layer = use_hard_layer
        
        # Configure how inputs should be sliced for each component
        self.input_slice_config = input_slice_config or {
            'hard_layer': {
                'time': slice(-2, -1),     # Second-to-last channel for time
                'property': slice(-1, None)  # Last channel for property
            }
        }
        
    def build(self, input_shape):
        """
        Build the layer components based on input shape.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # Initialize and build the hard layer if enabled
        if self.use_hard_layer:
            # Get default hard layer configuration
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
        
        # Initialize and build the PVT layer
        # Get fitting method and fluid type from configuration
        fitting_method = self.pvt_layer_config.get('fitting_method', 'polynomial')
        fluid_type = self.pvt_layer_config.get('fluid_type', 'DG')
        
        # Create the default PVT config
        default_pvt_config = get_configuration('pvt_layer', fluid_type=fluid_type, fitting_method=fitting_method)

        # Update with remaining user-provided configuration
        import collections.abc
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        updated_pvt_config = deep_update(default_pvt_config, self.pvt_layer_config)
        
        # Remove keys not accepted by PVTLayer constructor
        allowed_keys = {'fluid_type', 'fitting_method', 'polynomial_config', 'spline_config', 'spline_order', 'regularization_weight', 'name', 'min_input_threshold'}
        filtered_pvt_config = {k: v for k, v in updated_pvt_config.items() if k in allowed_keys}

        # Initialize the PVT layer
        self.pvt_layer = PVTLayer(**filtered_pvt_config)
        
        # Build PVT layer
        self.pvt_layer.build(input_shape)
        
        # Call the parent's build method
        super(PVTModuleWithHardLayer, self).build(input_shape)
    
    def call(self, inputs, training=False):
        """
        Forward pass through the module.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Either hard layer output or PVT layer output
        """
        # Process through hard layer if enabled
        if self.use_hard_layer:
            # Extract variables for hard layer
            if isinstance(inputs, list):
                # If inputs is already a list, use it directly
                input_vars = inputs[0]
                features = inputs[1]
                hard_layer_inputs = [input_vars, features]
            else:
                # Extract slices from the input for hard layer
                # Then use inputs as both input_vars and features
                hard_layer_inputs = [inputs, inputs]
            
            # Process through hard layer
            hard_output = self.hard_layer(hard_layer_inputs, training=training)
            
            # Pass hard layer output to PVT layer
            pvt_input = hard_output
        else:
            # If hard layer is not enabled, use direct input for PVT layer
            pvt_input = inputs
        
        # Process through PVT layer
        pvt_output = self.pvt_layer(pvt_input, training=training)
        return pvt_output
    
    def get_config(self):
        """
        Get the configuration of the layer.
        
        Returns:
            Dictionary of configuration parameters
        """
        config = super(PVTModuleWithHardLayer, self).get_config()
        config.update({
            'use_hard_layer': self.use_hard_layer,
            'hard_layer_config': self.hard_layer_config,
            'input_slice_config': self.input_slice_config,
            'pvt_layer_config': self.pvt_layer_config
        })
        return config


def test_pvt_module(input_shape=(None, 39, 39, 16), 
                    use_hard_layer=True,
                    fluid_type='DG',
                    fitting_method='polynomial'):
    """
    Test the module with hard layer and PVT layer.
    
    Args:
        input_shape: Input shape for testing (batch_size, height, width, channels)
                    This shape is similar to what would come from an encoder-decoder network
        use_hard_layer: Whether to use the hard layer
        fluid_type: Type of fluid for PVT layer ('DG' or 'GC')
        fitting_method: Fitting method for PVT layer ('polynomial' or 'spline')
        
    Returns:
        Test model with the module
    """
    import numpy as np
    import pandas as pd
    print("\n=== Testing PVT Module with Hard Layer ===")
    print(f"Hard Layer: {'Enabled' if use_hard_layer else 'Disabled'}")
    print(f"Fluid Type: {fluid_type}")
    print(f"Fitting Method: {fitting_method}")
    
    # Get configurations from default settings
    pvt_module_config = get_configuration('pvt_module', use_rbf=False, 
                                         fluid_type=fluid_type, 
                                         fitting_method=fitting_method)
    
    # Extract individual configurations
    hard_layer_config = pvt_module_config['hard_layer_config']
    pvt_layer_config = pvt_module_config['pvt_layer_config']
    input_slice_config = pvt_module_config['input_slice_config']
    
    # For spline fitting, check if we need to create test data
    if fitting_method.lower() == 'spline' and 'spline_config' not in pvt_layer_config:
        print("Spline configuration not found in default config, creating test data...")
        
        # Import required functions for creating test data
        try:
            # Try to import from PVT_Layer_Subclassed
            from PVT_Layer_Subclassed import create_test_pvt_dataframe
            
            # Try data_processing module first for DataSummary
            try:
                import data_processing as dp
                # Create test pvt data
                test_df = create_test_pvt_dataframe(fluid_type=fluid_type)
                print(f"Created test PVT data with {len(test_df)} samples")
                
                # Create spline config
                spline_config = dp.DataSummary(ts_f=test_df, ts_l=None)
                pvt_layer_config['spline_config'] = spline_config
                print("Using test PVT data for spline configuration")
                
            except ImportError:
                # If data_processing is not available, implement a simple lookup
                print("data_processing module not found, using SimpleDataSummary")
                import numpy as np
                
                # Create test pvt data
                test_df = create_test_pvt_dataframe(fluid_type=fluid_type)
                
                # Simple lookup class
                class SimpleDataSummary:
                    def __init__(self, df):
                        self.df = df
                        self.properties = [col for col in df.columns if col != 'pre']
                    
                    def lookup(self, property_name, pressure, temperature=None):
                        """Simple interpolation for property values"""
                        if property_name not in self.properties:
                            raise ValueError(f"Unknown property: {property_name}")
                        
                        # Find closest pressure points for interpolation
                        pressures = self.df['pre'].values
                        idx = np.searchsorted(pressures, pressure)
                        
                        # Handle boundary cases and interpolation
                        if idx == 0:
                            value = self.df[property_name].iloc[0]
                            # Simple derivative approximation
                            if idx < len(pressures) - 1:
                                derivative = (self.df[property_name].iloc[1] - value) / (pressures[1] - pressures[0])
                            else:
                                derivative = 0.0
                        elif idx >= len(pressures):
                            value = self.df[property_name].iloc[-1]
                            # Simple derivative approximation
                            derivative = (value - self.df[property_name].iloc[-2]) / (pressures[-1] - pressures[-2])
                        else:
                            # Linear interpolation
                            p0, p1 = pressures[idx-1], pressures[idx]
                            v0, v1 = self.df[property_name].iloc[idx-1], self.df[property_name].iloc[idx]
                            
                            # Calculate interpolated value
                            ratio = (pressure - p0) / (p1 - p0) if p1 != p0 else 0
                            value = v0 + ratio * (v1 - v0)
                            
                            # Calculate derivative (slope)
                            derivative = (v1 - v0) / (p1 - p0) if p1 != p0 else 0
                        
                        return value, derivative
                
                spline_config = SimpleDataSummary(test_df)
                pvt_layer_config['spline_config'] = spline_config
                print("Using SimpleDataSummary with test data for spline configuration")
                
        except Exception as e:
            print(f"Warning: Failed to create test spline data: {str(e)}")
            print("Falling back to polynomial fitting...")
            fitting_method = 'polynomial'
            pvt_layer_config['fitting_method'] = 'polynomial'
            
            # Get polynomial configuration for the selected fluid type
            temp_config = get_configuration('pvt_layer', fluid_type=fluid_type, fitting_method='polynomial')
            pvt_layer_config['polynomial_config'] = temp_config['polynomial_config']
    
    # Create the PVT module as a layer
    complete_module = PVTModuleWithHardLayer(
        use_hard_layer=use_hard_layer,
        hard_layer_config=hard_layer_config,
        pvt_layer_config=pvt_layer_config,
        input_slice_config=input_slice_config['hard_layer']
    )
    
    # Create a Keras model with the module
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    outputs = complete_module(inputs)
    test_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary
    test_model.summary()
    
    # Create dummy data for testing a forward pass
    batch_size = 2
    test_input = tf.random.normal((batch_size,) + input_shape[1:])
    
    # Test a forward pass
    output = test_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print number of trainable variables
    pvt_layer_params = np.sum([np.prod(v.get_shape()) for v in complete_module.pvt_layer.trainable_weights])
    print(f"PVT Layer trainable parameters: {pvt_layer_params}")
    
    if use_hard_layer:
        hard_layer_params = np.sum([np.prod(v.get_shape()) for v in complete_module.hard_layer.trainable_weights])
        print(f"Hard Layer trainable parameters: {hard_layer_params}")
        print(f"Total trainable parameters: {pvt_layer_params + hard_layer_params}")
    else:
        print(f"Total trainable parameters: {pvt_layer_params}")
    
    # Print sample property information based on fluid type
    if fluid_type.upper() == 'DG':
        print("Properties: invBg (gas formation volume factor), invug (gas viscosity)")
    else:  # GC
        print("Properties: invBg, invBo, invug, invuo, Rs, Rv, Vro")
    
    return test_model


if __name__ == "__main__":
    # Test with Hard Layer enabled
    model_with_hard = test_pvt_module(
        input_shape=(None, 39, 39, 16),
        use_hard_layer=True,
        fluid_type='DG',
        fitting_method='polynomial'
    )
    
    # Test without Hard Layer
    model_without_hard = test_pvt_module(
        input_shape=(None, 39, 39, 16),
        use_hard_layer=False,
        fluid_type='DG',
        fitting_method='polynomial'
    )
    
    # Test with different fluid type
    model_gc = test_pvt_module(
        input_shape=(None, 39, 39, 16),
        use_hard_layer=True,
        fluid_type='GC',
        fitting_method='polynomial'
    )
    
    # Test with spline fitting
    model_spline = test_pvt_module(
        input_shape=(None, 39, 39, 16),
        use_hard_layer=True,
        fluid_type='DG',
        fitting_method='spline'
    )
