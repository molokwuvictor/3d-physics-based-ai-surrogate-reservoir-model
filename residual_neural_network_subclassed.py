#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under the MIT license.
#
# This module provides an efficient and trainable residual network layer that can be
# built with either CNN layers or Dense layers. In the dense mode the layers operate
# on the last dimension so that the input can be multidimensional (e.g., [39, 39, 1])
# without flattening. Two separate activation functions are supported for the hidden 
# layers and for the output layer. When latent output is enabled, the latent sample is 
# rescaled based on user-specified limits a and b, and the stacked tensor is broadcast 
# to match the spatial dimensions of the feature map.
#
# Now with support for temporal dimensions to handle inputs with shape (None, T, H, W, C).

import tensorflow as tf
import numpy as np

# Define a basic Residual Block that supports both CNN and Dense variants.
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 hidden_activation=tf.nn.swish,
                 use_batch_norm=False,
                 dropout_rate=0.0,
                 use_projection=False,
                 network_type="cnn",   # Accepts "cnn", "cnn3d" or "dense"
                 kernel_initializer="glorot_normal",
                 kernel_regularizer=None,
                 time_distributed=False,
                 name=None,
                 **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
        self.network_type = network_type.lower()
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.use_projection = use_projection
        self.time_distributed = time_distributed
        LayerWrapper = tf.keras.layers.TimeDistributed if time_distributed else (lambda x: x)

        # Create the two main layers for the residual branch.
        if self.network_type == "cnn":
            self.layer1 = LayerWrapper(tf.keras.layers.Conv2D(filters,
                                                 kernel_size,
                                                 strides=strides,
                                                 padding='same',
                                                 kernel_initializer=kernel_initializer,
                                                 kernel_regularizer=kernel_regularizer,
                                                 use_bias=not use_batch_norm,
                                                 name="conv1"))
            self.layer2 = LayerWrapper(tf.keras.layers.Conv2D(filters,
                                                 kernel_size,
                                                 strides=1,
                                                 padding='same',
                                                 kernel_initializer=kernel_initializer,
                                                 kernel_regularizer=kernel_regularizer,
                                                 use_bias=not use_batch_norm,
                                                 name="conv2"))
        elif self.network_type == "cnn3d":
            self.layer1 = LayerWrapper(tf.keras.layers.Conv3D(filters,
                                                 kernel_size,
                                                 strides=strides,
                                                 padding='same',
                                                 kernel_initializer=kernel_initializer,
                                                 kernel_regularizer=kernel_regularizer,
                                                 use_bias=not use_batch_norm,
                                                 name="conv3d1"))
            self.layer2 = LayerWrapper(tf.keras.layers.Conv3D(filters,
                                                 kernel_size,
                                                 strides=1,
                                                 padding='same',
                                                 kernel_initializer=kernel_initializer,
                                                 kernel_regularizer=kernel_regularizer,
                                                 use_bias=not use_batch_norm,
                                                 name="conv3d2"))
        elif self.network_type == "dense":
            self.layer1 = LayerWrapper(tf.keras.layers.Dense(filters,
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                use_bias=True,
                                                name="dense1"))
            self.layer2 = LayerWrapper(tf.keras.layers.Dense(filters,
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                use_bias=True,
                                                name="dense2"))
        else:
            raise ValueError(f"Unknown network_type: {network_type}")

        # Batch normalization layers if enabled.
        self.bn1 = LayerWrapper(tf.keras.layers.BatchNormalization(name="bn1")) if use_batch_norm else (lambda x, training=None: x)
        self.bn2 = LayerWrapper(tf.keras.layers.BatchNormalization(name="bn2")) if use_batch_norm else (lambda x, training=None: x)

        # Dropout layer if dropout_rate > 0.
        self.dropout = LayerWrapper(tf.keras.layers.Dropout(rate=dropout_rate, name="dropout")) if dropout_rate > 0.0 else (lambda x, training=None: x)

        # Optional projection layer for the skip connection.
        if self.use_projection:
            if self.network_type == "cnn":
                self.proj = LayerWrapper(tf.keras.layers.Conv2D(filters,
                                                   kernel_size=1,
                                                   strides=strides,
                                                   padding='same',
                                                   kernel_initializer=kernel_initializer,
                                                   kernel_regularizer=kernel_regularizer,
                                                   use_bias=False,
                                                   name="proj"))
            elif self.network_type == "cnn3d":
                self.proj = LayerWrapper(tf.keras.layers.Conv3D(filters,
                                                   kernel_size=1,
                                                   strides=strides,
                                                   padding='same',
                                                   kernel_initializer=kernel_initializer,
                                                   kernel_regularizer=kernel_regularizer,
                                                   use_bias=False,
                                                   name="proj3d"))
            else:
                self.proj = LayerWrapper(tf.keras.layers.Dense(filters,
                                                  kernel_initializer=kernel_initializer,
                                                  kernel_regularizer=kernel_regularizer,
                                                  name="proj"))
            self.bn_proj = LayerWrapper(tf.keras.layers.BatchNormalization(name="bn_proj")) if use_batch_norm else (lambda x, training=None: x)
        else:
            self.proj = None

    def call(self, inputs, training=False):
        shortcut = inputs

        # Main branch: layer1 -> BN -> activation -> dropout -> layer2 -> BN.
        x = self.layer1(inputs)
        x = self.bn1(x, training=training)
        x = self.hidden_activation(x)
        x = self.dropout(x, training=training)
        x = self.layer2(x)
        x = self.bn2(x, training=training)

        # Automatically check shape compatibility.
        if self.proj is not None:
            if self.network_type == "cnn":
                # Compare spatial dimensions and channels.
                if shortcut.shape[1:] != x.shape[1:]:
                    shortcut = self.proj(shortcut)
                    shortcut = self.bn_proj(shortcut, training=training)
            else:
                # For dense layers, check the last dimension.
                if shortcut.shape[-1] != x.shape[-1]:
                    shortcut = self.proj(shortcut)
        # Add the residual connection.
        x = tf.keras.layers.add([x, shortcut])
        return self.hidden_activation(x)


# Define the overall Residual Network Layer that stacks residual blocks.
class ResidualNetworkLayer(tf.keras.layers.Layer):
    def __init__(self,
                 num_blocks,
                 filters,
                 kernel_size=3,
                 hidden_activation=tf.nn.swish,
                 output_activation=None,
                 use_batch_norm=False,
                 dropout_rate=0.0,
                 latent_output=False,
                 output_filters=1,
                 include_output_layer=True,
                 network_type="cnn",   # "cnn" or "dense"
                 kernel_initializer="he_normal",
                 kernel_regularizer=None,
                 latent_a=0.0,
                 latent_b=1.0,
                 temporal=False,       # Added support for temporal dimension
                 output_distribution=False,  # When True, uses GlobalAvgPool + Dense + Softmax for output
                 number_of_output_bins=50,
                 name="resnet_layer",
                 **kwargs):
        """
        Args:
            num_blocks: Number of residual blocks to stack.
            filters: Number of filters (or units for dense) for each block.
            kernel_size: Convolution kernel size (ignored for dense).
            hidden_activation: Activation function used within the hidden (residual block) layers.
            output_activation: Activation function applied to the output branch (if latent_output is False).
            use_batch_norm: Whether to apply batch normalization.
            dropout_rate: Dropout rate applied within each block.
            latent_output: If True, produces a latent representation via global pooling and reparameterization.
            output_filters: The number of filters (or units) for the output branch.
            include_output_layer: Whether to apply a final output layer.
            network_type: "cnn" for convolutional layers or "dense" for Dense layers that operate elementwise on the last dimension.
            latent_a: Lower bound for rescaling the latent sample.
            latent_b: Upper bound for rescaling the latent sample.
            temporal: Whether input has a temporal dimension (e.g., shape of [None, T, H, W, C])
            name: Layer name.
        """
        super(ResidualNetworkLayer, self).__init__(name=name, **kwargs)
        self.latent_output = latent_output
        self.include_output_layer = include_output_layer
        self.network_type = network_type.lower()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.latent_a = latent_a
        self.latent_b = latent_b
        self.temporal = temporal
        self.output_distribution = output_distribution
        self.number_of_output_bins = number_of_output_bins

        # Create the sequential stack of residual blocks.
        self.blocks = []
        for i in range(num_blocks):
            use_proj = (i == 0)
            block = ResidualBlock(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=1,
                                  hidden_activation=hidden_activation,
                                  use_batch_norm=use_batch_norm,
                                  dropout_rate=dropout_rate,
                                  use_projection=use_proj,
                                  network_type=self.network_type,
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  time_distributed=self.temporal,
                                  name=f"res_block_{i+1}")
            self.blocks.append(block)

        # Configure the output branch.
        if self.include_output_layer:
            if self.output_distribution:
                num_bins = self.number_of_output_bins
                # Always instantiate both 2D and 3D pooling and reshape layers
                self.global_pool_2d = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', name="global_timestep_pool_2d")
                self.global_pool_3d = tf.keras.layers.GlobalAveragePooling3D(data_format='channels_last', name="global_timestep_pool_3d")
                self.timestep_reshape_2d = tf.keras.layers.Reshape((1, 1, num_bins), name="timestep_reshape_2d")
                self.timestep_reshape_3d = tf.keras.layers.Reshape((1, 1, 1, num_bins), name="timestep_reshape_3d")
                self.timestep_dense = tf.keras.layers.Dense(num_bins, name="timestep_dense")
                self.timestep_softmax = tf.keras.layers.Softmax(axis=-1, name="timestep_softmax")
            elif not self.latent_output:
                if self.network_type == "cnn":
                    if self.temporal:
                        self.out_layer = tf.keras.layers.TimeDistributed(
                            tf.keras.layers.Conv2D(output_filters,
                                kernel_size=1,
                                activation=None,
                                padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                name="output_layer")
                        )
                    else:
                        self.out_layer = tf.keras.layers.Conv2D(output_filters,
                                                            kernel_size=1,
                                                            activation=None,
                                                            padding='same',
                                                            kernel_initializer=kernel_initializer,
                                                            kernel_regularizer=kernel_regularizer,
                                                            name="output_layer")
                elif self.network_type == "cnn3d":
                    if self.temporal:
                        self.out_layer = tf.keras.layers.TimeDistributed(
                            tf.keras.layers.Conv3D(output_filters,
                                kernel_size=1,
                                activation=None,
                                padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                name="output_layer")
                        )
                    else:
                        self.out_layer = tf.keras.layers.Conv3D(output_filters,
                                                            kernel_size=1,
                                                            activation=None,
                                                            padding='same',
                                                            kernel_initializer=kernel_initializer,
                                                            kernel_regularizer=kernel_regularizer,
                                                            name="output_layer")
                else:
                    self.out_layer = tf.keras.layers.Dense(output_filters,
                                                           kernel_initializer=kernel_initializer,
                                                           kernel_regularizer=kernel_regularizer,
                                                           name="output_layer")
            else:
                # For latent output, use global pooling and Dense layers for z_mean and z_log_var.
                self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")
                self.z_mean_dense = tf.keras.layers.Dense(output_filters, name="z_mean")
                self.z_log_var_dense = tf.keras.layers.Dense(output_filters, name="z_log_var")
                self.latent_activation = tf.keras.layers.Activation(tf.nn.sigmoid, name='latent_activation')
                self.sampling = tf.keras.layers.Lambda(self._sampling, name="z_sampling")
        else:
            self.out_layer = None
            
        # Note: time_distributed logic is handled internally. Each block handles its own time_distributed logic.

    def _sampling(self, args):
        """ Reparameterization trick for VAE-like sampling from the latent space """
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # Rescale to [a, b]
        rescaled_z = (self.latent_b - self.latent_a) * tf.sigmoid(z) + self.latent_a
        return rescaled_z

    def call(self, inputs, training=False, return_skip_conn=False):
        """
        Forward pass through the ResidualNetworkLayer.
        
        Args:
            inputs: Input tensor with shape [batch, height, width, channels] for CNN
                   or [batch, ..., features] for dense, or [batch, time, ...] if temporal.
            training: Whether the layer should run in training mode.
            return_skip_conn: If True, return intermediate activations for skip connections.
            
        Returns:
            Output tensor, or tuple of (output, skip_connections) if return_skip_conn is True.
        """
        x = inputs
        skip_connections = {}
        for i, block in enumerate(self.blocks):
            x = block(x, training=training)
            skip_connections[f"block_{i}"] = x
        
        # Apply output layer if needed
        if self.include_output_layer:
            if self.output_distribution:
                # Dynamically select pooling and reshape layers based on input rank
                input_rank = len(x.shape)
                if input_rank == 4:
                    # (batch, H, W, C) => use 2D pooling
                    pool_layer = self.global_pool_2d
                    reshape_layer = self.timestep_reshape_2d
                elif input_rank == 5:
                    # (batch, D, H, W, C) => use 3D pooling
                    pool_layer = self.global_pool_3d
                    reshape_layer = self.timestep_reshape_3d
                else:
                    raise ValueError(f"Unsupported input rank for output_distribution: {input_rank}")
                pooled = pool_layer(x)
                dense_output = self.timestep_dense(pooled)
                reshaped = reshape_layer(dense_output)
                x = self.timestep_softmax(reshaped)
            elif not self.latent_output:
                x = self.out_layer(x)
                if self.output_activation is not None:
                    x = self.output_activation(x)
            else:
                # VAE-like latent representation
                pooled = self.global_pool(x)
                z_mean = self.z_mean_dense(pooled)
                z_log_var = self.z_log_var_dense(pooled)
                z = self.sampling([z_mean, z_log_var])
                
                # Get spatial dimensions from the input for broadcasting
                spatial_shape = tf.shape(x)[1:-1]
                
                # Broadcast the latent vector to match the spatial dimensions
                z_broadcasted = tf.reshape(z, [-1, 1, 1, tf.shape(z)[-1]])
                multiples = tf.concat([[1], spatial_shape, [1]], axis=0)
                x = tf.tile(z_broadcasted, multiples)
                
                # Apply final activation if needed
                if self.output_activation is not None:
                    x = self.output_activation(x)
            
            if return_skip_conn:
                return x, skip_connections
            return x


# Example usage:
if __name__ == '__main__':
    print("\n===== 2D CNN Non-Temporal Example =====\n")
    cnn_input = tf.keras.Input(shape=(64, 64, 3))
    resnet_cnn = ResidualNetworkLayer(num_blocks=3,
                                      filters=32,
                                      kernel_size=3,
                                      hidden_activation=tf.nn.swish,
                                      output_activation=tf.nn.sigmoid,
                                      use_batch_norm=True,
                                      dropout_rate=0.1,
                                      latent_output=True,
                                      output_filters=10,
                                      include_output_layer=True,
                                      network_type="cnn",
                                      latent_a=0.2,
                                      latent_b=0.8,
                                      name="custom_resnet_cnn")
    cnn_output = resnet_cnn(cnn_input)
    model_cnn = tf.keras.Model(cnn_input, cnn_output, name="ResNetModel_CNN")
    model_cnn.summary()

    print("\n===== Dense Non-Temporal Example =====\n")
    dense_input = tf.keras.Input(shape=(39, 39, 1))
    resnet_dense = ResidualNetworkLayer(num_blocks=2,
                                        filters=64,
                                        kernel_size=1,
                                        hidden_activation=tf.nn.relu,
                                        output_activation=tf.nn.softmax,
                                        use_batch_norm=False,
                                        dropout_rate=0.0,
                                        latent_output=False,
                                        output_filters=5,
                                        include_output_layer=True,
                                        network_type="dense",
                                        name="custom_resnet_dense")
    dense_output = resnet_dense(dense_input)
    model_dense = tf.keras.Model(dense_input, dense_output, name="ResNetModel_Dense")
    model_dense.summary()

    print("\n===== 3D CNN Non-Temporal Example =====\n")
    cnn3d_input = tf.keras.Input(shape=(1, 39, 39, 1))
    resnet_cnn3d = ResidualNetworkLayer(num_blocks=2,
                                        filters=8,
                                        kernel_size=3,
                                        hidden_activation=tf.nn.swish,
                                        output_activation=tf.nn.sigmoid,
                                        use_batch_norm=True,
                                        dropout_rate=0.1,
                                        latent_output=False,
                                        output_filters=2,
                                        include_output_layer=True,
                                        network_type="cnn3d",
                                        name="custom_resnet_cnn3d")
    cnn3d_output = resnet_cnn3d(cnn3d_input)
    model_cnn3d = tf.keras.Model(cnn3d_input, cnn3d_output, name="ResNetModel_CNN3D")
    model_cnn3d.summary()

    print("\n===== 2D CNN Temporal Example (TimeDistributed) =====\n")
    temporal_input = tf.keras.Input(shape=(1, 39, 39, 5))  # (batch, time, H, W, C)
    resnet_temporal = ResidualNetworkLayer(num_blocks=4,
                                           filters=32,
                                           kernel_size=3,
                                           hidden_activation=tf.nn.swish,
                                           output_activation=None,
                                           use_batch_norm=True,
                                           dropout_rate=0.0,
                                           latent_output=False,
                                           output_filters=4,
                                           include_output_layer=True,
                                           network_type="cnn",
                                           temporal=True,
                                           output_distribution=True,  # Enable time step module
                                           number_of_output_bins=50,  # Number of time step bins
                                           name="custom_resnet_temporal")
    temporal_output = resnet_temporal(temporal_input)
    model_temporal = tf.keras.Model(temporal_input, temporal_output, name="ResNetModel_Temporal")
    model_temporal.summary()

    print("\n===== 3D CNN Temporal Example (TimeDistributed) =====\n")
    temporal3d_input = tf.keras.Input(shape=(3, 1, 39, 39, 5))  # (batch, time, D, H, W, C)
    resnet_temporal3d = ResidualNetworkLayer(num_blocks=2,
                                             filters=4,
                                             kernel_size=3,
                                             hidden_activation=tf.nn.relu,
                                             output_activation=None,
                                             use_batch_norm=False,
                                             dropout_rate=0.0,
                                             latent_output=False,
                                             output_filters=2,
                                             include_output_layer=True,
                                             network_type="cnn3d",
                                             temporal=True,
                                             name="custom_resnet_temporal3d")
    temporal3d_output = resnet_temporal3d(temporal3d_input)
    model_temporal3d = tf.keras.Model(temporal3d_input, temporal3d_output, name="ResNetModel_Temporal3D")
    model_temporal3d.summary()

    # Describe the output shape and functionality
    print("\nTime Step Module Output Shape:", temporal_output.shape)
    print("The output represents a categorical distribution over", resnet_temporal.number_of_output_bins, "time step bins.")
    print("Each bin represents a specific time step value, and the network predicts the probability of each bin.")
    print("The output can be used to sample time step values or take the expectation for a deterministic value.")
    
