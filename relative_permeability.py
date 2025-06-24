# -*- coding: utf-8 -*-
"""
Created on Wed May 14 02:05:54 2025

@author: User
"""

import tensorflow as tf
import numpy as np

# RelativePermeability class (copied from artifact for completeness)
class RelativePermeability:
    """
    Computes relative permeabilities (krog, krgo) and gas saturation (sg) using Corey functions.
    Takes End_Points and Corey_Exp dictionaries as inputs.
    All computations are performed in TensorFlow graph mode with @tf.function.
    """
    def __init__(self, end_points=None, corey_exponents=None, dtype=tf.float32):
        # Default dictionaries
        default_end_points = {
            'kro_Somax': 0.90, 'krg_Sorg': 0.80, 'krg_Swmin': 0.90,
            'Swmin': 0.22, 'Sorg': 0.2, 'Sgc': 0.05, 'Socr': 0.2, 'So_max': 0.28
        }
        default_corey_exponents = {'nog': 3.0, 'ng': 6.0, 'nw': 2.0}

        # Use provided dictionaries or defaults
        self.end_points = end_points if end_points is not None else default_end_points
        self.corey_exponents = corey_exponents if corey_exponents is not None else default_corey_exponents
        self.dtype = dtype

        # Convert End_Points to tensors
        self.kro_somax = tf.constant(self.end_points.get('kro_Somax', default_end_points['kro_Somax']), dtype=dtype, name='kro_somax')
        self.krg_sorg = tf.constant(self.end_points.get('krg_Sorg', default_end_points['krg_Sorg']), dtype=dtype, name='krg_sorg')
        self.krg_swmin = tf.constant(self.end_points.get('krg_Swmin', default_end_points['krg_Swmin']), dtype=dtype, name='krg_swmin')
        self.swmin = tf.constant(self.end_points.get('Swmin', default_end_points['Swmin']), dtype=dtype, name='swmin')
        self.sorg = tf.constant(self.end_points.get('Sorg', default_end_points['Sorg']), dtype=dtype, name='sorg')
        self.sgc = tf.constant(self.end_points.get('Sgc', default_end_points['Sgc']), dtype=dtype, name='sgc')
        self.socr = tf.constant(self.end_points.get('Socr', default_end_points['Socr']), dtype=dtype, name='socr')
        self.so_max = tf.constant(self.end_points.get('So_max', default_end_points['So_max']) * (1.0 - self.swmin), dtype=dtype, name='so_max')

        # Convert Corey_Exp to tensors
        self.nog = tf.constant(self.corey_exponents.get('nog', default_corey_exponents['nog']), dtype=dtype, name='nog')
        self.ng = tf.constant(self.corey_exponents.get('ng', default_corey_exponents['ng']), dtype=dtype, name='ng')
        self.nw = tf.constant(self.corey_exponents.get('nw', default_corey_exponents['nw']), dtype=dtype, name='nw')
        
        self.one = tf.constant(1.0, dtype=dtype)

    @tf.function
    def compute_krog_krgo(self, sg):
        """
        Computes relative permeabilities krog and krgo for given gas saturation sg.
        Args:
            sg: tf.Tensor, gas saturation (scalar or vector).
        Returns:
            Tuple of (krog, krgo) tensors.
        """
        # Compute base krog and krgo using Corey functions
        so =self.one -sg - self.swmin  # Oil saturation
        krog = self.kro_somax * tf.math.pow((so - self.sorg) / (1.0 - self.swmin - self.sorg), self.nog)
        krgo = self.krg_sorg * tf.math.pow((sg - self.sgc) / (1.0 - self.sgc - self.swmin - self.sorg), self.ng)

        # Compute krog and krgo at maximum liquid dropout
        krog_somax_ld = self.kro_somax * tf.math.pow((self.so_max - self.sorg) / (1.0 - self.swmin - self.sorg), self.nog)
        krgo_somax_ld = self.krg_sorg * tf.math.pow((1.0 - self.swmin - self.so_max - self.sgc) / (1.0 - self.sgc - self.swmin - self.sorg), self.ng)

        # Endpoint saturation checks
        sorg_eff = tf.math.maximum(self.sorg, self.socr)
        krog = tf.where(so <= (self.swmin + sorg_eff), tf.zeros_like(krog), krog)
        krgo = tf.where(sg > (1.0 - (self.swmin + self.sorg)), tf.ones_like(krgo) * self.krg_swmin, krgo)

        # Bound krog and krgo
        krog = tf.math.maximum(tf.math.minimum(krog, self.kro_somax), 0.0)
        krgo = tf.math.maximum(tf.math.minimum(krgo, self.krg_swmin), 0.0)

        return krog, krgo

    @tf.function
    def compute_sg(self, krg_kro, tol=1e-6, max_iter=100):
        """
        Computes gas saturation (sg) given krg/kro ratio using bisection.
        Args:
            krg_kro: tf.Tensor, ratio of krg/kro (scalar or vector).
            tol: float, tolerance for convergence.
            max_iter: int, maximum bisection iterations.
        Returns:
            tf.Tensor, gas saturation sg that satisfies krg_kro = krgo(sg)/krog(sg).
        """
        # Define bounds for sg
        sg_min = self.sgc
        sg_max = 1.0 - self.swmin

        # Initialize bisection bounds
        low = tf.ones_like(krg_kro) * sg_min
        high = tf.ones_like(krg_kro) * sg_max
        mid = (low + high) / 2.0

        def condition(i, low, high, mid):
            return tf.logical_and(i < max_iter, tf.reduce_any(high - low > tol))

        def body(i, low, high, mid):
            # Compute krgo/krog at mid
            krog, krgo = self.compute_krog_krgo(mid)
            ratio = krgo / tf.where(krog > 0.0, krog, tf.ones_like(krog) * 1e-10)  # Avoid division by zero
            # Update bounds: if ratio > krg_kro, sg is too high
            low_new = tf.where(ratio > krg_kro, low, mid)
            high_new = tf.where(ratio > krg_kro, mid, high)
            mid_new = (low_new + high_new) / 2.0
            return i + 1, low_new, high_new, mid_new

        # Run bisection loop
        i = tf.constant(0, dtype=tf.int32)
        _, _, _, sg = tf.while_loop(
            condition,
            body,
            loop_vars=[i, low, high, mid],
            shape_invariants=[
                tf.TensorShape([]),
                krg_kro.shape,
                krg_kro.shape,
                krg_kro.shape
            ]
        )

        # Clip sg to valid range
        sg = tf.clip_by_value(sg, sg_min, sg_max)
        return sg

# Test with shape (4, 1, 39, 39, 1)
if __name__ == '__main__':
    # Define input dictionaries
    end_points = {
        'kro_Somax': 0.90, 'krg_Sorg': 0.80, 'krg_Swmin': 0.90,
        'Swmin': 0.22, 'Sorg': 0.2, 'Sgc': 0.05, 'socr': 0.2, 'so_max': 0.28
    }
    corey_exponents = {'nog': 3.0, 'ng': 6.0, 'nw': 2.0}

    # Initialize RelativePermeability
    rel_perm = RelativePermeability(end_points=end_points, corey_exponents=corey_exponents)

    # Generate sg tensor with shape (4, 1, 39, 39, 1)
    sg_min = rel_perm.sgc.numpy()  # 0.05
    sg_max = (1.0 - rel_perm.swmin.numpy())  # 0.78
    sg_values = tf.linspace(sg_min, sg_max, 39 * 39)  # Linearly spaced values
    sg_values = tf.reshape(sg_values, (1, 1, 39, 39, 1))  # Reshape to (1, 1, 39, 39, 1)
    sg_test = tf.tile(sg_values, [4, 1, 1, 1, 1])  # Tile to (4, 1, 39, 39, 1)
    print("Input sg shape:", sg_test.shape)
    print("Input sg min/max:", tf.reduce_min(sg_test).numpy(), tf.reduce_max(sg_test).numpy())

    # Test compute_krog_krgo
    krog, krgo = rel_perm.compute_krog_krgo(sg_test)
    print("\nTest compute_krog_krgo:")
    print("krog shape:", krog.shape)
    print("krgo shape:", krgo.shape)
    print("krog min/max:", tf.reduce_min(krog).numpy(), tf.reduce_max(krog).numpy())
    print("krgo min/max:", tf.reduce_min(krgo).numpy(), tf.reduce_max(krgo).numpy())
    print("Sample krog [0, 0, 0, 0, 0]:", krog[0, 0, 0, 0, 0].numpy())
    print("Sample krgo [0, 0, 0, 0, 0]:", krgo[0, 0, 0, 0, 0].numpy())

    # Test compute_sg
    krg_kro = krgo / tf.where(krog > 0.0, krog, tf.ones_like(krog) * 1e-10)  # Compute ratios
    sg_computed = rel_perm.compute_sg(krg_kro)
    print("\nTest compute_sg:")
    print("krg_kro shape:", krg_kro.shape)
    print("Computed sg shape:", sg_computed.shape)
    print("Computed sg min/max:", tf.reduce_min(sg_computed).numpy(), tf.reduce_max(sg_computed).numpy())
    print("Sample computed sg [0, 0, 0, 0, 0]:", sg_computed[0, 0, 0, 0, 0].numpy())
    print("Sample input sg [0, 0, 0, 0, 0]:", sg_test[0, 0, 0, 0, 0].numpy())

    # Verify accuracy
    error = tf.reduce_mean(tf.abs(sg_computed - sg_test))
    print("Mean absolute error (computed sg vs input sg):", error.numpy())