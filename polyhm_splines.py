# Adapted: Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# ==============================================================================
import tensorflow as tf
import numpy as np

EPSILON = 1e-10

class PolyharmonicSplineInterpolationLayer(tf.keras.layers.Layer):
    """
    PolyharmonicSplineInterpolationLayer

    This layer performs polyharmonic spline interpolation.
    
    The interpolant is of the form:
          f(x) = Σ_i w_i φ(|x - c_i|) + v^T x + b
    where φ is defined via a polyharmonic radial basis function.
    
    In this design:
      - The control (training) data (experimental data) are provided as 1D tensors.
        For example, train_points is a 1D tensor (of shape [n] or [n,1]) and 
        train_values is a 1D tensor (of shape [n] or [n,1]). These represent the
        experimental relationship.
      - The layer promotes these control points into a batched form of shape [B_c, n, 1],
        where if no batch is provided (B_c = 1), they are assumed to be shared across all queries.
      - The query input is multi-dimensional (e.g. shape [B, 39, 39, 1]) and each query
        value is a scalar (coordinate dimension d = 1).
      - The output is reshaped to match the query spatial dimensions.
    
    All core interpolation functions (_phi, _solve_interpolation, _apply_interpolation)
    are preserved.
    
    Example usage:
    
       # Control data (experimental) are 1D.
       train_points = tf.linspace(0.0, 1.0, 10)   # shape [10]
       train_values = tf.sin(train_points * 6.28)   # shape [10]
       # (They can also be given as shape [10,1].)
       
       # Instantiate the layer.
       # (Internally these will be reshaped to shape [1, 10, 1].)
       spline_layer = PolyharmonicSplineInterpolationLayer(train_points, train_values,
                                                            order=2, regularization_weight=0.001)
       
       # Query data: e.g., an image-like tensor of shape [batch, 39, 39, 1].
       query = tf.random.uniform([4, 39, 39, 1], minval=0.0, maxval=1.0)
       # After interpolation, output shape is [4, 39, 39, k], typically k = 1.
       output = spline_layer(query)
       tf.print("Output shape:", tf.shape(output))
    """
    def __init__(self, train_points, train_values, order=2, regularization_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.regularization_weight = regularization_weight
        
        # Convert control data to tensors.
        # If they are 1D (or 2D with last dimension 1), we add a batch dimension.
        #self.train_points = tf.Variable(train_points, dtype=tf.float32, trainable=False)
        self.train_points = tf.convert_to_tensor(train_points, dtype=tf.float32)
        if self.train_points.shape.rank == 1:
            # Convert shape [n] -> [1, n, 1]
            self.train_points = tf.reshape(self.train_points, [1, -1, 1])
        elif self.train_points.shape.rank == 2:
            # Assume shape [n, 1] -> [1, n, 1]
            self.train_points = tf.expand_dims(self.train_points, axis=0)
        #self.train_values = tf.Variable(train_values, dtype=tf.float32, trainable=False)
        self.train_values = tf.convert_to_tensor(train_values, dtype=tf.float32)
        if self.train_values.shape.rank == 1:
            self.train_values = tf.reshape(self.train_values, [1, -1, 1])
        elif self.train_values.shape.rank == 2:
            self.train_values = tf.expand_dims(self.train_values, axis=0)
        
        # Now, control data have shape [B_c, n, 1] where typically B_c == 1.
        self.w = None
        self.v = None

    @staticmethod
    def _phi(r, order):
        r_safe = tf.maximum(r, EPSILON)
        if order == 1:
            return tf.sqrt(r_safe)
        elif order == 2:
            return 0.5 * r_safe * tf.math.log(r_safe)
        elif order == 4:
            return 0.5 * tf.square(r_safe) * tf.math.log(r_safe)
        elif order % 2 == 0:
            return 0.5 * tf.pow(r_safe, 0.5 * order) * tf.math.log(r_safe)
        else:
            return tf.pow(r_safe, 0.5 * order)

    @staticmethod
    def _cross_squared_distance_matrix(x, y):
        # x: [B, n, d], y: [B, m, d] -> [B, n, m]
        x_norm = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        y_norm = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)
        xy = tf.matmul(x, y, transpose_b=True)
        return x_norm - 2 * xy + tf.transpose(y_norm, [0, 2, 1])

    @staticmethod
    def _pairwise_squared_distance_matrix(x):
        return PolyharmonicSplineInterpolationLayer._cross_squared_distance_matrix(x, x)

    @staticmethod
    def _solve_interpolation(train_points, train_values, order, regularization_weight):
        # train_points: [B, n, d] (here d is 1), train_values: [B, n, k]
        B, n, d = tf.unstack(tf.shape(train_points), num=3)
        k = train_values.shape[-1]
        if d is None or k is None:
            raise ValueError("Control data dimensions must be statically known.")
        
        c = train_points  # [B, n, d]
        f = train_values  # [B, n, k]
        matrix_a = PolyharmonicSplineInterpolationLayer._phi(
            PolyharmonicSplineInterpolationLayer._pairwise_squared_distance_matrix(c),
            order
        )  # [B, n, n]
        if regularization_weight > 0:
            eye = tf.eye(n, batch_shape=[B], dtype=c.dtype)
            matrix_a += regularization_weight * eye
        
        ones = tf.ones([B, n, 1], dtype=c.dtype)
        matrix_b = tf.concat([c, ones], axis=2)  # [B, n, d+1], here d+1 = 2.
        
        left_block = tf.concat([matrix_a, tf.transpose(matrix_b, [0, 2, 1])], axis=1)  # [B, n+d+1, n]
        num_b_cols = matrix_b.shape[2]  # d+1 = 2.
        zeros_rhs = tf.zeros([B, num_b_cols, num_b_cols], dtype=c.dtype)
        right_block = tf.concat([matrix_b, zeros_rhs], axis=1)  # [B, n+d+1, d+1]
        lhs = tf.concat([left_block, right_block], axis=2)  # [B, n+d+1, n+d+1]
        
        zeros_rhs2 = tf.zeros([B, num_b_cols, k], dtype=c.dtype)
        rhs = tf.concat([f, zeros_rhs2], axis=1)  # [B, n+d+1, k]
        
        sol = tf.linalg.solve(lhs, rhs)  # [B, n+d+1, k]
        w = sol[:, :n, :]  # [B, n, k]
        v = sol[:, n:, :]  # [B, d+1, k]
        return w, v

    @staticmethod
    def _apply_interpolation(query_points, train_points, w, v, order):
        # query_points: [B, m, d], train_points: [B, n, d]
        pairwise_dists = PolyharmonicSplineInterpolationLayer._cross_squared_distance_matrix(query_points, train_points)
        phi_vals = PolyharmonicSplineInterpolationLayer._phi(pairwise_dists, order)  # [B, m, n]
        rbf_term = tf.matmul(phi_vals, w)  # [B, m, k]
        ones = tf.ones_like(query_points[..., :1], dtype=train_points.dtype)
        query_points_pad = tf.concat([query_points, ones], axis=2)  # [B, m, d+1]
        linear_term = tf.matmul(query_points_pad, v)  # [B, m, k]
        return rbf_term + linear_term

    def build(self, input_shape):
        # No tensor ops here; only validate shapes or set non-tensor attributes.
        super().build(input_shape)

    @tf.function 
    def call(self, query_points):
        """
        Evaluates the interpolant on query_points.
    
        In this case, query_points are multi-dimensional, e.g. [B, 39, 39, 16],
        while the control (training) data (and thus the interpolation model) are 1D 
        (coordinate dimension 1). We flatten all non-batch dimensions so that 
        the query becomes of shape [B, m_total, 1], apply interpolation, and then 
        reshape the result back to the original shape [B, 39, 39, 16].
    
        Args:
          query_points: A tensor of shape [B, ..., d_query] where d_query may be > 1.
    
        Returns:
          A tensor with the same shape as query_points containing the interpolated values.
        """
        # Get dynamic shape and rank.
        q_shape = tf.shape(query_points)    # e.g., [B, 39, 39, 16]
        q_rank = tf.rank(query_points)
        B = q_shape[0]
        
        # Flatten all dimensions except batch.
        # Here, m_total = product of dims 1...q_rank-1 (e.g., 39*39*16)
        m_total = tf.reduce_prod(q_shape[1:])  
        flat_query = tf.reshape(query_points, [B, m_total, 1])  # Shape: [B, m_total, 1]
        
        # Compute w and v inside call() to prevent out-of-scope errors in TensorFlow graph mode.
        w, v = self._solve_interpolation(self.train_points, self.train_values, self.order, self.regularization_weight)
        
        # Tile control points and weights to match the query batch size.
        # (If control data have batch size 1, they will be broadcasted to match B.)
        train_pts = tf.tile(self.train_points, [B, 1, 1])  # Expected shape: [B, n, 1]
        w = tf.tile(w, [B, 1, 1])                      # Expected shape: [B, n, k] (typically k=1)
        v = tf.tile(v, [B, 1, 1])                      # Expected shape: [B, d+1, k]
        
        # Apply the interpolation method defined in your layer.
        # This should operate on flat inputs of shape [B, m_total, 1] and produce an output [B, m_total, k].
        flat_output = self._apply_interpolation(flat_query, train_pts, w, v, self.order)
        
        # Reshape the flat output back to the original shape.
        # Since we flattened everything except the batch, we expect flat_output.shape[1] = m_total.
        # We then reshape back to q_shape, which is [B, 39, 39, 16] in this example.
        output = tf.reshape(flat_output, q_shape)
        return output

# ---------------------------
# Example test:
if __name__ == '__main__':
    # Simulated control data (experimental).
    # For example, let there be 10 control points and 1D coordinate (d = 1):
    n = 10
    B_control = 1  # control data share a single batch.
    train_points = tf.linspace(0.0, 1.0, n)  # shape [n]
    train_values = tf.sin(train_points * 6.28)  # shape [n]
    # These will be promoted in __init__ to shape [1, n, 1].
    
    # Create query data: multidimensional, e.g., an "image" of shape [B, 39, 39, 1].
    B = 4
    # Set the global seed for reproducibility
    tf.random.set_seed(42)
    
    # Define your parameters
    B = 4  # Example batch size
    
    # Generate random query points with a fixed operation-level seed
    query_points = tf.random.uniform(
        shape=[B, 39, 39, 16],
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
        seed=123  # Operation-level seed
    )
    
    # Create and test the spline layer.
    spline_layer = PolyharmonicSplineInterpolationLayer(train_points, train_values, order=2, regularization_weight=0.001)
    output = spline_layer(query_points)
    
    tf.print("Query shape:", tf.shape(query_points))
    tf.print("Output shape:", tf.shape(output))
    tf.print("Sample output (first batch, first 5 elements):", output[0, :5, :])
