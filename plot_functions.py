# -*- coding: utf-8 -*-
"""
Created on Fri May  9 02:45:49 2025

@author: User
"""
import math
from typing import List, Tuple, Union, Dict, Optional

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TimeDistributed, Conv2D


def build_cnn_time_model(t: int, h: int, w: int, c_in: int) -> Model:
    """
    Builds a Keras model that accepts an input tensor of shape
      (batch, t, h, w, c_in) and returns (batch, t, h, w, 1).
    Convolutions are applied across the spatial axes (h, w)
    for each time step via TimeDistributed wrapper.
    """
    # Input: batch x t x h x w x c_in
    inp = Input(shape=(t, h, w, c_in), name="input_sequence")

    # Apply per-time-step spatial convolutions
    x = TimeDistributed(
        Conv2D(16, (3, 3), padding="same", activation="relu"),
        name="conv1_per_time"
    )(inp)
    x = TimeDistributed(
        Conv2D(8, (3, 3), padding="same", activation="relu"),
        name="conv2_per_time"
    )(x)

    # Final 1-channel output per spatial location, per time
    out = TimeDistributed(
        Conv2D(1, (1, 1), padding="same", activation="sigmoid"),
        name="reconstruction"
    )(x)

    model = Model(inp, out, name="time_spatial_cnn")
    model.compile(optimizer="adam", loss="mse")
    return model



class ModelPlotter:
    def __init__(
        self,
        model_map: Dict[str, 'tf.keras.Model'],
        test_pairs: List[Tuple[np.ndarray, Union[np.ndarray, Dict[str, np.ndarray]]]],
        batch_size: int = 64,
        dpi: int = 100,
        font_type: str = 'Arial',
        font_size: float = 12.0,
        x_unit_label: str = '',
        y_unit_label: str = '',
    ):
        """
        Initialize the ModelPlotter.

        Args:
            model_map: Dictionary mapping keys (lowercase) to Keras models.
            test_pairs: List of (features, labels) tuples for plotting. Assumes one pair.
            batch_size: Size of batches for data generation (default: 64).
            dpi: Dots per inch for plot resolution (default: 100).
            font_type: Font type for all plot labels (default: 'Arial').
            font_size: Font size for all plot labels (default: 12.0).
            x_unit_label: Unit label for x-axis (default: '').
            y_unit_label: Unit label for y-axis (default: '').
        """
        self.models = {k.lower(): v for k, v in model_map.items()}
        self.test_pairs = test_pairs
        self.batch_size = batch_size
        self.dpi = dpi
        self.font_type = font_type
        self.font_size = font_size
        self.x_unit_label = x_unit_label
        self.y_unit_label = y_unit_label
        self.features, self.labels = test_pairs[0]  # Shape: (a, b, t, h, w, c), (a, b, t, h, w) or dict
        self.n0, self.n1 = self.features.shape[0], self.features.shape[1]  # a_slices, b

        # Set global font settings
        self._update_font_settings()

    def _update_font_settings(self):
        """Update Matplotlib font settings based on current font_type and font_size."""
        plt.rcParams['font.family'] = self.font_type
        plt.rcParams['font.size'] = self.font_size

    def set_font_settings(self, font_size: Optional[float] = None, font_type: Optional[str] = None):
        """
        Update the font size and/or font type and refresh global font settings.

        Args:
            font_size: New font size for all plot labels (default: None, keeps current).
            font_type: New font type for all plot labels (default: None, keeps current).
        """
        if font_size is not None:
            if font_size <= 0:
                raise ValueError("font_size must be positive")
            self.font_size = font_size
        if font_type is not None:
            self.font_type = font_type
        self._update_font_settings()

    def set_unit_labels(self, x_unit_label: Optional[str] = None, y_unit_label: Optional[str] = None):
        """
        Update the x and y unit labels.

        Args:
            x_unit_label: New unit label for x-axis (default: None, keeps current).
            y_unit_label: New unit label for y-axis (default: None, keeps current).
        """
        if x_unit_label is not None:
            self.x_unit_label = x_unit_label
        if y_unit_label is not None:
            self.y_unit_label = y_unit_label

    def _compute_time_points(
        self,
        a_indices: List[int],
        b_indices: List[int],
        time_slice_index: int = -2,
        nonnormalization: Optional[object] = None
    ) -> np.ndarray:
        """
        Compute time points by averaging features over t, h, w, and a specified c channel.

        Args:
            a_indices: List of valid a indices.
            b_indices: List of valid b indices.
            time_slice_index: Index of the c dimension to select (default: -2, time feature).
            nonnormalization: Class with an unnormalize method to unnormalize time values (default: None).

        Returns:
            np.ndarray: Time points with shape (len(a_indices), len(b_indices)).
        """
        a, b, t, h, w, c = self.features.shape
        if not (-c <= time_slice_index < c):
            raise ValueError(f"time_slice_index={time_slice_index} out of bounds for c={c}")

        # Select features for specified a and b indices
        x_subset = self.features[np.ix_(a_indices, b_indices, range(t), range(h), range(w), [time_slice_index])]
        # Average over t, h, w (axes 2, 3, 4)
        time_points = np.mean(x_subset, axis=(2, 3, 4)).squeeze(axis=-1)  # Shape: (len(a_indices), len(b_indices))

        # Unnormalize if a nonnormalization class is provided
        if nonnormalization is not None:
            if not hasattr(nonnormalization, 'unnormalize'):
                raise ValueError("nonnormalization class must have an unnormalize method")
            try:
                time_points = nonnormalization.unnormalize(time_points)
            except Exception as e:
                raise ValueError(f"Failed to unnormalize time points: {str(e)}")

        return time_points

    def plot_line(
        self,
        key: str,
        a_indices: Optional[Union[int, List[int]]] = None,
        b_indices: Optional[Union[int, List[int]]] = None,
        avg: bool = True,
        indices: Optional[List[Tuple[int, int, int]]] = None,
        figsize: Tuple[float, float] = (10, 5),
        color_pred: str = '#1f77b4',  # Blue, professional default
        color_true: str = '#ff7f0e',  # Orange, contrasting default
        linestyle_pred: str = '--',    # Dashed line for predicted
        marker_true: str = 's',       # Square marker for true
        linewidth_pred: float = 1.5,  # Thinner line for predicted
        markersize_true: float = 4.0, # Smaller size for true markers
        markerfacecolor_true: Optional[str] = None,  # Unfilled markers for true
        markeredgecolor_true: Optional[str] = None,  # Matches color_true by default
        time_slice_index: int = -2,   # Default c channel for time feature
        nonnormalization: Optional[object] = None,  # Optional nonnormalization class
        xlabel: str = 'Time',
        ylabel: str = 'Value',
        title: Optional[str] = None,
    ):
        """
        Plot XY line plot of model predictions (as lines) vs true values (as markers) against time points.

        Args:
            key: Model key (case-insensitive).
            a_indices: List of a indices, sampling interval (int), or None for all (default: None).
            b_indices: List of b indices, sampling interval (int), or None for all (default: None).
            avg: If True, plot spatial average; if False, use indices (default: True).
            indices: List of (t, h, w) tuples for specific points (required if avg=False).
            figsize: Figure size (default: (10, 5)).
            color_pred: Color for predicted line (default: '#1f77b4', blue).
            color_true: Color for true markers (default: '#ff7f0e', orange).
            linestyle_pred: Line style for predicted series (default: '--', dashed).
            marker_true: Marker style for true series (default: 's', square).
            linewidth_pred: Line width for predicted series (default: 1.5).
            markersize_true: Marker size for true series (default: 4.0).
            markerfacecolor_true: Face color for true markers (default: None, unfilled).
            markeredgecolor_true: Edge color for true markers (default: None, matches color_true).
            time_slice_index: Index of c dimension for time feature (default: -2).
            nonnormalization: Class with unnormalize method for time points (default: None).
            xlabel: X-axis base label (default: 'Time').
            ylabel: Y-axis base label (default: 'Value').
            title: Plot title (default: None).
        """
        key = key.lower()
        if key not in self.models:
            raise ValueError(f"No model found for specified key '{key}'")
        
        model = self.models[key]
        a, b, t, h, w, c = self.features.shape
        
        # Process a_indices
        if a_indices is None:
            a_indices_list = list(range(a))
        elif isinstance(a_indices, int):
            if a_indices <= 0:
                raise ValueError("a_indices interval must be positive")
            a_indices_list = list(range(0, a, a_indices))
        elif isinstance(a_indices, list):
            valid_a_indices = [idx for idx in a_indices if 0 <= idx < a]
            skipped = set(a_indices) - set(valid_a_indices)
            if skipped:
                print(f"Skipping non-existent a indices: {sorted(skipped)}")
            if not valid_a_indices:
                raise ValueError("No valid a indices provided")
            a_indices_list = valid_a_indices
        else:
            raise ValueError("a_indices must be None, an integer, or a list of integers")
        
        # Process b_indices
        if b_indices is None:
            b_indices_list = list(range(b))
        elif isinstance(b_indices, int):
            if b_indices <= 0:
                raise ValueError("b_indices interval must be positive")
            b_indices_list = list(range(0, b, b_indices))
        elif isinstance(b_indices, list):
            valid_b_indices = [idx for idx in b_indices if 0 <= idx < b]
            skipped = set(b_indices) - set(valid_b_indices)
            if skipped:
                print(f"Skipping non-existent b indices: {sorted(skipped)}")
            if not valid_b_indices:
                raise ValueError("No valid b indices provided")
            b_indices_list = valid_b_indices
        else:
            raise ValueError("b_indices must be None, an integer, or a list of integers")
        
        # Compute time points
        time_points = self._compute_time_points(
            a_indices_list, b_indices_list, time_slice_index, nonnormalization
        )
        
        # Set default markeredgecolor_true to color_true if not specified
        if markeredgecolor_true is None:
            markeredgecolor_true = color_true
        
        # Format axis labels with units
        x_label_final = f"{xlabel} ({self.x_unit_label})" if self.x_unit_label else xlabel
        y_label_final = f"{ylabel} ({self.y_unit_label})" if self.y_unit_label else ylabel
        
        # Plot for each a
        for a_idx_idx, a_idx in enumerate(a_indices_list):
            # Select features for specific a and b_indices
            x_a = self.features[a_idx, b_indices_list, :, :, :, :]  # Shape: (len(b_indices_list), t, h, w, c)
            y_true_a = self.labels[key][a_idx, b_indices_list, :, :, :]  # Shape: (len(b_indices_list), t, h, w)
            
            # Predict for selected b indices
            y_pred_a = model.predict(x_a, batch_size=self.batch_size)  # Shape: (len(b_indices_list), t, h, w, 1)
            y_pred_a = y_pred_a.squeeze(axis=-1)  # Shape: (len(b_indices_list), t, h, w)
            
            # Get time points for this a_idx
            t_a = time_points[a_idx_idx, :]  # Shape: (len(b_indices_list),)
            
            plt.figure(figsize=figsize, dpi=self.dpi)
            if avg:
                # Compute spatial average over (t, h, w)
                true_series = y_true_a.mean(axis=(1, 2, 3))  # Shape: (len(b_indices_list),)
                pred_series = y_pred_a.mean(axis=(1, 2, 3))  # Shape: (len(b_indices_list),)
                plt.plot(
                    t_a, pred_series,
                    label='Predicted', color=color_pred,
                    linestyle=linestyle_pred, linewidth=linewidth_pred
                )
                plt.plot(
                    t_a, true_series,
                    label='True', color=color_true,
                    marker=marker_true, linestyle='none', markersize=markersize_true,
                    markerfacecolor=markerfacecolor_true, markeredgecolor=markeredgecolor_true
                )
            else:
                if not indices or not isinstance(indices, list):
                    raise ValueError("Must supply a non-empty list of (t, h, w) indices when avg=False")
                for idx, (t_idx, h_idx, w_idx) in enumerate(indices):
                    if not (0 <= t_idx < t and 0 <= h_idx < h and 0 <= w_idx < w):
                        print(f"Warning: Index (t={t_idx}, h={h_idx}, w={w_idx}) out of bounds. Skipping.")
                        continue
                    true_pt = y_true_a[:, t_idx, h_idx, w_idx]  # Shape: (len(b_indices_list),)
                    pred_pt = y_pred_a[:, t_idx, h_idx, w_idx]  # Shape: (len(b_indices_list),)
                    plt.plot(
                        t_a, pred_pt,
                        label=f'Pred ({t_idx},{h_idx},{w_idx})',
                        color=plt.cm.tab10(idx % 10),
                        linestyle=linestyle_pred, linewidth=linewidth_pred
                    )
                    plt.plot(
                        t_a, true_pt,
                        label=f'True ({t_idx},{h_idx},{w_idx})',
                        color=plt.cm.tab10(idx % 10),
                        marker=marker_true, linestyle='none', markersize=markersize_true,
                        markerfacecolor=markerfacecolor_true, markeredgecolor=markeredgecolor_true
                    )
            
            plt.xlabel(x_label_final)
            plt.ylabel(y_label_final)
            plt.title(f"{title or 'XY Line Plot'} (a={a_idx})")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_images(
        self,
        key: str,
        b_indices: Optional[Union[int, List[int]]],
        a_indices: Optional[Union[int, List[int]]] = None,
        t_index: int = 0,
        max_rows: int = 5,
        figsize_per: Tuple[float, float] = (4, 4),
        cmap_pred_obs: str = 'viridis',
        cmap_residual: str = 'Blues',
        residual_max: float = 5.0,
        suptitle: Optional[str] = None,
        time_slice_index: int = -2,
        nonnormalization: Optional[object] = None,
    ):
        """
        Plot 2D images of predicted, observed, and residual error with time point labels.

        Args:
            key: Model key (case-insensitive).
            b_indices: List of b indices, sampling interval (int), or None for all.
            a_indices: List of a indices, sampling interval (int), or None for all (default: None).
            t_index: Time step index to plot (default: 0, corresponding to t=1).
            max_rows: Maximum rows per figure for pagination (default: 5).
            figsize_per: Size per subplot (default: (4, 4)).
            cmap_pred_obs: Color map for predicted and observed plots (default: 'viridis').
            cmap_residual: Color map for residual plots (default: 'Blues').
            residual_max: Maximum value for residual color bar (default: 5.0%).
            suptitle: Custom super title for plots (default: None).
            time_slice_index: Index of c dimension for time feature (default: -2).
            nonnormalization: Class with unnormalize method for time points (default: None).
        """
        key = key.lower()
        if key not in self.models:
            raise ValueError(f"No model found for specified key '{key}'")
        
        model = self.models[key]
        a, b, t, h, w, c = self.features.shape
        
        # Process a_indices
        if a_indices is None:
            a_indices_list = list(range(a))
        elif isinstance(a_indices, int):
            if a_indices <= 0:
                raise ValueError("a_indices interval must be positive")
            a_indices_list = list(range(0, a, a_indices))
        elif isinstance(a_indices, list):
            valid_a_indices = [idx for idx in a_indices if 0 <= idx < a]
            skipped = set(a_indices) - set(valid_a_indices)
            if skipped:
                print(f"Skipping non-existent a indices: {sorted(skipped)}")
            if not valid_a_indices:
                raise ValueError("No valid a indices provided")
            a_indices_list = valid_a_indices
        else:
            raise ValueError("a_indices must be None, an integer, or a list of integers")
        
        # Process b_indices
        if b_indices is None:
            b_indices_list = list(range(b))
        elif isinstance(b_indices, int):
            if b_indices <= 0:
                raise ValueError("b_indices interval must be positive")
            b_indices_list = list(range(0, b, b_indices))
        elif isinstance(b_indices, list):
            valid_b_indices = [idx for idx in b_indices if 0 <= idx < b]
            skipped = set(b_indices) - set(valid_b_indices)
            if skipped:
                print(f"Skipping non-existent b indices: {sorted(skipped)}")
            if not valid_b_indices:
                raise ValueError("No valid b indices provided")
            b_indices_list = valid_b_indices
        else:
            raise ValueError("b_indices must be None, an integer, or a list of integers")
        
        if not 0 <= t_index < t:
            raise ValueError(f"t_index={t_index} out of bounds (0 to {t-1})")
        
        # Compute time points
        time_points = self._compute_time_points(
            a_indices_list, b_indices_list, time_slice_index, nonnormalization
        )
        
        for a_idx_idx, a_idx in enumerate(a_indices_list):
            x_selected = self.features[a_idx, b_indices_list, :, :, :, :]  # Shape: (len(b_indices_list), t, h, w, c)
            
            y_pred = model.predict(x_selected, batch_size=self.batch_size)  # Shape: (len(b_indices_list), t, h, w, 1)
            y_pred = y_pred[:, t_index, :, :].squeeze(axis=-1)  # Shape: (len(b_indices_list), h, w)
            
            y_true = self.labels[key][a_idx, b_indices_list, t_index, :, :]  # Shape: (len(b_indices_list), h, w)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                residual = np.abs((y_pred - y_true) / y_true) * 100
                residual[np.isinf(residual)] = np.nan
            
            # Compute average error for each grid
            avg_errors = np.nanmean(residual, axis=(1, 2))  # Shape: (len(b_indices_list),)
            
            obs_min = np.nanmin(y_true)
            obs_max = np.nanmax(y_true)
            residual_min = 0
            residual_max_val = residual_max
            
            pages = math.ceil(len(b_indices_list) / max_rows)
            
            for pg in range(pages):
                start = pg * max_rows
                end = min((pg + 1) * max_rows, len(b_indices_list))
                n_rows = end - start
                
                fig = plt.figure(figsize=(figsize_per[0] * 3.6, figsize_per[1] * n_rows), dpi=self.dpi)
                gs = gridspec.GridSpec(n_rows, 3, wspace=0.3, hspace=0.3)
                
                for i in range(n_rows):
                    b_idx_idx = start + i
                    t_val = time_points[a_idx_idx, b_idx_idx]
                    avg_error = avg_errors[b_idx_idx]
                    
                    ax1 = fig.add_subplot(gs[i, 0])
                    im1 = ax1.imshow(y_pred[b_idx_idx], cmap=cmap_pred_obs, vmin=obs_min, vmax=obs_max)
                    ax1.axis('off')
                    ax1.set_title(f"Predicted time={t_val:.2f}" if i == 0 else f"time={t_val:.2f}")
                    cbar_ax1 = ax1.inset_axes([1.05, 0, 0.05, 1])
                    fig.colorbar(im1, cax=cbar_ax1, label='Value')
                    cbar_ax1.yaxis.label.set_size(self.font_size * 0.8)
                    
                    ax2 = fig.add_subplot(gs[i, 1])
                    im2 = ax2.imshow(y_true[b_idx_idx], cmap=cmap_pred_obs, vmin=obs_min, vmax=obs_max)
                    ax2.axis('off')
                    ax2.set_title(f"Observed time={t_val:.2f}" if i == 0 else f"time={t_val:.2f}")
                    cbar_ax2 = ax2.inset_axes([1.05, 0, 0.05, 1])
                    fig.colorbar(im2, cax=cbar_ax2, label='Value')
                    cbar_ax2.yaxis.label.set_size(self.font_size * 0.8)
                    
                    ax3 = fig.add_subplot(gs[i, 2])
                    im3 = ax3.imshow(residual[b_idx_idx], cmap=cmap_residual, vmin=residual_min, vmax=residual_max_val)
                    ax3.axis('off')
                    ax3.set_title(
                        f"Residual ({avg_error:.2f}%) time={t_val:.2f}" if i == 0 else f"({avg_error:.2f}%) time={t_val:.2f}"
                    )
                    cbar_ax3 = ax1.inset_axes([1.05, 0, 0.05, 1])
                    fig.colorbar(im3, cax=cbar_ax3, label='Residual (%)')
                    cbar_ax3.yaxis.label.set_size(self.font_size * 0.8)
                    
                    ax1.set_ylabel(f'time={t_val:.2f}', rotation=0, labelpad=40, va='center')
                
                title = f"{suptitle} (a={a_idx}, page {pg+1}/{pages})" if suptitle else f"a={a_idx}, page {pg+1}/{pages}"
                fig.suptitle(title, y = 0.925, fontsize = self.font_size*1.2)
                plt.tight_layout()
                plt.show()
                
if __name__ == "__main__":
    # Instantiate models for two keys
    t = 1   # temporal dimension
    h, w, c = 39, 39, 5
    model_map = {
        'a': build_cnn_time_model(t, h, w, c),
        'b': build_cnn_time_model(t, h, w, c),
    }

    # Sanity check:
    dummy_input = np.random.rand(2, t, h, w, c).astype(np.float32)
    out = model_map['a'].predict(dummy_input, batch_size=1)
    print("Sanity check input shape:", dummy_input.shape)
    print("Sanity check output shape:", out.shape)  # expect (2, 23, 39, 39, 1)

    # Define test data groups
    n0, n1, t, h, w, c = 12, 23, 1, 39, 39, 5
    
    # Define two feature-label pairs
    features1 = np.random.rand(n0, n1, t, h, w, c)
    labels1 = {'a': np.random.rand(n0, n1, t, h, w),
               'b': np.random.rand(n0, n1, t, h, w)}
    
    test_pairs = [(features1, labels1), ]
    
    # --- 5) Instantiate and plot! ---
    
    plotter = ModelPlotter(
        model_map=model_map,
        test_pairs=test_pairs,
        
    )
    
    # # Line plot of the spatial average:
    # plotter.plot_line(
    #     key='a',
    #     interval=2,
    #     avg=True,
    #     figsize=(8,4),
    #     title='Model (a) vs True – Spatial Average'
    # )
    
    # Line plot at two specific grid-points:
    plotter.set_unit_labels(x_unit_label='s', y_unit_label='m/s')
    plotter.set_font_settings(font_size=10.0, font_type='Times New Roman')
    plotter.plot_line(
        key='b',
        b_indices = 3,
        avg=False,
        indices=[(0,10,10), (5,20,20)],
        figsize=(8,4),
        title='Model (b) vs True – Selected Points'
    )
    
    # 2D image plot at time‐points [0, 5, 10, 15, 20]:
    plotter.set_font_settings(font_size=16.0, font_type='Times New Roman')
    plotter.plot_images(
        key='a',
        figsize_per = (3.5, 3.5),
        a_indices=[0, 5, 10, ],
        b_indices=[0, 5, 10, 15, 20,],
        suptitle='Pred vs True Images (a)'
    )