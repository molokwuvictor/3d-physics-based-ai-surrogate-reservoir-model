# Plotting Tools Documentation

This directory contains various plotting and visualization tools for evaluating the performance of the trained AI-based surrogate reservoir models (SRMs).  

## (TODO) Main Plotting Utilities

### `plot_functions.py`

A comprehensive plotting library for visualizing model predictions, errors, and reservoir properties. This consists of the `ModelPlotter` class with the following methods:

- Visualizing 2D and 3D reservoir properties.
- Comparing the AI-based SRM predictions with reservoir simulation data.
- Calculating and visualizing error metrics (absolute, relative L2, RMSE).
- Generating time series predictions for the different grid block indices.

#### Key Features:

- Class object supports multi-realization, multi-timestep reservoir simulations.
- Customizable plotting parameters (colormap, scale, titles).
- Built-in normalization and denormalization functionality for the AI-based model inputs. 
- Effective handling of different property types (pressure, saturation, etc.).

#### Example Usage:

```python
from plot_functions import ModelPlotter

# Initialize plotter with your data
plotter = ModelPlotter(
    features=input_data,  # Shape: (a, b, t, h, w, c)
    labels={'pressure': true_pressure, 'saturation': true_saturation},
    label_ranges={'pressure': (min_p, max_p), 'saturation': (0, 1)},
    label_names={'pressure': 'Pressure (psi)', 'saturation': 'Water Saturation'}
)

# Plot predictions for a specific model and property
plotter.plot_2d_prediction(
    model=pressure_model,
    key='pressure',
    a_indices=0,  # First realization
    b_indices=1,  # Second well scenario
    t_index=5,    # Sixth timestep
    title='Pressure Prediction at Day 150'
)

# Calculate and visualize errors
plotter.plot_error_distribution(model, key='pressure')
```

This folder also include tools for visualizing the training performance metrics, such as the `plot_timestep_log.py`, `loss_history_plotter.py`, and `plot_functions_tf.py` scripts.

### `plot_timestep_log.py`

A specialized script for analyzing the average time step used by the main pressure module in solving the discretized PDEs during training. It processes the logs to extract timestep-specific information and then generates:

- Boxplots showing distribution of values across timesteps.
- Moving average trends for monitored values.
- Statistical summaries of log data.

#### Key Features:

- Automatic extraction of numerical data from log files.
- Configurable sampling size for large logs.
- Support for different visualization modes (boxplots, line plots).

## Other Visualization Utilities

The repository also contains other specialized plotting tools:

- `loss_history_plotter.py`: For visualizing training and validation loss curves
- `plot_functions_tf.py`: TensorFlow-specific visualization utilities 

## Dependencies

These plotting tools depend on the following Python libraries:
- matplotlib
- numpy
- tensorflow (for some components)
- pandas (for data manipulation)



