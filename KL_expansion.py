# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 23:30:10 2025

@author: Victor PhD
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

def generate_kl_log_normal_real_params_3D(n_realizations, 
                                          Nx=30, Ny=30, Nz=30,
                                          Lx=100.0, Ly=50.0, Lz=20.0,
                                          real_mean=3.0, real_std=1.0,
                                          corr_length_fac=0.2, 
                                          energy_threshold=0.95,
                                          seed=2000,
                                          reverse_order=False,
                                          cond_values=None,
                                          dtype=np.float32):
    """
    Generate n realizations of a 3D permeability field (strictly positive) modeled as log-normal.
    The underlying Gaussian (log-permeability) field is generated via a Karhunen–Loève (KL) expansion.
    
    The physical permeability parameters are first converted to log-space using:
    
        sigma_log = sqrt( ln(1 + (real_std/real_mean)^2) )
        mu_log    = ln(real_mean) - 0.5 * sigma_log^2
    
    The simulation is performed in x,y,z ordering so that the grid (and the output field)
    has shape (Nx, Ny, Nz), where the first dimension corresponds to x, second to y, third to z.
    
    Optionally, conditional simulation is performed if cond_values is provided. In cond_values,
    keys are tuples of observed indices (x_index, y_index, z_index) in the default ordering
    and values are the corresponding known physical permeability. These observations are used to
    condition the simulated log-space field via a kriging adjustment.
    
    Finally, if reverse_order is True, a final transposeobut is applied so that the output becomes
    (Nz, Ny, Nx) (i.e. z,y,x ordering).
    
    Parameters:
    -----------
    n_realizations : int
        Number of realizations to generate.
    Nx, Ny, Nz : int
        Number of grid points in the x, y, and z directions.
    Lx, Ly, Lz : float
        Physical dimensions along x, y, and z directions.
    real_mean : float
        Mean permeability (physical).
    real_std : float
        Standard deviation of permeability (physical).
    corr_length_fac : float
        Correlation length for the exponential covariance kernel (in log-space).
    energy_threshold : float
        Fraction of the total energy (variance) to capture.
    seed : int, optional
        Seed for reproducibility.
    reverse_order : bool, optional
        If True, perform a final transpose to output fields in (Nz,Ny,Nx) (z, y, x).
    cond_values : dict or None
        Conditional observations as { (x_index, y_index, z_index): value, ... }.
    dtype : numpy.dtype, optional
        Data type for arrays (default: np.float32).
    
    Returns:
    --------
    permeability_fields : ndarray
        Array of shape (n_realizations, Nx, Ny, Nz) by default, or transposed if reverse_order is True.
    num_modes : int
        Number of KL modes used.
    grid : tuple of ndarrays
        The meshgrid arrays (X, Y, Z) corresponding to the simulation in the default ordering.
        X, Y, Z each have shape (Nx, Ny, Nz) by default.
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Convert physical permeability parameters to log-space
    sigma_log = np.sqrt(np.log(1 + (real_std/real_mean)**2)).astype(dtype)
    mu_log = np.log(real_mean) - 0.5 * sigma_log**2
    
    # Determine the correlation length using the correlation length factor
    corr_length = corr_length_fac * np.max((Lx, Ly, Lz)).astype(dtype)
    
    # Create grid points along x, y, z with specified dtype
    x = np.linspace(

0, Lx, Nx, dtype=dtype)
    y = np.linspace(0, Ly, Ny, dtype=dtype)
    z = np.linspace(0, Lz, Nz, dtype=dtype)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')   # Shapes: (Nx, Ny, Nz)
    
    # Flatten grid points in (x,y,z) order
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(dtype)
    
    # Define the exponential covariance function
    def exponential_covariance(p1, p2, corr_length, std):
        dists = cdist(p1, p2, metric='euclidean')
        return (std**2) * np.exp(-dists / corr_length).astype(dtype)
    
    # Compute full covariance matrix (for all grid points)
    C = exponential_covariance(points, points, corr_length, sigma_log)
    
    # Eigen decomposition
    eigvals, eigvecs = eigh(C)
    eigvals = eigvals[::-1].astype(dtype)
    eigvecs = eigvecs[:, ::-1].astype(dtype)
    
    # Determine number of modes to retain
    total_energy = np.sum(eigvals)
    cumulative_energy = np.cumsum(eigvals)
    energy_ratio = cumulative_energy / total_energy
    num_modes = np.searchsorted(energy_ratio, energy_threshold) + 1
    eigvals = eigvals[:num_modes]
    eigvecs = eigvecs[:, :num_modes]
    print(f"Using {num_modes} KL modes to capture {energy_threshold*100:.0f}% of total energy.")
    
    sqrt_eigvals = np.sqrt(eigvals).astype(dtype)
    
    # Generate unconditional log-space realizations
    log_fields = []
    for _ in range(n_realizations):
        xi = np.random.randn(num_modes).astype(dtype)
        log_field = mu_log + (eigvecs @ (sqrt_eigvals * xi))
        log_field = log_field.reshape(X.shape).astype(dtype)  # shape (Nx, Ny, Nz)
        log_fields.append(log_field)
    log_fields = np.array(log_fields, dtype=dtype)
    
    # Apply conditional correction if cond_values is provided
    if cond_values is not None:
        obs_indices = []
        obs_log_vals = []
        for key, val in cond_values.items():
            i, j, k = key
            if 0 <= i < Nx and 0 <= j < Ny and 0 <= k < Nz:
                idx = np.ravel_multi_index((i, j, k), dims=X.shape)
                obs_indices.append(idx)
                obs_log_vals.append(np.log(val).astype(dtype))
            else:
                print(f"Warning: observation index {key} out of bounds and will be skipped.")
        if len(obs_indices) > 0:
            obs_indices = np.array(obs_indices)
            obs_log_vals = np.array(obs_log_vals, dtype=dtype)
            C_obs = C[np.ix_(obs_indices, obs_indices)].astype(dtype)
            C_obs_inv = np.linalg.pinv(C_obs).astype(dtype)
            C_all_obs = C[:, obs_indices].astype(dtype)
            for i in range(n_realizations):
                u = log_fields[i].ravel()
                u_obs = u[obs_indices]
                correction = C_all_obs @ (C_obs_inv @ (obs_log_vals - u_obs))
                u_cond = u + correction
                log_fields[i] = u_cond.reshape(X.shape).astype(dtype)
        else:
            print("No valid observations provided; skipping conditioning.")
    
    # Transform to physical permeability
    permeability_fields = np.exp(log_fields).astype(dtype)
    
    grid = (X, Y, Z)
    # If reverse_order is True, perform a final transpose
    if reverse_order:
        permeability_fields = np.transpose(permeability_fields, axes=(0, 3, 2, 1))  # (Nx,Ny,Nz) becomes (Nz,Ny,Nx)
        X = np.transpose(X, axes=(2, 1, 0))
        Y = np.transpose(Y, axes=(2, 1, 0))
        Z = np.transpose(Z, axes=(2, 1, 0))
        grid = (X, Y, Z)
    
    return permeability_fields, num_modes, grid

import matplotlib.gridspec as gridspec
import math

def plot_realizations_3D(permeability_fields, 
                        z_slices=None,
                        realization_indices=None,
                        Lx=100.0, Ly=50.0, Lz=20.0,
                        title="Permeability Realizations (2D Slice)",
                        scale_method="percentile",
                        vmin=None, vmax=None,
                        cbar_label="Permeability (real scale)"):
    """
    Plots 2D slices (in the x-y plane) at specified z indices from 3D permeability realizations,
    with optional selective realization plotting.

    Parameters:
    -----------
    permeability_fields : ndarray
        Array with shape (n_realizations, Nx, Ny, Nz).
    z_slices : int, list of int, or None
        z-slice index/indices to plot (zero-based along z). If None, the middle slice is used.
    realization_indices : tuple (start, stop, step), list of int, or None
        Realizations to plot. If tuple, uses range(start, stop, step). If list, plots specified indices.
        If None, plots all realizations.
    Lx, Ly, Lz : float
        Physical extents used for setting the "extent" in imshow (x: [0,Lx], y: [0,Ly]).
    title : str
        Title for each figure (the slice index is appended).
    scale_method : str or tuple
        "global", "percentile", or a tuple (vmin, vmax) for the colorbar scaling.
    vmin, vmax : float, optional
        Overrides if scale_method is a tuple.
    cbar_label : str
        Label for the colorbar.
    """
    n_realizations = permeability_fields.shape[0]
    Nx, Ny, Nz = permeability_fields.shape[1:]

    # Handle z_slices
    if z_slices is None:
        z_slices = [Nz // 2]
    elif isinstance(z_slices, int):
        z_slices = [z_slices]
    elif not all(isinstance(z, int) and 0 <= z < Nz for z in z_slices):
        raise ValueError(f"z_slices must be integers in [0, {Nz-1}]")

    # Handle realization_indices
    if realization_indices is None:
        realizations = list(range(n_realizations))
    elif isinstance(realization_indices, tuple):
        start, stop, step = realization_indices
        realizations = [i for i in range(start, stop, step) if 0 <= i < n_realizations]
    elif isinstance(realization_indices, list):
        realizations = [i for i in realization_indices if 0 <= i < n_realizations]
    else:
        raise ValueError("realization_indices must be None, a tuple (start, stop, step), or a list of indices")

    if not realizations:
        raise ValueError("No valid realization indices to plot")

    extent = [0, Lx, 0, Ly]  # For imshow: x from 0 to Lx, y from 0 to Ly

    for slice_index in z_slices:
        # Extract 2D slice (x-y plane) for selected realizations
        slice_fields = permeability_fields[realizations, :, :, slice_index]  # shape: (n_selected, Nx, Ny)

        # Determine colorbar scaling
        if isinstance(scale_method, tuple):
            vmin, vmax = scale_method
        elif scale_method == "global":
            vmin = slice_fields.min()
            vmax = slice_fields.max()
        elif scale_method == "percentile":
            vmin = np.percentile(slice_fields, 2)
            vmax = np.percentile(slice_fields, 98)
        else:
            raise ValueError("scale_method must be 'global', 'percentile', or a (vmin, vmax) tuple")

        # Set up figure layout
        n_selected = len(realizations)
        n_cols = math.ceil(np.sqrt(n_selected))
        n_rows = math.ceil(n_selected / n_cols)
        fig = plt.figure(figsize=(min(n_cols * 3, 10), min(n_rows * 3, 10)))  # Cap figure size
        gs = gridspec.GridSpec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.05], wspace=0.1, hspace=0.2)

        # Plot each selected realization
        for idx, realization_idx in enumerate(realizations):
            row, col = divmod(idx, n_cols)
            ax = plt.subplot(gs[row, col])
            im = ax.imshow(slice_fields[idx].T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Realization {realization_idx + 1}', fontsize=12)

        # Add colorbar
        cax = plt.subplot(gs[:, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=10)

        fig.suptitle(f"{title} (z-slice index = {slice_index} of Lz={Lz})", fontsize=16, y=0.95   )
        plt.tight_layout(rect=[0, 0, 0.98, 0.96])
        plt.show()

def plot_model_3d_grid(permeability_fields, grid, Lx, Ly, Lz, sample_index=0, cmap='viridis'):
    """
    Creates a 3D voxel visualization of one permeability field using the real physical 
    dimensions (Lx, Ly, Lz) and the number of grid divisions (Nx, Ny, Nz). 
    
    This function constructs edge arrays using np.linspace so that voxels cover the full domain.
    Each voxel is colored according to the permeability value.
    
    Parameters:
    -----------
    permeability_fields : ndarray
        Array with shape (n_realizations, Nx, Ny, Nz) in x,y,z ordering.
    grid : tuple of ndarrays
        The meshgrid arrays (X, Y, Z) in x,y,z ordering (each with shape (Nx, Ny, Nz)).
    Lx, Ly, Lz : float
        Real physical extents along the x, y, and z directions.
    sample_index : int, optional
        Index of the realization to plot.
    cmap : str, optional
        Colormap.
    """
    import matplotlib.colors as colors
    # Extract grid dimensions (assumes fields are in x,y,z order)
    X, Y, Z = grid  # each of shape (Nx, Ny, Nz)
    field = permeability_fields[sample_index]  # shape (Nx, Ny, Nz)
    Nx, Ny, Nz = field.shape
    
    # Compute edge arrays directly using the real dimensions
    x_edges = np.linspace(0, Lx, Nx + 1, dtype=field.dtype)
    y_edges = np.linspace(0, Ly, Ny + 1, dtype=field.dtype)
    z_edges = np.linspace(0, Lz, Nz + 1, dtype=field.dtype)
    # Build a full 3D mesh for the voxel edges; these arrays have shape (Nx+1, Ny+1, Nz+1)
    X_edges, Y_edges, Z_edges = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')
    
    # Create a boolean array that is True for every voxel. Its shape must be (Nx, Ny, Nz)
    filled = np.ones(field.shape, dtype=bool)
    
    # Normalize the field values for colormap mapping
    norm = colors.Normalize(vmin=field.min(), vmax=field.max())
    facecolors = plt.cm.get_cmap(cmap)(norm(field))
    
    # Create the 3D voxel plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(X_edges, Y_edges, Z_edges, filled, facecolors=facecolors, edgecolor='k')
    
    # Add thinner colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Permeability", shrink=0.6, orientation='horizontal', aspect=30)
    
    ax.set_title(f"3D Voxel Visualization (Realization {sample_index+1})", fontsize=16)
    
    # Ensure aspect ratio reflects physical dimensions
    ax.set_box_aspect([Lx, Ly, Lz])  # Scale axes according to physical lengths
    ax.set_xlabel("X (Lx)")
    ax.set_ylabel("Y (Ly)")
    ax.set_zlabel("Z (Lz)")
    plt.tight_layout()
    plt.show()

# Example usage:
# Set global DPI for all figures
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

if __name__ == "__main__":
    # Generate 10 realizations of a 3D permeability field using x,y,z ordering
    n_realizations = 100
    permeability_fields_3D, num_modes, grid = generate_kl_log_normal_real_params_3D(
        n_realizations,
        Nx=39, Ny=39, Nz=1,
        Lx=2900.0, Ly=2900.0, Lz=80.0,
        real_mean=3.0, real_std=3,
        corr_length_fac=0.1, energy_threshold=0.95,
        seed=2000,
        reverse_order=False,
        cond_values={(29, 29, 0): 2.0, (29, 9, 0): 1.5, (9, 9, 0): 1.0, (9, 29, 0): 0.5},
        dtype=np.float32
    )

    # Plot specified 2D slices from the 3D realizations (slicing along the z dimension)
    plot_realizations_3D(permeability_fields_3D, realization_indices=(0, 100, 5), z_slices=[0], 
                         Lx=2900.0, Ly=2900.0, Lz=80.0, title="3D Permeability Realizations")
    
    # Create a 3D grid (voxel) visualization of the first realization
    plot_model_3d_grid(permeability_fields_3D, grid, Lx=2900.0, Ly=2900.0, Lz=80.0, sample_index=0, cmap='viridis')