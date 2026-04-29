import os
import time
import threading
import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

# These will be injected after import, avoiding circular dependency
_suppress_c_stderr = None
_restore_c_stderr = None
_prediction_error_score = None
_is_int_str = None
get_gpu_info = None
compute_persistence_landscapes = None
compute_cross_layer_wasserstein = None
_HAS_RIPSER = False
_HAS_GUDHI = False

@torch.no_grad()
def compute_layer_jacobi_fields(model, input_ids, device, max_tokens=1024, tokenizer=None):
    """
    Compute the Jacobi field of the space deformation itself at each layer.
    
    For 2D hidden states: returns raw coordinates and output positions for
    direct morphed-space visualization (no PCA, no Jacobian decomposition).
    For 3D/4D hidden states: returns multiple 2D coordinate-slice fields per layer,
    each rendered like the 2D case (warped grid with exact coordinates).
    For 5D+ hidden states: PCA projection with full Jacobian analysis.
    
    This is the coordinator function — delegates to specialized sub-functions.
    """
    model.eval()
    hidden_states = _extract_hidden_states(model, input_ids)
    if hidden_states is None:
        return []

    token_strings = _decode_token_strings(input_ids, tokenizer)
    D = hidden_states[0].shape[-1]
    is_2d = (D == 2)

    projection = _compute_projection_basis(hidden_states, D, is_2d)
    mode = projection.get('mode', 'pca')

    fields = []
    for ell in range(len(hidden_states) - 1):
        h_in, h_out, token_strings_layer = _prepare_layer_pair(
            hidden_states, ell, max_tokens, token_strings
        )
        delta = h_out - h_in

        if mode == '2d':
            h_in_2d, h_out_2d, delta_2d = _project_to_2d(
                h_in, h_out, delta, projection, is_2d=True
            )
            field = _compute_2d_field(
                h_in, h_out, delta, h_in_2d, h_out_2d, delta_2d,
                projection['pca_basis'], ell, token_strings_layer
            )
            fields.append(field)

        elif mode == 'sliced':
            slice_fields = _compute_sliced_fields(
                h_in, h_out, delta, projection, ell, token_strings_layer
            )
            fields.append(slice_fields)

        else:
            # PCA mode (5D+)
            h_in_2d, h_out_2d, delta_2d = _project_to_2d(
                h_in, h_out, delta, projection, is_2d=False
            )
            field = _compute_nd_field(
                h_in, h_out, delta, h_in_2d, delta_2d,
                projection, ell, token_strings_layer
            )
            fields.append(field)

    return fields

def _compute_sliced_fields(h_in, h_out, delta, projection, ell, token_strings_layer):
    """
    Compute Jacobi field dicts for all 2D coordinate slices of a 3D or 4D space.
    
    Each slice is treated like the 2D special case: exact coordinates (no PCA),
    warped grid, token displacement lines, volume change coloring.
    
    Returns a dict with:
      - 'layer': layer index
      - 'is_sliced': True
      - 'D': original dimensionality
      - 'slice_fields': list of field dicts, one per coordinate pair
    """
    T, D = h_in.shape
    slice_pairs = projection['slice_pairs']

    slice_field_list = []
    for dim_a, dim_b in slice_pairs:
        # Extract the 2D slice from full-D hidden states
        h_in_slice = h_in[:, [dim_a, dim_b]].cpu().numpy()
        h_out_slice = h_out[:, [dim_a, dim_b]].cpu().numpy()
        delta_slice = delta[:, [dim_a, dim_b]].cpu().numpy()

        # Build warped grid for this slice
        grid_data = _build_2d_warped_grid(h_in_slice, h_out_slice)

        # Compute per-token volume change using the 2D slice Jacobian
        per_token_volume = _compute_slice_per_token_volume(
            h_in, h_out, dim_a, dim_b, T, D
        )

        # Compute global Jacobian metrics (full-D, shared across slices)
        global_metrics = _compute_global_jacobian_metrics(h_in, h_out, delta, D)

        # Build the field dict (same structure as 2D case, plus slice info)
        pca_basis_placeholder = torch.eye(2, D)
        field = _assemble_2d_field_dict(
            ell, h_in_slice, h_out_slice, delta_slice, per_token_volume,
            grid_data, global_metrics, pca_basis_placeholder,
            token_strings_layer, T
        )
        # Add slice-specific metadata
        field['is_2d'] = True  # render like 2D
        field['is_slice'] = True
        field['slice_dims'] = (dim_a, dim_b)
        field['slice_label'] = f"dim{dim_a}\u00d7dim{dim_b}"
        field['D_original'] = D

        slice_field_list.append(field)

    return {
        'layer': ell,
        'is_sliced': True,
        'D': D,
        'slice_fields': slice_field_list,
        'slice_pairs': slice_pairs,
        # Keep global metrics for the layer title
        'anisotropy': slice_field_list[0]['anisotropy'] if slice_field_list else 1.0,
        'singular_values': slice_field_list[0]['singular_values'] if slice_field_list else np.ones(D),
        'log_det': slice_field_list[0]['log_det'] if slice_field_list else 0.0,
        'token_strings': token_strings_layer,
    }

def _compute_slice_per_token_volume(h_in, h_out, dim_a, dim_b, T, D):
    """
    Compute per-token volume change (det of local 2\u00d72 Jacobian) for a specific
    2D coordinate slice (dim_a, dim_b) of the full-D hidden states.
    
    Uses nearest-neighbor least squares in the full-D space, then extracts
    the 2\u00d72 sub-Jacobian for the selected dimensions.
    """
    per_token_volume = np.ones(T)
    for t in range(T):
        J_local = _estimate_local_jacobian(h_in, h_out, t, T, D, k_min=4)
        # Extract the 2x2 sub-Jacobian for dims (dim_a, dim_b) -> (dim_a, dim_b)
        J_2x2 = J_local[np.ix_([dim_a, dim_b], [dim_a, dim_b])]
        try:
            per_token_volume[t] = abs(np.linalg.det(J_2x2))
        except Exception:
            per_token_volume[t] = 1.0
    return per_token_volume

# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1: Top-level helpers called by the coordinator
# ═══════════════════════════════════════════════════════════════════════════

def _extract_hidden_states(model, input_ids):
    """Run the model forward pass and return hidden states, or None if insufficient."""
    output = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) < 2:
        return None
    return hidden_states


def _decode_token_strings(input_ids, tokenizer):
    """Decode input_ids into human-readable token strings."""
    if tokenizer is None:
        return None
    try:
        flat_ids = input_ids.reshape(-1).tolist()
        return [tokenizer.decode([tid]).strip() or f"<{tid}>" for tid in flat_ids]
    except Exception:
        return None


def _compute_projection_basis(hidden_states, D, is_2d):
    """
    Compute the shared PCA basis (or identity for 2D) used across all layers.
    
    For 2D: identity (no projection needed).
    For 3D/4D: returns slice pairs for direct coordinate-pair visualization.
    For 5D+: PCA projection with full Jacobian analysis.
    
    Returns a dict with keys: pca_basis, mean, var_explained, use_pca, mode,
    and optionally slice_pairs for 3D/4D.
    """
    if is_2d:
        return {
            'pca_basis': torch.eye(2, D),
            'mean': torch.zeros(D),
            'var_explained': 1.0,
            'use_pca': False,
            'mode': '2d',
            'slice_pairs': [(0, 1)],
        }

    if D in (3, 4):
        from itertools import combinations
        slice_pairs = list(combinations(range(D), 2))
        return {
            'pca_basis': torch.eye(2, D),  # placeholder, not used for slicing
            'mean': torch.zeros(D),
            'var_explained': 1.0,
            'use_pca': False,
            'mode': 'sliced',
            'slice_pairs': slice_pairs,
            'D': D,
        }

    # 5D+ case: PCA
    all_points = _gather_all_layer_points(hidden_states)
    pca_basis, mean, var_explained = _fit_shared_pca(all_points)

    return {
        'pca_basis': pca_basis,
        'mean': mean,
        'var_explained': var_explained,
        'use_pca': True,
        'mode': 'pca',
    }

def _prepare_layer_pair(hidden_states, ell, max_tokens, token_strings):
    """
    Extract and flatten the input/output hidden state pair for a layer,
    applying token subsampling if needed.
    
    Returns: (h_in, h_out, token_strings_layer)
    """
    h_in = hidden_states[ell].detach().float().reshape(-1, hidden_states[ell].shape[-1])
    h_out = hidden_states[ell + 1].detach().float().reshape(-1, hidden_states[ell + 1].shape[-1])

    T = h_in.shape[0]
    if T > max_tokens:
        idx = torch.linspace(0, T - 1, max_tokens).long()
        h_in = h_in[idx]
        h_out = h_out[idx]
        token_strings_layer = (
            [token_strings[i] for i in idx.tolist()] if token_strings else None
        )
    else:
        token_strings_layer = token_strings

    return h_in, h_out, token_strings_layer


def _project_to_2d(h_in, h_out, delta, projection, is_2d):
    """Project hidden states to 2D using PCA basis or direct slicing."""
    pca_basis = projection['pca_basis']
    mean = projection['mean']

    if not is_2d:
        h_in_2d = ((h_in.cpu() - mean) @ pca_basis.T).numpy()
        h_out_2d = ((h_out.cpu() - mean) @ pca_basis.T).numpy()
        delta_2d = (delta.cpu() @ pca_basis.T).numpy()
    else:
        h_in_2d = h_in.cpu().numpy()[:, :2]
        h_out_2d = h_out.cpu().numpy()[:, :2]
        delta_2d = delta.cpu().numpy()[:, :2]

    return h_in_2d, h_out_2d, delta_2d


def _compute_2d_field(h_in, h_out, delta, h_in_2d, h_out_2d, delta_2d,
                      pca_basis, ell, token_strings_layer):
    """
    Compute the full Jacobi field dict for the 2D special case.
    Morphed-space visualization with warped grid.
    """
    T, D = h_in.shape

    grid_data = _build_2d_warped_grid(h_in_2d, h_out_2d)
    per_token_volume = _compute_2d_per_token_volume(h_in, h_out, T, D)
    global_metrics = _compute_global_jacobian_metrics(h_in, h_out, delta, D)

    return _assemble_2d_field_dict(
        ell, h_in_2d, h_out_2d, delta_2d, per_token_volume,
        grid_data, global_metrics, pca_basis, token_strings_layer, T
    )


def _compute_nd_field(h_in, h_out, delta, h_in_2d, delta_2d,
                      projection, ell, token_strings_layer):
    """
    Compute the full Jacobi field dict for the 3+D case.
    Full Jacobian decomposition with PCA projection.
    """
    T, D = h_in.shape
    pca_basis = projection['pca_basis']
    var_explained = projection['var_explained']

    per_token_data = _compute_nd_per_token_jacobians(h_in, h_out, pca_basis, T, D)
    global_metrics = _compute_global_jacobian_metrics(h_in, h_out, delta, D)
    grid_data = _build_nd_interpolated_grid(h_in_2d, per_token_data, global_metrics, pca_basis)
    decomposed_grid = _decompose_grid_jacobians(grid_data)
    scalar_metrics = _compute_nd_scalar_metrics(global_metrics, D)

    return _assemble_nd_field_dict(
        ell, h_in_2d, delta_2d, per_token_data, grid_data,
        decomposed_grid, global_metrics, scalar_metrics,
        pca_basis, var_explained, token_strings_layer
    )


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 2: Helpers called by Level 1 functions
# ═══════════════════════════════════════════════════════════════════════════

def _gather_all_layer_points(hidden_states):
    """Flatten and concatenate all hidden states for shared PCA."""
    all_points = []
    for hs in hidden_states:
        pts = hs.detach().float().cpu().reshape(-1, hs.shape[-1])
        all_points.append(pts)
    return torch.cat(all_points, dim=0)


def _fit_shared_pca(all_concat):
    """Fit a 2-component PCA on concatenated points. Returns (basis, mean, var_explained)."""
    mean = all_concat.mean(0)
    centered = all_concat - mean

    try:
        U, S_pca, Vh = torch.linalg.svd(centered, full_matrices=False)
        pca_basis = Vh[:2, :]  # (2, D)
        total_var = (S_pca ** 2).sum()
        var_explained = ((S_pca[:2] ** 2).sum() / total_var).item()
    except Exception:
        pca_basis = torch.eye(2, centered.shape[1])
        var_explained = 0.0

    return pca_basis, mean, var_explained


def _build_2d_warped_grid(h_in_2d, h_out_2d):
    """
    Build a regular grid in input space and warp it through the layer
    using RBF interpolation. Returns a dict with grid arrays.
    """
    bounds = _compute_grid_bounds(h_in_2d, pad_frac=0.2)
    grid_n = 24

    gx = np.linspace(bounds['x_min'], bounds['x_max'], grid_n)
    gy = np.linspace(bounds['y_min'], bounds['y_max'], grid_n)
    grid_x_in, grid_y_in = np.meshgrid(gx, gy)

    grid_x_out, grid_y_out = _warp_grid_rbf(
        h_in_2d, h_out_2d, grid_x_in, grid_y_in, grid_n
    )

    return {
        'grid_x_in': grid_x_in,
        'grid_y_in': grid_y_in,
        'grid_x_out': grid_x_out,
        'grid_y_out': grid_y_out,
        'grid_n': grid_n,
        'x_lim': (bounds['x_min'], bounds['x_max']),
        'y_lim': (bounds['y_min'], bounds['y_max']),
    }


def _compute_2d_per_token_volume(h_in, h_out, T, D):
    """Compute per-token volume change (det of local Jacobian) for 2D case."""
    per_token_volume = np.ones(T)
    for t in range(T):
        J_local = _estimate_local_jacobian(h_in, h_out, t, T, D, k_min=4)
        try:
            per_token_volume[t] = abs(np.linalg.det(J_local))
        except Exception:
            per_token_volume[t] = 1.0
    return per_token_volume


def _compute_global_jacobian_metrics(h_in, h_out, delta, D):
    """
    Compute the global Jacobian (identity + residual) and its SVD metrics.
    Returns a dict with J_global, singular_values, anisotropy, log_det.
    """
    J_global = _fit_global_jacobian(h_in, delta, D)
    singular_values, anisotropy, log_det = _svd_metrics(J_global, D)

    return {
        'J_global': J_global,
        'singular_values': singular_values,
        'anisotropy': anisotropy,
        'log_det': log_det,
    }


def _compute_nd_per_token_jacobians(h_in, h_out, pca_basis, T, D):
    """
    Compute per-token local Jacobians projected to 2D, plus volume,
    rotation, and shear magnitudes.
    """
    P = pca_basis  # (2, D)
    per_token_J_2d = np.zeros((T, 2, 2))
    per_token_volume = np.zeros(T)
    per_token_rotation = np.zeros(T)
    per_token_shear_mag = np.zeros(T)

    k = min(max(D // 2, 8), T - 1)

    for t in range(T):
        J_local = _estimate_local_jacobian_torch(h_in, h_out, t, T, D, k)
        J_2d = (P @ J_local @ P.T).numpy()
        per_token_J_2d[t] = J_2d

        volume, rotation, shear_mag = _decompose_2x2_jacobian(J_2d)
        per_token_volume[t] = volume
        per_token_rotation[t] = rotation
        per_token_shear_mag[t] = shear_mag

    return {
        'per_token_J_2d': per_token_J_2d,
        'per_token_volume': per_token_volume,
        'per_token_rotation': per_token_rotation,
        'per_token_shear_mag': per_token_shear_mag,
    }


def _build_nd_interpolated_grid(h_in_2d, per_token_data, global_metrics, pca_basis):
    """
    Build a grid and interpolate the full 2×2 Jacobian at each grid point
    using RBF interpolation of per-token Jacobians.
    """
    from scipy.interpolate import RBFInterpolator

    bounds = _compute_grid_bounds(h_in_2d, pad_frac=0.2)
    grid_n = 32

    gx = np.linspace(bounds['x_min'], bounds['x_max'], grid_n)
    gy = np.linspace(bounds['y_min'], bounds['y_max'], grid_n)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    per_token_J_2d = per_token_data['per_token_J_2d']
    grid_J = _interpolate_jacobian_field(
        h_in_2d, per_token_J_2d, grid_pts, grid_n,
        global_metrics['J_global'], pca_basis
    )

    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_pts': grid_pts,
        'grid_J': grid_J,
        'grid_n': grid_n,
        'x_lim': (bounds['x_min'], bounds['x_max']),
        'y_lim': (bounds['y_min'], bounds['y_max']),
    }


def _decompose_grid_jacobians(grid_data):
    """
    Decompose the interpolated 2×2 Jacobian at each grid point into
    volume, rotation, shear, stretch directions, and displacement.
    """
    grid_x = grid_data['grid_x']
    grid_y = grid_data['grid_y']
    grid_J = grid_data['grid_J']
    grid_n = grid_data['grid_n']

    result = _init_grid_decomposition_arrays(grid_n)

    for gi in range(grid_n):
        for gj in range(grid_n):
            J_loc = grid_J[gi, gj]
            pos = np.array([grid_x[gi, gj], grid_y[gi, gj]])
            _decompose_single_grid_point(J_loc, pos, gi, gj, result)

    return result


def _compute_nd_scalar_metrics(global_metrics, D):
    """Compute divergence, curl, and shear from the global Jacobian."""
    J_global = global_metrics['J_global']

    divergence = torch.trace(J_global).item()

    J_sym = (J_global + J_global.T) / 2
    J_antisym = (J_global - J_global.T) / 2
    curl = torch.norm(J_antisym, p='fro').item()

    trace_sym = torch.trace(J_sym)
    shear_tensor = J_sym - (trace_sym / D) * torch.eye(D)
    shear = torch.norm(shear_tensor, p='fro').item()

    return {
        'divergence': divergence,
        'curl': curl,
        'shear': shear,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 3: Lowest-level utility functions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_grid_bounds(points_2d, pad_frac=0.2):
    """Compute padded axis limits from 2D points."""
    x_range = points_2d[:, 0]
    y_range = points_2d[:, 1]
    x_span = max(np.ptp(x_range), 0.1)
    y_span = max(np.ptp(y_range), 0.1)
    return {
        'x_min': x_range.min() - pad_frac * x_span,
        'x_max': x_range.max() + pad_frac * x_span,
        'y_min': y_range.min() - pad_frac * y_span,
        'y_max': y_range.max() + pad_frac * y_span,
    }


def _warp_grid_rbf(h_in_2d, h_out_2d, grid_x_in, grid_y_in, grid_n):
    """Warp a grid through the layer mapping using RBF interpolation."""
    from scipy.interpolate import RBFInterpolator

    try:
        rbf_out_x = RBFInterpolator(
            h_in_2d, h_out_2d[:, 0],
            kernel='thin_plate_spline', smoothing=1.0
        )
        rbf_out_y = RBFInterpolator(
            h_in_2d, h_out_2d[:, 1],
            kernel='thin_plate_spline', smoothing=1.0
        )
        grid_pts_in = np.stack([grid_x_in.ravel(), grid_y_in.ravel()], axis=1)
        grid_x_out = rbf_out_x(grid_pts_in).reshape(grid_n, grid_n)
        grid_y_out = rbf_out_y(grid_pts_in).reshape(grid_n, grid_n)
    except Exception:
        grid_x_out = grid_x_in.copy()
        grid_y_out = grid_y_in.copy()

    return grid_x_out, grid_y_out


def _estimate_local_jacobian(h_in, h_out, t, T, D, k_min=4):
    """
    Estimate the local Jacobian at token t using nearest-neighbor least squares.
    Returns a numpy array of shape (D, D).
    """
    dists = torch.norm(h_in - h_in[t], dim=1)
    dists[t] = float('inf')
    k = min(max(k_min, D + 2), T - 1)
    nn_idx = torch.argsort(dists)[:k]

    dh_in_local = (h_in[nn_idx] - h_in[t]).cpu().numpy()
    dh_out_local = (h_out[nn_idx] - h_out[t]).cpu().numpy()

    try:
        J_local, _, _, _ = np.linalg.lstsq(dh_in_local, dh_out_local, rcond=None)
    except Exception:
        J_local = np.eye(D)

    return J_local


def _estimate_local_jacobian_torch(h_in, h_out, t, T, D, k):
    """
    Estimate the local Jacobian at token t using torch least squares.
    Returns a torch tensor of shape (D, D).
    """
    dists = torch.norm(h_in - h_in[t], dim=1)
    dists[t] = float('inf')
    nn_idx = torch.argsort(dists)[:k]

    dh_in = (h_in[nn_idx] - h_in[t]).cpu()
    dh_out = (h_out[nn_idx] - h_out[t]).cpu()

    try:
        J_local = torch.linalg.lstsq(dh_in, dh_out).solution
    except Exception:
        J_local = torch.eye(D)

    return J_local


def _fit_global_jacobian(h_in, delta, D):
    """Fit the global Jacobian as identity + least-squares residual."""
    h_in_c = h_in - h_in.mean(0)
    delta_c = delta - delta.mean(0)

    try:
        J_residual = torch.linalg.lstsq(h_in_c.cpu(), delta_c.cpu()).solution
        J_global = torch.eye(D) + J_residual
    except Exception:
        J_global = torch.eye(D)

    return J_global


def _svd_metrics(J_global, D):
    """Compute singular values, anisotropy, and log-det from a Jacobian."""
    try:
        S_global = torch.linalg.svdvals(J_global)
        singular_values = S_global.cpu().numpy()
        anisotropy = (S_global.max() / S_global.min().clamp(min=1e-10)).item()
        log_det = torch.log(S_global.clamp(min=1e-10)).sum().item()
    except Exception:
        singular_values = np.ones(D)
        anisotropy = 1.0
        log_det = 0.0

    return singular_values, anisotropy, log_det


def _decompose_2x2_jacobian(J_2d):
    """
    Decompose a 2×2 Jacobian into volume change, rotation angle, and shear magnitude.
    Returns: (volume, rotation, shear_mag)
    """
    try:
        U_j, sigma_j, Vht_j = np.linalg.svd(J_2d)
        R_2d = U_j @ Vht_j
    except Exception:
        sigma_j = np.array([1.0, 1.0])
        R_2d = np.eye(2)
        Vht_j = np.eye(2)

    volume = sigma_j[0] * sigma_j[1]
    rotation = np.arctan2(R_2d[1, 0], R_2d[0, 0])

    S_2d = Vht_j.T @ np.diag(sigma_j) @ Vht_j
    S_trace = np.trace(S_2d)
    S_traceless = S_2d - (S_trace / 2.0) * np.eye(2)
    shear_mag = np.linalg.norm(S_traceless, 'fro')

    return volume, rotation, shear_mag


def _interpolate_jacobian_field(h_in_2d, per_token_J_2d, grid_pts, grid_n,
                                J_global, pca_basis):
    """Interpolate per-token 2×2 Jacobians onto a regular grid using RBF."""
    from scipy.interpolate import RBFInterpolator

    grid_J = np.zeros((grid_n, grid_n, 2, 2))

    try:
        for i in range(2):
            for j in range(2):
                rbf = RBFInterpolator(
                    h_in_2d, per_token_J_2d[:, i, j],
                    kernel='thin_plate_spline', smoothing=1.0
                )
                grid_J[:, :, i, j] = rbf(grid_pts).reshape(grid_n, grid_n)
    except Exception:
        J_2d_global = (pca_basis.numpy() @ J_global.numpy() @ pca_basis.numpy().T)
        grid_J[:, :] = J_2d_global

    return grid_J


def _init_grid_decomposition_arrays(grid_n):
    """Initialize all arrays needed for grid-point Jacobian decomposition."""
    return {
        'grid_volume': np.zeros((grid_n, grid_n)),
        'grid_rotation': np.zeros((grid_n, grid_n)),
        'grid_shear': np.zeros((grid_n, grid_n)),
        'grid_max_stretch_val': np.zeros((grid_n, grid_n)),
        'grid_min_stretch_val': np.zeros((grid_n, grid_n)),
        'grid_max_stretch_dir': np.zeros((grid_n, grid_n, 2)),
        'grid_disp_x': np.zeros((grid_n, grid_n)),
        'grid_disp_y': np.zeros((grid_n, grid_n)),
        'grid_Jm1_ex_x': np.zeros((grid_n, grid_n)),
        'grid_Jm1_ex_y': np.zeros((grid_n, grid_n)),
        'grid_Jm1_ey_x': np.zeros((grid_n, grid_n)),
        'grid_Jm1_ey_y': np.zeros((grid_n, grid_n)),
    }


def _decompose_single_grid_point(J_loc, pos, gi, gj, result):
    """Decompose the Jacobian at a single grid point and store into result arrays."""
    F = J_loc - np.eye(2)

    # Displacement
    disp = F @ pos
    result['grid_disp_x'][gi, gj] = disp[0]
    result['grid_disp_y'][gi, gj] = disp[1]

    # Basis vector deformations
    result['grid_Jm1_ex_x'][gi, gj] = F[0, 0]
    result['grid_Jm1_ex_y'][gi, gj] = F[1, 0]
    result['grid_Jm1_ey_x'][gi, gj] = F[0, 1]
    result['grid_Jm1_ey_y'][gi, gj] = F[1, 1]

    # SVD decomposition
    try:
        U_g, sigma_g, Vht_g = np.linalg.svd(J_loc)
        R_g = U_g @ Vht_g
    except Exception:
        sigma_g = np.array([1.0, 1.0])
        R_g = np.eye(2)
        Vht_g = np.eye(2)

    result['grid_volume'][gi, gj] = sigma_g[0] * sigma_g[1]
    result['grid_rotation'][gi, gj] = np.arctan2(R_g[1, 0], R_g[0, 0])
    result['grid_max_stretch_val'][gi, gj] = sigma_g[0]
    result['grid_min_stretch_val'][gi, gj] = sigma_g[1]
    result['grid_max_stretch_dir'][gi, gj] = Vht_g[0]

    # Shear
    S_g = Vht_g.T @ np.diag(sigma_g) @ Vht_g
    S_trace = np.trace(S_g)
    S_traceless = S_g - (S_trace / 2.0) * np.eye(2)
    result['grid_shear'][gi, gj] = np.linalg.norm(S_traceless, 'fro')


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLY: Build the final field dictionaries
# ═══════════════════════════════════════════════════════════════════════════

def _assemble_2d_field_dict(ell, h_in_2d, h_out_2d, delta_2d, per_token_volume,
                            grid_data, global_metrics, pca_basis,
                            token_strings_layer, T):
    """Assemble the output dict for a 2D layer."""
    J_global = global_metrics['J_global']

    return {
        'layer': ell,
        'is_2d': True,
        'h_in_2d': h_in_2d,
        'h_out_2d': h_out_2d,
        'per_token_delta_2d': delta_2d,
        'per_token_volume': per_token_volume,
        # Grid
        'grid_x_in': grid_data['grid_x_in'],
        'grid_y_in': grid_data['grid_y_in'],
        'grid_x_out': grid_data['grid_x_out'],
        'grid_y_out': grid_data['grid_y_out'],
        'grid_n': grid_data['grid_n'],
        'x_lim': grid_data['x_lim'],
        'y_lim': grid_data['y_lim'],
        # Global metrics
        'jacobian': J_global.cpu().numpy(),
        'singular_values': global_metrics['singular_values'],
        'anisotropy': global_metrics['anisotropy'],
        'log_det': global_metrics['log_det'],
        'var_explained': 1.0,
        'use_pca': False,
        'token_strings': token_strings_layer,
        # Compatibility keys
        'divergence': torch.trace(J_global).item(),
        'curl': 0.0,
        'shear': 0.0,
        'per_token_divergence': per_token_volume,
        'per_token_curl': np.zeros(T),
        'per_token_shear': np.zeros(T),
        'per_token_shear_mag': np.zeros(T),
        'per_token_rotation': np.zeros(T),
        'principal_directions': pca_basis.numpy(),
    }

def _assemble_nd_field_dict(ell, h_in_2d, delta_2d, per_token_data, grid_data,
                            decomposed_grid, global_metrics, scalar_metrics,
                            pca_basis, var_explained, token_strings_layer):
    """Assemble the output dict for a 3+D layer."""
    return {
        'layer': ell,
        'is_2d': False,
        'jacobian': global_metrics['J_global'].cpu().numpy(),
        'divergence': scalar_metrics['divergence'],
        'curl': scalar_metrics['curl'],
        'shear': scalar_metrics['shear'],
        'singular_values': global_metrics['singular_values'],
        'log_det': global_metrics['log_det'],
        'anisotropy': global_metrics['anisotropy'],
        # ── Token data ──────────────────────────────────────
        'h_in_2d': h_in_2d,
        'per_token_J_2d': per_token_data['per_token_J_2d'],
        'per_token_volume': per_token_data['per_token_volume'],
        'per_token_rotation': per_token_data['per_token_rotation'],
        'per_token_shear_mag': per_token_data['per_token_shear_mag'],
        'per_token_delta_2d': delta_2d,
        # ── Grid data ───────────────────────────────────────
        'grid_x': grid_data['grid_x'],
        'grid_y': grid_data['grid_y'],
        'grid_J': grid_data['grid_J'],
        'grid_volume': decomposed_grid['grid_volume'],
        'grid_rotation': decomposed_grid['grid_rotation'],
        'grid_shear': decomposed_grid['grid_shear'],
        'grid_max_stretch_val': decomposed_grid['grid_max_stretch_val'],
        'grid_min_stretch_val': decomposed_grid['grid_min_stretch_val'],
        'grid_max_stretch_dir': decomposed_grid['grid_max_stretch_dir'],
        'grid_disp_x': decomposed_grid['grid_disp_x'],
        'grid_disp_y': decomposed_grid['grid_disp_y'],
        'grid_Jm1_ex_x': decomposed_grid['grid_Jm1_ex_x'],
        'grid_Jm1_ex_y': decomposed_grid['grid_Jm1_ex_y'],
        'grid_Jm1_ey_x': decomposed_grid['grid_Jm1_ey_x'],
        'grid_Jm1_ey_y': decomposed_grid['grid_Jm1_ey_y'],
        'x_lim': grid_data['x_lim'],
        'y_lim': grid_data['y_lim'],
        'var_explained': var_explained,
        'use_pca': True,
        # ── Compatibility ───────────────────────────────────
        'per_token_divergence': per_token_data['per_token_volume'],
        'per_token_curl': per_token_data['per_token_rotation'],
        'per_token_shear': per_token_data['per_token_shear_mag'],
        'principal_directions': pca_basis.numpy(),
        'token_strings': token_strings_layer,
    }

class LivePlotter:
    def __init__(self, enabled: bool = True, update_every: int = 5,
                 topo_enabled: bool = False, topo_every: int = 50,
                 topo_max_points: int = 200, topo_pca_dim: int = 30,
                 suppress_window: bool = False, plot_file: str = "training_plot.png",
                 model_info: dict = None, kelp_every: int = 25):
        self.enabled = enabled
        self._model_info = model_info or {}
        self._avg_line_diffs = None
        self._wass_cbar = None  # colorbar handle for Wasserstein heatmap
        self._last_predictions = []
        self.update_every = update_every
        self._global_batch = 0
        self._ema_train = None
        self.suppress_window = suppress_window
        self.plot_file = plot_file
        self._window_closed = False
        self._reopen_requested = False
        self._lock = threading.Lock()
        self._abs_diffs_history: List[List[int]] = []

        self._save_dpi = 150
        self._fig_width = 26    # inches
        self._fig_height = 18   # inches

        # ── Kelp forest config ──────────────────────────────────────────
        self.kelp_every = kelp_every
        self._kelp_step = 0
        self._kelp_data = None
        self._kelp_time_offset = 0.0
        self._jacobi_subaxes = []  # per-layer inset axes for Jacobi PCA
        self._jacobi_data = None
        self._jacobi_drawn_step = -1  # ← ADD THIS: initialize the redraw guard

        # ── TDA config ──────────────────────────────────────────────────
        self.topo_enabled = topo_enabled and _HAS_RIPSER and _HAS_GUDHI and enabled
        self.topo_every = topo_every
        self.topo_max_points = topo_max_points
        self.topo_pca_dim = topo_pca_dim
        self._topo_step = 0
        self._topo_dgms = None
        self._topo_layer_name = ""

        if topo_enabled and not _HAS_GUDHI:
            console.print("[yellow][TopoPlotter] gudhi not installed. "
                          "pip install gudhi  — TDA panels disabled.[/]")

        if not enabled:
            return

        if topo_enabled and not _HAS_RIPSER:
            console.print("[yellow][TopoPlotter] ripser not installed. "
                          "pip install ripser  — TDA panels disabled.[/]")

        _suppress_c_stderr()
        try:
            if suppress_window:
                matplotlib.use("Agg")
            else:
                try:
                    matplotlib.use("TkAgg")
                except ImportError:
                    matplotlib.use("Agg")
                    self.suppress_window = True
                    console.print(
                        "[bold yellow]⚠ Could not load TkAgg backend (no display available). "
                        "Falling back to Agg (headless mode). Plots will be saved to file only.[/]"
                    )

            self.plt = plt
        finally:
            _restore_c_stderr()

        if not suppress_window:
            self.plt.ion()

        # ── Data stores (initialized BEFORE _create_figure) ─────────────
        self.train_epoch_losses: List[float] = []
        self.val_epoch_losses: List[float] = []
        self.batch_ema: List[float] = []
        self.lr_history: List[float] = []

        self.val_batch_raw: List[float] = []
        self._val_epoch_batch_buf: List[float] = []
        self._val_epoch_avg_xs: List[float] = []
        self._val_epoch_avg_ys: List[float] = []

        self._create_figure()

        if not suppress_window:
            self._window_watch_thread = threading.Thread(
                    target=self._watch_window, daemon=True
                    )
            self._window_watch_thread.start()

    def get_state(self) -> dict:
        """Serialize all plotter data for checkpoint saving."""
        return {
            "train_epoch_losses": list(self.train_epoch_losses),
            "val_epoch_losses": list(self.val_epoch_losses),
            "batch_ema": list(self.batch_ema),
            "lr_history": list(self.lr_history),
            "val_batch_raw": list(self.val_batch_raw),
            "_val_epoch_avg_xs": list(self._val_epoch_avg_xs),
            "_val_epoch_avg_ys": list(self._val_epoch_avg_ys),
            "_abs_diffs_history": [list(d) for d in self._abs_diffs_history],
            "_global_batch": self._global_batch,
            "_ema_train": self._ema_train,
            "_kelp_step": self._kelp_step,
            "_kelp_time_offset": self._kelp_time_offset,
            "_topo_step": self._topo_step,
        }

    def restore_state(self, state: dict):
        """Restore plotter data from a checkpoint, then redraw everything."""
        if not state or not self.enabled:
            return

        self.train_epoch_losses = state.get("train_epoch_losses", [])
        self.val_epoch_losses = state.get("val_epoch_losses", [])
        self.batch_ema = state.get("batch_ema", [])
        self.lr_history = state.get("lr_history", [])
        self.val_batch_raw = state.get("val_batch_raw", [])
        self._val_epoch_avg_xs = state.get("_val_epoch_avg_xs", [])
        self._val_epoch_avg_ys = state.get("_val_epoch_avg_ys", [])
        self._abs_diffs_history = state.get("_abs_diffs_history", [])
        self._global_batch = state.get("_global_batch", 0)
        self._ema_train = state.get("_ema_train", None)
        self._kelp_step = state.get("_kelp_step", 0)
        self._kelp_time_offset = state.get("_kelp_time_offset", 0.0)
        self._topo_step = state.get("_topo_step", 0)
        self._jacobi_drawn_step = -1  # ← ADD THIS: force redraw after restore

        # Redraw all lines from restored data
        if self.train_epoch_losses:
            epochs = list(range(1, len(self.train_epoch_losses) + 1))
            self.line_train_epoch.set_data(epochs, self.train_epoch_losses)
            self.line_val_epoch.set_data(epochs, self.val_epoch_losses)
            self.line_lr.set_data(epochs, self.lr_history)

        if self.batch_ema:
            xs = list(range(len(self.batch_ema)))
            self.line_batch_ema.set_data(xs, self.batch_ema)

        if self.val_batch_raw:
            vxs = list(range(len(self.val_batch_raw)))
            self.line_val_raw.set_data(vxs, self.val_batch_raw)
            self.line_val_epoch_avg.set_data(self._val_epoch_avg_xs,
                                             self._val_epoch_avg_ys)

        # Restore the diff scatter plot
        self._restore_data()

        self._refresh()
        console.print("  [green]✓ Plot state restored — graphs continue from where they left off[/]")

    def _refresh(self):
        if not self.enabled:
            return

        if not self.suppress_window:
            self._check_reopen()

        _suppress_c_stderr()
        try:
            for ax in self._plot_axes:
                if ax is self.ax_diffs:
                    continue
                ax.relim()
                ax.autoscale_view()

            # ── Protect Jacobi sub-axes positions ──────────────────
            # Save positions before draw/pause, restore after
            saved_positions = []
            for sub_ax in self._jacobi_subaxes:
                try:
                    saved_positions.append((sub_ax, sub_ax.get_position()))
                except Exception:
                    pass

            if not self.suppress_window and self._is_window_alive():
                self.plt.pause(0.001)
            else:
                self.fig.canvas.draw()

            # Restore Jacobi sub-axes positions after redraw
            for sub_ax, pos in saved_positions:
                try:
                    sub_ax.set_position(pos)
                except Exception:
                    pass
        finally:
            _restore_c_stderr()

    def _flush_canvas(self):
        """
        Force a synchronous canvas repaint.
        Uses draw() + flush_events() instead of draw_idle().
        """
        if not self.enabled:
            return

        _suppress_c_stderr()
        try:
            if not self.suppress_window and self._is_window_alive():
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                self.fig.canvas.draw()
        except Exception:
            # Fallback: try draw_idle which is more forgiving of layout issues
            try:
                self.fig.canvas.draw_idle()
                if not self.suppress_window and self._is_window_alive():
                    self.fig.canvas.flush_events()
            except Exception:
                pass
        finally:
            _restore_c_stderr()

    def _remove_jacobi_subaxes(self):
        """Safely remove all Jacobi inset sub-axes from the figure."""
        for old_ax in self._jacobi_subaxes:
            try:
                old_ax.remove()
            except Exception:
                try:
                    self.fig.delaxes(old_ax)
                except Exception:
                    pass
        self._jacobi_subaxes = []

    @torch.no_grad()
    def update_jacobi_fields(self, model: nn.Module, input_ids: torch.Tensor, tokenizer=None):
        """
        Extract Jacobi fields from the residual stream and render them.
        Called every batch; only recomputes every kelp_every batches.
        """
        if not self.enabled:
            return

        self._kelp_step += 1

        should_draw = True
        if not should_draw:
            return

        was_training = model.training
        model.eval()

        try:
            fields = compute_layer_jacobi_fields(
                model, input_ids, input_ids.device, max_tokens=1024,
                tokenizer=tokenizer
            )

            if not fields:
                return

            # Use _kelp_step as draw_key — guaranteed unique and monotonically
            # increasing, unlike _global_batch which is modified by update_batch
            draw_key = self._kelp_step

            self._jacobi_data = {
                'fields': fields,
                'step': self._kelp_step,
                'draw_key': draw_key,
            }

            # Invalidate the guard so _draw_jacobi_fields will execute
            self._jacobi_drawn_step = -1

            # Draw the new fields
            self._draw_jacobi_fields(
                self.ax_kelp, self._jacobi_data, draw_key
            )

            # Force a synchronous repaint so the new sub-axes actually appear
            self._flush_canvas()

            # ── Save Jacobi data and images ─────────────────────────────
            try:
                import liveplotter as _lp_self
                _run_dir = getattr(_lp_self, 'run_dir', None)

                if _run_dir is not None:
                    jacobi_dir = os.path.join(_run_dir, "jacobi_data")
                    os.makedirs(jacobi_dir, exist_ok=True)
                    step = self._kelp_step

                    for f in fields:
                        layer = f['layer']
                        save_dict = {}
                        for key, val in f.items():
                            if isinstance(val, np.ndarray):
                                save_dict[key] = val
                            elif isinstance(val, (int, float)):
                                save_dict[key] = np.array(val)
                            elif isinstance(val, tuple) and len(val) == 2:
                                save_dict[key] = np.array(val)
                        save_dict['step'] = np.array(step)
                        save_dict['layer_idx'] = np.array(layer)

                        np.savez_compressed(
                            os.path.join(jacobi_dir, f"jacobi_step{step:06d}_layer{layer:02d}.npz"),
                            **save_dict,
                        )

                    try:
                        self._save_jacobi_layer_images(self._jacobi_data)
                    except Exception as e:
                        console.print(f"[yellow]⚠ Could not save Jacobi layer images: {e}[/]")
                else:
                    console.print(f"[red]⚠ Could not save Jacobi data: no run dir found[/]")
            except Exception as e:
                console.print(f"[yellow]⚠ Could not save Jacobi data: {e}[/]")

        except Exception as e:
            import traceback
            console.print(f"[yellow]\u26a0 Jacobi field error at step {self._kelp_step}: {e}[/]")
            traceback.print_exc()
        finally:
            if was_training:
                model.train()

    def _draw_jacobi_fields_sliced(self, ax, fields, step, draw_key):
        """
        Render Jacobi fields for 3D/4D models using multiple 2D coordinate slices.
        
        Layout: rows = layers, columns = coordinate-pair slices.
        Each cell is rendered like the 2D special case (warped grid, tokens, etc.).
        """
        try:
            self.fig.set_tight_layout(False)
        except Exception:
            pass

        n_layers = len(fields)
        first = fields[0]
        D_orig = first['D']
        slice_pairs = first['slice_pairs']
        n_slices = len(slice_pairs)

        # Build title
        slice_labels = [f"d{a}×d{b}" for a, b in slice_pairs]
        ax.set_title(
            f"Jacobi Fields — {D_orig}D Space, {n_slices} coordinate slices "
            f"(step {step})  [{', '.join(slice_labels)}]",
            fontsize=10, fontweight="bold", color="#c0d8e8",
        )

        # Layout: n_layers rows × n_slices columns, plus legend on right
        legend_width_frac = 0.10
        field_width_frac = 1.0 - legend_width_frac

        n_cols = n_slices
        n_rows = min(n_layers, 6)

        pad_x = 0.02
        pad_y = 0.10
        pad_bottom = 0.03

        total_h = 1.0 - pad_y - pad_bottom
        total_w = field_width_frac - 2 * pad_x

        cell_w = total_w / n_cols
        cell_h = total_h / n_rows
        inset_margin = 0.004

        parent_bbox = ax.get_position()
        px0, py0 = parent_bbox.x0, parent_bbox.y0
        pw, ph = parent_bbox.width, parent_bbox.height

        for ell, layer_data in enumerate(fields):
            row = ell % n_rows
            slice_field_list = layer_data['slice_fields']

            for si, sf in enumerate(slice_field_list):
                col = si

                fx = px0 + pw * (pad_x + col * cell_w + inset_margin)
                # Top row (layer 0) at top of panel, bottom row at bottom
                fy = py0 + ph * (pad_bottom + (n_rows - 1 - row) * cell_h + inset_margin)
                fw = pw * (cell_w - 2 * inset_margin)
                fh = ph * (cell_h - 2 * inset_margin)

                sub_ax = self.fig.add_axes([fx, fy, fw, fh])
                sub_ax.set_facecolor("#0d1117")
                sub_ax.set_in_layout(False)
                self._jacobi_subaxes.append(sub_ax)

                # Draw using the 2D panel renderer
                self._draw_single_jacobi_panel_2d(sub_ax, sf, ell, jacobi_data={'draw_key': draw_key})

                # Override the title to include slice info
                dim_a, dim_b = sf['slice_dims']
                layer_label = "Emb→L1" if ell == 0 else f"L{ell}→{ell+1}"
                mean_vol = sf['per_token_volume'].mean() if 'per_token_volume' in sf else 1.0
                aniso_val = sf.get('anisotropy', 1.0)
                sub_ax.set_title(
                    f"{layer_label} [d{dim_a}×d{dim_b}]  det={mean_vol:.2f}  "
                    f"σ₁/σ₂={aniso_val:.1f}",
                    fontsize=6, fontweight="bold", color="#aaccee", pad=2,
                )

        # ── Column labels on the top edge (slice labels) ────────────────
        for si, (dim_a, dim_b) in enumerate(slice_pairs):
            col = si
            label_x = px0 + pw * (pad_x + col * cell_w + cell_w * 0.5)
            label_y = py0 + ph * (pad_bottom + total_h + 0.01)
            ax.text(
                (label_x - px0) / pw, (label_y - py0) / ph,
                f"d{dim_a}×d{dim_b}",
                fontsize=7, fontweight='bold', color='#88aacc',
                ha='center', va='bottom',
                transform=ax.transAxes,
            )

        # ── Row labels on the left edge (layer labels) ──────────────────
        for ell in range(min(n_layers, n_rows)):
            row = ell
            label_y = py0 + ph * (pad_bottom + (n_rows - 1 - row) * cell_h + cell_h * 0.5)
            label_x = px0 + pw * 0.005
            layer_label = "Emb→L1" if ell == 0 else f"L{ell}→{ell+1}"
            ax.text(
                (label_x - px0) / pw, (label_y - py0) / ph,
                layer_label,
                fontsize=7, fontweight='bold', color='#88aacc',
                ha='left', va='center', rotation=90,
                transform=ax.transAxes,
            )

        # ── Legend (same as 2D) ─────────────────────────────────────
        legend_x = px0 + pw * (field_width_frac + 0.01)
        legend_w = pw * (legend_width_frac - 0.02)
        legend_text_y = py0 + ph * 0.03
        legend_text_h = ph * 0.94

        legend_text_ax = self.fig.add_axes(
            [legend_x, legend_text_y, legend_w, legend_text_h]
        )
        legend_text_ax.set_facecolor("#0a0a1a")
        legend_text_ax.set_xlim(0, 1)
        legend_text_ax.set_ylim(0, 1)
        legend_text_ax.axis('off')
        legend_text_ax.set_in_layout(False)
        self._jacobi_subaxes.append(legend_text_ax)

        legend_items = [
            (f"{D_orig}D SLICED VIEW", "", "#c0d8e8", True),
            ("", "", "#000000", False),
            ("Each column = one 2D", "", "#88aacc", False),
            ("coordinate slice", "", "#88aacc", False),
            (f"({n_slices} slices from", "", "#88aacc", False),
            (f" C({D_orig},2) pairs)", "", "#88aacc", False),
            ("Each row = one layer", "", "#88aacc", False),
            ("", "", "#000000", False),
            ("── gray grid", "input space", "#888888", False),
            ("── colored grid", "output space", "#44aaff", False),
            ("", "", "#000000", False),
            ("GRID COLORING", "", "#c0d8e8", True),
            ("red tint", "expansion", "#ff6666", False),
            ("blue tint", "contraction", "#6688ff", False),
            ("green", "~preserving", "#66cc66", False),
            ("", "", "#000000", False),
            ("TOKENS", "", "#c0d8e8", True),
            ("○ hollow", "input pos", "#aaaaaa", False),
            ("● filled", "output pos", "#ffcc44", False),
            ("── line", "displacement", "#ffffff", False),
            ("red ring", "expanding", "#ff4444", False),
            ("blue ring", "contracting", "#4488ff", False),
            ("label", "token text", "#f0f0f0", False),
        ]

        n_items = len(legend_items)
        line_spacing = 0.95 / max(n_items, 1)

        for i, (symbol, desc, color, is_header) in enumerate(legend_items):
            y = 0.97 - i * line_spacing
            if not symbol and not desc:
                continue
            if is_header:
                legend_text_ax.text(
                    0.05, y, symbol,
                    fontsize=7, fontweight='bold', color=color,
                    fontfamily='sans-serif', va='top',
                    transform=legend_text_ax.transAxes,
                )
            else:
                legend_text_ax.text(
                    0.05, y, symbol,
                    fontsize=6, fontweight='bold', color=color,
                    fontfamily='monospace', va='top',
                    transform=legend_text_ax.transAxes,
                )
                if desc:
                    legend_text_ax.text(
                        0.55, y, desc,
                        fontsize=5.5, color='#999999',
                        fontfamily='sans-serif', va='top',
                        transform=legend_text_ax.transAxes,
                    )


    def _draw_jacobi_fields(self, ax, jacobi_data, draw_key):
        """
        Render the Jacobi field visualization per layer.

        For 2D models: shows the actual morphed space (warped grid).
        For 3D/4D models: shows multiple 2D coordinate-slice panels per layer,
            each rendered like the 2D case.
        For 5D+ models: HSV-encoded deformation field with PCA projection.
        """
        if ax is None or jacobi_data is None:
            return

        if self._jacobi_drawn_step == draw_key:
            return
        self._jacobi_drawn_step = draw_key

        self._remove_jacobi_subaxes()

        fields = jacobi_data['fields']
        step = jacobi_data.get('step', draw_key)
        n_layers = len(fields)

        ax.clear()
        ax.set_facecolor("#0a0a1a")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if n_layers == 0:
            ax.text(0.5, 0.5, "Waiting for hidden states...",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    color="#6a9ab8", transform=ax.transAxes)
            return

        # Detect mode from the first field entry
        first = fields[0]
        is_2d = isinstance(first, dict) and first.get('is_2d', False) and not first.get('is_sliced', False)
        is_sliced = isinstance(first, dict) and first.get('is_sliced', False)

        # If the coordinator returned a list-of-dicts with 'slice_fields', it's sliced mode
        if not is_sliced and not is_2d:
            # Check if it's a plain nd field (5D+ PCA)
            is_nd = isinstance(first, dict) and not first.get('is_2d', False)
        else:
            is_nd = False

        from matplotlib.colors import hsv_to_rgb

        # ═══════════════════════════════════════════════════════════════
        # 3D/4D SLICED CASE: multiple 2D coordinate-slice panels per layer
        # ═══════════════════════════════════════════════════════════════
        if is_sliced:
            self._draw_jacobi_fields_sliced(ax, fields, step, draw_key)
            return

        # ═══════════════════════════════════════════════════════════════
        # 2D SPECIAL CASE: morphed-space visualization
        # ═══════════════════════════════════════════════════════════════
        if is_2d:
            var_exp = first.get('var_explained', 0)
            use_pca = first.get('use_pca', True)
            proj_label = "exact 2D — morphed space"

            ax.set_title(
                f"Jacobi Fields — Space Deformation (step {step}, "
                f"{proj_label})",
                fontsize=10, fontweight="bold", color="#c0d8e8",
            )

            try:
                self.fig.set_tight_layout(False)
            except Exception:
                pass

            legend_width_frac = 0.10
            field_width_frac = 1.0 - legend_width_frac
            n_cols = min(n_layers, 6)

            pad_x = 0.02
            pad_y = 0.10
            pad_bottom = 0.03

            total_h = 1.0 - pad_y - pad_bottom
            row_h = total_h

            cell_w = (field_width_frac - 2 * pad_x) / n_cols
            inset_margin = 0.005

            parent_bbox = ax.get_position()
            px0, py0 = parent_bbox.x0, parent_bbox.y0
            pw, ph = parent_bbox.width, parent_bbox.height

            for ell, f in enumerate(fields):
                col = ell % n_cols

                fx = px0 + pw * (pad_x + col * cell_w + inset_margin)
                fy_top = py0 + ph * (pad_bottom + inset_margin)
                fw = pw * (cell_w - 2 * inset_margin)
                fh = ph * (row_h - 2 * inset_margin)

                sub_ax = self.fig.add_axes([fx, fy_top, fw, fh])
                sub_ax.set_facecolor("#0d1117")
                sub_ax.set_in_layout(False)
                self._jacobi_subaxes.append(sub_ax)

                self._draw_single_jacobi_panel_2d(sub_ax, f, ell, jacobi_data)

            # ── Simple text legend for 2D mode ─────────────────────
            legend_x = px0 + pw * (field_width_frac + 0.01)
            legend_w = pw * (legend_width_frac - 0.02)
            legend_text_y = py0 + ph * 0.03
            legend_text_h = ph * 0.94

            legend_text_ax = self.fig.add_axes(
                [legend_x, legend_text_y, legend_w, legend_text_h]
            )
            legend_text_ax.set_facecolor("#0a0a1a")
            legend_text_ax.set_xlim(0, 1)
            legend_text_ax.set_ylim(0, 1)
            legend_text_ax.axis('off')
            legend_text_ax.set_in_layout(False)
            self._jacobi_subaxes.append(legend_text_ax)

            legend_items_2d = [
                ("2D MORPHED SPACE", "", "#c0d8e8", True),
                ("", "", "#000000", False),
                ("── gray grid", "original (input) space", "#888888", False),
                ("── colored grid", "warped (output) space", "#44aaff", False),
                ("", "", "#000000", False),
                ("GRID COLORING", "", "#c0d8e8", True),
                ("red tint", "local expansion", "#ff6666", False),
                ("blue tint", "local contraction", "#6688ff", False),
                ("green", "~volume preserving", "#66cc66", False),
                ("", "", "#000000", False),
                ("TOKENS", "", "#c0d8e8", True),
                ("○ hollow", "input position", "#aaaaaa", False),
                ("● filled", "output position", "#ffcc44", False),
                ("── line", "displacement vector", "#ffffff", False),
                ("red ring", "expanding (det J > 1)", "#ff4444", False),
                ("blue ring", "contracting (det J < 1)", "#4488ff", False),
                ("label", "token text", "#f0f0f0", False),
            ]

            n_items = len(legend_items_2d)
            line_spacing = 0.95 / max(n_items, 1)

            for i, (symbol, desc, color, is_header) in enumerate(legend_items_2d):
                y = 0.97 - i * line_spacing
                if not symbol and not desc:
                    continue
                if is_header:
                    legend_text_ax.text(
                        0.05, y, symbol,
                        fontsize=7, fontweight='bold', color=color,
                        fontfamily='sans-serif', va='top',
                        transform=legend_text_ax.transAxes,
                    )
                else:
                    legend_text_ax.text(
                        0.05, y, symbol,
                        fontsize=6, fontweight='bold', color=color,
                        fontfamily='monospace', va='top',
                        transform=legend_text_ax.transAxes,
                    )
                    legend_text_ax.text(
                        0.55, y, desc,
                        fontsize=5.5, color='#999999',
                        fontfamily='sans-serif', va='top',
                        transform=legend_text_ax.transAxes,
                    )

            return  # Done — skip the 3+D path entirely

        # ═══════════════════════════════════════════════════════════════
        # 5D+ PATH: existing HSV deformation field visualization
        # ═══════════════════════════════════════════════════════════════
        var_exp = first.get('var_explained', 0)
        use_pca = first.get('use_pca', True)
        proj_label = f"PCA var={var_exp:.0%}" if use_pca else "raw 2D (exact)"

        ax.set_title(
            f"Jacobi Fields — Space Deformation (step {step}, "
            f"{proj_label})",
            fontsize=10, fontweight="bold", color="#c0d8e8",
        )

        try:
            self.fig.set_tight_layout(False)
        except Exception:
            pass

        legend_width_frac = 0.12
        field_width_frac = 1.0 - legend_width_frac

        n_cols = min(n_layers, 6)

        pad_x = 0.02
        pad_y = 0.10
        pad_bottom = 0.03

        total_h = 1.0 - pad_y - pad_bottom
        row_h = total_h

        cell_w = (field_width_frac - 2 * pad_x) / n_cols
        inset_margin = 0.005

        parent_bbox = ax.get_position()
        px0, py0 = parent_bbox.x0, parent_bbox.y0
        pw, ph = parent_bbox.width, parent_bbox.height

        for ell, f in enumerate(fields):
            col = ell % n_cols

            fx = px0 + pw * (pad_x + col * cell_w + inset_margin)
            fy_top = py0 + ph * (pad_bottom + inset_margin)
            fw = pw * (cell_w - 2 * inset_margin)
            fh = ph * (row_h - 2 * inset_margin)

            sub_ax = self.fig.add_axes([fx, fy_top, fw, fh])
            sub_ax.set_facecolor("#0d1117")
            sub_ax.set_in_layout(False)
            self._jacobi_subaxes.append(sub_ax)

            self._draw_single_jacobi_panel(sub_ax, f, ell, jacobi_data, hsv_to_rgb)

        # ═══════════════════════════════════════════════════════════════
        # COLOR WHEEL LEGEND (right side)
        # ═══════════════════════════════════════════════════════════════
        self._draw_jacobi_legend(ax, px0, py0, pw, ph,
                                 field_width_frac, legend_width_frac,
                                 hsv_to_rgb, use_pca)

    def _draw_cumulative_panel(self, cum_ax, ref_h_in_2d, cumulative_delta, f, ell, step):
        """
        Draw the cumulative space-movements panel.

        Token positions are ABSOLUTELY FIXED at their layer-0 positions.
        Arrows show the cumulative displacement that space has undergone
        from layer 0 through the current layer.
        """
        from matplotlib.colors import hsv_to_rgb

        n_tokens = ref_h_in_2d.shape[0]

        # ── Axis limits: use the reference positions with some padding ──
        x_range = ref_h_in_2d[:, 0]
        y_range = ref_h_in_2d[:, 1]
        pad_frac = 0.25
        x_span = max(np.ptp(x_range), 0.1)
        y_span = max(np.ptp(y_range), 0.1)
        x_min = x_range.min() - pad_frac * x_span
        x_max = x_range.max() + pad_frac * x_span
        y_min = y_range.min() - pad_frac * y_span
        y_max = y_range.max() + pad_frac * y_span

        # ── Background: dark ────────────────────────────────────────────
        cum_ax.set_facecolor("#0d1117")

        # ── Draw arrows from fixed positions showing cumulative movement ─
        mag = np.sqrt(cumulative_delta[:, 0]**2 + cumulative_delta[:, 1]**2)
        mag_max = max(mag.max(), 1e-10)
        mag_norm = mag / mag_max

        # Color arrows by direction (HSV hue = angle, saturation = magnitude)
        angles = np.arctan2(cumulative_delta[:, 1], cumulative_delta[:, 0])
        hues = (angles + np.pi) / (2 * np.pi)

        for i in range(n_tokens):
            if mag[i] < 1e-8:
                continue  # skip zero-displacement tokens

            # Arrow color from HSV
            h_val = hues[i]
            s_val = 0.85
            v_val = 0.4 + 0.6 * mag_norm[i]
            rgb = hsv_to_rgb(np.array([[[h_val, s_val, v_val]]]))[0, 0]

            alpha = 0.3 + 0.6 * mag_norm[i]

            cum_ax.annotate(
                '',
                xy=(ref_h_in_2d[i, 0] + cumulative_delta[i, 0],
                    ref_h_in_2d[i, 1] + cumulative_delta[i, 1]),
                xytext=(ref_h_in_2d[i, 0], ref_h_in_2d[i, 1]),
                arrowprops=dict(
                    arrowstyle='->', color=rgb,
                    lw=0.8 + 1.2 * mag_norm[i],
                    mutation_scale=8,
                    alpha=alpha,
                ),
                zorder=3,
            )

        # ── Fixed token markers (always at reference positions) ─────────
        cum_ax.scatter(
            ref_h_in_2d[:, 0], ref_h_in_2d[:, 1],
            s=12, color='#ffffff', edgecolors='#3a5a7a',
            linewidths=0.4, zorder=6, alpha=0.9,
        )

        # ── Sparse token labels ─────────────────────────────────────────
        token_strings_f = f.get('token_strings', None)
        n_labels = self._get_n_labels(n_tokens)

        rng = np.random.RandomState(seed=(ell * 17 + 42) & 0xFFFFFFFF)
        label_idx = rng.choice(n_tokens, size=min(n_labels, n_tokens), replace=False)

        for li in label_idx:
            tx, ty = ref_h_in_2d[li, 0], ref_h_in_2d[li, 1]
            if token_strings_f is not None and li < len(token_strings_f):
                label_text = token_strings_f[li]
                if len(label_text) > 10:
                    label_text = label_text[:9] + "\u2026"
            else:
                label_text = f"t{li}"
            if not label_text.strip():
                label_text = "\u2423"

            cum_ax.annotate(
                label_text,
                (tx, ty),
                fontsize=5.5,
                color='#cccccc',
                alpha=0.8,
                fontweight='bold',
                fontfamily='monospace',
                xytext=(3, 3),
                textcoords='offset points',
                zorder=8,
                bbox=dict(
                    boxstyle='round,pad=0.12',
                    facecolor='#0d1117',
                    edgecolor='#2a3a4a',
                    alpha=0.7,
                    linewidth=0.3,
                ),
            )

        # ── Title ───────────────────────────────────────────────────────
        layer_label = "Emb\u2192L1" if ell == 0 else f"L0\u2192L{ell+1}"
        mean_cum_mag = mag.mean()
        max_cum_mag = mag.max()

        cum_ax.set_title(
            f"Cumul. {layer_label}  \u03bc|\u0394|={mean_cum_mag:.3f}  max={max_cum_mag:.3f}",
            fontsize=6, fontweight="bold", color="#88ccaa", pad=2,
        )

        cum_ax.set_xlim(x_min, x_max)
        cum_ax.set_ylim(y_min, y_max)
        cum_ax.set_aspect('auto')
        cum_ax.tick_params(labelsize=0, length=0)
        for spine in cum_ax.spines.values():
            spine.set_color('#1a3a2a')
            spine.set_linewidth(0.5)

    def _draw_single_jacobi_panel(self, sub_ax, f, ell, jacobi_data, hsv_to_rgb):
        """Draw a single layer's Jacobi field panel (the existing top-row visualization)."""
        h_in_2d = f['h_in_2d']
        grid_x = f['grid_x']
        grid_y = f['grid_y']
        grid_volume = f['grid_volume']
        grid_disp_x = f['grid_disp_x']
        grid_disp_y = f['grid_disp_y']
        grid_max_stretch_val = f['grid_max_stretch_val']
        grid_min_stretch_val = f['grid_min_stretch_val']
        grid_max_stretch_dir = f['grid_max_stretch_dir']
        x_lim = f['x_lim']
        y_lim = f['y_lim']
        per_token_volume = f['per_token_volume']
        per_token_shear = f['per_token_shear_mag']

        grid_n = grid_x.shape[0]

        # ── 1. COLORFUL BACKGROUND: HSV encoding of deformation ───
        disp_angle = np.arctan2(grid_disp_y, grid_disp_x)
        disp_mag = np.sqrt(grid_disp_x**2 + grid_disp_y**2)

        mag_norm = np.arcsinh(disp_mag * 2.0)
        mag_max = max(mag_norm.max(), 1e-10)
        mag_norm = mag_norm / mag_max

        H = (disp_angle + np.pi) / (2 * np.pi)
        S_hsv = np.ones_like(H) * 0.85
        V_hsv = 0.15 + 0.85 * mag_norm

        hsv_img = np.stack([H, S_hsv, V_hsv], axis=-1)
        rgb_img = hsv_to_rgb(hsv_img)

        log_vol = np.log(np.clip(grid_volume, 1e-6, None))
        lv_absmax = max(np.abs(log_vol).max(), 1e-6)
        lv_norm = np.clip(log_vol / lv_absmax, -1, 1)

        vol_blend = 0.2
        rgb_img[:, :, 0] = np.clip(
            rgb_img[:, :, 0] + vol_blend * np.clip(lv_norm, 0, 1), 0, 1
        )
        rgb_img[:, :, 2] = np.clip(
            rgb_img[:, :, 2] + vol_blend * np.clip(-lv_norm, 0, 1), 0, 1
        )

        sub_ax.imshow(
            rgb_img,
            extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
            origin='lower', aspect='auto', interpolation='bilinear', alpha=0.9,
        )

        # ── 2. STREAMLINES ────────────────────────────────────────
        gx_1d = np.linspace(x_lim[0], x_lim[1], grid_n)
        gy_1d = np.linspace(y_lim[0], y_lim[1], grid_n)

        try:
            speed = np.sqrt(grid_disp_x**2 + grid_disp_y**2)
            lw = 0.3 + 1.2 * speed / max(speed.max(), 1e-10)
            sub_ax.streamplot(
                gx_1d, gy_1d, grid_disp_x, grid_disp_y,
                color='white', linewidth=lw, density=0.7,
                arrowsize=0.6, arrowstyle='->', zorder=2, minlength=0.2,
            )
        except Exception:
            pass

        # ── 3. STRETCH WHISKERS ───────────────────────────────────
        whisker_step = max(1, grid_n // 8)
        x_span = x_lim[1] - x_lim[0]
        y_span = y_lim[1] - y_lim[0]
        whisker_len = min(x_span, y_span) / (grid_n / whisker_step) * 0.3

        for gi in range(whisker_step // 2, grid_n, whisker_step):
            for gj in range(whisker_step // 2, grid_n, whisker_step):
                cx = grid_x[gi, gj]
                cy = grid_y[gi, gj]
                d = grid_max_stretch_dir[gi, gj]
                s_max = grid_max_stretch_val[gi, gj]
                s_min = grid_min_stretch_val[gi, gj]

                ratio = s_max / max(s_min, 1e-10)
                if ratio < 1.05:
                    continue

                length = whisker_len * min(ratio - 1.0, 3.0) / 3.0

                if ratio < 1.5:
                    wcolor, walpha = '#ffff44', 0.4
                elif ratio < 3.0:
                    wcolor, walpha = '#ff8800', 0.6
                else:
                    wcolor, walpha = '#ff2222', 0.8

                x0 = cx - length * d[0]
                y0 = cy - length * d[1]
                x1 = cx + length * d[0]
                y1 = cy + length * d[1]

                sub_ax.plot(
                    [x0, x1], [y0, y1],
                    color=wcolor, linewidth=1.0, alpha=walpha,
                    zorder=4, solid_capstyle='round',
                )

        # ── 4. TOKEN MARKERS ─────────────────────────────────────
        shear_max = max(per_token_shear.max(), 1e-6)
        shear_norm = per_token_shear / shear_max

        sub_ax.scatter(
            h_in_2d[:, 0], h_in_2d[:, 1],
            s=10, c=shear_norm, cmap='magma',
            vmin=0, vmax=1, edgecolors='none', zorder=6, alpha=0.9,
        )

        # ── 4b. SPARSE TOKEN LABELS ──────────────────────────────
        token_strings_f = f.get('token_strings', None)
        n_tokens = h_in_2d.shape[0]

        fig_dpi = self.fig.dpi
        fig_w_in, fig_h_in = self.fig.get_size_inches()
        # Estimate panel pixel size from figure-fraction coords
        # (fw and fh are in figure fraction; we don't have them here,
        #  so use a reasonable estimate based on sub_ax position)
        pos = sub_ax.get_position()
        panel_w_px = pos.width * fig_w_in * fig_dpi
        panel_h_px = pos.height * fig_h_in * fig_dpi
        panel_diag_px = math.sqrt(panel_w_px**2 + panel_h_px**2)

        base_font = panel_diag_px * 0.020
        token_scale = max(0.6, min(1.5, 25.0 / max(n_tokens, 1)))
        label_fontsize = max(7.0, min(18.0, base_font * token_scale))

        # ── How many labels ───────────────────────────────────
        n_labels = self._get_n_labels(n_tokens)
        if panel_diag_px < 200:
            n_labels = min(n_labels, 6)

        # ── Seed: shifts each render, but stable within a layer
        draw_key = jacobi_data.get('draw_key', 0)
        rng = np.random.RandomState(
            seed=(draw_key * 7 + ell * 31) & 0xFFFFFFFF
        )
        label_idx = rng.choice(
            n_tokens, size=min(n_labels, n_tokens), replace=False
        )

        offset_px = max(3.0, label_fontsize * 0.7)

        for li in label_idx:
            tx, ty = h_in_2d[li, 0], h_in_2d[li, 1]

            # ── Human-readable label ──────────────────────────
            if token_strings_f is not None and li < len(token_strings_f):
                label_text = token_strings_f[li]
                if len(label_text) > 14:
                    label_text = label_text[:13] + "…"
            else:
                label_text = f"t{li}"

            # Whitespace-only → visible placeholder
            if not label_text.strip():
                label_text = "␣"

            sub_ax.annotate(
                label_text,
                (tx, ty),
                fontsize=label_fontsize,
                color='#f0f0f0',
                alpha=0.92,
                fontweight='bold',
                fontfamily='monospace',
                xytext=(offset_px, offset_px),
                textcoords='offset points',
                zorder=8,
                bbox=dict(
                    boxstyle=f'round,pad={max(0.12, label_fontsize * 0.018):.2f}',
                    facecolor='#0d1117',
                    edgecolor='#3a5a7a',
                    alpha=0.75,
                    linewidth=0.4,
                ),
            )

        # ═══════════════════════════════════════════════════════
        # 5. LAYER LABEL
        # ═══════════════════════════════════════════════════════
        layer_label = "Emb→L1" if ell == 0 else f"L{ell}→{ell+1}"
        mean_vol = per_token_volume.mean()
        aniso_val = f['anisotropy']

        sub_ax.set_title(
            f"{layer_label}  det={mean_vol:.2f}  σ₁/σₙ={aniso_val:.1f}",
            fontsize=7, fontweight="bold", color="#aaccee", pad=2,
        )

        sub_ax.set_xlim(*x_lim)
        sub_ax.set_ylim(*y_lim)
        sub_ax.set_aspect('auto')
        sub_ax.tick_params(labelsize=0, length=0)
        for spine in sub_ax.spines.values():
            spine.set_color('#2a3a4a')
            spine.set_linewidth(0.5)

    def _draw_cumulative_panel(self, cum_ax, ref_h_in_2d, cumulative_delta, f, ell, step):
        """
        Draw the cumulative space-movements panel.
        
        Token positions are ABSOLUTELY FIXED at their layer-0 positions.
        Arrows show the cumulative displacement that space has undergone
        from layer 0 through the current layer.
        """
        from matplotlib.colors import hsv_to_rgb

        n_tokens = ref_h_in_2d.shape[0]

        # ── Axis limits: use the reference positions with some padding ──
        x_range = ref_h_in_2d[:, 0]
        y_range = ref_h_in_2d[:, 1]
        pad_frac = 0.25
        x_span = max(np.ptp(x_range), 0.1)
        y_span = max(np.ptp(y_range), 0.1)
        x_min = x_range.min() - pad_frac * x_span
        x_max = x_range.max() + pad_frac * x_span
        y_min = y_range.min() - pad_frac * y_span
        y_max = y_range.max() + pad_frac * y_span

        # ── Background: dark ────────────────────────────────────────────
        cum_ax.set_facecolor("#0d1117")

        # ── Draw arrows from fixed positions showing cumulative movement ─
        mag = np.sqrt(cumulative_delta[:, 0]**2 + cumulative_delta[:, 1]**2)
        mag_max = max(mag.max(), 1e-10)
        mag_norm = mag / mag_max

        # Color arrows by direction (HSV hue = angle, saturation = magnitude)
        angles = np.arctan2(cumulative_delta[:, 1], cumulative_delta[:, 0])
        hues = (angles + np.pi) / (2 * np.pi)

        for i in range(n_tokens):
            if mag[i] < 1e-8:
                continue  # skip zero-displacement tokens

            # Arrow color from HSV
            h_val = hues[i]
            s_val = 0.85
            v_val = 0.4 + 0.6 * mag_norm[i]
            rgb = hsv_to_rgb(np.array([[[h_val, s_val, v_val]]]))[0, 0]

            alpha = 0.3 + 0.6 * mag_norm[i]

            cum_ax.annotate(
                '',
                xy=(ref_h_in_2d[i, 0] + cumulative_delta[i, 0],
                    ref_h_in_2d[i, 1] + cumulative_delta[i, 1]),
                xytext=(ref_h_in_2d[i, 0], ref_h_in_2d[i, 1]),
                arrowprops=dict(
                    arrowstyle='->', color=rgb,
                    lw=0.8 + 1.2 * mag_norm[i],
                    mutation_scale=8,
                    alpha=alpha,
                ),
                zorder=3,
            )

        # ── Fixed token markers (always at reference positions) ─────────
        cum_ax.scatter(
            ref_h_in_2d[:, 0], ref_h_in_2d[:, 1],
            s=12, color='#ffffff', edgecolors='#3a5a7a',
            linewidths=0.4, zorder=6, alpha=0.9,
        )

        # ── Sparse token labels ─────────────────────────────────────────
        token_strings_f = f.get('token_strings', None)
        n_labels = self._get_n_labels(n_tokens)

        rng = np.random.RandomState(seed=(ell * 17 + 42) & 0xFFFFFFFF)
        label_idx = rng.choice(n_tokens, size=min(n_labels, n_tokens), replace=False)

        for li in label_idx:
            tx, ty = ref_h_in_2d[li, 0], ref_h_in_2d[li, 1]
            if token_strings_f is not None and li < len(token_strings_f):
                label_text = token_strings_f[li]
                if len(label_text) > 10:
                    label_text = label_text[:9] + "…"
            else:
                label_text = f"t{li}"
            if not label_text.strip():
                label_text = "␣"

            cum_ax.annotate(
                label_text,
                (tx, ty),
                fontsize=5.5,
                color='#cccccc',
                alpha=0.8,
                fontweight='bold',
                fontfamily='monospace',
                xytext=(3, 3),
                textcoords='offset points',
                zorder=8,
                bbox=dict(
                    boxstyle='round,pad=0.12',
                    facecolor='#0d1117',
                    edgecolor='#2a3a4a',
                    alpha=0.7,
                    linewidth=0.3,
                ),
            )

        # ── Title ───────────────────────────────────────────────────────
        layer_label = "Emb→L1" if ell == 0 else f"L0→L{ell+1}"
        mean_cum_mag = mag.mean()
        max_cum_mag = mag.max()

        cum_ax.set_title(
            f"Cumul. {layer_label}  μ|Δ|={mean_cum_mag:.3f}  max={max_cum_mag:.3f}",
            fontsize=6, fontweight="bold", color="#88ccaa", pad=2,
        )

        cum_ax.set_xlim(x_min, x_max)
        cum_ax.set_ylim(y_min, y_max)
        cum_ax.set_aspect('auto')
        cum_ax.tick_params(labelsize=0, length=0)
        for spine in cum_ax.spines.values():
            spine.set_color('#1a3a2a')
            spine.set_linewidth(0.5)

    def _draw_jacobi_legend(self, ax, px0, py0, pw, ph,
                            field_width_frac, legend_width_frac,
                            hsv_to_rgb, use_pca):
        """Draw the color wheel legend and text legend on the right side."""
        legend_x = px0 + pw * (field_width_frac + 0.01)
        legend_w = pw * (legend_width_frac - 0.02)

        wheel_size = min(legend_w, ph * 0.18)
        wheel_y = py0 + ph * 0.78
        wheel_ax = self.fig.add_axes(
            [legend_x + (legend_w - wheel_size) * 0.5, wheel_y,
             wheel_size, wheel_size]
        )
        wheel_ax.set_facecolor("#0a0a1a")
        wheel_ax.set_in_layout(False)
        self._jacobi_subaxes.append(wheel_ax)

        # Build the color wheel as a Cartesian RGBA image
        wheel_res = 128
        wx = np.linspace(-1, 1, wheel_res)
        wy = np.linspace(-1, 1, wheel_res)
        WX, WY = np.meshgrid(wx, wy)
        W_angle = np.arctan2(WY, WX)
        W_radius = np.sqrt(WX**2 + WY**2)

        W_H = (W_angle + np.pi) / (2 * np.pi)
        W_S = np.ones_like(W_H) * 0.85
        W_V = 0.15 + 0.85 * np.clip(W_radius, 0, 1)

        mask = W_radius > 1.0
        W_S[mask] = 0.0
        W_V[mask] = 0.0

        hsv_wheel = np.stack([W_H, W_S, W_V], axis=-1)
        rgb_wheel = hsv_to_rgb(hsv_wheel)

        alpha_channel = np.ones((wheel_res, wheel_res))
        alpha_channel[mask] = 0.0
        rgba_wheel = np.concatenate(
            [rgb_wheel, alpha_channel[:, :, np.newaxis]], axis=-1
        )

        wheel_ax.imshow(
            rgba_wheel,
            extent=[-1.0, 1.0, -1.0, 1.0],
            origin='lower',
            aspect='equal',
            interpolation='bilinear',
        )

        # Direction labels
        if use_pca:
            lbl_right, lbl_left = '+PC1\n→', '←\n−PC1'
            lbl_up, lbl_down = '↑ +PC2', '↓ −PC2'
        else:
            lbl_right, lbl_left = '+dim0\n→', '←\n−dim0'
            lbl_up, lbl_down = '↑ +dim1', '↓ −dim1'

        lbl_cfg = dict(fontsize=6.5, fontweight='bold', color='white',
                       ha='center', va='center')
        wheel_ax.text(1.35, 0.0, lbl_right, **lbl_cfg)
        wheel_ax.text(-1.35, 0.0, lbl_left, **lbl_cfg)
        wheel_ax.text(0.0, 1.35, lbl_up, **lbl_cfg)
        wheel_ax.text(0.0, -1.35, lbl_down, **lbl_cfg)

        wheel_ax.text(0.0, 0.0, 'weak', fontsize=5, color='#666666',
                      ha='center', va='center', fontstyle='italic')

        wheel_ax.set_xlim(-1.7, 1.7)
        wheel_ax.set_ylim(-1.7, 1.7)
        wheel_ax.axis('off')

        wheel_ax.set_title("Deformation\nDirection & Strength",
                           fontsize=8, fontweight='bold', color='#c0d8e8',
                           pad=8)

        # ═══════════════════════════════════════════════════════════════
        # TEXT LEGEND below the color wheel
        # ═══════════════════════════════════════════════════════════════
        legend_text_y = py0 + ph * 0.03
        legend_text_h = ph * 0.73
        legend_text_ax = self.fig.add_axes(
            [legend_x, legend_text_y, legend_w, legend_text_h]
        )
        legend_text_ax.set_facecolor("#0a0a1a")
        legend_text_ax.set_xlim(0, 1)
        legend_text_ax.set_ylim(0, 1)
        legend_text_ax.axis('off')
        legend_text_ax.set_in_layout(False)
        self._jacobi_subaxes.append(legend_text_ax)

        legend_items = [
            ("TOP ROW", "", "#c0d8e8", True),
            ("██ Hue", "deformation direction", "#ffffff", False),
            ("██ Bright", "strong deformation", "#dddddd", False),
            ("██ Dark", "weak / no deformation", "#555555", False),
            ("██ Red tint", "expanding (det J > 1)", "#ff6666", False),
            ("██ Blue tint", "contracting (det J < 1)", "#6688ff", False),
            ("", "", "#000000", False),
            ("OVERLAYS", "", "#c0d8e8", True),
            ("── white", "streamlines (flow)", "#ffffff", False),
            ("── yellow", "mild anisotropic stretch", "#ffff44", False),
            ("── orange", "moderate stretch", "#ff8800", False),
            ("── red", "extreme stretch", "#ff2222", False),
            ("", "", "#000000", False),
            ("BOTTOM ROW", "", "#88ccaa", True),
            ("→ arrows", "cumulative displacement", "#88ccaa", False),
            ("● white", "fixed token positions", "#ffffff", False),
            ("hue", "direction of movement", "#aaddcc", False),
            ("length", "magnitude of movement", "#aaddcc", False),
            ("", "", "#000000", False),
            ("TOKENS", "", "#c0d8e8", True),
            ("● dot", "shear magnitude (magma)", "#dd6644", False),
        ]

        n_items = len(legend_items)
        line_spacing = 0.95 / max(n_items, 1)

        for i, (symbol, desc, color, is_header) in enumerate(legend_items):
            y = 0.97 - i * line_spacing
            if not symbol and not desc:
                continue

            if is_header:
                legend_text_ax.text(
                    0.05, y, symbol,
                    fontsize=7, fontweight='bold', color=color,
                    fontfamily='sans-serif', va='top',
                    transform=legend_text_ax.transAxes,
                )
            else:
                legend_text_ax.text(
                    0.05, y, symbol,
                    fontsize=6, fontweight='bold', color=color,
                    fontfamily='monospace', va='top',
                    transform=legend_text_ax.transAxes,
                )
                legend_text_ax.text(
                    0.55, y, desc,
                    fontsize=5.5, color='#999999',
                    fontfamily='sans-serif', va='top',
                    transform=legend_text_ax.transAxes,
                )


    def _redraw_jacobi(self):
        """
        Redraw the Jacobi fields from cached data.
        Called from _restore_data after figure recreation.
        """
        if self._jacobi_data is None:
            return
        if not hasattr(self, 'ax_kelp') or self.ax_kelp is None:
            console.print("[yellow]\u26a0 Jacobi: ax_kelp is None, cannot draw[/]")
            return
        try:
            # Invalidate the guard so the redraw actually happens
            self._jacobi_drawn_step = -1
            self._draw_jacobi_fields(
                self.ax_kelp, self._jacobi_data,
                self._jacobi_data.get('draw_key', self._kelp_step)
            )
            self._flush_canvas()
        except Exception as e:
            import traceback
            console.print(f"[yellow]\u26a0 Jacobi redraw error: {e}[/]")
            traceback.print_exc()


    def accumulate_predictions(self, predictions: list):
        """Accumulate predictions across batches (call clear_predictions() at epoch start)."""
        if not self.enabled:
            return
        self._last_predictions.extend(predictions)
        # Keep only the most recent 20 for display
        if len(self._last_predictions) > 20:
            self._last_predictions = self._last_predictions[-20:]
        self._draw_predictions()
        self._refresh()

    def clear_predictions(self):
        """Clear accumulated predictions (call at start of each epoch)."""
        self._last_predictions = []

    # ── Figure layout helpers ───────────────────────────────────────────

    def _build_figure_and_gridspec(self):
        """Create the figure and return named axes.

        Layout (3 rows × 3 cols):
          Row 0: [epoch loss] [batch loss (train EMA)] [learning rate]
          Row 1: [jacobi fields ──────────────────────] [val batch loss]
          Row 2: [predictions ────────────] [pred diffs] [model info]
        """
        fig = self.plt.figure(figsize=(26, 18))
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                              height_ratios=[1, 1.3, 1])
        axes = {
            "epoch":   fig.add_subplot(gs[0, 0]),
            "batch":   fig.add_subplot(gs[0, 1]),
            "lr":      fig.add_subplot(gs[0, 2]),
            "kelp":    fig.add_subplot(gs[1, 0:2]),   # jacobi fields — wide
            "val":     fig.add_subplot(gs[1, 2]),      # val batch loss — right
            "preds":   fig.add_subplot(gs[2, 0]),
            "diffs":   fig.add_subplot(gs[2, 1]),
            "info":    fig.add_subplot(gs[2, 2]),
            "barcode": None,
            "bd":      None,
        }
        return fig, axes

    def _setup_diffs_axis(self, ax):
        """Configure the prediction error score axis (no legend)."""
        ax.set_title("Prediction Error Score (0=perfect, .1=tiny diff, .9=huge diff, 1=unparseable)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Prediction Update #")
        #ax.set_ylabel(""0)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        self.ax_diffs = ax
        self._scatter_diffs = None

    def _setup_kelp_axis(self, ax):
        """Configure the Jacobi field axis with placeholder text."""
        ax.set_title("Jacobi Fields — Per-Layer Space Deformation",
                      fontsize=10, fontweight="bold", color="#c0d8e8")
        ax.set_facecolor("#0a0a1a")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "Waiting for hidden states...",
                ha="center", va="center", fontsize=11, alpha=0.4,
                color="#6a9ab8", transform=ax.transAxes)
        self.ax_kelp = ax
        self._jacobi_subaxes = []

    def _setup_epoch_axis(self, ax):
        """Configure the epoch loss axis and create its line artists."""
        ax.set_title("Epoch Loss", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        self.line_train_epoch, = ax.plot(
            [], [], label="Train", color="steelblue", linewidth=2,
        )
        self.line_val_epoch, = ax.plot(
            [], [], label="Val", color="tomato", linewidth=2,
        )
        ax.legend(loc="upper right", fontsize=8)

    def _setup_batch_axis(self, ax):
        """Configure the batch EMA loss axis and create its line artist."""
        ax.set_title("Batch Loss (Train EMA)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        self.line_batch_ema, = ax.plot(
            [], [], label="Train EMA", color="steelblue", linewidth=2,
        )
        ax.legend(loc="upper right", fontsize=8)

    def _setup_lr_axis(self, ax):
        """Configure the learning rate axis and create its line artist."""
        ax.set_title("Learning Rate", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, -3))
        self.line_lr, = ax.plot(
            [], [], label="LR", color="seagreen", linewidth=2,
        )
        ax.legend(loc="upper right", fontsize=8)

    def _setup_val_axis(self, ax):
        """Configure the validation batch loss axis and create its line artists."""
        ax.set_title("Batch Loss (Val)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Global Val Batch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        self.line_val_raw, = ax.plot(
            [], [], label="Val Batch", color="tomato", linewidth=1.0, alpha=0.5,
        )
        self.line_val_epoch_avg, = ax.plot(
            [], [], label="Val Epoch Avg", color="darkred", linewidth=2.0,
            marker="o", markersize=4,
        )
        ax.legend(loc="upper right", fontsize=8)

    def _setup_preds_axis(self, ax):
        """Configure the predictions table panel."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Predictions (Training + Validation samples)",
                      fontsize=10, fontweight="bold")

    def _setup_info_axis(self, ax):
        """Configure the model info text panel."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Model / System Info", fontsize=10, fontweight="bold")

    def _collect_plot_axes(self, named_axes):
        data_axes = [
            named_axes["epoch"],
            named_axes["batch"],
            named_axes["lr"],
            named_axes["val"],
        ]
        if self.ax_diffs is not None:
            data_axes.append(self.ax_diffs)
        return data_axes

    def _apply_figure_chrome(self):
        """Set suptitle, window title, tight_layout, close event, and force draw."""
        self.fig.suptitle("LLVM IR GPT Training", fontsize=14, fontweight="bold")
        try:
            self.fig.canvas.manager.set_window_title("LLVM IR GPT \u2014 Live Training")
        except Exception:
            pass

        # Apply tight_layout ONCE for initial positioning of the grid axes,
        # then immediately disable it so it doesn't fight with manually
        # positioned Jacobi sub-axes on subsequent draw() calls.
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        try:
            self.fig.set_tight_layout(False)
        except Exception:
            pass

        if not self.suppress_window:
            self.fig.canvas.mpl_connect("close_event", self._on_close)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)


    # ── The director ────────────────────────────────────────────────────

    def _create_figure(self):
        """Create (or recreate) the matplotlib figure and all axes."""
        # 1. Build the figure skeleton
        self.fig, named_axes = self._build_figure_and_gridspec()

        # 2. Store optional topo axes
        self.ax_barcode = named_axes["barcode"]
        self.ax_bd = named_axes["bd"]

        # 3. Configure each axis
        self._setup_diffs_axis(named_axes["diffs"])
        self._jacobi_subaxes = []
        # Do NOT reset _jacobi_data here — preserve cached data for restore
        self._jacobi_drawn_step = -1  # force redraw on next draw call
        self._setup_kelp_axis(named_axes["kelp"])
        self._setup_epoch_axis(named_axes["epoch"])
        self._setup_batch_axis(named_axes["batch"])
        self._setup_lr_axis(named_axes["lr"])
        self._setup_val_axis(named_axes["val"])
        self._setup_preds_axis(named_axes["preds"])
        self._setup_info_axis(named_axes["info"])

        # 4. Store text-only axes for later redraws
        self.ax_preds = named_axes["preds"]
        self.ax_info = named_axes["info"]

        # 5. Build the list of data-bearing axes (for relim/autoscale)
        self._plot_axes = self._collect_plot_axes(named_axes)
        self.axes = np.array(self._plot_axes)

        # 6. Populate text panels with current data
        self._draw_predictions()
        self._draw_model_info()

        # 7. Apply window chrome, layout, and event bindings
        self._apply_figure_chrome()

        # 8. Restore any accumulated data onto the new figure
        self._restore_data()

    def _restore_data(self):
        """Re-plot all accumulated data onto current line objects."""
        self._draw_predictions()
        self._draw_model_info()

        # Restore diff plot (scatter only)
        if hasattr(self, 'ax_diffs') and self.ax_diffs is not None and self._abs_diffs_history:
            n_updates = len(self._abs_diffs_history)
            max_scatter_window = 1_000_000

            scatter_start = max(0, n_updates - max_scatter_window)
            all_scatter_x = []
            all_scatter_y = []
            for i in range(scatter_start, n_updates):
                for s in self._abs_diffs_history[i]:
                    all_scatter_x.append(i)
                    all_scatter_y.append(s)

            if all_scatter_x:
                n_scatter = len(all_scatter_x)
                scatter_alpha = 0.30 if n_scatter < 500 else (0.15 if n_scatter < 2000 else 0.08)
                scatter_size = 14 if n_scatter < 500 else (10 if n_scatter < 2000 else 6)
                self._scatter_diffs = self.ax_diffs.scatter(
                    all_scatter_x, all_scatter_y,
                    s=scatter_size, alpha=scatter_alpha, color="steelblue", zorder=1,
                    edgecolors="none",
                )

            x_lo = max(scatter_start - 0.5, -0.5)
            self.ax_diffs.set_xlim(x_lo, max(n_updates - 0.5, 0.5))
            self.ax_diffs.set_ylim(-0.05, 1.05)

        # Restore average line
        if self._abs_diffs_history:
            avg_xs = list(range(scatter_start, n_updates))
            avg_ys = [
                sum(self._abs_diffs_history[i]) / len(self._abs_diffs_history[i])
                for i in range(scatter_start, n_updates)
            ]
            if avg_xs:
                smoothed_ys = []
                ema = avg_ys[0]
                alpha = max(0.01, min(0.2, 50.0 / max(len(avg_ys), 1)))
                for y in avg_ys:
                    ema = alpha * y + (1 - alpha) * ema
                    smoothed_ys.append(ema)

                self._avg_line_diffs, = self.ax_diffs.plot(
                    avg_xs, smoothed_ys,
                    color="black", linewidth=1.8, alpha=0.85, zorder=3,
                )

    def _on_close(self, event):
        """Called when the user closes the matplotlib window."""
        with self._lock:
            self._window_closed = True
        console.print(
                "\n[bold yellow]📊 Plot window closed. "
                "Press [bold white]r[/bold white] to reopen it.[/]\n"
                )

    def _watch_window(self):
        """Background thread: periodically pumps the matplotlib event loop
        so that close events are detected even between batch updates."""
        while True:
            time.sleep(0.5)
            # We do NOT touch the flags here.
            # The main thread's _check_reopen() is the sole consumer.

    def request_reopen(self):
        """Call this (e.g. from EpochController) to request window reopen."""
        with self._lock:
            if self._window_closed:
                self._reopen_requested = True

    def _check_reopen(self):
        """Check if we need to reopen the window. Must be called from main thread."""
        with self._lock:
            needs_reopen = self._reopen_requested and self._window_closed
            if needs_reopen:
                self._reopen_requested = False
                self._window_closed = False

        if needs_reopen:
            console.print("[bold green]📊 Reopening plot window...[/]")
            self.plt.ion()
            self._create_figure()

    def _is_window_alive(self) -> bool:
        """Check if the figure window is still open."""
        with self._lock:
            return not self._window_closed

    # ── Save figure to file ─────────────────────────────────────────────
    def _save_to_file(self):
        """Save the current figure to disk at a fixed size, independent of window state."""
        if not self.enabled:
            return
        try:
            save_path = self.plot_file
            if run_dir is not None:
                save_path = os.path.join(run_dir, self.plot_file)

            # ── Rotate: move current latest → next numbered history file ──
            if os.path.exists(save_path):
                base, ext = os.path.splitext(save_path)
                seq = 1
                while os.path.exists(f"{base}-{seq:08d}{ext}"):
                    seq += 1
                history_path = f"{base}-{seq:08d}{ext}"
                os.rename(save_path, history_path)

            # ── Force a fixed size regardless of interactive window state ──
            original_size = self.fig.get_size_inches()
            self.fig.set_size_inches(self._fig_width, self._fig_height)

            self.fig.savefig(
                save_path,
                dpi=self._save_dpi,
                bbox_inches="tight",
                facecolor=self.fig.get_facecolor(),
                edgecolor="none",
            )

            # ── Restore the interactive window size so display isn't affected ──
            if not self.suppress_window:
                self.fig.set_size_inches(original_size)

        except Exception as e:
            pass  # Don't crash training for a file write error

    # ── Batch update (train) ────────────────────────────────────────────
    def update_batch(self, batch_loss: float):
        if not self.enabled:
            return
        self._global_batch += 1

        alpha = 0.05
        if self._ema_train is None:
            self._ema_train = batch_loss
        else:
            self._ema_train = alpha * batch_loss + (1 - alpha) * self._ema_train
        self.batch_ema.append(self._ema_train)

        if self._global_batch % self.update_every == 0:
            xs = list(range(len(self.batch_ema)))
            self.line_batch_ema.set_data(xs, self.batch_ema)
            self._refresh()
            self._save_to_file()  # ← ADD THIS: re-save plot after each batch update

    # ── Val batch ───────────────────────────────────────────────────────
    def update_val_batch(self, val_batch_loss: float):
        if not self.enabled:
            return
        self.val_batch_raw.append(val_batch_loss)
        self._val_epoch_batch_buf.append(val_batch_loss)

    def finish_val_epoch(self):
        if not self.enabled:
            return
        if self._val_epoch_batch_buf:
            avg = sum(self._val_epoch_batch_buf) / len(self._val_epoch_batch_buf)
            end_x = len(self.val_batch_raw) - 1
            start_x = end_x - len(self._val_epoch_batch_buf) + 1
            mid_x = (start_x + end_x) / 2.0
            self._val_epoch_avg_xs.append(mid_x)
            self._val_epoch_avg_ys.append(avg)
            self._val_epoch_batch_buf = []

        vxs = list(range(len(self.val_batch_raw)))
        self.line_val_raw.set_data(vxs, self.val_batch_raw)
        self.line_val_epoch_avg.set_data(self._val_epoch_avg_xs,
                                         self._val_epoch_avg_ys)

    # ── Epoch update ────────────────────────────────────────────────────
    def update_epoch(self, train_loss: float, val_loss: float, lr: float):
        if not self.enabled:
            return

        self.train_epoch_losses.append(train_loss)
        self.val_epoch_losses.append(val_loss)
        self.lr_history.append(lr)

        epochs = list(range(1, len(self.train_epoch_losses) + 1))
        self.line_train_epoch.set_data(epochs, self.train_epoch_losses)
        self.line_val_epoch.set_data(epochs, self.val_epoch_losses)
        self.line_lr.set_data(epochs, self.lr_history)

        self._refresh()

        # Always save to file after each epoch
        self._save_to_file()

    # ════════════════════════════════════════════════════════════════════
    # TDA methods (unchanged from original)
    # ════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def update_topo(self, model: nn.Module, input_ids: torch.Tensor):
        """
        Compute per-layer persistence landscapes and cross-layer Wasserstein
        heatmap, then render them onto the two TDA panels.
        """
        if not self.topo_enabled:
            return

        self._topo_step += 1
        if self._topo_step != 1 and self._topo_step % self.topo_every != 0:
            return

        was_training = model.training
        model.eval()

        try:
            output = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states
            if hidden_states is None or len(hidden_states) < 2:
                return

            # ── Persistence Landscapes (replaces barcode) ───────────────
            n_landscapes = 5
            resolution = 100
            landscapes, sample_range = compute_persistence_landscapes(
                    hidden_states,
                    n_landscapes=n_landscapes,
                    resolution=resolution,
                    max_points=self.topo_max_points,
                    pca_dim=self.topo_pca_dim,
                    )
            self._draw_persistence_landscapes(
                    landscapes, sample_range, n_landscapes, resolution,
                    )

            # ── Wasserstein Heatmap (replaces birth/death) ──────────────
            W = compute_cross_layer_wasserstein(
                    hidden_states,
                    max_points=self.topo_max_points,
                    pca_dim=self.topo_pca_dim,
                    )
            self._draw_wasserstein_heatmap(W)

            self._refresh()

        except Exception as e:
            # Silently skip — don't crash training for a viz failure
            import traceback
            traceback.print_exc()
        finally:
            if was_training:
                model.train()

    def update_prediction_diffs(self, predictions: list):
        if not self.enabled:
            return

        scores = []

        for expected, predicted, _ in predictions:
            # ── FIX: Clean expected the same way as predicted ───────────
            exp_cleaned = ''.join(
                c for c in expected if c.isascii() and c.isprintable()
            ).strip()

            try:
                exp_val = int(exp_cleaned)
            except (ValueError, TypeError, AttributeError):
                # Skip predictions where expected is unparseable
                # (this was the source of the "empty expected" bug)
                continue
            # ── END FIX ─────────────────────────────────────────────────

            pred_cleaned = ''.join(
                c for c in predicted if c.isascii() and c.isprintable()
            ).strip()

            try:
                pred_val = int(pred_cleaned)
            except (ValueError, TypeError):
                # Garbage, but shorter garbage is slightly less wrong
                garbage_len = len(pred_cleaned)
                # Exponential decay: short → ~0.93, long → ~1.0
                length_discount = min(0.07, 0.07 * math.exp(-0.3 * max(garbage_len, 0)))
                scores.append(1.0 - length_discount)
                continue

            scores.append(_prediction_error_score(exp_val, pred_val))

        if not scores:
            return

        self._abs_diffs_history.append(scores)

        if not hasattr(self, 'ax_diffs') or self.ax_diffs is None:
            return

        ax = self.ax_diffs
        n_updates = len(self._abs_diffs_history)

        max_scatter_window = 1_000_000

        if self._scatter_diffs is not None:
            try:
                self._scatter_diffs.remove()
            except (ValueError, AttributeError):
                pass
            self._scatter_diffs = None

        # Remove old average line if it exists
        if hasattr(self, '_avg_line_diffs') and self._avg_line_diffs is not None:
            try:
                self._avg_line_diffs.remove()
            except (ValueError, AttributeError):
                pass
            self._avg_line_diffs = None

        scatter_start = max(0, n_updates - max_scatter_window)
        all_scatter_x = []
        all_scatter_y = []
        for i in range(scatter_start, n_updates):
            for s in self._abs_diffs_history[i]:
                all_scatter_x.append(i)
                all_scatter_y.append(s)

        n_scatter = len(all_scatter_x)
        if n_scatter > 5000:
            scatter_alpha, scatter_size = 0.08, 6
        elif n_scatter > 2000:
            scatter_alpha, scatter_size = 0.15, 10
        elif n_scatter > 500:
            scatter_alpha, scatter_size = 0.22, 12
        else:
            scatter_alpha, scatter_size = 0.30, 14

        if all_scatter_x:
            import matplotlib.colors as mcolors
            cmap = plt.cm.RdYlGn_r
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            colors = [cmap(norm(y)) for y in all_scatter_y]

            self._scatter_diffs = ax.scatter(
                all_scatter_x, all_scatter_y,
                s=scatter_size, alpha=scatter_alpha, c=colors, zorder=1,
                edgecolors="none",
            )

        # ── Compute and plot the average line ───────────────────────────
        avg_xs = list(range(scatter_start, n_updates))
        avg_ys = [
            sum(self._abs_diffs_history[i]) / len(self._abs_diffs_history[i])
            for i in range(scatter_start, n_updates)
        ]

        if avg_xs:
            smoothed_ys = []
            ema = avg_ys[0]
            alpha = max(0.01, min(0.2, 50.0 / max(len(avg_ys), 1)))
            for y in avg_ys:
                ema = alpha * y + (1 - alpha) * ema
                smoothed_ys.append(ema)

            self._avg_line_diffs, = ax.plot(
                avg_xs, smoothed_ys,
                color="black", linewidth=1.8, alpha=0.85, zorder=3,
                label="Mean score (EMA)",
            )

        x_lo = max(scatter_start - 0.5, -0.5)
        x_hi = max(n_updates - 0.5, 0.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.05, 1.05)

        from matplotlib.lines import Line2D
        window_label = f"Individual (last {min(max_scatter_window, n_updates)})"
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=6, alpha=0.5, linestyle='None', label=window_label),
            Line2D([0], [0], color='black', linewidth=1.8, alpha=0.85,
                   label='Mean score (EMA)'),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=7, framealpha=0.7)

    @staticmethod
    def _downsample_line_indices(
            n_total: int,
            max_points: int = 1500,
            recent_full_res: int = 300,
            ) -> List[int]:
        """
        Two-tier downsampling strategy for line data:
          - The most recent `recent_full_res` points are kept at full resolution.
          - Older points are uniformly downsampled so the total never exceeds
            `max_points`.
          - The first and last points of the old region are always included
            to avoid visual jumps.

        This keeps recent trends sharp while preventing matplotlib from
        choking on tens of thousands of line segments.
        """
        if n_total <= max_points:
            return list(range(n_total))

        # Split into old region and recent region
        recent_start = max(0, n_total - recent_full_res)
        n_recent = n_total - recent_start
        budget_for_old = max_points - n_recent

        if recent_start == 0 or budget_for_old <= 2:
            # Everything is "recent" or no budget for old
            # Just uniformly downsample everything
            step = max(1, n_total // max_points)
            indices = list(range(0, n_total, step))
            if indices[-1] != n_total - 1:
                indices.append(n_total - 1)
            return indices

        # Old region: [0, recent_start) — downsample to budget_for_old points
        old_step = max(1, recent_start // budget_for_old)
        old_indices = list(range(0, recent_start, old_step))
        # Ensure the boundary point is included for continuity
        if old_indices[-1] != recent_start - 1:
            old_indices.append(recent_start - 1)

        # Recent region: [recent_start, n_total) — full resolution
        recent_indices = list(range(recent_start, n_total))

        return old_indices + recent_indices

    def _draw_predictions(self):
        """Draw predictions as a compact table with summary stats above."""
        ax = self.ax_preds
        if ax is None:
            return
        ax.clear()
        ax.axis("off")

        if not self._last_predictions:
            ax.set_title("Predictions (Training + Validation samples)",
                         fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, "Waiting for predictions...",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    transform=ax.transAxes)
            return

        # ── Compute summary stats ──────────────────────────────────────
        n_correct = 0
        n_garbage = 0
        n_total = len(self._last_predictions)
        for exp, pred, _ in self._last_predictions:
            exp_c = ''.join(c for c in exp if c.isascii() and c.isprintable()).strip()
            pred_c = ''.join(c for c in pred if c.isascii() and c.isprintable()).strip()
            if _is_int_str(exp_c) and _is_int_str(pred_c):
                if int(exp_c) == int(pred_c):
                    n_correct += 1
            if not _is_int_str(pred_c):
                n_garbage += 1

        accuracy = n_correct / n_total * 100 if n_total > 0 else 0

        # ── Title includes accuracy (no separate text to overlap) ──────
        summary = f"Accuracy: {n_correct}/{n_total} ({accuracy:.1f}%)"
        if n_garbage > 0:
            summary += f"  ·  ⚠ {n_garbage} unparseable"

        title_color = "green" if accuracy >= 50 else ("darkorange" if accuracy >= 20 else "red")
        ax.set_title(
            f"Predictions — {summary}",
            fontsize=9, fontweight="bold", color=title_color,
        )

        # ── Table header ───────────────────────────────────────────────
        header_y = 0.95
        col_x = [0.01, 0.05, 0.27, 0.49, 0.69, 0.85]
        headers = ["", "Expected", "Got", "Diff", "Score", ""]

        for cx, h in zip(col_x, headers):
            ax.text(cx, header_y, h,
                    fontsize=7, fontweight="bold", fontfamily="monospace",
                    color="black", alpha=0.6, transform=ax.transAxes,
                    verticalalignment="top")

        # Separator line
        ax.plot([0.01, 0.98], [0.91, 0.91],
                color='gray', linewidth=0.5, alpha=0.4,
                transform=ax.transAxes, clip_on=False)

        # ── Table rows ─────────────────────────────────────────────────
        n_show = min(len(self._last_predictions), 14)
        row_height = min(0.06, 0.88 / max(n_show, 1))
        y_start = 0.88

        for i, (expected, predicted, is_correct) in enumerate(
                self._last_predictions[-n_show:]
        ):
            y = y_start - i * row_height
            if y < 0.01:
                break

            exp_clean = ''.join(
                c for c in expected if c.isascii() and c.isprintable()
            ).strip() or "(empty)"
            pred_clean = ''.join(
                c for c in predicted if c.isascii() and c.isprintable()
            ).strip() or "(empty)"

            exp_parseable = _is_int_str(exp_clean)
            pred_parseable = _is_int_str(pred_clean)

            if exp_parseable and pred_parseable:
                exp_val = int(exp_clean)
                pred_val = int(pred_clean)
                score = _prediction_error_score(exp_val, pred_val)
                diff = abs(exp_val - pred_val)

                if score == 0.0:
                    color, marker = "green", "✓"
                elif score < 0.2:
                    color, marker = "#228B22", "≈"
                elif score < 0.5:
                    color, marker = "darkorange", "~"
                else:
                    color, marker = "red", "✗"

                exp_str = f"{exp_val}"
                pred_str = f"{pred_val}"
                diff_str = f"{diff}"
                score_str = f"{score:.2f}"

            elif exp_parseable and not pred_parseable:
                color, marker = "red", "✗"
                exp_str = f"{int(exp_clean)}"
                pred_str = pred_clean[:12]
                diff_str = "—"
                score_str = "1.00"

            elif not exp_parseable and pred_parseable:
                color, marker = "red", "✗"
                exp_str = exp_clean[:12]
                pred_str = f"{int(pred_clean)}"
                diff_str = "—"
                score_str = "—"

            else:
                color, marker = "red", "✗"
                exp_str = exp_clean[:12]
                pred_str = pred_clean[:12]
                diff_str = "—"
                score_str = "1.00"

            row_data = [marker, exp_str, pred_str, diff_str, score_str, ""]
            for cx, val in zip(col_x, row_data):
                ax.text(cx, y, val,
                        fontsize=6.5, fontfamily="monospace",
                        color=color, transform=ax.transAxes,
                        verticalalignment="top")

    def _draw_model_info(self):
        """Draw basic model information + GPU/system stats."""
        ax = self.ax_info
        if ax is None:
            return
        ax.clear()
        ax.axis("off")
        ax.set_title("Model / System Info", fontsize=10, fontweight="bold")

        if not self._model_info:
            ax.text(0.5, 0.5, "No model info available",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    transform=ax.transAxes)
            return

        info_lines = [
                f"Parameters:    {self._model_info.get('params', '?'):,}",
                f"d_model:       {self._model_info.get('d_model', '?')}",
                f"n_heads:       {self._model_info.get('n_heads', '?')}",
                f"n_layers:      {self._model_info.get('n_layers', '?')}",
                f"max_seq_len:   {self._model_info.get('max_seq_len', '?')}",
                f"vocab_size:    {self._model_info.get('vocab_size', '?')}",
                f"device:        {self._model_info.get('device', '?')}",
                ]

        # Add GPU info if available, otherwise show CPU/system info
        gpu = get_gpu_info()
        info_lines.append(f"───────────────────────────")
        if gpu:
            if "gpu_name" in gpu:
                info_lines.append(f"GPU:           {gpu['gpu_name']}")
            info_lines.append(
                    f"GPU Mem:       {gpu.get('gpu_mem_used_mb', '?')} / "
                    f"{gpu.get('gpu_mem_total_mb', '?')} MB "
                    f"({gpu.get('gpu_mem_free_mb', '?')} MB free)"
                    )
            info_lines.append(f"GPU Util:      {gpu.get('gpu_util_pct', '?')}%")
            if "gpu_temp_c" in gpu:
                info_lines.append(f"GPU Temp:      {gpu['gpu_temp_c']}°C")
            if "gpu_power_w" in gpu:
                info_lines.append(f"GPU Power:     {gpu['gpu_power_w']} W")
        else:
            # No nvidia-smi — show CPU/system info instead
            import platform
            info_lines.append(f"GPU:           N/A (no nvidia-smi)")
            if torch.cuda.is_available():
                info_lines.append(f"CUDA:          {torch.version.cuda}")
                info_lines.append(
                        f"CUDA Device:   {torch.cuda.get_device_name(0)}"
                        )
                mem = torch.cuda.mem_get_info(0)
                free_mb = mem[0] // (1024 * 1024)
                total_mb = mem[1] // (1024 * 1024)
                used_mb = total_mb - free_mb
                info_lines.append(
                        f"CUDA Mem:      {used_mb} / {total_mb} MB "
                        f"({free_mb} MB free)"
                        )
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info_lines.append(f"Backend:       MPS (Apple Silicon)")
            else:
                info_lines.append(f"Backend:       CPU only")
                try:
                    cpu_count = os.cpu_count()
                    info_lines.append(f"CPU Cores:     {cpu_count}")
                except Exception:
                    pass

        # Fixed line spacing instead of np.linspace
        line_height = 0.08
        y_start = 0.95

        for i, line in enumerate(info_lines):
            y_pos = y_start - i * line_height
            if y_pos < 0.0:
                break  # don't draw lines that would go off the bottom
            ax.text(0.05, y_pos, line,
                    fontsize=7, fontfamily="monospace",
                    color="black",
                    transform=ax.transAxes,
                    verticalalignment="center")


    def update_predictions(self, predictions: list):
        """
        Update the predictions panel.

        Args:
            predictions: List of tuples (expected_str, predicted_str, is_correct)
        """
        if not self.enabled:
            return
        self._last_predictions = predictions
        self._draw_predictions()
        self._refresh()


    def set_model_info(self, info: dict):
        self._model_info = info
        if not self.enabled:
            return
        self._draw_model_info()
        self._refresh()

    def _draw_persistence_landscapes(self, landscapes_per_layer, sample_range, n_landscapes, resolution):
        """Draw overlaid persistence landscapes (one curve per layer) on ax_barcode."""
        ax = self.ax_barcode
        if ax is None:
            return

        ax.clear()
        ax.set_title(
                f"Persistence Landscapes (H₁) — step {self._topo_step}",
                fontsize=9, fontweight="bold",
                )

        if not landscapes_per_layer:
            ax.text(0.5, 0.5, "No H₁ features detected",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    transform=ax.transAxes)
            return

        n_layers = len(landscapes_per_layer)
        x_grid = np.linspace(sample_range[0], sample_range[1], resolution)

        # Use a colormap: early layers cool, late layers warm
        cmap = plt.cm.viridis
        colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

        for li, ls_flat in enumerate(landscapes_per_layer):
            # ls_flat has shape (n_landscapes * resolution,)
            # Reshape to (n_landscapes, resolution) and plot only k=1 (the dominant landscape)
            ls_matrix = ls_flat.reshape(n_landscapes, resolution)
            dominant = ls_matrix[0]  # k=1 landscape

            if np.max(np.abs(dominant)) < 1e-12:
                continue  # skip flat-zero layers

            ax.plot(
                    x_grid, dominant,
                    color=colors[li],
                    linewidth=1.4,
                    alpha=0.7,
                    label=f"L{li}" if li % max(1, n_layers // 6) == 0 else None,
                    )

        ax.set_xlabel("Filtration value", fontsize=8)
        ax.set_ylabel("λ₁(t)", fontsize=8)
        ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.6)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)


    def _draw_wasserstein_heatmap(self, W_matrix):
        """Update the cross-layer Wasserstein-1 distance heatmap in-place."""
        ax = self.ax_bd
        if ax is None:
            return

        if W_matrix is None or W_matrix.size == 0:
            return

        # ── Normalize to [0, 1] ─────────────────────────────────────────
        w_max = W_matrix.max()
        if w_max > 0:
            W_normed = W_matrix / w_max
        else:
            W_normed = W_matrix

        # ── Update image data in-place (NO new imshow, NO new colorbar) ─
        if self._wass_im is not None:
            self._wass_im.set_data(W_normed)
            self._wass_im.set_extent((-0.5, W_normed.shape[1] - 0.5,
                                      -0.5, W_normed.shape[0] - 0.5))
        else:
            # Fallback: first call before _create_figure set it up
            self._wass_im = ax.imshow(
                    W_normed,
                    cmap="inferno",
                    interpolation="nearest",
                    origin="lower",
                    aspect="equal",
                    vmin=0.0,
                    vmax=1.0,
                    )

        # ── Restore axes position (undo any drift) ──────────────────────
        if self._wass_ax_pos is not None:
            ax.set_position(self._wass_ax_pos)

        # ── Update title and tick labels ────────────────────────────────
        ax.set_title(
                f"Wasserstein-1 Distance (layers) — step {self._topo_step}\n"
                f"[max={w_max:.2f}]",
                fontsize=9, fontweight="bold",
                )

        n = W_normed.shape[0]
        tick_positions = list(range(n))
        tick_labels = [f"L{i}" for i in range(n)]
        if n > 10:
            for i in range(len(tick_labels)):
                if i % 2 != 0:
                    tick_labels[i] = ""

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Layer", fontsize=8)


    def _save_jacobi_layer_images(self, jacobi_data):
        """Save each Jacobi field layer as a standalone PNG with token labels."""
        import liveplotter as _lp_self
        _run_dir = getattr(_lp_self, 'run_dir', None)
        if _run_dir is None:
            return

        fields = jacobi_data.get('fields', [])
        step = jacobi_data.get('step', 0)
        if not fields:
            return

        jacobi_img_dir = os.path.join(_run_dir, "jacobi_images")
        os.makedirs(jacobi_img_dir, exist_ok=True)

        from matplotlib.colors import hsv_to_rgb
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        for entry in fields:
            # Handle sliced mode (3D/4D)
            if isinstance(entry, dict) and entry.get('is_sliced', False):
                layer = entry['layer']
                slice_field_list = entry['slice_fields']
                for sf in slice_field_list:
                    dim_a, dim_b = sf['slice_dims']
                    self._save_jacobi_layer_image_2d_slice(
                        sf, step, jacobi_img_dir, layer, dim_a, dim_b
                    )
            elif isinstance(entry, dict) and entry.get('is_2d', False):
                self._save_jacobi_layer_image_2d(entry, step, jacobi_img_dir)
            else:
                self._save_jacobi_layer_image_3d(entry, step, jacobi_img_dir, hsv_to_rgb)

    def _save_jacobi_layer_image_2d_slice(self, f, step, jacobi_img_dir, layer, dim_a, dim_b):
        """
        Save a standalone PNG for a single 2D coordinate slice of a 3D/4D
        Jacobi layer. Rendered identically to the 2D special case.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        h_in_2d = f['h_in_2d']
        h_out_2d = f['h_out_2d']
        delta_2d = f['per_token_delta_2d']
        per_token_volume = f['per_token_volume']
        grid_x_in = f['grid_x_in']
        grid_y_in = f['grid_y_in']
        grid_x_out = f['grid_x_out']
        grid_y_out = f['grid_y_out']
        grid_n = f['grid_n']
        x_lim = f['x_lim']
        y_lim = f['y_lim']
        aniso_val = f['anisotropy']
        mean_vol = per_token_volume.mean()
        D_orig = f.get('D_original', '?')

        fig_s = Figure(figsize=(6, 6))
        FigureCanvasAgg(fig_s)
        ax_s = fig_s.add_subplot(1, 1, 1)
        ax_s.set_facecolor("#0d1117")

        # ── Compute view limits ─────────────────────────────────────
        all_x = np.concatenate([grid_x_in.ravel(), grid_x_out.ravel(),
                                h_in_2d[:, 0], h_out_2d[:, 0]])
        all_y = np.concatenate([grid_y_in.ravel(), grid_y_out.ravel(),
                                h_in_2d[:, 1], h_out_2d[:, 1]])
        pad_frac = 0.05
        x_span = max(np.ptp(all_x), 0.1)
        y_span = max(np.ptp(all_y), 0.1)
        view_x_min = all_x.min() - pad_frac * x_span
        view_x_max = all_x.max() + pad_frac * x_span
        view_y_min = all_y.min() - pad_frac * y_span
        view_y_max = all_y.max() + pad_frac * y_span

        # ── 1. Original (input) grid — gray ─────────────────────────
        for i in range(grid_n):
            ax_s.plot(grid_x_in[i, :], grid_y_in[i, :],
                      color='#555555', linewidth=0.4, alpha=0.5, zorder=1)
            ax_s.plot(grid_x_in[:, i], grid_y_in[:, i],
                      color='#555555', linewidth=0.4, alpha=0.5, zorder=1)

        # ── 2. Warped (output) grid — colored by local volume ───────
        grid_local_vol = np.ones((grid_n, grid_n))
        for i in range(grid_n - 1):
            for j in range(grid_n - 1):
                dx_in = grid_x_in[i, j+1] - grid_x_in[i, j]
                dy_in = grid_y_in[i, j+1] - grid_y_in[i, j]
                dx_in2 = grid_x_in[i+1, j] - grid_x_in[i, j]
                dy_in2 = grid_y_in[i+1, j] - grid_y_in[i, j]
                area_in = abs(dx_in * dy_in2 - dy_in * dx_in2)

                dx_out = grid_x_out[i, j+1] - grid_x_out[i, j]
                dy_out = grid_y_out[i, j+1] - grid_y_out[i, j]
                dx_out2 = grid_x_out[i+1, j] - grid_x_out[i, j]
                dy_out2 = grid_y_out[i+1, j] - grid_y_out[i, j]
                area_out = abs(dx_out * dy_out2 - dy_out * dx_out2)

                if area_in > 1e-12:
                    grid_local_vol[i, j] = area_out / area_in
                else:
                    grid_local_vol[i, j] = 1.0

        log_vol = np.log(np.clip(grid_local_vol, 1e-6, None))
        lv_absmax = max(np.abs(log_vol).max(), 1e-6)

        def _vol_to_color(lv):
            t = np.clip(lv / lv_absmax, -1, 1)
            if t > 0.05:
                r = 0.3 + 0.7 * t
                g = 0.25 * (1 - t)
                b = 0.2 * (1 - t)
            elif t < -0.05:
                at = abs(t)
                r = 0.2 * (1 - at)
                g = 0.25 * (1 - at)
                b = 0.3 + 0.7 * at
            else:
                r, g, b = 0.2, 0.55, 0.45
            return (r, g, b)

        for i in range(grid_n):
            for j in range(grid_n - 1):
                lv = log_vol[min(i, grid_n - 2), j]
                color = _vol_to_color(lv)
                ax_s.plot(
                    [grid_x_out[i, j], grid_x_out[i, j+1]],
                    [grid_y_out[i, j], grid_y_out[i, j+1]],
                    color=color, linewidth=0.8, alpha=0.85, zorder=2,
                )
            for j in range(grid_n):
                if i < grid_n - 1:
                    lv = log_vol[i, min(j, grid_n - 2)]
                    color = _vol_to_color(lv)
                    ax_s.plot(
                        [grid_x_out[i, j], grid_x_out[i+1, j]],
                        [grid_y_out[i, j], grid_y_out[i+1, j]],
                        color=color, linewidth=0.8, alpha=0.85, zorder=2,
                    )

        # ── 3. Token displacement lines + dots ──────────────────────
        T = h_in_2d.shape[0]
        for t in range(T):
            mag = np.sqrt(delta_2d[t, 0]**2 + delta_2d[t, 1]**2)
            if mag < 1e-8:
                continue
            ax_s.plot(
                [h_in_2d[t, 0], h_out_2d[t, 0]],
                [h_in_2d[t, 1], h_out_2d[t, 1]],
                color='#ffffff', linewidth=0.6, alpha=0.4, zorder=3,
            )

        ax_s.scatter(h_in_2d[:, 0], h_in_2d[:, 1],
                     s=18, facecolors='none', edgecolors='#aaaaaa',
                     linewidths=0.7, zorder=5, alpha=0.9)

        ax_s.scatter(h_out_2d[:, 0], h_out_2d[:, 1],
                     s=14, color='#ffcc44', edgecolors='none',
                     zorder=6, alpha=0.9)

        vol_hi = np.percentile(per_token_volume, 80)
        vol_lo = np.percentile(per_token_volume, 20)
        expanding = per_token_volume > vol_hi
        contracting = per_token_volume < vol_lo

        if expanding.any():
            ax_s.scatter(h_out_2d[expanding, 0], h_out_2d[expanding, 1],
                         s=32, facecolors='none', edgecolors='#ff4444',
                         linewidths=0.9, zorder=7, alpha=0.85)
        if contracting.any():
            ax_s.scatter(h_out_2d[contracting, 0], h_out_2d[contracting, 1],
                         s=32, facecolors='none', edgecolors='#4488ff',
                         linewidths=0.9, zorder=7, alpha=0.85)

        # ── 4. Token labels ─────────────────────────────────────────
        token_strings_f = f.get('token_strings', None)
        n_tokens = T
        panel_diag_px = math.sqrt(720**2 + 720**2)

        base_font = panel_diag_px * 0.020
        token_scale = max(0.6, min(1.5, 25.0 / max(n_tokens, 1)))
        label_fontsize = max(7.0, min(18.0, base_font * token_scale))

        n_labels = self._get_n_labels(n_tokens)

        rng = np.random.RandomState(
            seed=(step * 7 + layer * 31 + dim_a * 113 + dim_b * 197) & 0xFFFFFFFF
        )
        label_idx = rng.choice(
            n_tokens, size=min(n_labels, n_tokens), replace=False
        )

        offset_px = max(3.0, label_fontsize * 0.7)

        for li in label_idx:
            tx, ty = h_out_2d[li, 0], h_out_2d[li, 1]

            if token_strings_f is not None and li < len(token_strings_f):
                label_text = token_strings_f[li]
                if len(label_text) > 14:
                    label_text = label_text[:13] + "\u2026"
            else:
                label_text = f"t{li}"

            if not label_text.strip():
                label_text = "\u2423"

            ax_s.annotate(
                label_text,
                (tx, ty),
                fontsize=label_fontsize,
                color='#f0f0f0',
                alpha=0.92,
                fontweight='bold',
                fontfamily='monospace',
                xytext=(offset_px, offset_px),
                textcoords='offset points',
                zorder=8,
                bbox=dict(
                    boxstyle=f'round,pad={max(0.12, label_fontsize * 0.018):.2f}',
                    facecolor='#0d1117',
                    edgecolor='#3a5a7a',
                    alpha=0.75,
                    linewidth=0.4,
                ),
            )

        # ── 5. Title ────────────────────────────────────────────────
        layer_label = "Emb\u2192L1" if layer == 0 else f"L{layer}\u2192{layer+1}"
        ax_s.set_title(
            f"{layer_label} [d{dim_a}\u00d7d{dim_b}]  det={mean_vol:.2f}  "
            f"\u03c3\u2081/\u03c3\u2082={aniso_val:.1f}  "
            f"(step {step}, {D_orig}D slice)",
            fontsize=9, fontweight="bold", color="#aaccee", pad=4)

        ax_s.set_xlim(view_x_min, view_x_max)
        ax_s.set_ylim(view_y_min, view_y_max)
        ax_s.set_aspect('auto')
        ax_s.tick_params(labelsize=0, length=0)
        for spine in ax_s.spines.values():
            spine.set_color('#2a3a4a')
            spine.set_linewidth(0.5)

        fname = os.path.join(
            jacobi_img_dir,
            f"jacobi_step{step:06d}_layer{layer:02d}_d{dim_a}xd{dim_b}.png"
        )
        fig_s.savefig(fname, dpi=120, bbox_inches='tight', facecolor='#0a0a1a', edgecolor='none')
        del fig_s


    def _save_jacobi_layer_image_2d(self, f, step, jacobi_img_dir):
        """Save a standalone PNG for a 2D morphed-space Jacobi layer."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        layer = f['layer']
        h_in_2d = f['h_in_2d']
        h_out_2d = f['h_out_2d']
        delta_2d = f['per_token_delta_2d']
        per_token_volume = f['per_token_volume']
        grid_x_in = f['grid_x_in']
        grid_y_in = f['grid_y_in']
        grid_x_out = f['grid_x_out']
        grid_y_out = f['grid_y_out']
        grid_n = f['grid_n']
        x_lim = f['x_lim']
        y_lim = f['y_lim']
        aniso_val = f['anisotropy']
        mean_vol = per_token_volume.mean()

        fig_s = Figure(figsize=(6, 6))
        FigureCanvasAgg(fig_s)
        ax_s = fig_s.add_subplot(1, 1, 1)
        ax_s.set_facecolor("#0d1117")

        # ── Compute view limits ─────────────────────────────────────
        all_x = np.concatenate([grid_x_in.ravel(), grid_x_out.ravel(),
                                h_in_2d[:, 0], h_out_2d[:, 0]])
        all_y = np.concatenate([grid_y_in.ravel(), grid_y_out.ravel(),
                                h_in_2d[:, 1], h_out_2d[:, 1]])
        pad_frac = 0.05
        x_span = max(np.ptp(all_x), 0.1)
        y_span = max(np.ptp(all_y), 0.1)
        view_x_min = all_x.min() - pad_frac * x_span
        view_x_max = all_x.max() + pad_frac * x_span
        view_y_min = all_y.min() - pad_frac * y_span
        view_y_max = all_y.max() + pad_frac * y_span

        # ── 1. Original (input) grid — gray ─────────────────────────
        for i in range(grid_n):
            ax_s.plot(grid_x_in[i, :], grid_y_in[i, :],
                      color='#555555', linewidth=0.4, alpha=0.5, zorder=1)
            ax_s.plot(grid_x_in[:, i], grid_y_in[:, i],
                      color='#555555', linewidth=0.4, alpha=0.5, zorder=1)

        # ── 2. Warped (output) grid — colored by local volume ───────
        grid_local_vol = np.ones((grid_n, grid_n))
        for i in range(grid_n - 1):
            for j in range(grid_n - 1):
                dx_in = grid_x_in[i, j+1] - grid_x_in[i, j]
                dy_in = grid_y_in[i, j+1] - grid_y_in[i, j]
                dx_in2 = grid_x_in[i+1, j] - grid_x_in[i, j]
                dy_in2 = grid_y_in[i+1, j] - grid_y_in[i, j]
                area_in = abs(dx_in * dy_in2 - dy_in * dx_in2)

                dx_out = grid_x_out[i, j+1] - grid_x_out[i, j]
                dy_out = grid_y_out[i, j+1] - grid_y_out[i, j]
                dx_out2 = grid_x_out[i+1, j] - grid_x_out[i, j]
                dy_out2 = grid_y_out[i+1, j] - grid_y_out[i, j]
                area_out = abs(dx_out * dy_out2 - dy_out * dx_out2)

                if area_in > 1e-12:
                    grid_local_vol[i, j] = area_out / area_in
                else:
                    grid_local_vol[i, j] = 1.0

        log_vol = np.log(np.clip(grid_local_vol, 1e-6, None))
        lv_absmax = max(np.abs(log_vol).max(), 1e-6)

        def _vol_to_color(lv):
            t = np.clip(lv / lv_absmax, -1, 1)
            if t > 0.05:
                r = 0.3 + 0.7 * t
                g = 0.25 * (1 - t)
                b = 0.2 * (1 - t)
            elif t < -0.05:
                at = abs(t)
                r = 0.2 * (1 - at)
                g = 0.25 * (1 - at)
                b = 0.3 + 0.7 * at
            else:
                r, g, b = 0.2, 0.55, 0.45
            return (r, g, b)

        for i in range(grid_n):
            for j in range(grid_n - 1):
                lv = log_vol[min(i, grid_n - 2), j]
                color = _vol_to_color(lv)
                ax_s.plot(
                    [grid_x_out[i, j], grid_x_out[i, j+1]],
                    [grid_y_out[i, j], grid_y_out[i, j+1]],
                    color=color, linewidth=0.8, alpha=0.85, zorder=2,
                )
            for j in range(grid_n):
                if i < grid_n - 1:
                    lv = log_vol[i, min(j, grid_n - 2)]
                    color = _vol_to_color(lv)
                    ax_s.plot(
                        [grid_x_out[i, j], grid_x_out[i+1, j]],
                        [grid_y_out[i, j], grid_y_out[i+1, j]],
                        color=color, linewidth=0.8, alpha=0.85, zorder=2,
                    )

        # ── 3. Token displacement lines + dots ──────────────────────
        T = h_in_2d.shape[0]
        for t in range(T):
            mag = np.sqrt(delta_2d[t, 0]**2 + delta_2d[t, 1]**2)
            if mag < 1e-8:
                continue
            ax_s.plot(
                [h_in_2d[t, 0], h_out_2d[t, 0]],
                [h_in_2d[t, 1], h_out_2d[t, 1]],
                color='#ffffff', linewidth=0.6, alpha=0.4, zorder=3,
            )

        ax_s.scatter(h_in_2d[:, 0], h_in_2d[:, 1],
                     s=18, facecolors='none', edgecolors='#aaaaaa',
                     linewidths=0.7, zorder=5, alpha=0.9)

        ax_s.scatter(h_out_2d[:, 0], h_out_2d[:, 1],
                     s=14, color='#ffcc44', edgecolors='none',
                     zorder=6, alpha=0.9)

        vol_hi = np.percentile(per_token_volume, 80)
        vol_lo = np.percentile(per_token_volume, 20)
        expanding = per_token_volume > vol_hi
        contracting = per_token_volume < vol_lo

        if expanding.any():
            ax_s.scatter(h_out_2d[expanding, 0], h_out_2d[expanding, 1],
                         s=32, facecolors='none', edgecolors='#ff4444',
                         linewidths=0.9, zorder=7, alpha=0.85)
        if contracting.any():
            ax_s.scatter(h_out_2d[contracting, 0], h_out_2d[contracting, 1],
                         s=32, facecolors='none', edgecolors='#4488ff',
                         linewidths=0.9, zorder=7, alpha=0.85)

        # ── 4. Token labels ─────────────────────────────────────────
        token_strings_f = f.get('token_strings', None)
        n_tokens = T
        panel_diag_px = math.sqrt(720**2 + 720**2)

        base_font = panel_diag_px * 0.020
        token_scale = max(0.6, min(1.5, 25.0 / max(n_tokens, 1)))
        label_fontsize = max(7.0, min(18.0, base_font * token_scale))

        n_labels = self._get_n_labels(n_tokens)

        rng = np.random.RandomState(
            seed=(step * 7 + layer * 31) & 0xFFFFFFFF
        )
        label_idx = rng.choice(
            n_tokens, size=min(n_labels, n_tokens), replace=False
        )

        offset_px = max(3.0, label_fontsize * 0.7)

        for li in label_idx:
            tx, ty = h_out_2d[li, 0], h_out_2d[li, 1]

            if token_strings_f is not None and li < len(token_strings_f):
                label_text = token_strings_f[li]
                if len(label_text) > 14:
                    label_text = label_text[:13] + "\u2026"
            else:
                label_text = f"t{li}"

            if not label_text.strip():
                label_text = "\u2423"

            ax_s.annotate(
                label_text,
                (tx, ty),
                fontsize=label_fontsize,
                color='#f0f0f0',
                alpha=0.92,
                fontweight='bold',
                fontfamily='monospace',
                xytext=(offset_px, offset_px),
                textcoords='offset points',
                zorder=8,
                bbox=dict(
                    boxstyle=f'round,pad={max(0.12, label_fontsize * 0.018):.2f}',
                    facecolor='#0d1117',
                    edgecolor='#3a5a7a',
                    alpha=0.75,
                    linewidth=0.4,
                ),
            )

        # ── 5. Title ────────────────────────────────────────────────
        layer_label = "Emb\u2192L1" if layer == 0 else f"L{layer}\u2192{layer+1}"
        ax_s.set_title(
            f"{layer_label}  det={mean_vol:.2f}  \u03c3\u2081/\u03c3\u2082={aniso_val:.1f}  "
            f"(step {step}, exact 2D)",
            fontsize=9, fontweight="bold", color="#aaccee", pad=4)

        ax_s.set_xlim(view_x_min, view_x_max)
        ax_s.set_ylim(view_y_min, view_y_max)
        ax_s.set_aspect('auto')
        ax_s.tick_params(labelsize=0, length=0)
        for spine in ax_s.spines.values():
            spine.set_color('#2a3a4a')
            spine.set_linewidth(0.5)

        fname = os.path.join(jacobi_img_dir, f"jacobi_step{step:06d}_layer{layer:02d}.png")
        fig_s.savefig(fname, dpi=120, bbox_inches='tight', facecolor='#0a0a1a', edgecolor='none')
        del fig_s

    def _save_jacobi_layer_image_3d(self, f, step, jacobi_img_dir, hsv_to_rgb):
        """Save a standalone PNG for a 3+D Jacobi layer (existing HSV path)."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        layer = f['layer']
        h_in_2d = f['h_in_2d']
        grid_x = f['grid_x']
        grid_y = f['grid_y']
        grid_volume = f['grid_volume']
        grid_disp_x = f['grid_disp_x']
        grid_disp_y = f['grid_disp_y']
        grid_max_stretch_val = f['grid_max_stretch_val']
        grid_min_stretch_val = f['grid_min_stretch_val']
        grid_max_stretch_dir = f['grid_max_stretch_dir']
        x_lim = f['x_lim']
        y_lim = f['y_lim']
        per_token_volume = f['per_token_volume']
        per_token_shear = f['per_token_shear_mag']
        grid_n = grid_x.shape[0]
        var_exp = f.get('var_explained', 0)
        use_pca = f.get('use_pca', True)
        aniso_val = f['anisotropy']
        mean_vol = per_token_volume.mean()

        fig_s = Figure(figsize=(6, 6))
        FigureCanvasAgg(fig_s)
        ax_s = fig_s.add_subplot(1, 1, 1)
        ax_s.set_facecolor("#0d1117")

        # ── Background HSV ──────────────────────────────────────
        disp_angle = np.arctan2(grid_disp_y, grid_disp_x)
        disp_mag = np.sqrt(grid_disp_x**2 + grid_disp_y**2)
        mag_norm = np.arcsinh(disp_mag * 2.0)
        mag_max = max(mag_norm.max(), 1e-10)
        mag_norm = mag_norm / mag_max

        H = (disp_angle + np.pi) / (2 * np.pi)
        S_hsv = np.ones_like(H) * 0.85
        V_hsv = 0.15 + 0.85 * mag_norm

        hsv_img = np.stack([H, S_hsv, V_hsv], axis=-1)
        rgb_img = hsv_to_rgb(hsv_img)

        log_vol = np.log(np.clip(grid_volume, 1e-6, None))
        lv_absmax = max(np.abs(log_vol).max(), 1e-6)
        lv_norm = np.clip(log_vol / lv_absmax, -1, 1)
        vol_blend = 0.2
        rgb_img[:, :, 0] = np.clip(rgb_img[:, :, 0] + vol_blend * np.clip(lv_norm, 0, 1), 0, 1)
        rgb_img[:, :, 2] = np.clip(rgb_img[:, :, 2] + vol_blend * np.clip(-lv_norm, 0, 1), 0, 1)

        ax_s.imshow(rgb_img, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
                     origin='lower', aspect='auto', interpolation='bilinear', alpha=0.9)

        # ── Streamlines ─────────────────────────────────────────
        gx_1d = np.linspace(x_lim[0], x_lim[1], grid_n)
        gy_1d = np.linspace(y_lim[0], y_lim[1], grid_n)
        try:
            speed = np.sqrt(grid_disp_x**2 + grid_disp_y**2)
            lw = 0.3 + 1.2 * speed / max(speed.max(), 1e-10)
            ax_s.streamplot(gx_1d, gy_1d, grid_disp_x, grid_disp_y,
                            color='white', linewidth=lw, density=0.7,
                            arrowsize=0.6, arrowstyle='->', zorder=2, minlength=0.2)
        except Exception:
            pass

        # ── Stretch whiskers ────────────────────────────────────
        whisker_step = max(1, grid_n // 8)
        x_span = x_lim[1] - x_lim[0]
        y_span = y_lim[1] - y_lim[0]
        whisker_len = min(x_span, y_span) / (grid_n / whisker_step) * 0.3

        for gi in range(whisker_step // 2, grid_n, whisker_step):
            for gj in range(whisker_step // 2, grid_n, whisker_step):
                cx, cy = grid_x[gi, gj], grid_y[gi, gj]
                d = grid_max_stretch_dir[gi, gj]
                s_max = grid_max_stretch_val[gi, gj]
                s_min = grid_min_stretch_val[gi, gj]
                ratio = s_max / max(s_min, 1e-10)
                if ratio < 1.05:
                    continue
                length = whisker_len * min(ratio - 1.0, 3.0) / 3.0
                if ratio < 1.5:
                    wcolor, walpha = '#ffff44', 0.4
                elif ratio < 3.0:
                    wcolor, walpha = '#ff8800', 0.6
                else:
                    wcolor, walpha = '#ff2222', 0.8
                ax_s.plot([cx - length*d[0], cx + length*d[0]],
                          [cy - length*d[1], cy + length*d[1]],
                          color=wcolor, linewidth=1.0, alpha=walpha, zorder=4,
                          solid_capstyle='round')

        # ── Token markers ───────────────────────────────────────
        shear_max = max(per_token_shear.max(), 1e-6)
        shear_norm = per_token_shear / shear_max
        ax_s.scatter(h_in_2d[:, 0], h_in_2d[:, 1], s=10, c=shear_norm,
                     cmap='magma', vmin=0, vmax=1, edgecolors='none', zorder=6, alpha=0.9)

        # ── Expanding / contracting rings ───────────────────────
        vol_hi = np.percentile(per_token_volume, 85)
        vol_lo = np.percentile(per_token_volume, 15)
        expanding = per_token_volume > vol_hi
        contracting = per_token_volume < vol_lo
        if expanding.any():
            ax_s.scatter(h_in_2d[expanding, 0], h_in_2d[expanding, 1],
                         s=28, facecolors='none', edgecolors='#ff4444', linewidths=0.8, zorder=7)
        if contracting.any():
            ax_s.scatter(h_in_2d[contracting, 0], h_in_2d[contracting, 1],
                         s=28, facecolors='none', edgecolors='#4488ff', linewidths=0.8, zorder=7)

        # ── Token labels ────────────────────────────────────────
        token_strings_f = f.get('token_strings', None)
        n_tokens = h_in_2d.shape[0]
        panel_diag_px = math.sqrt(720**2 + 720**2)

        base_font = panel_diag_px * 0.020
        token_scale = max(0.6, min(1.5, 25.0 / max(n_tokens, 1)))
        label_fontsize = max(7.0, min(18.0, base_font * token_scale))

        n_labels = self._get_n_labels(n_tokens)

        rng = np.random.RandomState(
            seed=(step * 7 + layer * 31) & 0xFFFFFFFF
        )
        label_idx = rng.choice(
            n_tokens, size=min(n_labels, n_tokens), replace=False
        )

        offset_px = max(3.0, label_fontsize * 0.7)

        for li in label_idx:
            tx, ty = h_in_2d[li, 0], h_in_2d[li, 1]

            if token_strings_f is not None and li < len(token_strings_f):
                label_text = token_strings_f[li]
                if len(label_text) > 14:
                    label_text = label_text[:13] + "\u2026"
            else:
                label_text = f"t{li}"

            if not label_text.strip():
                label_text = "\u2423"

            ax_s.annotate(
                label_text,
                (tx, ty),
                fontsize=label_fontsize,
                color='#f0f0f0',
                alpha=0.92,
                fontweight='bold',
                fontfamily='monospace',
                xytext=(offset_px, offset_px),
                textcoords='offset points',
                zorder=8,
                bbox=dict(
                    boxstyle=f'round,pad={max(0.12, label_fontsize * 0.018):.2f}',
                    facecolor='#0d1117',
                    edgecolor='#3a5a7a',
                    alpha=0.75,
                    linewidth=0.4,
                ),
            )

        # ── Title ───────────────────────────────────────────────
        layer_label = "Emb\u2192L1" if layer == 0 else f"L{layer}\u2192{layer+1}"
        proj_label = f"PCA var={var_exp:.0%}" if use_pca else "raw 2D"
        ax_s.set_title(
            f"{layer_label}  det={mean_vol:.2f}  \u03c3\u2081/\u03c3\u2099={aniso_val:.1f}  "
            f"(step {step}, {proj_label})",
            fontsize=9, fontweight="bold", color="#aaccee", pad=4)

        ax_s.set_xlim(*x_lim)
        ax_s.set_ylim(*y_lim)
        ax_s.set_aspect('auto')
        ax_s.tick_params(labelsize=0, length=0)
        for spine in ax_s.spines.values():
            spine.set_color('#2a3a4a')
            spine.set_linewidth(0.5)

        fname = os.path.join(jacobi_img_dir, f"jacobi_step{step:06d}_layer{layer:02d}.png")
        fig_s.savefig(fname, dpi=120, bbox_inches='tight', facecolor='#0a0a1a', edgecolor='none')
        del fig_s

    def _get_n_labels(self, n_tokens):
        min_nr = 10
        max_nr = 200
        every_nth = 2
        return min(max(n_tokens // every_nth, min_nr), max_nr)

    def _draw_single_jacobi_panel_2d(self, sub_ax, f, ell, jacobi_data):
        """
        Draw a single layer's Jacobi field panel for the 2D special case.
        
        Instead of PCA + HSV deformation encoding, this draws:
          1. The original (input) grid in gray
          2. The warped (output) grid colored by local volume change
          3. Token dots at input and output positions with displacement lines
          4. Token labels
        """
        h_in_2d = f['h_in_2d']
        h_out_2d = f['h_out_2d']
        delta_2d = f['per_token_delta_2d']
        per_token_volume = f['per_token_volume']
        grid_x_in = f['grid_x_in']
        grid_y_in = f['grid_y_in']
        grid_x_out = f['grid_x_out']
        grid_y_out = f['grid_y_out']
        grid_n = f['grid_n']
        x_lim = f['x_lim']
        y_lim = f['y_lim']

        # ── Compute axis limits that encompass both input and output grids ──
        all_x = np.concatenate([grid_x_in.ravel(), grid_x_out.ravel(),
                                h_in_2d[:, 0], h_out_2d[:, 0]])
        all_y = np.concatenate([grid_y_in.ravel(), grid_y_out.ravel(),
                                h_in_2d[:, 1], h_out_2d[:, 1]])
        pad_frac = 0.05
        x_span = max(np.ptp(all_x), 0.1)
        y_span = max(np.ptp(all_y), 0.1)
        view_x_min = all_x.min() - pad_frac * x_span
        view_x_max = all_x.max() + pad_frac * x_span
        view_y_min = all_y.min() - pad_frac * y_span
        view_y_max = all_y.max() + pad_frac * y_span

        # ════════════════════════════════════════════════════════════════
        # 1. ORIGINAL (INPUT) GRID — gray, thin lines
        # ════════════════════════════════════════════════════════════════
        for i in range(grid_n):
            # Horizontal lines (constant row)
            sub_ax.plot(grid_x_in[i, :], grid_y_in[i, :],
                        color='#555555', linewidth=0.4, alpha=0.5, zorder=1)
            # Vertical lines (constant column)
            sub_ax.plot(grid_x_in[:, i], grid_y_in[:, i],
                        color='#555555', linewidth=0.4, alpha=0.5, zorder=1)

        # ════════════════════════════════════════════════════════════════
        # 2. WARPED (OUTPUT) GRID — colored by local volume change
        # ════════════════════════════════════════════════════════════════
        # Compute local volume change at each grid cell from the warped grid
        # using finite differences of the warped coordinates
        grid_local_vol = np.ones((grid_n, grid_n))
        for i in range(grid_n - 1):
            for j in range(grid_n - 1):
                # Input cell vectors
                dx_in = grid_x_in[i, j+1] - grid_x_in[i, j]
                dy_in = grid_y_in[i, j+1] - grid_y_in[i, j]
                dx_in2 = grid_x_in[i+1, j] - grid_x_in[i, j]
                dy_in2 = grid_y_in[i+1, j] - grid_y_in[i, j]
                area_in = abs(dx_in * dy_in2 - dy_in * dx_in2)

                # Output cell vectors
                dx_out = grid_x_out[i, j+1] - grid_x_out[i, j]
                dy_out = grid_y_out[i, j+1] - grid_y_out[i, j]
                dx_out2 = grid_x_out[i+1, j] - grid_x_out[i, j]
                dy_out2 = grid_y_out[i+1, j] - grid_y_out[i, j]
                area_out = abs(dx_out * dy_out2 - dy_out * dx_out2)

                if area_in > 1e-12:
                    grid_local_vol[i, j] = area_out / area_in
                else:
                    grid_local_vol[i, j] = 1.0

        # Map volume change to color:
        #   expansion (vol > 1) → red
        #   contraction (vol < 1) → blue
        #   ~preserving (vol ≈ 1) → green/cyan
        log_vol = np.log(np.clip(grid_local_vol, 1e-6, None))
        lv_absmax = max(np.abs(log_vol).max(), 1e-6)

        def _vol_to_color(lv):
            """Map log-volume-change to RGB color."""
            t = np.clip(lv / lv_absmax, -1, 1)
            if t > 0.05:
                # Expansion → red tint
                r = 0.3 + 0.7 * t
                g = 0.25 * (1 - t)
                b = 0.2 * (1 - t)
            elif t < -0.05:
                # Contraction → blue tint
                at = abs(t)
                r = 0.2 * (1 - at)
                g = 0.25 * (1 - at)
                b = 0.3 + 0.7 * at
            else:
                # ~Volume preserving → green/cyan
                r, g, b = 0.2, 0.55, 0.45
            return (r, g, b)

        for i in range(grid_n):
            for j in range(grid_n - 1):
                lv = log_vol[min(i, grid_n - 2), j]
                color = _vol_to_color(lv)
                # Horizontal warped lines (constant row)
                sub_ax.plot(
                    [grid_x_out[i, j], grid_x_out[i, j+1]],
                    [grid_y_out[i, j], grid_y_out[i, j+1]],
                    color=color, linewidth=0.8, alpha=0.85, zorder=2,
                )
            for j in range(grid_n):
                if i < grid_n - 1:
                    lv = log_vol[i, min(j, grid_n - 2)]
                    color = _vol_to_color(lv)
                    # Vertical warped lines (constant column)
                    sub_ax.plot(
                        [grid_x_out[i, j], grid_x_out[i+1, j]],
                        [grid_y_out[i, j], grid_y_out[i+1, j]],
                        color=color, linewidth=0.8, alpha=0.85, zorder=2,
                    )

        # ════════════════════════════════════════════════════════════════
        # 3. TOKEN DISPLACEMENT LINES + DOTS
        # ════════════════════════════════════════════════════════════════
        T = h_in_2d.shape[0]

        # Displacement lines (input → output)
        for t in range(T):
            mag = np.sqrt(delta_2d[t, 0]**2 + delta_2d[t, 1]**2)
            if mag < 1e-8:
                continue
            sub_ax.plot(
                [h_in_2d[t, 0], h_out_2d[t, 0]],
                [h_in_2d[t, 1], h_out_2d[t, 1]],
                color='#ffffff', linewidth=0.6, alpha=0.4, zorder=3,
            )

        # Input positions — hollow circles
        sub_ax.scatter(
            h_in_2d[:, 0], h_in_2d[:, 1],
            s=18, facecolors='none', edgecolors='#aaaaaa',
            linewidths=0.7, zorder=5, alpha=0.9,
        )

        # Output positions — filled circles
        sub_ax.scatter(
            h_out_2d[:, 0], h_out_2d[:, 1],
            s=14, color='#ffcc44', edgecolors='none',
            zorder=6, alpha=0.9,
        )

        # Expanding / contracting rings on OUTPUT positions
        vol_hi = np.percentile(per_token_volume, 80)
        vol_lo = np.percentile(per_token_volume, 20)
        expanding = per_token_volume > vol_hi
        contracting = per_token_volume < vol_lo

        if expanding.any():
            sub_ax.scatter(
                h_out_2d[expanding, 0], h_out_2d[expanding, 1],
                s=32, facecolors='none', edgecolors='#ff4444',
                linewidths=0.9, zorder=7, alpha=0.85,
            )
        if contracting.any():
            sub_ax.scatter(
                h_out_2d[contracting, 0], h_out_2d[contracting, 1],
                s=32, facecolors='none', edgecolors='#4488ff',
                linewidths=0.9, zorder=7, alpha=0.85,
            )

        # ════════════════════════════════════════════════════════════════
        # 4. SPARSE TOKEN LABELS
        # ════════════════════════════════════════════════════════════════
        token_strings_f = f.get('token_strings', None)
        n_tokens = T

        fig_dpi = self.fig.dpi
        fig_w_in, fig_h_in = self.fig.get_size_inches()
        pos = sub_ax.get_position()
        panel_w_px = pos.width * fig_w_in * fig_dpi
        panel_h_px = pos.height * fig_h_in * fig_dpi
        panel_diag_px = math.sqrt(panel_w_px**2 + panel_h_px**2)

        base_font = panel_diag_px * 0.020
        token_scale = max(0.6, min(1.5, 25.0 / max(n_tokens, 1)))
        label_fontsize = max(7.0, min(18.0, base_font * token_scale))

        n_labels = self._get_n_labels(n_tokens)
        if panel_diag_px < 200:
            n_labels = min(n_labels, 6)

        draw_key = jacobi_data.get('draw_key', 0)
        rng = np.random.RandomState(
            seed=(draw_key * 7 + ell * 31) & 0xFFFFFFFF
        )
        label_idx = rng.choice(
            n_tokens, size=min(n_labels, n_tokens), replace=False
        )

        offset_px = max(3.0, label_fontsize * 0.7)

        for li in label_idx:
            # Label at the OUTPUT position (where the token ended up)
            tx, ty = h_out_2d[li, 0], h_out_2d[li, 1]

            if token_strings_f is not None and li < len(token_strings_f):
                label_text = token_strings_f[li]
                if len(label_text) > 14:
                    label_text = label_text[:13] + "\u2026"
            else:
                label_text = f"t{li}"

            if not label_text.strip():
                label_text = "\u2423"

            sub_ax.annotate(
                label_text,
                (tx, ty),
                fontsize=label_fontsize,
                color='#f0f0f0',
                alpha=0.92,
                fontweight='bold',
                fontfamily='monospace',
                xytext=(offset_px, offset_px),
                textcoords='offset points',
                zorder=8,
                bbox=dict(
                    boxstyle=f'round,pad={max(0.12, label_fontsize * 0.018):.2f}',
                    facecolor='#0d1117',
                    edgecolor='#3a5a7a',
                    alpha=0.75,
                    linewidth=0.4,
                ),
            )

        # ════════════════════════════════════════════════════════════════
        # 5. LAYER LABEL / TITLE
        # ════════════════════════════════════════════════════════════════
        layer_label = "Emb\u2192L1" if ell == 0 else f"L{ell}\u2192{ell+1}"
        mean_vol = per_token_volume.mean()
        aniso_val = f['anisotropy']

        sub_ax.set_title(
            f"{layer_label}  det={mean_vol:.2f}  \u03c3\u2081/\u03c3\u2082={aniso_val:.1f}",
            fontsize=7, fontweight="bold", color="#aaccee", pad=2,
        )

        sub_ax.set_xlim(view_x_min, view_x_max)
        sub_ax.set_ylim(view_y_min, view_y_max)
        sub_ax.set_aspect('auto')
        sub_ax.tick_params(labelsize=0, length=0)
        for spine in sub_ax.spines.values():
            spine.set_color('#2a3a4a')
            spine.set_linewidth(0.5)

    # ── Finalize ────────────────────────────────────────────────────────
    def finalize(self):
        if not self.enabled:
            return

        # Always save final plot to file
        self._save_to_file()
        console.print(f"[bold green]📊 Final plot saved to: {self.plot_file}[/]")

        if not self.suppress_window:
            self.plt.ioff()
            self.plt.show()

