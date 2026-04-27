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
def compute_layer_jacobi_fields(model, input_ids, device, max_tokens=1024):
    """
    Compute the Jacobi field of the space deformation itself at each layer.
    
    Interpretation: tokens are FIXED OBSERVERS. The space (representation
    manifold) deforms around them as it passes through each layer.
    
    At each point on a 2D grid (in PCA space), we interpolate the local
    2×2 Jacobian from nearby token observations, then decompose it:
    
        J = R · S   (polar decomposition)
        S = volumetric + shear
    
    We store everything needed for a colorful vector field rendering:
    - The displacement field (J-I) applied to basis vectors → shows how
      space is being pulled/pushed at each point
    - Volume change (det J) → expansion vs contraction
    - Principal stretch directions and magnitudes → anisotropy
    - Rotation angle → local curl/twist of space
    """
    model.eval()
    output = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = output.hidden_states

    if hidden_states is None or len(hidden_states) < 2:
        return []

    fields = []

    # ── Shared PCA basis from all layers for consistent 2D view ─────
    all_points = []
    for hs in hidden_states:
        pts = hs.detach().float().cpu()
        pts = pts.reshape(-1, pts.shape[-1])  # (B*T, D)
        all_points.append(pts)

    all_concat = torch.cat(all_points, dim=0)
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

    for ell in range(len(hidden_states) - 1):
        h_in = hidden_states[ell].detach().float()         # (B, T, D)
        h_in = h_in.reshape(-1, h_in.shape[-1])            # (B*T, D)
        h_out = hidden_states[ell + 1].detach().float()      # (B, T, D)  ← FIXED
        h_out = h_out.reshape(-1, h_out.shape[-1])           # (B*T, D)  ← FIXED

        T, D = h_in.shape
        if T > max_tokens:
            idx = torch.linspace(0, T - 1, max_tokens).long()
            h_in = h_in[idx]
            h_out = h_out[idx]
            T = max_tokens

        delta = h_out - h_in

        # Project to 2D for visualization
        h_in_2d = ((h_in.cpu() - mean) @ pca_basis.T).numpy()   # (T, 2)

        # ── Per-token LOCAL Jacobian via neighbors ──────────────────
        per_token_J_2d = np.zeros((T, 2, 2))
        per_token_volume = np.zeros(T)
        per_token_rotation = np.zeros(T)
        per_token_shear_mag = np.zeros(T)

        # Global Jacobian for aggregate metrics
        h_in_c = h_in - h_in.mean(0)
        delta_c = delta - delta.mean(0)

        try:
            J_residual = torch.linalg.lstsq(h_in_c.cpu(), delta_c.cpu()).solution
            J_global = torch.eye(D) + J_residual
        except Exception:
            J_global = torch.eye(D)

        P = pca_basis  # (2, D)

        for t in range(T):
            dists = torch.norm(h_in - h_in[t], dim=1)
            dists[t] = float('inf')
            k = min(max(D // 2, 8), T - 1)
            nn_idx = torch.argsort(dists)[:k]

            dh_in = (h_in[nn_idx] - h_in[t]).cpu()
            dh_out = (h_out[nn_idx] - h_out[t]).cpu()

            try:
                J_local = torch.linalg.lstsq(dh_in, dh_out).solution
            except Exception:
                J_local = torch.eye(D)

            J_2d = (P @ J_local @ P.T).numpy()
            per_token_J_2d[t] = J_2d

            try:
                U_j, sigma_j, Vht_j = np.linalg.svd(J_2d)
                R_2d = U_j @ Vht_j
            except Exception:
                sigma_j = np.array([1.0, 1.0])
                R_2d = np.eye(2)

            per_token_volume[t] = sigma_j[0] * sigma_j[1]
            per_token_rotation[t] = np.arctan2(R_2d[1, 0], R_2d[0, 0])

            S_2d = Vht_j.T @ np.diag(sigma_j) @ Vht_j if 'Vht_j' in dir() else np.eye(2)
            S_trace = np.trace(S_2d)
            S_traceless = S_2d - (S_trace / 2.0) * np.eye(2)
            per_token_shear_mag[t] = np.linalg.norm(S_traceless, 'fro')

        # ── Build grid and interpolate the full 2×2 Jacobian ───────
        from scipy.interpolate import RBFInterpolator

        grid_n = 32
        x_range = h_in_2d[:, 0]
        y_range = h_in_2d[:, 1]
        pad_frac = 0.2
        x_span = max(np.ptp(x_range), 0.1)
        y_span = max(np.ptp(y_range), 0.1)
        x_min = x_range.min() - pad_frac * x_span
        x_max = x_range.max() + pad_frac * x_span
        y_min = y_range.min() - pad_frac * y_span
        y_max = y_range.max() + pad_frac * y_span

        gx = np.linspace(x_min, x_max, grid_n)
        gy = np.linspace(y_min, y_max, grid_n)
        grid_x, grid_y = np.meshgrid(gx, gy)
        grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        # Interpolate each component of the 2×2 Jacobian
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

        # ── Decompose at each grid point ───────────────────────────
        grid_volume = np.zeros((grid_n, grid_n))
        grid_rotation = np.zeros((grid_n, grid_n))
        grid_shear = np.zeros((grid_n, grid_n))
        grid_max_stretch_val = np.zeros((grid_n, grid_n))
        grid_min_stretch_val = np.zeros((grid_n, grid_n))
        grid_max_stretch_dir = np.zeros((grid_n, grid_n, 2))
        # The displacement field: (J - I) applied to the local position
        # This shows how space is being PULLED at each point
        grid_disp_x = np.zeros((grid_n, grid_n))
        grid_disp_y = np.zeros((grid_n, grid_n))
        # Also: (J - I) applied to e_x and e_y separately for a richer field
        grid_Jm1_ex_x = np.zeros((grid_n, grid_n))
        grid_Jm1_ex_y = np.zeros((grid_n, grid_n))
        grid_Jm1_ey_x = np.zeros((grid_n, grid_n))
        grid_Jm1_ey_y = np.zeros((grid_n, grid_n))

        for gi in range(grid_n):
            for gj in range(grid_n):
                J_loc = grid_J[gi, gj]
                F = J_loc - np.eye(2)  # deformation gradient minus identity

                # Displacement = F · position (how space moves at this point)
                pos = np.array([grid_x[gi, gj], grid_y[gi, gj]])
                disp = F @ pos
                grid_disp_x[gi, gj] = disp[0]
                grid_disp_y[gi, gj] = disp[1]

                # F applied to basis vectors (for the vector field)
                grid_Jm1_ex_x[gi, gj] = F[0, 0]
                grid_Jm1_ex_y[gi, gj] = F[1, 0]
                grid_Jm1_ey_x[gi, gj] = F[0, 1]
                grid_Jm1_ey_y[gi, gj] = F[1, 1]

                try:
                    U_g, sigma_g, Vht_g = np.linalg.svd(J_loc)
                    R_g = U_g @ Vht_g
                except Exception:
                    sigma_g = np.array([1.0, 1.0])
                    R_g = np.eye(2)
                    Vht_g = np.eye(2)

                grid_volume[gi, gj] = sigma_g[0] * sigma_g[1]
                grid_rotation[gi, gj] = np.arctan2(R_g[1, 0], R_g[0, 0])
                grid_max_stretch_val[gi, gj] = sigma_g[0]
                grid_min_stretch_val[gi, gj] = sigma_g[1]
                grid_max_stretch_dir[gi, gj] = Vht_g[0]

                S_g = Vht_g.T @ np.diag(sigma_g) @ Vht_g
                S_trace = np.trace(S_g)
                S_traceless = S_g - (S_trace / 2.0) * np.eye(2)
                grid_shear[gi, gj] = np.linalg.norm(S_traceless, 'fro')

        # ── Global metrics ─────────────────────────────────────────
        try:
            S_global = torch.linalg.svdvals(J_global)
            singular_values = S_global.cpu().numpy()
            anisotropy = (S_global.max() / S_global.min().clamp(min=1e-10)).item()
            log_det = torch.log(S_global.clamp(min=1e-10)).sum().item()
        except Exception:
            singular_values = np.ones(D)
            anisotropy = 1.0
            log_det = 0.0

        divergence = torch.trace(J_global).item()
        J_sym = (J_global + J_global.T) / 2
        J_antisym = (J_global - J_global.T) / 2
        curl = torch.norm(J_antisym, p='fro').item()
        trace_sym = torch.trace(J_sym)
        shear_tensor = J_sym - (trace_sym / D) * torch.eye(D)
        shear = torch.norm(shear_tensor, p='fro').item()

        fields.append({
            'layer': ell,
            'jacobian': J_global.cpu().numpy(),
            'divergence': divergence,
            'curl': curl,
            'shear': shear,
            'singular_values': singular_values,
            'log_det': log_det,
            'anisotropy': anisotropy,
            # ── Token data ──────────────────────────────────────
            'h_in_2d': h_in_2d,
            'per_token_J_2d': per_token_J_2d,
            'per_token_volume': per_token_volume,
            'per_token_rotation': per_token_rotation,
            'per_token_shear_mag': per_token_shear_mag,
            # ── Grid data ───────────────────────────────────────
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_J': grid_J,
            'grid_volume': grid_volume,
            'grid_rotation': grid_rotation,
            'grid_shear': grid_shear,
            'grid_max_stretch_val': grid_max_stretch_val,
            'grid_min_stretch_val': grid_min_stretch_val,
            'grid_max_stretch_dir': grid_max_stretch_dir,
            'grid_disp_x': grid_disp_x,
            'grid_disp_y': grid_disp_y,
            'grid_Jm1_ex_x': grid_Jm1_ex_x,
            'grid_Jm1_ex_y': grid_Jm1_ex_y,
            'grid_Jm1_ey_x': grid_Jm1_ey_x,
            'grid_Jm1_ey_y': grid_Jm1_ey_y,
            'x_lim': (x_min, x_max),
            'y_lim': (y_min, y_max),
            'var_explained': var_explained,
            # ── Compatibility ───────────────────────────────────
            'per_token_divergence': per_token_volume,
            'per_token_curl': per_token_rotation,
            'per_token_shear': per_token_shear_mag,
            'principal_directions': pca_basis.numpy(),
        })

    return fields

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
                matplotlib.use("TkAgg")
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
    def update_jacobi_fields(self, model: nn.Module, input_ids: torch.Tensor):
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
                model, input_ids, input_ids.device, max_tokens=1024
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

            try:
                if run_dir is not None:
                    jacobi_dir = os.path.join(run_dir, "jacobi_data")
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
            except Exception as e:
                console.print(f"[yellow]⚠ Could not save Jacobi data: {e}[/]")

        except Exception as e:
            import traceback
            console.print(f"[yellow]\u26a0 Jacobi field error at step {self._kelp_step}: {e}[/]")
            traceback.print_exc()
        finally:
            if was_training:
                model.train()

    def _draw_jacobi_fields(self, ax, jacobi_data, draw_key):
        """
        Render the Jacobi field as a colorful vector field image per layer.

        Each layer panel shows:
        1. BACKGROUND: HSV color field
           - Hue = direction of space deformation (angle of (J-I)·r)
           - Brightness = magnitude of deformation
           - Warm/cool tint for volume expansion/contraction
        2. STREAMLINES: displacement field (J-I)·r as continuous curves
        3. STRETCH WHISKERS: max stretch direction at sparse grid points
        4. TOKEN MARKERS: fixed observers, ringed by volume change
        5. COLOR WHEEL LEGEND + TEXT LEGEND on the right
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

        var_exp = fields[0].get('var_explained', 0) if fields else 0
        ax.set_title(
            f"Jacobi Fields \u2014 Space Deformation (step {step}, "
            f"PCA var={var_exp:.0%})",
            fontsize=10, fontweight="bold", color="#c0d8e8",
        )

        if n_layers == 0:
            ax.text(0.5, 0.5, "Waiting for hidden states...",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    color="#6a9ab8", transform=ax.transAxes)
            return

        try:
            self.fig.set_tight_layout(False)
        except Exception:
            pass

        from matplotlib.colors import hsv_to_rgb

        # ═══════════════════════════════════════════════════════════════
        # Layout: reserve right edge for the color wheel legend
        # ═══════════════════════════════════════════════════════════════
        legend_width_frac = 0.12
        field_width_frac = 1.0 - legend_width_frac

        n_cols = min(n_layers, 6)
        n_rows = math.ceil(n_layers / n_cols)

        pad_x = 0.02
        pad_y = 0.10
        pad_bottom = 0.03
        cell_w = (field_width_frac - 2 * pad_x) / n_cols
        cell_h = (1.0 - pad_y - pad_bottom) / n_rows
        inset_margin = 0.005

        parent_bbox = ax.get_position()
        px0, py0 = parent_bbox.x0, parent_bbox.y0
        pw, ph = parent_bbox.width, parent_bbox.height

        for ell, f in enumerate(fields):
            col = ell % n_cols
            row = n_rows - 1 - (ell // n_cols)

            fx = px0 + pw * (pad_x + col * cell_w + inset_margin)
            fy = py0 + ph * (pad_bottom + row * cell_h + inset_margin)
            fw = pw * (cell_w - 2 * inset_margin)
            fh = ph * (cell_h - 2 * inset_margin)

            sub_ax = self.fig.add_axes([fx, fy, fw, fh])
            sub_ax.set_facecolor("#0d1117")
            sub_ax.set_in_layout(False)
            self._jacobi_subaxes.append(sub_ax)

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
            per_token_rotation = f['per_token_rotation']

            grid_n = grid_x.shape[0]

            # ═══════════════════════════════════════════════════════
            # 1. COLORFUL BACKGROUND: HSV encoding of deformation
            # ═══════════════════════════════════════════════════════
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

            # Volume change tint
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
                origin='lower',
                aspect='auto',
                interpolation='bilinear',
                alpha=0.9,
            )

            # ═══════════════════════════════════════════════════════
            # 2. STREAMLINES
            # ═══════════════════════════════════════════════════════
            gx_1d = np.linspace(x_lim[0], x_lim[1], grid_n)
            gy_1d = np.linspace(y_lim[0], y_lim[1], grid_n)

            try:
                speed = np.sqrt(grid_disp_x**2 + grid_disp_y**2)
                lw = 0.3 + 1.2 * speed / max(speed.max(), 1e-10)

                sub_ax.streamplot(
                    gx_1d, gy_1d, grid_disp_x, grid_disp_y,
                    color='white',
                    linewidth=lw,
                    density=0.7,
                    arrowsize=0.6,
                    arrowstyle='->',
                    zorder=2,
                    minlength=0.2,
                )
            except Exception:
                pass

            # ═══════════════════════════════════════════════════════
            # 3. STRETCH WHISKERS
            # ═══════════════════════════════════════════════════════
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

            # ═══════════════════════════════════════════════════════
            # 4. TOKEN MARKERS
            # ═══════════════════════════════════════════════════════
            shear_max = max(per_token_shear.max(), 1e-6)
            shear_norm = per_token_shear / shear_max

            sub_ax.scatter(
                h_in_2d[:, 0], h_in_2d[:, 1],
                s=10, c=shear_norm, cmap='magma',
                vmin=0, vmax=1,
                edgecolors='none',
                zorder=6, alpha=0.9,
            )

            vol_threshold_high = np.percentile(per_token_volume, 85)
            vol_threshold_low = np.percentile(per_token_volume, 15)
            expanding = per_token_volume > vol_threshold_high
            contracting = per_token_volume < vol_threshold_low

            if expanding.any():
                sub_ax.scatter(
                    h_in_2d[expanding, 0], h_in_2d[expanding, 1],
                    s=28, facecolors='none', edgecolors='#ff4444',
                    linewidths=0.8, zorder=7,
                )
            if contracting.any():
                sub_ax.scatter(
                    h_in_2d[contracting, 0], h_in_2d[contracting, 1],
                    s=28, facecolors='none', edgecolors='#4488ff',
                    linewidths=0.8, zorder=7,
                )

            # ═══════════════════════════════════════════════════════
            # 5. LAYER LABEL
            # ═══════════════════════════════════════════════════════
            layer_label = "Emb\u2192L1" if ell == 0 else f"L{ell}\u2192{ell+1}"
            mean_vol = per_token_volume.mean()
            aniso_val = f['anisotropy']

            sub_ax.set_title(
                f"{layer_label}  det={mean_vol:.2f}  \u03c3\u2081/\u03c3\u2099={aniso_val:.1f}",
                fontsize=7, fontweight="bold", color="#aaccee", pad=2,
            )

            sub_ax.set_xlim(*x_lim)
            sub_ax.set_ylim(*y_lim)
            sub_ax.set_aspect('auto')
            sub_ax.tick_params(labelsize=0, length=0)
            for spine in sub_ax.spines.values():
                spine.set_color('#2a3a4a')
                spine.set_linewidth(0.5)

        # ═══════════════════════════════════════════════════════════════
        # 6. COLOR WHEEL LEGEND (Cartesian imshow, not polar pcolormesh)
        # ═══════════════════════════════════════════════════════════════
        legend_x = px0 + pw * (field_width_frac + 0.01)
        legend_w = pw * (legend_width_frac - 0.02)

        wheel_size = min(legend_w, ph * 0.22)
        wheel_y = py0 + ph * 0.73
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

        # Mask outside the unit circle
        mask = W_radius > 1.0
        W_S[mask] = 0.0
        W_V[mask] = 0.0

        hsv_wheel = np.stack([W_H, W_S, W_V], axis=-1)
        rgb_wheel = hsv_to_rgb(hsv_wheel)

        # RGBA with transparent outside
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

        # Direction labels at cardinal points
        lbl_cfg = dict(fontsize=6.5, fontweight='bold', color='white',
                       ha='center', va='center')
        wheel_ax.text(1.35, 0.0, '+PC1\n\u2192', **lbl_cfg)
        wheel_ax.text(-1.35, 0.0, '\u2190\n\u2212PC1', **lbl_cfg)
        wheel_ax.text(0.0, 1.35, '\u2191 +PC2', **lbl_cfg)
        wheel_ax.text(0.0, -1.35, '\u2193 \u2212PC2', **lbl_cfg)

        # Center label
        wheel_ax.text(0.0, 0.0, 'weak', fontsize=5, color='#666666',
                      ha='center', va='center', fontstyle='italic')

        wheel_ax.set_xlim(-1.7, 1.7)
        wheel_ax.set_ylim(-1.7, 1.7)
        wheel_ax.axis('off')

        wheel_ax.set_title("Deformation\nDirection & Strength",
                           fontsize=8, fontweight='bold', color='#c0d8e8',
                           pad=8)

        # ═══════════════════════════════════════════════════════════════
        # 7. TEXT LEGEND below the color wheel
        # ═══════════════════════════════════════════════════════════════
        legend_text_y = py0 + ph * 0.03
        legend_text_h = ph * 0.68
        legend_text_ax = self.fig.add_axes(
            [legend_x, legend_text_y, legend_w, legend_text_h]
        )
        legend_text_ax.set_facecolor("#0a0a1a")
        legend_text_ax.set_xlim(0, 1)
        legend_text_ax.set_ylim(0, 1)
        legend_text_ax.axis('off')
        legend_text_ax.set_in_layout(False)
        self._jacobi_subaxes.append(legend_text_ax)

        # ── Legend items: (symbol, description, color, is_header) ──
        # For non-header items, symbol and description are rendered
        # on the SAME line to avoid overlap.
        legend_items = [
            ("BACKGROUND", "", "#c0d8e8", True),
            ("\u2588\u2588 Hue", "deformation direction", "#ffffff", False),
            ("\u2588\u2588 Bright", "strong deformation", "#dddddd", False),
            ("\u2588\u2588 Dark", "weak / no deformation", "#555555", False),
            ("\u2588\u2588 Red tint", "expanding (det J > 1)", "#ff6666", False),
            ("\u2588\u2588 Blue tint", "contracting (det J < 1)", "#6688ff", False),
            ("", "", "#000000", False),
            ("OVERLAYS", "", "#c0d8e8", True),
            ("\u2500\u2500 white", "streamlines (flow)", "#ffffff", False),
            ("\u2500\u2500 yellow", "mild anisotropic stretch", "#ffff44", False),
            ("\u2500\u2500 orange", "moderate stretch", "#ff8800", False),
            ("\u2500\u2500 red", "extreme stretch", "#ff2222", False),
            ("", "", "#000000", False),
            ("TOKENS", "", "#c0d8e8", True),
            ("\u25cf dot", "shear magnitude (magma)", "#dd6644", False),
            ("\u25cb red ring", "expanding (top 15%)", "#ff4444", False),
            ("\u25a1 blue ring", "contracting (top 15%)", "#4488ff", False),
        ]

        n_items = len(legend_items)
        line_spacing = 0.95 / max(n_items, 1)

        for i, (symbol, desc, color, is_header) in enumerate(legend_items):
            y = 0.97 - i * line_spacing
            if not symbol and not desc:
                continue  # spacer line

            if is_header:
                # Section header — bold, larger font, standalone
                legend_text_ax.text(
                    0.05, y, symbol,
                    fontsize=7, fontweight='bold', color=color,
                    fontfamily='sans-serif', va='top',
                    transform=legend_text_ax.transAxes,
                )
            else:
                # Symbol + description on ONE line to prevent overlap
                # Symbol in its own color, then description in gray
                # We render the colored symbol first, then the gray desc
                # next to it using a fixed x-offset.
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
                scores.append(1.0)  # unparseable → 1.0
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

