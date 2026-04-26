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

        # ── Kelp forest config ──────────────────────────────────────────
        self.kelp_every = kelp_every
        self._kelp_step = 0
        self._kelp_data = None
        self._kelp_time_offset = 0.0

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

    def _draw_kelp_forest(self):
        """
        Render the kelp forest visualization onto self.ax_kelp.

        Fibre-bundle interpretation (per paper §2):
          - X axis  = token index i ∈ M  (base manifold — the "sea floor")
          - Y axis  = layer depth ℓ = 0 … L  (embedding → output)
          - Each kelp strand = one fibre F_i = (V_i^(0), …, V_i^(L))
          - Within each strand, there are exactly n_layers discrete segments
          - Sway amplitude  ∝ mean activation norm   (proxy for ‖Φ^(ℓ)‖, divergence)
          - Sway frequency  ∝ std of activation norms (proxy for shear anisotropy)
          - Lean / twist    ∝ cosine drift            (proxy for Procrustes deviation / connection strength)
          - Color intensity ∝ per-token norm at that layer
          - ALL values are deterministic — derived solely from hidden_states

        No randomness anywhere. Every visual element is a function of the
        network's actual activation geometry.
        """
        ax = self.ax_kelp
        if ax is None or self._kelp_data is None:
            return

        ax.clear()
        ax.set_facecolor("#020a1a")
        ax.set_title(
                f"Kelp Forest — Fibre Bundle Dynamics  (step {self._kelp_data['step']})",
                fontsize=10, fontweight="bold", color="#c0d8e8",
                )

        data = self._kelp_data
        n_layers = data["n_layers"]       # number of hidden states (embedding + L blocks)
        n_tokens = data["n_tokens"]
        layer_stats = data["layer_stats"]
        t_anim = self._kelp_time_offset

        if n_tokens == 0 or n_layers < 1:
            ax.text(0.5, 0.5, "Not enough data",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    color="#6a9ab8", transform=ax.transAxes)
            return

        # ── Normalize stats for visual mapping ──────────────────────────
        all_mean_norms = [s["mean_norm"] for s in layer_stats]
        max_mean_norm = max(all_mean_norms) if max(all_mean_norms) > 0 else 1.0

        all_std_norms = [s["std_norm"] for s in layer_stats]
        max_std_norm = max(all_std_norms) if max(all_std_norms) > 0 else 1.0

        all_drifts = [s["cosine_drift"] for s in layer_stats]
        max_drift = max(abs(d) for d in all_drifts) if any(d != 0 for d in all_drifts) else 1.0

        # Global token norm for color normalization
        all_token_norms = []
        for s in layer_stats:
            all_token_norms.extend(s["token_norms"].tolist())
        global_max_tnorm = max(all_token_norms) if all_token_norms and max(all_token_norms) > 0 else 1.0

        # ── Layout parameters ───────────────────────────────────────────
        x_margin = 0.08
        y_floor = 0.05
        y_top = 0.92
        total_height = y_top - y_floor

        # ── Draw layer zone backgrounds ─────────────────────────────────
        # Each layer gets a band whose brightness is derived from that
        # layer's mean norm — NOT arbitrary colors.
        for li in range(n_layers):
            frac_lo = li / n_layers
            frac_hi = (li + 1) / n_layers
            y_lo = y_floor + frac_lo * total_height
            y_hi = y_floor + frac_hi * total_height

            # Brightness from mean norm of this layer (deterministic)
            norm_frac = layer_stats[li]["mean_norm"] / max_mean_norm
            # Map to a subtle blue-dark range
            r = 0.02 + 0.04 * norm_frac
            g = 0.06 + 0.06 * norm_frac
            b = 0.10 + 0.10 * norm_frac
            zone_alpha = 0.3 + 0.2 * norm_frac
            ax.axhspan(y_lo, y_hi, facecolor=(r, g, b), alpha=zone_alpha, zorder=0)

        # ── Draw sea floor (deterministic sine waves) ───────────────────
        floor_xs = np.linspace(0, 1, 200)
        floor_ys = y_floor + 0.008 * np.sin(floor_xs * 15 + t_anim * 0.3) + \
                0.004 * np.sin(floor_xs * 37 + t_anim * 0.7)
        ax.fill_between(floor_xs, 0, floor_ys, color="#1a0a30", alpha=0.8)
        ax.plot(floor_xs, floor_ys, color="#4a2a6a", linewidth=1.0, alpha=0.6)

        # ── Draw light rays (positions derived from layer stats) ────────
        # Use the per-layer mean norms to position rays deterministically
        from matplotlib.patches import Polygon as MplPolygon
        n_rays = min(4, n_layers)
        for ray_i in range(n_rays):
            # Position derived from the layer's mean norm
            stat_idx = int(ray_i * (n_layers - 1) / max(n_rays - 1, 1))
            norm_val = layer_stats[stat_idx]["mean_norm"] / max_mean_norm
            ray_x = 0.1 + ray_i * (0.8 / max(n_rays - 1, 1)) + 0.03 * np.sin(t_anim * 0.2 + norm_val * 6.28)
            ray_width = 0.02 + 0.015 * norm_val
            sway = 0.02 * np.sin(t_anim * 0.15 + ray_i * 1.618)
            ray_verts = [
                    (ray_x - ray_width, 1.0),
                    (ray_x + ray_width, 1.0),
                    (ray_x + ray_width * 2 + sway, 0.0),
                    (ray_x - ray_width * 2 + sway, 0.0),
                    ]
            ray_patch = MplPolygon(ray_verts, closed=True,
                                   facecolor="#4080b0", alpha=0.03)
            ax.add_patch(ray_patch)

        # ── Draw kelp strands (fibres) ──────────────────────────────────
        # Each strand = one fibre F_i, with EXACTLY n_layers discrete
        # segments. Each segment's geometry comes from that layer's stats.

        max_display_tokens = min(n_tokens, 32)
        if n_tokens > max_display_tokens:
            token_indices = np.linspace(0, n_tokens - 1, max_display_tokens, dtype=int)
        else:
            token_indices = np.arange(n_tokens)

        actual_spacing = (1.0 - 2 * x_margin) / max(len(token_indices) - 1, 1)

        # Sub-segments per layer for smooth curves within each layer zone
        sub_segments_per_layer = 8

        from matplotlib.collections import LineCollection

        for ti_display, ti in enumerate(token_indices):
            base_x = x_margin + ti_display * actual_spacing

            # Deterministic phase unique to this token, derived from its
            # embedding-layer norm (the actual network value, not random)
            tn_embed = layer_stats[0]["token_norms"]
            tok_embed_norm = tn_embed[min(ti, len(tn_embed) - 1)] if len(tn_embed) > 0 else 0
            phase = (tok_embed_norm / global_max_tnorm) * 6.2832  # map to [0, 2π]

            # Build path: for each layer, create sub_segments_per_layer points
            all_xs = []
            all_ys = []
            all_colors = []
            all_layer_ids = []

            for li in range(n_layers):
                mean_n = layer_stats[li]["mean_norm"]
                std_n = layer_stats[li]["std_norm"]
                drift = layer_stats[li]["cosine_drift"]

                # Per-token norm at this layer
                tn = layer_stats[li]["token_norms"]
                tok_norm = tn[min(ti, len(tn) - 1)] if len(tn) > 0 else 0

                # Sway parameters — ALL from real network values
                amplitude = (mean_n / max_mean_norm) * 0.04
                freq = 1.0 + 2.0 * (std_n / max_std_norm)
                lean = (drift / max_drift) * 0.03 if max_drift > 0 else 0

                for si in range(sub_segments_per_layer):
                    sub_frac = si / sub_segments_per_layer
                    # Global vertical fraction
                    global_frac = (li + sub_frac) / n_layers
                    y_val = y_floor + global_frac * total_height

                    # Height-dependent amplitude scaling (taller = more sway)
                    height_scale = 0.3 + 0.7 * global_frac

                    sway = (amplitude * height_scale *
                            np.sin(t_anim * freq * 0.8 + phase + global_frac * 3.0)
                            + amplitude * 0.4 * height_scale *
                            np.sin(t_anim * freq * 1.3 + phase * 2 + global_frac * 5.0)
                            + lean * global_frac *
                            np.sin(t_anim * 0.5 + phase))

                    all_xs.append(base_x + sway)
                    all_ys.append(y_val)
                    all_colors.append(tok_norm / global_max_tnorm)
                    all_layer_ids.append(li)

            # Add the final point at the top
            li_last = n_layers - 1
            tn_last = layer_stats[li_last]["token_norms"]
            tok_norm_last = tn_last[min(ti, len(tn_last) - 1)] if len(tn_last) > 0 else 0
            mean_n_last = layer_stats[li_last]["mean_norm"]
            std_n_last = layer_stats[li_last]["std_norm"]
            drift_last = layer_stats[li_last]["cosine_drift"]
            amplitude_last = (mean_n_last / max_mean_norm) * 0.04
            freq_last = 1.0 + 2.0 * (std_n_last / max_std_norm)
            lean_last = (drift_last / max_drift) * 0.03 if max_drift > 0 else 0

            sway_top = (amplitude_last * np.sin(t_anim * freq_last * 0.8 + phase + 3.0)
                        + amplitude_last * 0.4 * np.sin(t_anim * freq_last * 1.3 + phase * 2 + 5.0)
                        + lean_last * np.sin(t_anim * 0.5 + phase))
            all_xs.append(base_x + sway_top)
            all_ys.append(y_top)
            all_colors.append(tok_norm_last / global_max_tnorm)
            all_layer_ids.append(li_last)

            all_xs = np.array(all_xs)
            all_ys = np.array(all_ys)
            all_colors = np.array(all_colors)
            all_layer_ids = np.array(all_layer_ids)

            # Build line segments
            points = np.array([all_xs, all_ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color: green hue, brightness from token norm at that layer
            # Per-layer hue shift derived from that layer's cosine drift
            colors_rgba = []
            for ci in range(len(segments)):
                intensity = 0.25 + 0.75 * all_colors[ci]
                li_ci = all_layer_ids[ci]
                # Hue shift from cosine drift (deterministic, from network)
                drift_ci = layer_stats[li_ci]["cosine_drift"]
                drift_norm = abs(drift_ci) / max_drift if max_drift > 0 else 0
                r = 0.05 + 0.12 * drift_norm
                g = 0.15 + 0.35 * intensity
                b = 0.10 + 0.15 * drift_norm
                a = 0.4 + 0.5 * intensity
                colors_rgba.append((r, g, b, a))

            # Strand thickness: thicker at base, thinner at top
            fracs = np.linspace(0, 1, len(segments))
            linewidths = 2.5 - 1.5 * fracs

            lc = LineCollection(segments, colors=colors_rgba, linewidths=linewidths,
                                capstyle="round", joinstyle="round")
            ax.add_collection(lc)

            # ── Draw nodes at EACH layer boundary ───────────────────────
            # These are the discrete fibre morphism points Φ^(ℓ)
            for li in range(n_layers):
                # Find the exact position of this layer boundary on the strand
                seg_idx = li * sub_segments_per_layer
                seg_idx = min(seg_idx, len(all_xs) - 1)
                node_x = all_xs[seg_idx]
                node_y = all_ys[seg_idx]

                # Node size from layer's mean norm
                node_size = 2.0 + 3.0 * (layer_stats[li]["mean_norm"] / max_mean_norm)

                # Color from per-token norm at this layer
                tn_li = layer_stats[li]["token_norms"]
                tok_n = tn_li[min(ti, len(tn_li) - 1)] if len(tn_li) > 0 else 0
                node_intensity = tok_n / global_max_tnorm

                node_color = (
                        0.2 + 0.3 * node_intensity,
                        0.5 + 0.4 * node_intensity,
                        0.3 + 0.2 * node_intensity,
                        0.6 + 0.3 * node_intensity,
                        )
                ax.plot(node_x, node_y, 'o',
                        color=node_color, markersize=node_size,
                        markeredgecolor=(0.4, 0.8, 0.5, 0.3),
                        markeredgewidth=0.5, zorder=4)

            # ── Draw fronds at layer midpoints (deterministic) ──────────
            # One frond per layer, positioned at the midpoint of each
            # layer zone. Direction derived from cosine drift sign.
            for li in range(n_layers):
                mid_seg = li * sub_segments_per_layer + sub_segments_per_layer // 2
                mid_seg = min(mid_seg, len(all_xs) - 1)
                fx = all_xs[mid_seg]
                fy = all_ys[mid_seg]

                drift_li = layer_stats[li]["cosine_drift"]
                # Frond direction: sign of drift determines left/right
                frond_sign = 1.0 if drift_li >= 0 else -1.0
                # Alternate sides for even/odd tokens
                if ti_display % 2 == 1:
                    frond_sign *= -1.0

                frond_angle = frond_sign * (0.4 + 0.3 * abs(drift_li) / max_drift) if max_drift > 0 else frond_sign * 0.4
                frond_angle += 0.1 * np.sin(t_anim * 0.7 + phase + li)

                # Frond length from std norm (more variation = longer frond)
                frond_len = 0.012 + 0.01 * (layer_stats[li]["std_norm"] / max_std_norm)
                frond_dx = np.cos(frond_angle) * frond_len
                frond_dy = np.sin(frond_angle) * frond_len

                frond_intensity = layer_stats[li]["mean_norm"] / max_mean_norm
                ax.plot([fx, fx + frond_dx], [fy, fy + frond_dy],
                        color=(0.10 + 0.08 * frond_intensity,
                               0.35 + 0.15 * frond_intensity,
                               0.20 + 0.08 * frond_intensity,
                               0.35 + 0.2 * frond_intensity),
                        linewidth=1.0)

            # ── Root anchor ─────────────────────────────────────────────
            from matplotlib.patches import Ellipse
            anchor = Ellipse((base_x, y_floor), width=0.012, height=0.008,
                             facecolor="#2a1540", edgecolor="#4a2a6a",
                             linewidth=0.5, alpha=0.7)
            ax.add_patch(anchor)

        # ── Layer boundary lines ────────────────────────────────────────
        for li in range(n_layers + 1):
            frac = li / n_layers
            y_line = y_floor + frac * total_height

            if li == 0 or li == n_layers:
                ax.axhline(y=y_line, color="#5a8aaa", linewidth=1.0,
                           linestyle="-", alpha=0.4, zorder=2)
            else:
                ax.axhline(y=y_line, color="#4a7a9a", linewidth=0.8,
                           linestyle="--", alpha=0.35, zorder=2)

            # Layer labels
            if li < n_layers:
                label_y = y_floor + (li + 0.5) / n_layers * total_height
                if li == 0:
                    label = "Embed"
                elif li == n_layers - 1:
                    label = "Output"
                else:
                    label = f"L{li}"

                ax.text(0.995, label_y, label,
                        fontsize=7, color="#6a9aba", alpha=0.6,
                        ha="right", va="center", transform=ax.get_yaxis_transform(),
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#0a1a2a",
                                  edgecolor="none", alpha=0.5))

        # ── Water caustics (deterministic, from layer stats) ────────────
        # One caustic per layer, position and size from that layer's stats
        from matplotlib.patches import Circle
        for li in range(min(n_layers, 8)):
            norm_frac = layer_stats[li]["mean_norm"] / max_mean_norm
            std_frac = layer_stats[li]["std_norm"] / max_std_norm
            cx = 0.1 + li * (0.8 / max(n_layers - 1, 1)) + 0.03 * np.sin(t_anim * 0.4 + norm_frac * 6.28)
            cy = y_floor + (li + 0.5) / n_layers * total_height
            caustic_alpha = 0.01 + 0.015 * std_frac
            caustic_radius = 0.04 + 0.03 * norm_frac
            caustic = Circle((cx, cy), radius=caustic_radius,
                             facecolor="#60b0d0",
                             alpha=max(0, min(caustic_alpha, 0.03)),
                             edgecolor="none")
            ax.add_patch(caustic)

        # ── Floating particles (deterministic, from per-token norms) ────
        # Instead of random particles, place one "spore" per token per
        # layer, at a position derived from that token's norm at that layer.
        # This makes every particle a real data point.
        n_display = len(token_indices)
        for li in range(n_layers):
            tn_li = layer_stats[li]["token_norms"]
            for ti_d, ti in enumerate(token_indices):
                tok_n = tn_li[min(ti, len(tn_li) - 1)] if len(tn_li) > 0 else 0
                norm_frac = tok_n / global_max_tnorm

                # Position: offset from the strand, derived from norm
                # Small deterministic displacement so particles don't overlap strands
                px = x_margin + ti_d * actual_spacing + 0.015 * np.sin(norm_frac * 6.28 + li * 1.618)
                py = y_floor + (li + 0.5) / n_layers * total_height + 0.005 * np.cos(norm_frac * 6.28 + ti * 1.618)
                py = np.clip(py, y_floor, y_top)

                p_alpha = 0.08 + 0.15 * norm_frac
                p_size = 0.3 + 1.2 * norm_frac

                ax.plot(px, py, '.', color="#80c0e0",
                        alpha=min(p_alpha, 0.25), markersize=p_size, zorder=5)

        # ── Stats annotation ────────────────────────────────────────────
        mean_norms_str = ", ".join(f"{s['mean_norm']:.1f}" for s in layer_stats[:6])
        if n_layers > 6:
            mean_norms_str += " …"
        drift_str = ", ".join(f"{s['cosine_drift']:.3f}" for s in layer_stats[:6])
        if n_layers > 6:
            drift_str += " …"

        ax.text(0.01, 0.98,
                f"Layers: {n_layers}  Tokens: {n_tokens}  Fibres: {len(token_indices)}\n"
                f"Mean‖h‖: [{mean_norms_str}]\n"
                f"Drift:   [{drift_str}]",
                fontsize=5.5, color="#5a8aaa", alpha=0.5,
                ha="left", va="top", transform=ax.transAxes,
                fontfamily="monospace")

        # ── Axis limits and cleanup ─────────────────────────────────────
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([])
        ax.set_yticks([])


    @torch.no_grad()
    def update_kelp_forest(self, model: nn.Module, input_ids: torch.Tensor):
        """
        Extract hidden-state statistics from the model and render the kelp forest.

        Each kelp strand = one layer's embedding space.
        The strand is rooted at the "index manifold" (bottom axis = token position).
        Sway is driven by per-layer statistics:
          - mean activation magnitude  → lateral sway amplitude
          - std of activations         → sway frequency multiplier
          - cross-layer cosine drift   → additional twist
        """
        if not self.enabled:
            return

        self._kelp_step += 1
        if self._kelp_step != 1 and self._kelp_step % self.kelp_every != 0:
            return

        was_training = model.training
        model.eval()

        try:
            output = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states
            if hidden_states is None or len(hidden_states) < 2:
                return

            # Gather per-layer statistics
            layer_stats = []
            prev_mean_vec = None
            for li, hs in enumerate(hidden_states):
                # hs shape: (B, T, D)
                flat = hs.float().reshape(-1, hs.shape[-1])  # (B*T, D)
                norms = flat.norm(dim=-1)  # (B*T,)
                mean_norm = norms.mean().item()
                std_norm = norms.std().item()
                mean_vec = flat.mean(dim=0)  # (D,)

                # Cosine similarity to previous layer's mean vector
                cosine_drift = 0.0
                if prev_mean_vec is not None:
                    cos = F.cosine_similarity(
                            mean_vec.unsqueeze(0), prev_mean_vec.unsqueeze(0)
                            ).item()
                    cosine_drift = 1.0 - cos  # 0 = identical, ~2 = opposite

                # Per-token norms for this layer (take first batch element)
                token_norms = hs[0].float().norm(dim=-1).cpu().numpy()  # (T,)

                layer_stats.append({
                    "mean_norm": mean_norm,
                    "std_norm": std_norm,
                    "cosine_drift": cosine_drift,
                    "token_norms": token_norms,
                    })
                prev_mean_vec = mean_vec

            self._kelp_data = {
                    "layer_stats": layer_stats,
                    "n_layers": len(hidden_states),
                    "n_tokens": hidden_states[0].shape[1],
                    "step": self._kelp_step,
                    }

            # Advance the sway "time" so animation progresses between updates
            self._kelp_time_offset += 1.0

            self._draw_kelp_forest()
            self._refresh()

        except Exception:
            pass
        finally:
            if was_training:
                model.train()


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
        """Create the figure and return named axes based on topo mode."""
        if self.topo_enabled:
            fig = self.plt.figure(figsize=(26, 22))
            gs = fig.add_gridspec(4, 3, hspace=0.50, wspace=0.35)
            axes = {
                "epoch":   fig.add_subplot(gs[0, 0]),
                "batch":   fig.add_subplot(gs[0, 1]),
                "lr":      fig.add_subplot(gs[0, 2]),
                "val":     fig.add_subplot(gs[1, 0]),
                "barcode": fig.add_subplot(gs[1, 1]),
                "bd":      fig.add_subplot(gs[1, 2]),
                "kelp":    fig.add_subplot(gs[2, 0:2]),
                "diffs":   fig.add_subplot(gs[2, 2]),
                "preds":   fig.add_subplot(gs[3, 0:2]),
                "info":    fig.add_subplot(gs[3, 2]),
            }
        else:
            fig = self.plt.figure(figsize=(24, 18))
            gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
            axes = {
                "epoch":   fig.add_subplot(gs[0, 0]),
                "batch":   fig.add_subplot(gs[0, 1]),
                "lr":      fig.add_subplot(gs[0, 2]),
                "val":     fig.add_subplot(gs[1, 0]),
                "diffs":   fig.add_subplot(gs[1, 1]),
                "kelp":    fig.add_subplot(gs[1, 2]),
                "preds":   fig.add_subplot(gs[2, 0:2]),
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
        """Configure the kelp forest axis with placeholder text."""
        ax.set_title("Kelp Forest — Embedding Space Dynamics",
                      fontsize=10, fontweight="bold")
        ax.set_facecolor("#020a1a")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "Waiting for hidden states...",
                ha="center", va="center", fontsize=11, alpha=0.4,
                color="#6a9ab8", transform=ax.transAxes)
        self.ax_kelp = ax

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

    def _setup_barcode_axis(self, ax):
        """Configure the persistence landscape axis (topo mode only)."""
        if ax is None:
            return
        ax.set_title("Persistence Landscapes (H₁, all layers)",
                      fontsize=10, fontweight="bold")
        ax.text(0.5, 0.5, "Waiting for data...",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11, alpha=0.4)
        ax.grid(True, alpha=0.2)

    def _setup_bd_axis(self, ax):
        """Configure the Wasserstein heatmap axis with a permanent colorbar."""
        if ax is None:
            return
        ax.set_title("Wasserstein-1 Distance Heatmap (layers × layers)",
                      fontsize=10, fontweight="bold")
        self._wass_im = ax.imshow(
            np.zeros((1, 1)),
            cmap="inferno", interpolation="nearest",
            origin="lower", aspect="equal", vmin=0.0, vmax=1.0,
        )
        self._wass_cbar = self.fig.colorbar(
            self._wass_im, ax=ax, fraction=0.046, pad=0.04,
        )
        self._wass_cbar.ax.tick_params(labelsize=6)
        self._wass_cbar.set_label("Relative W₁", fontsize=7)
        self._wass_ax_pos = ax.get_position()

    def _setup_preds_axis(self, ax):
        """Configure the predictions text panel."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Last Batch Predictions (Expected → Got)",
                      fontsize=10, fontweight="bold")

    def _setup_info_axis(self, ax):
        """Configure the model info text panel."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Model / System Info", fontsize=10, fontweight="bold")

    def _collect_plot_axes(self, named_axes):
        """Build the list of axes that have data lines (for relim/autoscale)."""
        data_axes = [
            named_axes["epoch"],
            named_axes["batch"],
            named_axes["lr"],
            named_axes["val"],
        ]
        if self.ax_diffs is not None:
            data_axes.append(self.ax_diffs)
        if self.ax_barcode is not None:
            data_axes.append(self.ax_barcode)
        if self.ax_bd is not None:
            data_axes.append(self.ax_bd)
        return data_axes

    def _apply_figure_chrome(self):
        """Set suptitle, window title, tight_layout, close event, and force draw."""
        self.fig.suptitle("LLVM IR GPT Training", fontsize=14, fontweight="bold")
        try:
            self.fig.canvas.manager.set_window_title("LLVM IR GPT — Live Training")
        except Exception:
            pass

        self.fig.tight_layout()

        if not self.suppress_window:
            self.fig.canvas.mpl_connect("close_event", self._on_close)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    # ── The director ────────────────────────────────────────────────────

    def _create_figure(self):
        """Create (or recreate) the matplotlib figure and all axes.

        This method is a pure director — it delegates every piece of
        setup to a focused helper and orchestrates the data flow between
        them.
        """
        # 1. Build the figure skeleton
        self.fig, named_axes = self._build_figure_and_gridspec()

        # 2. Store optional topo axes
        self.ax_barcode = named_axes["barcode"]
        self.ax_bd = named_axes["bd"]

        # 3. Configure each axis (order doesn't matter — no dependencies)
        self._setup_diffs_axis(named_axes["diffs"])
        self._setup_kelp_axis(named_axes["kelp"])
        self._setup_epoch_axis(named_axes["epoch"])
        self._setup_batch_axis(named_axes["batch"])
        self._setup_lr_axis(named_axes["lr"])
        self._setup_val_axis(named_axes["val"])
        self._setup_barcode_axis(self.ax_barcode)
        self._setup_bd_axis(self.ax_bd)
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
        """Save the current figure to disk."""
        if not self.enabled:
            return
        try:
            save_path = self.plot_file
            if run_dir is not None:
                save_path = os.path.join(run_dir, self.plot_file)
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight")
        except Exception as e:
            pass  # Don't crash training for a file write error

    # ── Refresh helper ──────────────────────────────────────────────────
    def _refresh(self):
        if not self.enabled:
            return

        if not self.suppress_window:
            self._check_reopen()

        _suppress_c_stderr()
        try:
            for ax in self._plot_axes:
                if ax is self.ax_diffs:
                    # Skip relim/autoscale entirely for ax_diffs —
                    # we manage its limits manually in update_prediction_diffs
                    continue
                ax.relim()
                ax.autoscale_view()

            if not self.suppress_window and self._is_window_alive():
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            else:
                self.fig.canvas.draw()
        finally:
            _restore_c_stderr()


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
        """Draw the last batch's expected vs predicted with differences."""
        ax = self.ax_preds
        if ax is None:
            return
        ax.clear()
        ax.axis("off")
        ax.set_title(
            "Predictions (Training + Validation samples)",
            fontsize=10, fontweight="bold",
        )

        if not self._last_predictions:
            ax.text(0.5, 0.5, "Waiting for predictions...",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    transform=ax.transAxes)
            return

        n_show = min(len(self._last_predictions), 10)
        y_positions = np.linspace(0.95, 0.05, n_show)

        for i, (expected, predicted, is_correct) in enumerate(
                self._last_predictions[-n_show:]
        ):
            # ── FIX: Sanitize both strings before any parsing ───────────
            # Strip non-ASCII/non-printable (BPE byte-level artifacts)
            expected_clean = ''.join(
                c for c in expected if c.isascii() and c.isprintable()
            ).strip()
            predicted_clean = ''.join(
                c for c in predicted if c.isascii() and c.isprintable()
            ).strip()

            # Guard against empty strings
            if not expected_clean or expected_clean == "(empty)":
                expected_clean = "(empty)"
            if not predicted_clean or predicted_clean == "(empty)":
                predicted_clean = "(empty)"
            # ── END FIX ─────────────────────────────────────────────────

            # Try to parse both as integers
            exp_parseable = True
            pred_parseable = True
            try:
                exp_val = int(expected_clean)
            except (ValueError, TypeError):
                exp_parseable = False

            try:
                pred_val = int(predicted_clean)
            except (ValueError, TypeError):
                pred_parseable = False

            # ── Determine color and marker from NUMERIC comparison ──────
            if exp_parseable and pred_parseable:
                score = _prediction_error_score(exp_val, pred_val)
                diff = abs(exp_val - pred_val)

                if score == 0.0:
                    color, marker = "green", "✓"
                elif score < 0.2:
                    color, marker = "orange", "≈"
                elif score < 0.5:
                    color, marker = "darkorange", "~"
                else:
                    color, marker = "red", "✗"

                text = (
                    f"{marker}  expected: {exp_val:>6d}  │  "
                    f"got: {pred_val:>6d}  │  diff: {diff}  │  score: {score:.2f}"
                )

            elif exp_parseable and not pred_parseable:
                color = "red"
                marker = "✗"
                pred_display = predicted_clean[:20]
                if len(predicted_clean) > 20:
                    pred_display = pred_display[:20] + "…"
                text = (
                    f"{marker}  expected: {exp_val:>6d}  │  "
                    f"got: {pred_display:<20s}  │  ⚠ UNPARSEABLE"
                )

            elif not exp_parseable and pred_parseable:
                # ── FIX: Handle case where expected is unparseable but
                #    predicted is valid (e.g. expected="(empty)") ────────
                color = "red"
                marker = "✗"
                exp_display = expected_clean[:12]
                text = (
                    f"{marker}  expected: {exp_display:>12s}  │  "
                    f"got: {pred_val:>6d}  │  ⚠ EXPECTED INVALID"
                )

            else:
                color = "red"
                marker = "✗"
                # ── FIX: Use cleaned strings, never raw (may be empty) ──
                exp_display = expected_clean[:12] if expected_clean else "(empty)"
                pred_display = predicted_clean[:12] if predicted_clean else "(empty)"
                text = (
                    f"{marker}  expected: {exp_display:>12s}  │  "
                    f"got: {pred_display:>12s}  │  ⚠ BOTH INVALID"
                )

            ax.text(0.02, y_positions[i], text,
                    fontsize=8, fontfamily="monospace",
                    color=color, transform=ax.transAxes,
                    verticalalignment="center")

        # Summary stats — use numeric comparison with cleaned strings
        n_correct = 0
        n_garbage = 0
        for exp, pred, _ in self._last_predictions:
            exp_c = ''.join(c for c in exp if c.isascii() and c.isprintable()).strip()
            pred_c = ''.join(c for c in pred if c.isascii() and c.isprintable()).strip()
            if _is_int_str(exp_c) and _is_int_str(pred_c):
                if int(exp_c) == int(pred_c):
                    n_correct += 1
            if not _is_int_str(pred_c):
                n_garbage += 1

        n_total = len(self._last_predictions)
        accuracy = n_correct / n_total * 100 if n_total > 0 else 0

        summary = f"Accuracy: {n_correct}/{n_total} ({accuracy:.1f}%)"
        if n_garbage > 0:
            summary += f"  │  ⚠ {n_garbage} unparseable"

        ax.text(0.98, 0.01, summary,
                ha="right", va="bottom", fontsize=9, fontweight="bold",
                transform=ax.transAxes, alpha=0.7)

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

