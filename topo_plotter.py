# topo_plotter.py
"""
Live topological barcode visualization for layer activations and feature spaces.

Computes Vietoris-Rips persistent homology (H0, H1) on hidden-state point clouds
sampled during training, and renders barcodes + persistence diagrams in a
dedicated matplotlib window that updates every N batches.

Usage:
    Instantiate TopoPlotter alongside your LivePlotter, call .update() with
    the model and a sample batch, and it will handle the rest.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import threading
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

try:
    from persim import plot_diagrams
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TopoPlotter:
    """
    Second matplotlib window showing live topological barcodes of
    layer activations and feature spaces during training.

    For each transformer block (+ embedding + final LN), we:
      1. Collect hidden states for a small probe batch
      2. Subsample tokens to keep computation tractable
      3. Run Vietoris-Rips persistent homology (H0 and H1)
      4. Render barcodes (left column) and persistence diagrams (right column)

    The window updates every `update_every` batches.
    """

    def __init__(
        self,
        enabled: bool = True,
        update_every: int = 50,
        max_points: int = 200,
        max_homology_dim: int = 1,
        max_layers_to_show: int = 6,
        pca_components: int = 30,
        use_pca: bool = True,
    ):
        self.enabled = enabled and HAS_RIPSER
        self.update_every = update_every
        self.max_points = max_points
        self.max_homology_dim = max_homology_dim
        self.max_layers_to_show = max_layers_to_show
        self.pca_components = pca_components
        self.use_pca = use_pca

        self._global_step = 0
        self._lock = threading.Lock()

        # History: track topological summaries over time
        self.betti_history: Dict[str, List[List[int]]] = {}
        self.total_persistence_history: Dict[str, List[float]] = {}
        self._history_steps: List[int] = []

        if not self.enabled:
            if not HAS_RIPSER:
                print("[TopoPlotter] ripser not installed. "
                      "Install with: pip install ripser")
            return

        self._init_figure()

    def _init_figure(self):
        """Create the second matplotlib figure for topological visualization."""
        matplotlib.use("TkAgg")
        plt.ion()

        # We'll create a grid: rows = layers, cols = [barcode, persistence diagram, PCA scatter]
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.suptitle(
            "Topological Barcodes — Layer Activations",
            fontsize=14, fontweight="bold"
        )
        try:
            self.fig.canvas.manager.set_window_title(
                "LLVM IR GPT — Topological Analysis"
            )
        except Exception:
            pass

        # We'll dynamically create subplots when we know the number of layers
        self._axes_created = False
        self._n_display_layers = 0

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _ensure_axes(self, n_layers: int):
        """Create subplot grid once we know how many layers to display."""
        if self._axes_created and self._n_display_layers == n_layers:
            return

        self.fig.clear()
        self._n_display_layers = n_layers

        # 3 columns: barcode, persistence diagram, Betti number evolution
        self.axes_barcode = []
        self.axes_persistence = []
        self.axes_betti = []

        for i in range(n_layers):
            ax_bar = self.fig.add_subplot(n_layers, 3, i * 3 + 1)
            ax_per = self.fig.add_subplot(n_layers, 3, i * 3 + 2)
            ax_bet = self.fig.add_subplot(n_layers, 3, i * 3 + 3)

            self.axes_barcode.append(ax_bar)
            self.axes_persistence.append(ax_per)
            self.axes_betti.append(ax_bet)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._axes_created = True

    def _compute_persistence(
        self, points: np.ndarray
    ) -> Optional[Dict]:
        """
        Run Vietoris-Rips persistent homology on a point cloud.

        Returns dict with 'dgms' (list of diagrams per dimension)
        or None on failure.
        """
        if points.shape[0] < 3:
            return None

        try:
            # Optionally reduce dimensionality for speed
            if self.use_pca and points.shape[1] > self.pca_components:
                scaler = StandardScaler()
                points = scaler.fit_transform(points)
                n_components = min(self.pca_components, points.shape[0], points.shape[1])
                pca = PCA(n_components=n_components)
                points = pca.fit_transform(points)

            result = ripser(
                points,
                maxdim=self.max_homology_dim,
                thresh=np.percentile(
                    np.linalg.norm(
                        points[:, None] - points[None, :], axis=-1
                    ).flatten(),
                    50  # Use median distance as threshold to keep it tractable
                ),
            )
            return result
        except Exception as e:
            print(f"[TopoPlotter] Persistence computation failed: {e}")
            return None

    def _draw_barcode(self, ax, dgms: list, layer_name: str):
        """Draw a persistence barcode on the given axes."""
        ax.clear()
        ax.set_title(f"Barcode: {layer_name}", fontsize=9, fontweight="bold")

        colors = ["#2196F3", "#FF5722", "#4CAF50"]  # H0=blue, H1=red, H2=green
        labels = ["H₀ (components)", "H₁ (loops)", "H₂ (voids)"]

        y_offset = 0
        y_ticks = []
        y_labels = []

        for dim, dgm in enumerate(dgms):
            if dim > self.max_homology_dim:
                break

            finite_bars = dgm[np.isfinite(dgm[:, 1])]
            if len(finite_bars) == 0:
                continue

            # Sort by birth time
            sorted_idx = np.argsort(finite_bars[:, 0])
            finite_bars = finite_bars[sorted_idx]

            for bar in finite_bars:
                birth, death = bar
                ax.plot(
                    [birth, death],
                    [y_offset, y_offset],
                    color=colors[dim % len(colors)],
                    linewidth=2.5,
                    alpha=0.8,
                    solid_capstyle="butt",
                )
                y_offset += 1

            # Add dimension label
            mid_y = y_offset - len(finite_bars) / 2
            y_ticks.append(mid_y)
            y_labels.append(labels[dim] if dim < len(labels) else f"H_{dim}")

        ax.set_xlabel("Filtration value", fontsize=7)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.grid(True, alpha=0.2, axis="x")
        ax.tick_params(axis="x", labelsize=7)

    def _draw_persistence_diagram(self, ax, dgms: list, layer_name: str):
        """Draw a persistence diagram on the given axes."""
        ax.clear()
        ax.set_title(f"Persistence: {layer_name}", fontsize=9, fontweight="bold")

        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        labels = ["H₀", "H₁", "H₂"]

        all_vals = []
        for dim, dgm in enumerate(dgms):
            if dim > self.max_homology_dim:
                break
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                all_vals.extend(finite.flatten().tolist())
                ax.scatter(
                    finite[:, 0], finite[:, 1],
                    s=20, alpha=0.7,
                    color=colors[dim % len(colors)],
                    label=labels[dim] if dim < len(labels) else f"H_{dim}",
                    edgecolors="white", linewidth=0.3,
                )

        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.1 + 1e-6
            ax.plot(
                [lo - margin, hi + margin],
                [lo - margin, hi + margin],
                "k--", alpha=0.3, linewidth=1,
            )
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

        ax.set_xlabel("Birth", fontsize=7)
        ax.set_ylabel("Death", fontsize=7)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")

    def _draw_betti_evolution(self, ax, layer_name: str):
        """Draw Betti number evolution over training steps."""
        ax.clear()
        ax.set_title(f"Betti Evolution: {layer_name}", fontsize=9, fontweight="bold")

        if layer_name not in self.betti_history or len(self.betti_history[layer_name]) < 2:
            ax.text(0.5, 0.5, "Collecting...", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, alpha=0.5)
            return

        history = np.array(self.betti_history[layer_name])
        steps = self._history_steps[:len(history)]

        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        labels = ["β₀", "β₁", "β₂"]

        for dim in range(min(history.shape[1], self.max_homology_dim + 1)):
            ax.plot(
                steps, history[:, dim],
                color=colors[dim % len(colors)],
                label=labels[dim] if dim < len(labels) else f"β_{dim}",
                linewidth=1.5, alpha=0.8,
            )

        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Betti number", fontsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    def _extract_hidden_states(
        self, model, input_ids: torch.Tensor
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Forward pass with output_hidden_states=True, return list of
        (layer_name, point_cloud) pairs.
        """
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids, output_hidden_states=True)

        hidden_states = output.hidden_states  # tuple of tensors
        if hidden_states is None:
            return []

        results = []
        layer_names = ["Embed+Pos"]
        for i in range(len(hidden_states) - 1):
            layer_names.append(f"Block {i}")

        for idx, (name, hs) in enumerate(zip(layer_names, hidden_states)):
            # hs shape: (batch, seq_len, d_model)
            # Flatten batch and seq dimensions to get a point cloud
            points = hs.reshape(-1, hs.shape[-1]).cpu().numpy()

            # Subsample if too many points
            if points.shape[0] > self.max_points:
                indices = np.random.choice(
                    points.shape[0], self.max_points, replace=False
                )
                points = points[indices]

            results.append((name, points))

        # Also add the final layer norm output
        if output.last_hidden_state is not None:
            final = output.last_hidden_state.reshape(
                -1, output.last_hidden_state.shape[-1]
            ).cpu().numpy()
            if final.shape[0] > self.max_points:
                indices = np.random.choice(
                    final.shape[0], self.max_points, replace=False
                )
                final = final[indices]
            results.append(("Final LN", final))

        # Limit number of layers shown
        if len(results) > self.max_layers_to_show:
            # Always show first, last, and evenly spaced middle layers
            indices = np.linspace(0, len(results) - 1,
                                  self.max_layers_to_show, dtype=int)
            results = [results[i] for i in indices]

        return results

    def _compute_betti_numbers(self, dgms: list) -> List[int]:
        """Compute Betti numbers from persistence diagrams."""
        betti = []
        for dim, dgm in enumerate(dgms):
            if dim > self.max_homology_dim:
                break
            # Count bars with significant persistence
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) == 0:
                betti.append(0)
                continue
            persistences = finite[:, 1] - finite[:, 0]
            # Use a threshold: bars with persistence > 10% of max
            if len(persistences) > 0:
                threshold = np.percentile(persistences, 25)
                betti.append(int(np.sum(persistences > threshold)))
            else:
                betti.append(0)
        return betti

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        probe_input_ids: torch.Tensor,
        force: bool = False,
    ):
        """
        Call this every training batch. Actual computation only happens
        every `update_every` steps.

        Args:
            model: The TinyGPT model
            probe_input_ids: A small batch of input_ids to probe activations
            force: Force update regardless of step count
        """
        if not self.enabled:
            return

        self._global_step += 1
        if not force and (self._global_step % self.update_every != 0):
            return

        with self._lock:
            self._do_update(model, probe_input_ids)

    def _do_update(self, model, probe_input_ids):
        """Perform the actual topological computation and plot update."""
        was_training = model.training
        model.eval()

        try:
            # 1. Extract hidden states
            layer_data = self._extract_hidden_states(model, probe_input_ids)
            if not layer_data:
                return

            n_layers = len(layer_data)
            self._ensure_axes(n_layers)

            # 2. Compute persistence for each layer
            for i, (layer_name, points) in enumerate(layer_data):
                result = self._compute_persistence(points)
                if result is None:
                    continue

                dgms = result["dgms"]

                # Draw barcode
                self._draw_barcode(self.axes_barcode[i], dgms, layer_name)

                # Draw persistence diagram
                self._draw_persistence_diagram(
                    self.axes_persistence[i], dgms, layer_name
                )

                # Track Betti numbers
                betti = self._compute_betti_numbers(dgms)
                # Pad to consistent length
                while len(betti) <= self.max_homology_dim:
                    betti.append(0)

                if layer_name not in self.betti_history:
                    self.betti_history[layer_name] = []
                    self.total_persistence_history[layer_name] = []

                self.betti_history[layer_name].append(betti)

                # Total persistence
                total_pers = sum(
                    np.sum(dgm[np.isfinite(dgm[:, 1])][:, 1] -
                           dgm[np.isfinite(dgm[:, 1])][:, 0])
                    for dgm in dgms if len(dgm) > 0
                )
                self.total_persistence_history[layer_name].append(total_pers)

                # Draw Betti evolution
                self._draw_betti_evolution(self.axes_betti[i], layer_name)

            # Record step
            self._history_steps.append(self._global_step)

            # Update title with step info
            self.fig.suptitle(
                f"Topological Barcodes — Layer Activations  "
                f"(step {self._global_step})",
                fontsize=14, fontweight="bold",
            )

            self.fig.tight_layout(rect=[0, 0, 1, 0.95])
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"[TopoPlotter] Update failed: {e}")
        finally:
            if was_training:
                model.train()

    def finalize(self):
        """Call at end of training to keep the window open."""
        if not self.enabled:
            return
        plt.ioff()
        plt.show()
