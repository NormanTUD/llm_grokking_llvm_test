#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "scipy",
#   "scikit-learn",
#   "ripser",
#   "persim",
#   "rich",
# ]
# ///

"""
Jacobi Structure Formation Tracker
====================================

Scans ALL jacobi_step*_layer*.npz files in a run directory and produces:
  1. A step×layer HEATMAP of topological feature counts (H0, H1, H2)
  2. A step×layer HEATMAP of total persistence (lifetime of structures)
  3. FORMATION/DISAPPEARANCE timeline — marks where structures appear/vanish
  4. Per-layer evolution curves showing structure birth/death over training
  5. Phase transition detection — largest jumps in topology
  6. Geometric metric evolution (divergence, curl, shear) as step×layer grids
  7. Wasserstein distance between consecutive steps (same layer)
  8. "Grokking detector" — sudden jumps in topological complexity

Usage:
    python3 track_structures.py runs/0/jacobi_data/
    python3 track_structures.py runs/0/jacobi_data/ --output-dir structure_analysis
    python3 track_structures.py runs/*/jacobi_data/  # multiple runs
"""

import os
import sys
import glob
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc


def compute_exclude_newer_date(days_back=8):
    return (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")


def should_set_exclude_newer():
    return not os.environ.get("UV_EXCLUDE_NEWER")


def restart_with_uv(script_path, args, env):
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", script_path] + args, env)
    except FileNotFoundError:
        print("uv is not installed. Try:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def ensure_safe_env():
    if not should_set_exclude_newer():
        return
    past_date = compute_exclude_newer_date(8)
    os.environ["UV_EXCLUDE_NEWER"] = past_date
    restart_with_uv(sys.argv[0], sys.argv[1:], os.environ)


ensure_safe_env()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import FancyArrowPatch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, norm
from scipy.stats import entropy as scipy_entropy
from scipy.ndimage import gaussian_filter

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("WARNING: ripser not available. Install with: pip install ripser")
    sys.exit(1)

try:
    from persim import wasserstein as wasserstein_distance
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False
    print("WARNING: persim not available. Wasserstein distances disabled.")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich import box

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def parse_filename(filepath: str):
    """Extract step and layer from filename."""
    basename = os.path.basename(filepath)
    step_match = re.search(r'step(\d+)', basename)
    layer_match = re.search(r'layer(\d+)', basename)
    step = int(step_match.group(1)) if step_match else None
    layer = int(layer_match.group(1)) if layer_match else None
    return step, layer


def load_all_files(path: str):
    """Load all jacobi npz files, organized by (step, layer)."""
    if os.path.isdir(path):
        pattern = os.path.join(path, "jacobi_step*_layer*.npz")
        files = sorted(glob.glob(pattern))
    else:
        files = sorted(glob.glob(path))

    if not files:
        console.print(f"[red]No jacobi files found at: {path}[/]")
        return {}

    data_map = {}  # (step, layer) -> data dict
    for f in files:
        step, layer = parse_filename(f)
        if step is not None and layer is not None:
            data_map[(step, layer)] = f

    return data_map


def load_npz(filepath: str):
    """Load npz and return dict."""
    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.files:
        arr = data[key]
        if arr.ndim == 0:
            result[key] = arr.item()
        else:
            result[key] = arr
    data.close()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TOPOLOGY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_topology(data: dict, max_pts: int = 300):
    """Compute persistent homology for a single file's point cloud."""
    # Try output points first (post-deformation), then input
    if 'h_out_2d' in data:
        points = data['h_out_2d']
    elif 'h_in_2d' in data:
        points = data['h_in_2d']
    else:
        return None

    if points.shape[0] < 4:
        return None

    if points.shape[0] > max_pts:
        idx = np.random.choice(points.shape[0], max_pts, replace=False)
        points = points[idx]

    dists = pdist(points)
    if len(dists) == 0 or dists.max() == 0:
        return None

    thresh = np.percentile(dists[dists > 0], 95)

    try:
        result = ripser(points, maxdim=2, thresh=thresh)
        dgms = result['dgms']
    except Exception:
        return None

    topo = {'diagrams': dgms, 'threshold': thresh, 'n_points': points.shape[0]}

    # H0
    h0 = dgms[0]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    topo['h0_count'] = len(h0_finite)
    if len(h0_finite) > 0:
        h0_lt = h0_finite[:, 1] - h0_finite[:, 0]
        topo['h0_total_persistence'] = h0_lt.sum()
        topo['h0_max_lifetime'] = h0_lt.max()
    else:
        topo['h0_total_persistence'] = 0
        topo['h0_max_lifetime'] = 0

    # H1
    if len(dgms) > 1:
        h1 = dgms[1]
        h1_finite = h1[np.isfinite(h1[:, 1])]
        topo['h1_count'] = len(h1_finite)
        if len(h1_finite) > 0:
            h1_lt = h1_finite[:, 1] - h1_finite[:, 0]
            topo['h1_total_persistence'] = h1_lt.sum()
            topo['h1_max_lifetime'] = h1_lt.max()
            topo['h1_mean_lifetime'] = h1_lt.mean()
            # Significant = lifetime > 2× median
            if len(h1_lt) > 1:
                topo['h1_significant'] = int(np.sum(h1_lt > 2 * np.median(h1_lt)))
            else:
                topo['h1_significant'] = len(h1_lt)
        else:
            topo['h1_total_persistence'] = 0
            topo['h1_max_lifetime'] = 0
            topo['h1_mean_lifetime'] = 0
            topo['h1_significant'] = 0
    else:
        topo['h1_count'] = 0
        topo['h1_total_persistence'] = 0
        topo['h1_max_lifetime'] = 0
        topo['h1_mean_lifetime'] = 0
        topo['h1_significant'] = 0

    # H2
    if len(dgms) > 2:
        h2 = dgms[2]
        h2_finite = h2[np.isfinite(h2[:, 1])]
        topo['h2_count'] = len(h2_finite)
        if len(h2_finite) > 0:
            h2_lt = h2_finite[:, 1] - h2_finite[:, 0]
            topo['h2_total_persistence'] = h2_lt.sum()
            topo['h2_max_lifetime'] = h2_lt.max()
        else:
            topo['h2_total_persistence'] = 0
            topo['h2_max_lifetime'] = 0
    else:
        topo['h2_count'] = 0
        topo['h2_total_persistence'] = 0
        topo['h2_max_lifetime'] = 0

    # Persistence entropy
    all_lifetimes = []
    for dim_idx in range(min(len(dgms), 3)):
        finite = dgms[dim_idx][np.isfinite(dgms[dim_idx][:, 1])]
        if len(finite) > 0:
            all_lifetimes.extend((finite[:, 1] - finite[:, 0]).tolist())
    if all_lifetimes:
        lt_arr = np.array(all_lifetimes)
        lt_norm = lt_arr / (lt_arr.sum() + 1e-10)
        topo['persistence_entropy'] = scipy_entropy(lt_norm)
    else:
        topo['persistence_entropy'] = 0.0

    return topo


def compute_geometry(data: dict):
    """Compute geometric metrics from Jacobian and token data."""
    geo = {}

    if 'jacobian' in data:
        J = data['jacobian']
        D = J.shape[0]
        J_sym = (J + J.T) / 2
        J_antisym = (J - J.T) / 2
        geo['divergence'] = float(np.trace(J))
        geo['curl'] = float(norm(J_antisym, 'fro'))
        trace_sym = np.trace(J_sym)
        shear_tensor = J_sym - (trace_sym / D) * np.eye(D)
        geo['shear'] = float(norm(shear_tensor, 'fro'))

        try:
            eigenvalues = np.linalg.eigvals(J)
            geo['spectral_radius'] = float(np.max(np.abs(eigenvalues)))
            geo['n_complex_eig'] = int(np.sum(np.abs(eigenvalues.imag) > 1e-8))
        except Exception:
            geo['spectral_radius'] = 0
            geo['n_complex_eig'] = 0

        try:
            _, S, _ = svd(J)
            geo['condition_number'] = float(S[0] / max(S[-1], 1e-10))
            geo['log_det'] = float(np.sum(np.log(np.clip(S, 1e-10, None))))
        except Exception:
            geo['condition_number'] = 0
            geo['log_det'] = 0

    if 'per_token_volume' in data:
        vol = data['per_token_volume']
        geo['volume_mean'] = float(vol.mean())
        geo['volume_std'] = float(vol.std())
        geo['n_expanding'] = int(np.sum(vol > 1.1))
        geo['n_contracting'] = int(np.sum(vol < 0.9))

    if 'per_token_rotation' in data:
        rot = data['per_token_rotation']
        geo['rotation_mean'] = float(rot.mean())
        geo['rotation_std'] = float(rot.std())

    if 'per_token_shear_mag' in data:
        shear = data['per_token_shear_mag']
        geo['shear_mean'] = float(shear.mean())
        geo['shear_max'] = float(shear.max())

    if 'h_in_2d' in data and 'h_out_2d' in data:
        h_in = data['h_in_2d']
        h_out = data['h_out_2d']
        delta = h_out - h_in
        magnitudes = np.linalg.norm(delta, axis=1)
        geo['displacement_mean'] = float(magnitudes.mean())
        geo['displacement_max'] = float(magnitudes.max())
        if magnitudes.max() > 1e-8:
            unit_deltas = delta / (magnitudes[:, None] + 1e-10)
            geo['direction_coherence'] = float(np.linalg.norm(unit_deltas.mean(axis=0)))

    return geo


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class StructureTracker:
    """Tracks topological structure formation and disappearance across training."""

    def __init__(self, output_dir: str = "structure_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, data_map: dict):
        """Main analysis pipeline."""
        steps = sorted(set(s for s, l in data_map.keys()))
        layers = sorted(set(l for s, l in data_map.keys()))

        console.print(f"[green]Steps: {steps}[/]")
        console.print(f"[green]Layers: {layers}[/]")
        console.print(f"[green]Total files: {len(data_map)}[/]")

        n_steps = len(steps)
        n_layers = len(layers)

        # ── Compute topology and geometry for every (step, layer) ────
        topo_grid = {}   # (step, layer) -> topo dict
        geo_grid = {}    # (step, layer) -> geo dict
        dgm_grid = {}    # (step, layer) -> persistence diagrams (for Wasserstein)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Computing topology & geometry..."),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("compute", total=len(data_map))

            for (step, layer), filepath in data_map.items():
                data = load_npz(filepath)
                topo = compute_topology(data)
                geo = compute_geometry(data)

                if topo is not None:
                    topo_grid[(step, layer)] = topo
                    dgm_grid[(step, layer)] = topo['diagrams']
                geo_grid[(step, layer)] = geo

                progress.update(task, advance=1)

        console.print(f"[green]Topology computed for {len(topo_grid)}/{len(data_map)} files[/]")

        # ── Build matrices for heatmaps ─────────────────────────────────
        step_idx = {s: i for i, s in enumerate(steps)}
        layer_idx = {l: i for i, l in enumerate(layers)}

        # Topology matrices
        h1_count_matrix = np.zeros((n_steps, n_layers))
        h1_persistence_matrix = np.zeros((n_steps, n_layers))
        h1_max_lt_matrix = np.zeros((n_steps, n_layers))
        h1_significant_matrix = np.zeros((n_steps, n_layers))
        h0_count_matrix = np.zeros((n_steps, n_layers))
        h2_count_matrix = np.zeros((n_steps, n_layers))
        entropy_matrix = np.zeros((n_steps, n_layers))

        # Geometry matrices
        divergence_matrix = np.zeros((n_steps, n_layers))
        curl_matrix = np.zeros((n_steps, n_layers))
        shear_matrix = np.zeros((n_steps, n_layers))
        spectral_matrix = np.zeros((n_steps, n_layers))
        displacement_matrix = np.zeros((n_steps, n_layers))

        for (step, layer), topo in topo_grid.items():
            si = step_idx[step]
            li = layer_idx[layer]
            h1_count_matrix[si, li] = topo['h1_count']
            h1_persistence_matrix[si, li] = topo['h1_total_persistence']
            h1_max_lt_matrix[si, li] = topo['h1_max_lifetime']
            h1_significant_matrix[si, li] = topo['h1_significant']
            h0_count_matrix[si, li] = topo['h0_count']
            h2_count_matrix[si, li] = topo['h2_count']
            entropy_matrix[si, li] = topo['persistence_entropy']

        for (step, layer), geo in geo_grid.items():
            si = step_idx[step]
            li = layer_idx[layer]
            divergence_matrix[si, li] = geo.get('divergence', 0)
            curl_matrix[si, li] = geo.get('curl', 0)
            shear_matrix[si, li] = geo.get('shear', 0)
            spectral_matrix[si, li] = geo.get('spectral_radius', 0)
            displacement_matrix[si, li] = geo.get('displacement_mean', 0)

        # ── Generate all visualizations ─────────────────────────────────
        self._plot_topology_heatmaps(steps, layers, h1_count_matrix,
                                     h1_persistence_matrix, h1_significant_matrix,
                                     h0_count_matrix, h2_count_matrix, entropy_matrix)

        self._plot_geometry_heatmaps(steps, layers, divergence_matrix,
                                     curl_matrix, shear_matrix, spectral_matrix,
                                     displacement_matrix)

        self._plot_formation_timeline(steps, layers, h1_count_matrix,
                                      h1_persistence_matrix, h1_significant_matrix)

        self._plot_per_layer_evolution(steps, layers, h1_count_matrix,
                                       h1_persistence_matrix, entropy_matrix)

        self._plot_per_layer_geometry_evolution(steps, layers, divergence_matrix,
                                                curl_matrix, shear_matrix)

        self._detect_phase_transitions(steps, layers, h1_count_matrix,
                                        h1_persistence_matrix, entropy_matrix,
                                        curl_matrix)

        if HAS_PERSIM:
            self._plot_wasserstein_over_training(steps, layers, dgm_grid)

        self._plot_structure_birth_death(steps, layers, topo_grid)

        self._plot_combined_summary(steps, layers, h1_count_matrix,
                                    h1_persistence_matrix, curl_matrix,
                                    entropy_matrix)

        # ── Print summary table ─────────────────────────────────────────
        self._print_summary(steps, layers, topo_grid, geo_grid)

    # ─── HEATMAP VISUALIZATIONS ─────────────────────────────────────────

    def _plot_topology_heatmaps(self, steps, layers, h1_count, h1_pers,
                                 h1_sig, h0_count, h2_count, entropy):
        """Plot step×layer heatmaps for all topological metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='#0a0a1a')

        configs = [
            (axes[0, 0], h1_count, 'H₁ Loop Count', 'Reds', False),
            (axes[0, 1], h1_pers, 'H₁ Total Persistence', 'OrRd', False),
            (axes[0, 2], h1_sig, 'H₁ Significant Loops', 'magma', False),
            (axes[1, 0], h0_count, 'H₀ Components', 'Blues', False),
            (axes[1, 1], h2_count, 'H₂ Voids', 'Greens', False),
            (axes[1, 2], entropy, 'Persistence Entropy', 'viridis', False),
        ]

        for ax, matrix, title, cmap, use_log in configs:
            ax.set_facecolor('#0d1117')
            if use_log and matrix.max() > 0:
                im = ax.imshow(matrix.T, aspect='auto', cmap=cmap,
                               norm=LogNorm(vmin=max(matrix[matrix > 0].min(), 1e-6),
                                            vmax=matrix.max()),
                               origin='lower', interpolation='nearest')
            else:
                im = ax.imshow(matrix.T, aspect='auto', cmap=cmap,
                               origin='lower', interpolation='nearest')

            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(colors='#888888', labelsize=7)

            # Axis labels
            ax.set_xlabel('Training Step', color='#888888', fontsize=9)
            ax.set_ylabel('Layer', color='#888888', fontsize=9)
            ax.set_title(title, color='#c0d8e8', fontsize=11)

            # Tick labels
            n_steps = len(steps)
            n_layers = len(layers)
            if n_steps <= 25:
                ax.set_xticks(range(n_steps))
                ax.set_xticklabels(steps, fontsize=6, color='#888888', rotation=45)
            else:
                tick_idx = np.linspace(0, n_steps - 1, min(15, n_steps), dtype=int)
                ax.set_xticks(tick_idx)
                ax.set_xticklabels([steps[i] for i in tick_idx], fontsize=6,
                                   color='#888888', rotation=45)

            ax.set_yticks(range(n_layers))
            ax.set_yticklabels(layers, fontsize=7, color='#888888')

            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')
            ax.tick_params(colors='#888888', labelsize=7)

        fig.suptitle('Topological Structure Heatmaps (Step × Layer)',
                     color='#c0d8e8', fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'topology_heatmaps.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    def _plot_geometry_heatmaps(self, steps, layers, divergence, curl,
                                 shear, spectral, displacement):
        """Plot step×layer heatmaps for geometric metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='#0a0a1a')

        configs = [
            (axes[0, 0], divergence, 'Divergence (tr J)', 'RdBu_r', True),
            (axes[0, 1], curl, 'Curl (||J_antisym||)', 'Purples', False),
            (axes[0, 2], shear, 'Shear', 'Oranges', False),
            (axes[1, 0], spectral, 'Spectral Radius', 'Reds', False),
            (axes[1, 1], displacement, 'Mean Displacement', 'YlOrRd', False),
            (axes[1, 2], np.abs(divergence) + curl + shear,
             'Total Deformation (|div|+curl+shear)', 'inferno', False),
        ]

        for ax, matrix, title, cmap, symmetric in configs:
            ax.set_facecolor('#0d1117')
            if symmetric:
                vmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
                im = ax.imshow(matrix.T, aspect='auto', cmap=cmap,
                               vmin=-vmax, vmax=vmax,
                               origin='lower', interpolation='nearest')
            else:
                im = ax.imshow(matrix.T, aspect='auto', cmap=cmap,
                               origin='lower', interpolation='nearest')

            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(colors='#888888', labelsize=7)

            ax.set_xlabel('Training Step', color='#888888', fontsize=9)
            ax.set_ylabel('Layer', color='#888888', fontsize=9)
            ax.set_title(title, color='#c0d8e8', fontsize=11)

            n_steps = len(steps)
            n_layers = len(layers)
            if n_steps <= 25:
                ax.set_xticks(range(n_steps))
                ax.set_xticklabels(steps, fontsize=6, color='#888888', rotation=45)
            else:
                tick_idx = np.linspace(0, n_steps - 1, min(15, n_steps), dtype=int)
                ax.set_xticks(tick_idx)
                ax.set_xticklabels([steps[i] for i in tick_idx], fontsize=6,
                                   color='#888888', rotation=45)

            ax.set_yticks(range(n_layers))
            ax.set_yticklabels(layers, fontsize=7, color='#888888')

            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')
            ax.tick_params(colors='#888888', labelsize=7)

        fig.suptitle('Geometric Deformation Heatmaps (Step × Layer)',
                     color='#c0d8e8', fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'geometry_heatmaps.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── FORMATION / DISAPPEARANCE TIMELINE ─────────────────────────────

    def _plot_formation_timeline(self, steps, layers, h1_count, h1_pers, h1_sig):
        """
        Plot a timeline showing WHERE structures FORM and DISAPPEAR.
        Computes the difference between consecutive steps and marks
        positive changes (formation) and negative changes (disappearance).
        """
        n_steps = len(steps)
        n_layers = len(layers)

        if n_steps < 2:
            return

        # Compute step-to-step differences
        h1_diff = np.diff(h1_count, axis=0)       # (n_steps-1, n_layers)
        pers_diff = np.diff(h1_pers, axis=0)
        sig_diff = np.diff(h1_sig, axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(16, 14), facecolor='#0a0a1a')

        diff_configs = [
            (axes[0], h1_diff, 'H₁ Loop Count Change (Δ per step)'),
            (axes[1], pers_diff, 'H₁ Persistence Change (Δ per step)'),
            (axes[2], sig_diff, 'H₁ Significant Loop Change (Δ per step)'),
        ]

        for ax, diff_matrix, title in diff_configs:
            ax.set_facecolor('#0d1117')

            vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()), 1e-6)
            im = ax.imshow(diff_matrix.T, aspect='auto', cmap='RdYlGn',
                           vmin=-vmax, vmax=vmax,
                           origin='lower', interpolation='nearest')

            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Δ (green=formation, red=disappearance)',
                           color='#888888', fontsize=8)
            cbar.ax.tick_params(colors='#888888', labelsize=7)

            # Mark significant changes with markers
            for si in range(diff_matrix.shape[0]):
                for li in range(diff_matrix.shape[1]):
                    val = diff_matrix[si, li]
                    if abs(val) > 0.5:  # threshold for marking
                        marker = '▲' if val > 0 else '▼'
                        color = '#00ff00' if val > 0 else '#ff0000'
                        ax.text(si, li, marker, ha='center', va='center',
                                fontsize=8, color=color, fontweight='bold')

            ax.set_xlabel('Step Transition', color='#888888', fontsize=9)
            ax.set_ylabel('Layer', color='#888888', fontsize=9)
            ax.set_title(title, color='#c0d8e8', fontsize=11)

            # X-axis: show transitions
            if n_steps - 1 <= 25:
                trans_labels = [f'{steps[i]}→{steps[i+1]}' for i in range(n_steps - 1)]
                ax.set_xticks(range(n_steps - 1))
                ax.set_xticklabels(trans_labels, fontsize=5, color='#888888',
                                   rotation=45, ha='right')
            else:
                tick_idx = np.linspace(0, n_steps - 2, min(15, n_steps - 1), dtype=int)
                ax.set_xticks(tick_idx)
                ax.set_xticklabels([f'{steps[i]}→{steps[i+1]}' for i in tick_idx],
                                   fontsize=5, color='#888888', rotation=45, ha='right')

            ax.set_yticks(range(n_layers))
            ax.set_yticklabels(layers, fontsize=7, color='#888888')

            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')
            ax.tick_params(colors='#888888', labelsize=7)

        fig.suptitle('Structure Formation & Disappearance Timeline\n'
                     '(Green ▲ = formation, Red ▼ = disappearance)',
                     color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'formation_timeline.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── PER-LAYER EVOLUTION CURVES ─────────────────────────────────────

    def _plot_per_layer_evolution(self, steps, layers, h1_count, h1_pers, entropy):
        """Plot per-layer evolution curves showing how each layer's topology changes."""
        n_layers = len(layers)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0a0a1a')

        configs = [
            (axes[0], h1_count, 'H₁ Loop Count per Layer', 'tab10'),
            (axes[1], h1_pers, 'H₁ Total Persistence per Layer', 'tab10'),
            (axes[2], entropy, 'Persistence Entropy per Layer', 'tab10'),
        ]

        for ax, matrix, title, cmap_name in configs:
            ax.set_facecolor('#0d1117')
            cmap = plt.get_cmap(cmap_name)

            for li in range(n_layers):
                color = cmap(li / max(n_layers - 1, 1))
                ax.plot(steps, matrix[:, li], 'o-', color=color, lw=1.8,
                        markersize=4, label=f'Layer {layers[li]}', alpha=0.85)

            ax.set_xlabel('Training Step', color='#888888', fontsize=9)
            ax.set_ylabel('Value', color='#888888', fontsize=9)
            ax.set_title(title, color='#c0d8e8', fontsize=11)
            ax.legend(fontsize=7, facecolor='#0d1117', edgecolor='#2a3a4a',
                      labelcolor='#cccccc', loc='best', ncol=max(1, n_layers // 4))
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.15, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle('Per-Layer Topological Evolution Over Training',
                     color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'per_layer_evolution.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    def _plot_per_layer_geometry_evolution(self, steps, layers, divergence,
                                           curl, shear):
        """Plot per-layer geometry evolution curves."""
        n_layers = len(layers)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0a0a1a')

        configs = [
            (axes[0], divergence, 'Divergence per Layer'),
            (axes[1], curl, 'Curl per Layer'),
            (axes[2], shear, 'Shear per Layer'),
        ]

        cmap = plt.get_cmap('tab10')

        for ax, matrix, title in configs:
            ax.set_facecolor('#0d1117')

            for li in range(n_layers):
                color = cmap(li / max(n_layers - 1, 1))
                ax.plot(steps, matrix[:, li], 'o-', color=color, lw=1.8,
                        markersize=4, label=f'Layer {layers[li]}', alpha=0.85)

            ax.set_xlabel('Training Step', color='#888888', fontsize=9)
            ax.set_ylabel('Value', color='#888888', fontsize=9)
            ax.set_title(title, color='#c0d8e8', fontsize=11)
            ax.legend(fontsize=7, facecolor='#0d1117', edgecolor='#2a3a4a',
                      labelcolor='#cccccc', loc='best', ncol=max(1, n_layers // 4))
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.15, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle('Per-Layer Geometric Evolution Over Training',
                     color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'per_layer_geometry_evolution.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── PHASE TRANSITION DETECTION ─────────────────────────────────────

    def _detect_phase_transitions(self, steps, layers, h1_count, h1_pers,
                                   entropy, curl):
        """
        Detect phase transitions: sudden jumps in topological complexity.
        Reports the step and layer where the largest changes occur.
        """
        n_steps = len(steps)
        n_layers = len(layers)

        if n_steps < 3:
            return

        console.print("\n[bold yellow]═══ Phase Transition Detection ═══[/]")

        # Compute total topological signal per step (sum across layers)
        total_h1_per_step = h1_count.sum(axis=1)
        total_pers_per_step = h1_pers.sum(axis=1)
        total_entropy_per_step = entropy.sum(axis=1)
        total_curl_per_step = curl.sum(axis=1)

        # Find largest jumps
        signals = [
            ('H₁ Loop Count', total_h1_per_step),
            ('H₁ Persistence', total_pers_per_step),
            ('Persistence Entropy', total_entropy_per_step),
            ('Total Curl', total_curl_per_step),
        ]

        table = Table(title="Largest Step-to-Step Changes (Potential Phase Transitions)",
                      box=box.ROUNDED, show_lines=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Transition", style="yellow")
        table.add_column("Change", style="magenta")
        table.add_column("Direction", style="green")

        transitions_found = []

        for name, signal in signals:
            diffs = np.diff(signal)
            if len(diffs) == 0:
                continue

            # Find top-3 largest absolute changes
            top_indices = np.argsort(np.abs(diffs))[::-1][:3]

            for idx in top_indices:
                change = diffs[idx]
                if abs(change) < 1e-6:
                    continue
                direction = "↑ FORMATION" if change > 0 else "↓ DISAPPEARANCE"
                step_from = steps[idx]
                step_to = steps[idx + 1]
                table.add_row(
                    name,
                    f"Step {step_from} → {step_to}",
                    f"{change:+.4f}",
                    direction
                )
                transitions_found.append((name, step_from, step_to, change))

        console.print(table)

        # Per-layer phase transitions
        console.print("\n[bold]Per-Layer Largest Jumps:[/]")
        for li, layer in enumerate(layers):
            h1_diffs = np.diff(h1_count[:, li])
            if len(h1_diffs) > 0 and np.abs(h1_diffs).max() > 0:
                max_idx = np.argmax(np.abs(h1_diffs))
                max_change = h1_diffs[max_idx]
                if abs(max_change) > 0.5:
                    emoji = "🟢" if max_change > 0 else "🔴"
                    console.print(f"  {emoji} Layer {layer}: H₁ loops "
                                  f"{'formed' if max_change > 0 else 'disappeared'} "
                                  f"(Δ={max_change:+.0f}) at step "
                                  f"{steps[max_idx]}→{steps[max_idx+1]}")

        # Grokking detection: sudden large increase in topology
        if len(total_h1_per_step) > 5:
            diffs = np.diff(total_h1_per_step)
            mean_diff = np.mean(np.abs(diffs))
            std_diff = np.std(np.abs(diffs))

            for idx, d in enumerate(diffs):
                if abs(d) > mean_diff + 3 * std_diff and abs(d) > 2:
                    console.print(f"\n  [bold green]⚡ POSSIBLE GROKKING EVENT at "
                                  f"step {steps[idx]}→{steps[idx+1]}![/]")
                    console.print(f"     H₁ loops changed by {d:+.0f} "
                                  f"(3σ threshold: {mean_diff + 3*std_diff:.1f})")
                    console.print(f"     This suggests sudden emergence of "
                                  f"topological computational structures!")

        # Plot the phase transition summary
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0a0a1a')

        for ax, (name, signal) in zip(axes.flatten(), signals):
            ax.set_facecolor('#0d1117')

            # Plot signal
            ax.plot(steps, signal, 'o-', color='#44aaff', lw=2, markersize=5)

            # Highlight jumps
            diffs = np.diff(signal)
            if len(diffs) > 0:
                threshold = np.mean(np.abs(diffs)) + 2 * np.std(np.abs(diffs))
                for idx, d in enumerate(diffs):
                    if abs(d) > threshold:
                        color = '#00ff00' if d > 0 else '#ff0000'
                        ax.axvspan(steps[idx], steps[idx + 1], alpha=0.2, color=color)
                        ax.annotate(f'Δ={d:+.2f}',
                                    xy=(steps[idx + 1], signal[idx + 1]),
                                    fontsize=7, color=color,
                                    xytext=(5, 10), textcoords='offset points')

            ax.set_xlabel('Training Step', color='#888888', fontsize=9)
            ax.set_ylabel('Total (all layers)', color='#888888', fontsize=9)
            ax.set_title(name, color='#c0d8e8', fontsize=11)
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.15, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle('Phase Transition Detection — Total Signal Over Training',
                     color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'phase_transitions.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── WASSERSTEIN DISTANCE OVER TRAINING ─────────────────────────────

    def _plot_wasserstein_over_training(self, steps, layers, dgm_grid):
        """
        Compute Wasserstein distance between consecutive training steps
        for each layer. Shows how fast topology is changing.
        """
        n_steps = len(steps)
        n_layers = len(layers)

        if n_steps < 2:
            return

        # Wasserstein distance matrix: (n_steps-1) × n_layers
        wass_matrix = np.zeros((n_steps - 1, n_layers))

        for li, layer in enumerate(layers):
            for si in range(n_steps - 1):
                step_a = steps[si]
                step_b = steps[si + 1]

                dgms_a = dgm_grid.get((step_a, layer))
                dgms_b = dgm_grid.get((step_b, layer))

                if dgms_a is None or dgms_b is None:
                    continue

                total_dist = 0.0
                for dim in range(min(len(dgms_a), len(dgms_b), 2)):
                    d_a = dgms_a[dim]
                    d_b = dgms_b[dim]

                    # Filter to finite
                    d_a_fin = d_a[np.isfinite(d_a[:, 1])] if len(d_a) > 0 else np.empty((0, 2))
                    d_b_fin = d_b[np.isfinite(d_b[:, 1])] if len(d_b) > 0 else np.empty((0, 2))

                    if len(d_a_fin) > 0 and len(d_b_fin) > 0:
                        try:
                            total_dist += wasserstein_distance(d_a_fin, d_b_fin)
                        except Exception:
                            pass

                wass_matrix[si, li] = total_dist

        # Plot heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0a0a1a')

        # Heatmap
        ax = axes[0]
        ax.set_facecolor('#0d1117')
        im = ax.imshow(wass_matrix.T, aspect='auto', cmap='hot',
                       origin='lower', interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Wasserstein Distance', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)

        ax.set_xlabel('Step Transition', color='#888888', fontsize=9)
        ax.set_ylabel('Layer', color='#888888', fontsize=9)
        ax.set_title('Wasserstein Distance Between Consecutive Steps',
                     color='#c0d8e8', fontsize=11)

        if n_steps - 1 <= 25:
            trans_labels = [f'{steps[i]}→{steps[i+1]}' for i in range(n_steps - 1)]
            ax.set_xticks(range(n_steps - 1))
            ax.set_xticklabels(trans_labels, fontsize=5, color='#888888',
                               rotation=45, ha='right')
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layers, fontsize=7, color='#888888')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')
        ax.tick_params(colors='#888888', labelsize=7)

        # Per-layer curves
        ax2 = axes[1]
        ax2.set_facecolor('#0d1117')
        cmap = plt.get_cmap('tab10')
        for li, layer in enumerate(layers):
            color = cmap(li / max(n_layers - 1, 1))
            # x-axis: midpoints between steps
            x_vals = [(steps[i] + steps[i+1]) / 2 for i in range(n_steps - 1)]
            ax2.plot(x_vals, wass_matrix[:, li], 'o-', color=color, lw=1.8,
                     markersize=4, label=f'Layer {layer}', alpha=0.85)

        ax2.set_xlabel('Training Step (midpoint)', color='#888888', fontsize=9)
        ax2.set_ylabel('Wasserstein Distance', color='#888888', fontsize=9)
        ax2.set_title('Topological Change Rate per Layer',
                      color='#c0d8e8', fontsize=11)
        ax2.legend(fontsize=7, facecolor='#0d1117', edgecolor='#2a3a4a',
                   labelcolor='#cccccc', loc='best')
        ax2.tick_params(colors='#888888', labelsize=7)
        ax2.grid(True, alpha=0.15, color='#333333')
        for spine in ax2.spines.values():
            spine.set_color('#2a3a4a')

        fig.suptitle('Topological Change Rate Over Training (Wasserstein)',
                     color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'wasserstein_over_training.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── STRUCTURE BIRTH/DEATH TRACKING ─────────────────────────────────

    def _plot_structure_birth_death(self, steps, layers, topo_grid):
        """
        Track individual topological features across training steps.
        For each layer, plot when H1 features are "born" and "die"
        relative to training progression.
        """
        n_steps = len(steps)
        n_layers = len(layers)

        fig, axes = plt.subplots(n_layers, 1, figsize=(14, 4 * n_layers),
                                 facecolor='#0a0a1a')
        if n_layers == 1:
            axes = [axes]

        for li, layer in enumerate(layers):
            ax = axes[li]
            ax.set_facecolor('#0d1117')

            for si, step in enumerate(steps):
                topo = topo_grid.get((step, layer))
                if topo is None:
                    continue

                dgms = topo.get('diagrams', [])
                if len(dgms) < 2:
                    continue

                h1 = dgms[1]
                h1_finite = h1[np.isfinite(h1[:, 1])]

                if len(h1_finite) == 0:
                    continue

                # Plot each H1 feature as a vertical bar at this training step
                lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
                births = h1_finite[:, 0]
                deaths = h1_finite[:, 1]

                # Sort by lifetime
                sort_idx = np.argsort(lifetimes)[::-1]
                lifetimes_sorted = lifetimes[sort_idx]

                # Color by lifetime (longer = more significant)
                max_lt = lifetimes.max() if len(lifetimes) > 0 else 1.0

                for feat_idx, lt in enumerate(lifetimes_sorted[:15]):  # top 15
                    alpha = 0.3 + 0.7 * (lt / (max_lt + 1e-10))
                    size = 20 + 80 * (lt / (max_lt + 1e-10))
                    ax.scatter(si, feat_idx, s=size, c='#ff44aa',
                               alpha=alpha, edgecolors='none')

            ax.set_xlabel('Training Step Index', color='#888888', fontsize=9)
            ax.set_ylabel('Feature Rank (by lifetime)', color='#888888', fontsize=9)
            ax.set_title(f'Layer {layer} — H₁ Feature Presence Over Training',
                         color='#c0d8e8', fontsize=10)

            if n_steps <= 25:
                ax.set_xticks(range(n_steps))
                ax.set_xticklabels(steps, fontsize=6, color='#888888', rotation=45)

            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.1, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle('H₁ Feature Tracking Over Training\n'
                     '(Larger/brighter dots = longer-lived features)',
                     color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'structure_birth_death.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── COMBINED SUMMARY DASHBOARD ─────────────────────────────────────

    def _plot_combined_summary(self, steps, layers, h1_count, h1_pers,
                               curl, entropy):
        """
        A single combined dashboard showing the key signals together.
        """
        n_steps = len(steps)
        n_layers = len(layers)

        fig = plt.figure(figsize=(20, 12), facecolor='#0a0a1a')

        # Layout: 2×3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # 1. Total H1 over training (line)
        ax1 = fig.add_subplot(gs[0, 0], facecolor='#0d1117')
        total_h1 = h1_count.sum(axis=1)
        ax1.plot(steps, total_h1, 'o-', color='#ff44aa', lw=2, markersize=5)
        ax1.fill_between(steps, total_h1, alpha=0.1, color='#ff44aa')
        ax1.set_xlabel('Step', color='#888888', fontsize=9)
        ax1.set_ylabel('Total H₁ Loops', color='#888888', fontsize=9)
        ax1.set_title('Total Topological Complexity', color='#ff44aa', fontsize=10)
        ax1.tick_params(colors='#888888', labelsize=7)
        ax1.grid(True, alpha=0.15, color='#333333')
        for spine in ax1.spines.values():
            spine.set_color('#2a3a4a')

        # 2. H1 count heatmap
        ax2 = fig.add_subplot(gs[0, 1], facecolor='#0d1117')
        im2 = ax2.imshow(h1_count.T, aspect='auto', cmap='Reds',
                         origin='lower', interpolation='nearest')
        fig.colorbar(im2, ax=ax2, shrink=0.8).ax.tick_params(colors='#888888', labelsize=6)
        ax2.set_xlabel('Step', color='#888888', fontsize=9)
        ax2.set_ylabel('Layer', color='#888888', fontsize=9)
        ax2.set_title('H₁ Loops (step × layer)', color='#c0d8e8', fontsize=10)
        ax2.set_yticks(range(n_layers))
        ax2.set_yticklabels(layers, fontsize=7, color='#888888')
        ax2.tick_params(colors='#888888', labelsize=7)
        for spine in ax2.spines.values():
            spine.set_color('#2a3a4a')

        # 3. Curl heatmap
        ax3 = fig.add_subplot(gs[0, 2], facecolor='#0d1117')
        im3 = ax3.imshow(curl.T, aspect='auto', cmap='Purples',
                         origin='lower', interpolation='nearest')
        fig.colorbar(im3, ax=ax3, shrink=0.8).ax.tick_params(colors='#888888', labelsize=6)
        ax3.set_xlabel('Step', color='#888888', fontsize=9)
        ax3.set_ylabel('Layer', color='#888888', fontsize=9)
        ax3.set_title('Curl (step × layer)', color='#c0d8e8', fontsize=10)
        ax3.set_yticks(range(n_layers))
        ax3.set_yticklabels(layers, fontsize=7, color='#888888')
        ax3.tick_params(colors='#888888', labelsize=7)
        for spine in ax3.spines.values():
            spine.set_color('#2a3a4a')

        # 4. Persistence entropy over training
        ax4 = fig.add_subplot(gs[1, 0], facecolor='#0d1117')
        total_entropy = entropy.sum(axis=1)
        ax4.plot(steps, total_entropy, 'o-', color='#aa44ff', lw=2, markersize=5)
        ax4.fill_between(steps, total_entropy, alpha=0.1, color='#aa44ff')
        ax4.set_xlabel('Step', color='#888888', fontsize=9)
        ax4.set_ylabel('Total Persistence Entropy', color='#888888', fontsize=9)
        ax4.set_title('Topological Disorder', color='#aa44ff', fontsize=10)
        ax4.tick_params(colors='#888888', labelsize=7)
        ax4.grid(True, alpha=0.15, color='#333333')
        for spine in ax4.spines.values():
            spine.set_color('#2a3a4a')

        # 5. Step-to-step change magnitude
        ax5 = fig.add_subplot(gs[1, 1], facecolor='#0d1117')
        if n_steps > 1:
            h1_change = np.abs(np.diff(h1_count, axis=0)).sum(axis=1)
            pers_change = np.abs(np.diff(h1_pers, axis=0)).sum(axis=1)
            mid_steps = [(steps[i] + steps[i+1]) / 2 for i in range(n_steps - 1)]
            ax5.bar(mid_steps, h1_change, width=(steps[1]-steps[0])*0.4,
                    color='#ff44aa', alpha=0.7, label='H₁ count Δ')
            ax5.bar(mid_steps, pers_change, width=(steps[1]-steps[0])*0.4,
                    color='#ffaa44', alpha=0.5, label='H₁ persistence Δ',
                    bottom=h1_change)
            ax5.legend(fontsize=7, facecolor='#0d1117', edgecolor='#2a3a4a',
                       labelcolor='#cccccc')
        ax5.set_xlabel('Step', color='#888888', fontsize=9)
        ax5.set_ylabel('Total |Δ|', color='#888888', fontsize=9)
        ax5.set_title('Topological Change Magnitude', color='#c0d8e8', fontsize=10)
        ax5.tick_params(colors='#888888', labelsize=7)
        ax5.grid(True, alpha=0.15, color='#333333')
        for spine in ax5.spines.values():
            spine.set_color('#2a3a4a')

        # 6. Layer activity summary (which layers are most active)
        ax6 = fig.add_subplot(gs[1, 2], facecolor='#0d1117')
        layer_activity = h1_count.mean(axis=0)  # mean H1 per layer across all steps
        layer_variability = h1_count.std(axis=0)  # how much each layer changes
        x_pos = np.arange(n_layers)
        ax6.bar(x_pos - 0.2, layer_activity, width=0.4, color='#44aaff',
                alpha=0.8, label='Mean H₁')
        ax6.bar(x_pos + 0.2, layer_variability, width=0.4, color='#ff8844',
                alpha=0.8, label='Std H₁')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([f'L{l}' for l in layers], fontsize=7, color='#888888')
        ax6.set_xlabel('Layer', color='#888888', fontsize=9)
        ax6.set_ylabel('H₁ Features', color='#888888', fontsize=9)
        ax6.set_title('Layer Activity Profile', color='#c0d8e8', fontsize=10)
        ax6.legend(fontsize=7, facecolor='#0d1117', edgecolor='#2a3a4a',
                   labelcolor='#cccccc')
        ax6.tick_params(colors='#888888', labelsize=7)
        ax6.grid(True, alpha=0.15, color='#333333', axis='y')
        for spine in ax6.spines.values():
            spine.set_color('#2a3a4a')

        fig.suptitle('Structure Formation Summary Dashboard',
                     color='#c0d8e8', fontsize=14, y=0.98)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'combined_summary.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── SUMMARY TABLE ──────────────────────────────────────────────────

    def _print_summary(self, steps, layers, topo_grid, geo_grid):
        """Print a rich summary table of all findings."""
        console.print("\n")

        # Overall statistics
        table = Table(title="Structure Formation Summary",
                      box=box.DOUBLE_EDGE, show_lines=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Interpretation", style="green")

        total_h1 = sum(t.get('h1_count', 0) for t in topo_grid.values())
        total_sig = sum(t.get('h1_significant', 0) for t in topo_grid.values())
        total_h2 = sum(t.get('h2_count', 0) for t in topo_grid.values())
        max_h1_pers = max((t.get('h1_total_persistence', 0) for t in topo_grid.values()),
                         default=0)

        table.add_row("Total H₁ loops (all steps×layers)",
                      str(total_h1),
                      "Topological features detected")
        table.add_row("Significant H₁ loops",
                      str(total_sig),
                      "Long-lived structures (>2× median lifetime)")
        table.add_row("Total H₂ voids",
                      str(total_h2),
                      "Higher-dimensional cavities")
        table.add_row("Max H₁ persistence (single file)",
                      f"{max_h1_pers:.4f}",
                      "Strongest topological signal")
        table.add_row("Training steps analyzed",
                      str(len(steps)),
                      f"Steps {steps[0]} to {steps[-1]}")
        table.add_row("Layers analyzed",
                      str(len(layers)),
                      f"Layers {layers}")
        table.add_row("Total files processed",
                      str(len(topo_grid)),
                      f"Out of {len(steps) * len(layers)} possible")

        console.print(table)

        # Per-layer summary
        if len(layers) > 1:
            layer_table = Table(title="Per-Layer Summary (averaged over steps)",
                                box=box.ROUNDED, show_lines=True)
            layer_table.add_column("Layer", style="bold")
            layer_table.add_column("Mean H₁", style="magenta")
            layer_table.add_column("Mean Persistence", style="red")
            layer_table.add_column("Mean Curl", style="cyan")
            layer_table.add_column("Mean Divergence", style="blue")

            for layer in layers:
                h1_vals = [topo_grid[(s, layer)].get('h1_count', 0)
                           for s in steps if (s, layer) in topo_grid]
                pers_vals = [topo_grid[(s, layer)].get('h1_total_persistence', 0)
                             for s in steps if (s, layer) in topo_grid]
                curl_vals = [geo_grid[(s, layer)].get('curl', 0)
                             for s in steps if (s, layer) in geo_grid]
                div_vals = [geo_grid[(s, layer)].get('divergence', 0)
                            for s in steps if (s, layer) in geo_grid]

                layer_table.add_row(
                    str(layer),
                    f"{np.mean(h1_vals):.1f}" if h1_vals else "—",
                    f"{np.mean(pers_vals):.4f}" if pers_vals else "—",
                    f"{np.mean(curl_vals):.4f}" if curl_vals else "—",
                    f"{np.mean(div_vals):.4f}" if div_vals else "—",
                )

            console.print(layer_table)

        # Key findings
        console.print("\n[bold yellow]═══ Key Findings ═══[/]")

        if total_sig > 0:
            console.print(f"  [bold green]⚡ {total_sig} SIGNIFICANT TOPOLOGICAL "
                          f"STRUCTURES detected across training![/]")
            console.print(f"     These persistent loops may encode computational "
                          f"circuits (Conjecture 3)")
        else:
            console.print(f"  [yellow]No significant persistent loops detected.[/]")
            console.print(f"     The model may not have grokked yet, or structures "
                          f"exist in higher dimensions.")

        # Find where structures are most concentrated
        if topo_grid:
            best_key = max(topo_grid.keys(),
                           key=lambda k: topo_grid[k].get('h1_total_persistence', 0))
            best_topo = topo_grid[best_key]
            console.print(f"\n  📍 Strongest topology at step {best_key[0]}, "
                          f"layer {best_key[1]}:")
            console.print(f"     H₁ loops: {best_topo.get('h1_count', 0)}, "
                          f"persistence: {best_topo.get('h1_total_persistence', 0):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Track topological structure formation/disappearance across training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 track_structures.py runs/0/jacobi_data/
  python3 track_structures.py runs/0/jacobi_data/ --output-dir structure_analysis
  python3 track_structures.py runs/*/jacobi_data/   # multiple runs
        """,
    )

    parser.add_argument("paths", type=str, nargs='+',
                        help="Path(s) to jacobi_data directories")
    parser.add_argument("--output-dir", type=str, default="structure_analysis",
                        help="Output directory for plots and reports")
    parser.add_argument("--max-pts", type=int, default=300,
                        help="Max points for persistent homology (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max-files", type=int, default=None,
                        help="If set, subsample to this many equidistant files "
                             "(e.g., --max-files 10 on 1000 files picks every 100th)")

    args = parser.parse_args()

    np.random.seed(args.seed)

    console.print(Panel(
        "[bold white]Jacobi Structure Formation Tracker[/]\n"
        "[dim]Scanning all training steps × layers for topological structure[/]\n"
        "[dim]formation, disappearance, and phase transitions[/]",
        border_style="bold cyan",
        padding=(1, 4),
    ))

    # ── Load all files from all paths ───────────────────────────────────
    all_data_maps = {}

    for path in args.paths:
        console.print(f"\n[green]Scanning: {path}[/]")
        data_map = load_all_files(path)
        if data_map:
            all_data_maps.update(data_map)
            console.print(f"  Found {len(data_map)} files")

    if not all_data_maps:
        console.print("[bold red]No jacobi files found in any of the provided paths.[/]")
        sys.exit(1)

    # ── Subsample to equidistant files if --max-files is set ────────────
    if args.max_files is not None and args.max_files > 0:
        all_keys = sorted(all_data_maps.keys())  # sorted by (step, layer)
        total = len(all_keys)
        if args.max_files < total:
            # Pick every Nth file so we get exactly max_files equidistant samples
            step_size = total / args.max_files
            selected_indices = [int(round(i * step_size)) for i in range(args.max_files)]
            # Clamp to valid range
            selected_indices = [min(idx, total - 1) for idx in selected_indices]
            selected_keys = [all_keys[i] for i in selected_indices]
            all_data_maps = {k: all_data_maps[k] for k in selected_keys}
            console.print(f"[yellow]Subsampled to {len(all_data_maps)} equidistant files "
                          f"(every ~{total // args.max_files}th file from {total} total)[/]")

    console.print(f"\n[bold green]Total files to analyze: {len(all_data_maps)}[/]")


    # ── Run the tracker ─────────────────────────────────────────────────
    tracker = StructureTracker(output_dir=args.output_dir)
    tracker.run(all_data_maps)

    # ── Final summary ───────────────────────────────────────────────────
    n_plots = len(glob.glob(os.path.join(args.output_dir, '*.png')))
    console.print(Panel(
        f"[bold green]Structure tracking complete![/]\n"
        f"Output directory: {args.output_dir}/\n"
        f"Visualizations generated: {n_plots}\n"
        f"Files processed: {len(all_data_maps)}",
        title="[bold]Results",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
