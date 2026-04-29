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
#   "gudhi",
#   "networkx",
#   "rich",
#   "umap-learn",
# ]
# ///

"""
Jacobi Field Topology Analyzer (EXTENDED VISUALIZATIONS)
=========================================================

Now includes:
  - Persistence BARCODES (horizontal lifetime bars for H0, H1, H2)
  - Persistence LANDSCAPES (functional summaries of persistence)
  - Singular value SPECTRUM plots (waterfall + cumulative energy)
  - Displacement QUIVER fields (vector field of token motion)
  - Rotation/Shear POLAR plots (per-token angular decomposition)
  - Local curvature HEATMAPS on the warped grid
  - Vietoris-Rips FILTRATION snapshots (simplicial complex at multiple scales)
  - Token DENDROGRAM (hierarchical clustering of output positions)
  - Phase portrait (eigenvalue FLOW field)
  - RADAR/SPIDER plots (multi-metric per-layer fingerprints)
  - Correlation MATRIX of all geometric features
  - Persistence ENTROPY over layers
  - Bottleneck distance HEATMAP (alternative to Wasserstein)
  - STREAM plot of the displacement field (continuous flow lines)
  - 3D SURFACE plot of the area-change field
  - Per-token STRIP plot (volume, rotation, shear side by side)
  - Cumulative BETTI barcode (stacked area across layers)

Usage:
    python3 analyze_jacobi.py runs/0/jacobi_data/ --all
"""

import os
import sys
import glob
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

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

# ═══════════════════════════════════════════════════════════════════════════
# Heavy imports
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, det, norm
from scipy.stats import entropy as scipy_entropy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("WARNING: ripser not available. Persistent homology disabled.")

try:
    from persim import wasserstein as wasserstein_distance
    from persim import plot_diagrams
    try:
        from persim import bottleneck as bottleneck_distance
        HAS_BOTTLENECK = True
    except ImportError:
        HAS_BOTTLENECK = False
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False
    HAS_BOTTLENECK = False

try:
    import gudhi
    from gudhi.representations import Landscape
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    print("WARNING: gudhi not available. Persistence landscapes disabled.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich import box

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# NPZ FILE LOADING AND PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_filename(filepath: str) -> Dict:
    """Extract step and layer from filename like jacobi_step000100_layer02.npz"""
    basename = os.path.basename(filepath)
    info = {'path': filepath, 'basename': basename}
    import re
    step_match = re.search(r'step(\d+)', basename)
    layer_match = re.search(r'layer(\d+)', basename)
    if step_match:
        info['step'] = int(step_match.group(1))
    if layer_match:
        info['layer'] = int(layer_match.group(1))
    return info


def load_jacobi_npz(filepath: str) -> Dict:
    """Load a Jacobi field .npz file and return its contents as a dict."""
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


def discover_npz_files(path: str, step: Optional[int] = None,
                       layer: Optional[int] = None) -> List[Dict]:
    if os.path.isfile(path):
        info = parse_filename(path)
        info['data'] = load_jacobi_npz(path)
        return [info]
    if os.path.isdir(path):
        pattern = os.path.join(path, "jacobi_step*_layer*.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            console.print(f"[red]No jacobi_step*_layer*.npz files found in {path}[/]")
            return []
        results = []
        for f in files:
            info = parse_filename(f)
            if step is not None and info.get('step') != step:
                continue
            if layer is not None and info.get('layer') != layer:
                continue
            results.append(info)
        return results
    console.print(f"[red]Path not found: {path}[/]")
    return []


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS + VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class JacobiAnalyzer:
    """Main analyzer class with EXTENDED visualizations."""

    def __init__(self, output_dir: str = "jacobi_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.report_lines: List[str] = []

    def log(self, msg: str):
        self.report_lines.append(msg)
        console.print(f"  {msg}")

    def save_report(self, filename: str = "analysis_report.md"):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write("# Jacobi Field Topology Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            for line in self.report_lines:
                import re
                clean = re.sub(r'\[.*?\]', '', line)
                f.write(clean + "\n")
        console.print(f"\n[bold green]📄 Report saved to: {path}[/]")

    # ─── Single-file analysis ───────────────────────────────────────────

    def analyze_single(self, info: Dict) -> Dict:
        if 'data' not in info:
            info['data'] = load_jacobi_npz(info['path'])
        data = info['data']
        step = info.get('step', '?')
        layer = info.get('layer', '?')

        self.log(f"\n[bold cyan]═══ Step {step}, Layer {layer} ═══[/]")
        self.log(f"  File: {info['basename']}")
        self.log(f"  Keys: {list(data.keys())}")

        results = {'step': step, 'layer': layer, 'file': info['path']}
        results['geometry'] = self._analyze_geometry(data)
        results['jacobian'] = self._analyze_jacobian(data)
        results['tokens'] = self._analyze_tokens(data)
        if HAS_RIPSER:
            results['topology'] = self._analyze_topology(data)
        results['grid'] = self._analyze_grid_deformation(data)
        results['eigenvalues'] = self._analyze_eigenvalues(data)
        return results

    def _analyze_geometry(self, data: Dict) -> Dict:
        results = {}
        if 'h_in_2d' in data:
            h_in = data['h_in_2d']
            results['n_tokens'] = h_in.shape[0]
            results['dim'] = h_in.shape[1] if h_in.ndim > 1 else 2
            results['input_centroid'] = h_in.mean(axis=0)
            results['input_spread'] = h_in.std(axis=0)
            results['input_bbox'] = (h_in.min(axis=0), h_in.max(axis=0))
            if h_in.shape[0] > 1:
                dists = pdist(h_in)
                results['mean_pairwise_dist'] = dists.mean()
                results['max_pairwise_dist'] = dists.max()
                results['min_pairwise_dist'] = dists[dists > 0].min() if (dists > 0).any() else 0
            self.log(f"  Tokens: {results['n_tokens']}, Dim: {results['dim']}")
            self.log(f"  Input spread: {results['input_spread']}")
        if 'h_out_2d' in data:
            h_out = data['h_out_2d']
            results['output_centroid'] = h_out.mean(axis=0)
            results['output_spread'] = h_out.std(axis=0)
            if 'input_centroid' in results:
                shift = results['output_centroid'] - results['input_centroid']
                results['centroid_shift'] = shift
                results['centroid_shift_magnitude'] = np.linalg.norm(shift)
                self.log(f"  Centroid shift: {results['centroid_shift_magnitude']:.4f}")
        for key in ['anisotropy', 'log_det']:
            if key in data:
                results[key] = float(data[key]) if np.isscalar(data[key]) else float(data[key].item())
                self.log(f"  {key}: {results[key]:.4f}")
        return results

    def _analyze_jacobian(self, data: Dict) -> Dict:
        results = {}
        if 'jacobian' in data:
            J = data['jacobian']
            results['shape'] = J.shape
            D = J.shape[0]
            try:
                U, S, Vh = svd(J)
                results['singular_values'] = S
                results['condition_number'] = S[0] / max(S[-1], 1e-10)
                results['determinant'] = np.prod(S)
                results['log_determinant'] = np.sum(np.log(np.clip(S, 1e-10, None)))
                self.log(f"  Jacobian shape: {J.shape}")
                self.log(f"  Singular values: [{S[0]:.3f}, ..., {S[-1]:.3f}]")
                self.log(f"  Condition number: {results['condition_number']:.2f}")
            except Exception as e:
                self.log(f"  [yellow]SVD failed: {e}[/]")
            J_sym = (J + J.T) / 2
            J_antisym = (J - J.T) / 2
            results['divergence'] = np.trace(J)
            results['curl'] = norm(J_antisym, 'fro')
            trace_sym = np.trace(J_sym)
            shear_tensor = J_sym - (trace_sym / D) * np.eye(D)
            results['shear'] = norm(shear_tensor, 'fro')
            if D == 2:
                results['rotation_angle'] = np.arctan2(J_antisym[1, 0], J_sym[0, 0])
            self.log(f"  Divergence: {results['divergence']:.4f}, Curl: {results['curl']:.4f}, Shear: {results['shear']:.4f}")
            try:
                eigenvalues = np.linalg.eigvals(J)
                results['eigenvalues'] = eigenvalues
                results['eigenvalue_magnitudes'] = np.abs(eigenvalues)
                results['eigenvalue_phases'] = np.angle(eigenvalues)
                n_complex = np.sum(np.abs(eigenvalues.imag) > 1e-8)
                results['n_complex_eigenvalues'] = n_complex
                results['spectral_radius'] = np.max(np.abs(eigenvalues))
                self.log(f"  Spectral radius: {results['spectral_radius']:.4f}")
            except Exception as e:
                self.log(f"  [yellow]Eigenvalue computation failed: {e}[/]")
        for key in ['divergence', 'curl', 'shear']:
            if key in data and key not in results:
                val = data[key]
                results[key] = float(val)
        return results

    def _analyze_tokens(self, data: Dict) -> Dict:
        results = {}
        if 'per_token_volume' in data:
            vol = data['per_token_volume']
            results['volume_mean'] = vol.mean()
            results['volume_std'] = vol.std()
            results['volume_min'] = vol.min()
            results['volume_max'] = vol.max()
            results['n_expanding'] = int(np.sum(vol > 1.1))
            results['n_contracting'] = int(np.sum(vol < 0.9))
            results['n_preserving'] = int(np.sum((vol >= 0.9) & (vol <= 1.1)))
        if 'per_token_rotation' in data:
            rot = data['per_token_rotation']
            results['rotation_mean'] = rot.mean()
            results['rotation_std'] = rot.std()
            results['rotation_range'] = (rot.min(), rot.max())
        if 'per_token_shear_mag' in data:
            shear = data['per_token_shear_mag']
            results['shear_mean'] = shear.mean()
            results['shear_std'] = shear.std()
            results['shear_max'] = shear.max()
        if 'per_token_delta_2d' in data:
            delta = data['per_token_delta_2d']
            magnitudes = np.linalg.norm(delta, axis=1)
            results['displacement_mean'] = magnitudes.mean()
            results['displacement_std'] = magnitudes.std()
            results['displacement_max'] = magnitudes.max()
            if magnitudes.max() > 1e-8:
                unit_deltas = delta / (magnitudes[:, None] + 1e-10)
                mean_direction = unit_deltas.mean(axis=0)
                results['direction_coherence'] = np.linalg.norm(mean_direction)
        if 'token_strings' in data:
            results['token_strings'] = data['token_strings']
        return results

    def _analyze_topology(self, data: Dict) -> Dict:
        results = {}
        if 'h_in_2d' in data:
            points = data['h_in_2d']
        elif 'h_out_2d' in data:
            points = data['h_out_2d']
        else:
            return results
        if points.shape[0] < 4:
            return results
        max_pts = 300
        if points.shape[0] > max_pts:
            idx = np.random.choice(points.shape[0], max_pts, replace=False)
            points = points[idx]
        try:
            dists = pdist(points)
            thresh = np.percentile(dists[dists > 0], 95) if len(dists) > 0 and dists.max() > 0 else 1.0
            result = ripser(points, maxdim=2, thresh=thresh)
            dgms = result['dgms']
            results['diagrams'] = dgms
            results['threshold'] = thresh

            h0 = dgms[0]
            h0_finite = h0[np.isfinite(h0[:, 1])]
            results['h0_n_features'] = len(h0_finite)
            if len(h0_finite) > 0:
                h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
                results['h0_max_lifetime'] = h0_lifetimes.max()
                results['h0_mean_lifetime'] = h0_lifetimes.mean()
                results['h0_total_persistence'] = h0_lifetimes.sum()

            if len(dgms) > 1:
                h1 = dgms[1]
                h1_finite = h1[np.isfinite(h1[:, 1])]
                results['h1_n_features'] = len(h1_finite)
                if len(h1_finite) > 0:
                    h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
                    results['h1_max_lifetime'] = h1_lifetimes.max()
                    results['h1_mean_lifetime'] = h1_lifetimes.mean()
                    results['h1_total_persistence'] = h1_lifetimes.sum()
                    results['h1_lifetimes'] = h1_lifetimes
                    if len(h1_lifetimes) > 1:
                        median_lt = np.median(h1_lifetimes)
                        significant = h1_lifetimes > 2 * median_lt
                        results['h1_n_significant'] = int(significant.sum())
                    else:
                        results['h1_n_significant'] = len(h1_lifetimes)
                    self.log(f"  [bold magenta]H1 (LOOPS):[/] {results['h1_n_features']} features, "
                             f"{results['h1_n_significant']} significant")
                else:
                    results['h1_n_features'] = 0

            if len(dgms) > 2:
                h2 = dgms[2]
                h2_finite = h2[np.isfinite(h2[:, 1])]
                results['h2_n_features'] = len(h2_finite)
                if len(h2_finite) > 0:
                    h2_lifetimes = h2_finite[:, 1] - h2_finite[:, 0]
                    results['h2_max_lifetime'] = h2_lifetimes.max()
                    results['h2_total_persistence'] = h2_lifetimes.sum()

            # Betti curves
            n_scales = 20
            scales = np.linspace(0, thresh, n_scales)
            betti_0 = np.zeros(n_scales)
            betti_1 = np.zeros(n_scales)
            for si, s in enumerate(scales):
                betti_0[si] = np.sum((dgms[0][:, 0] <= s) & (dgms[0][:, 1] > s))
                if len(dgms) > 1 and len(dgms[1]) > 0:
                    betti_1[si] = np.sum((dgms[1][:, 0] <= s) & (dgms[1][:, 1] > s))
            results['betti_scales'] = scales
            results['betti_0'] = betti_0
            results['betti_1'] = betti_1

            # Persistence entropy
            all_lifetimes = []
            for dim_idx in range(min(len(dgms), 3)):
                finite = dgms[dim_idx][np.isfinite(dgms[dim_idx][:, 1])]
                if len(finite) > 0:
                    all_lifetimes.extend((finite[:, 1] - finite[:, 0]).tolist())
            if all_lifetimes:
                lt_arr = np.array(all_lifetimes)
                lt_norm = lt_arr / (lt_arr.sum() + 1e-10)
                results['persistence_entropy'] = scipy_entropy(lt_norm)
            else:
                results['persistence_entropy'] = 0.0

        except Exception as e:
            self.log(f"  [red]Persistent homology failed: {e}[/]")

        # Output topology
        if 'h_out_2d' in data:
            out_points = data['h_out_2d']
            if out_points.shape[0] > max_pts:
                idx = np.random.choice(out_points.shape[0], max_pts, replace=False)
                out_points = out_points[idx]
            try:
                result_out = ripser(out_points, maxdim=1, thresh=thresh)
                dgms_out = result_out['dgms']
                results['output_diagrams'] = dgms_out
                if len(dgms_out) > 1:
                    h1_out = dgms_out[1]
                    h1_out_finite = h1_out[np.isfinite(h1_out[:, 1])]
                    results['output_h1_n_features'] = len(h1_out_finite)
            except Exception:
                pass

        return results

    def _analyze_grid_deformation(self, data: Dict) -> Dict:
        results = {}
        has_grid_in = 'grid_x_in' in data and 'grid_y_in' in data
        has_grid_out = 'grid_x_out' in data and 'grid_y_out' in data
        if has_grid_in and has_grid_out:
            gx_in = data['grid_x_in']
            gy_in = data['grid_y_in']
            gx_out = data['grid_x_out']
            gy_out = data['grid_y_out']
            dx = gx_out - gx_in
            dy = gy_out - gy_in
            displacement_mag = np.sqrt(dx**2 + dy**2)
            results['grid_shape'] = gx_in.shape
            results['mean_displacement'] = displacement_mag.mean()
            results['max_displacement'] = displacement_mag.max()
            results['displacement_std'] = displacement_mag.std()
            grid_n = gx_in.shape[0]
            local_det = np.ones((grid_n - 1, grid_n - 1))
            for i in range(grid_n - 1):
                for j in range(grid_n - 1):
                    v1_in = np.array([gx_in[i, j+1] - gx_in[i, j], gy_in[i, j+1] - gy_in[i, j]])
                    v2_in = np.array([gx_in[i+1, j] - gx_in[i, j], gy_in[i+1, j] - gy_in[i, j]])
                    area_in = abs(np.cross(v1_in, v2_in))
                    v1_out = np.array([gx_out[i, j+1] - gx_out[i, j], gy_out[i, j+1] - gy_out[i, j]])
                    v2_out = np.array([gx_out[i+1, j] - gx_out[i, j], gy_out[i+1, j] - gy_out[i, j]])
                    area_out = abs(np.cross(v1_out, v2_out))
                    if area_in > 1e-12:
                        local_det[i, j] = area_out / area_in
            results['local_det'] = local_det
            results['mean_area_change'] = local_det.mean()
            results['max_area_change'] = local_det.max()
            results['min_area_change'] = local_det.min()
            self.log(f"  Grid: {grid_n}x{grid_n}, mean disp: {results['mean_displacement']:.4f}")
        return results

    def _analyze_eigenvalues(self, data: Dict) -> Dict:
        results = {}
        if 'singular_values' in data:
            sv = data['singular_values']
            if isinstance(sv, np.ndarray) and sv.ndim > 0:
                results['singular_values'] = sv
                results['condition_number'] = sv[0] / max(sv[-1], 1e-10)
                results['effective_rank'] = np.sum(sv > sv[0] * 0.01)
                sv_norm = sv / (sv.sum() + 1e-10)
                results['sv_entropy'] = scipy_entropy(sv_norm)
        return results

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZATION — ORIGINAL + NEW
    # ═══════════════════════════════════════════════════════════════════

    def visualize_single(self, info: Dict, results: Dict):
        step = results.get('step', '?')
        layer = results.get('layer', '?')
        prefix = f"step{step}_layer{layer}"
        data = info['data'] if 'data' in info else load_jacobi_npz(info['path'])

        # Original plots
        self._plot_point_cloud(data, results, prefix)
        self._plot_warped_grid(data, results, prefix)
        if 'topology' in results and 'diagrams' in results['topology']:
            self._plot_persistence(results['topology'], prefix)
            # NEW: Barcodes
            self._plot_barcodes(results['topology'], prefix)
            # NEW: Persistence landscape
            self._plot_persistence_landscape(results['topology'], prefix)
            # NEW: Filtration snapshots
            self._plot_filtration_snapshots(data, results['topology'], prefix)
        if 'topology' in results and 'betti_scales' in results['topology']:
            self._plot_betti_curves(results['topology'], prefix)
        if 'jacobian' in results and 'eigenvalues' in results['jacobian']:
            self._plot_eigenvalues(results['jacobian'], prefix)
            # NEW: Phase portrait
            self._plot_phase_portrait(results['jacobian'], prefix)
        if 'per_token_volume' in data:
            self._plot_volume_histogram(data, prefix)
        if 'grid' in results and 'local_det' in results['grid']:
            self._plot_area_change_heatmap(data, results['grid'], prefix)
            # NEW: 3D surface of area change
            self._plot_area_change_3d(results['grid'], prefix)
            # NEW: Local curvature heatmap
            self._plot_curvature_heatmap(data, results['grid'], prefix)

        # NEW: Displacement quiver field
        self._plot_displacement_quiver(data, prefix)
        # NEW: Displacement stream plot
        self._plot_displacement_stream(data, prefix)
        # NEW: Singular value spectrum (waterfall + cumulative)
        self._plot_sv_spectrum(results, data, prefix)
        # NEW: Per-token rotation/shear polar plot
        self._plot_rotation_shear_polar(data, prefix)
        # NEW: Token dendrogram
        self._plot_token_dendrogram(data, prefix)
        # NEW: Per-token strip plot
        self._plot_token_strip(data, prefix)
        # NEW: Radar/spider plot of layer metrics
        self._plot_radar(results, prefix)

    # ─── ORIGINAL PLOTS (kept from your code) ──────────────────────────

    def _plot_point_cloud(self, data: Dict, results: Dict, prefix: str):
        if 'h_in_2d' not in data:
            return
        h_in = data['h_in_2d']
        fig, axes = plt.subplots(1, 2 if 'h_out_2d' in data else 1,
                                 figsize=(14, 6), facecolor='#0a0a1a')
        ax = axes[0] if isinstance(axes, np.ndarray) else axes
        ax.set_facecolor('#0d1117')
        ax.scatter(h_in[:, 0], h_in[:, 1], s=20, c='#44aaff', alpha=0.8, edgecolors='none')
        if 'token_strings' in data:
            ts = data['token_strings']
            for i in range(min(len(ts), h_in.shape[0])):
                if i % max(1, h_in.shape[0] // 15) == 0:
                    ax.annotate(str(ts[i])[:12], (h_in[i, 0], h_in[i, 1]),
                                fontsize=6, color='#cccccc', alpha=0.8,
                                xytext=(3, 3), textcoords='offset points')

        ax.set_title('Input Point Cloud', color='#c0d8e8', fontsize=10)
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        if 'h_out_2d' in data:
            h_out = data['h_out_2d']
            ax2 = axes[1]
            ax2.set_facecolor('#0d1117')

            for i in range(h_in.shape[0]):
                dx = h_out[i, 0] - h_in[i, 0]
                dy = h_out[i, 1] - h_in[i, 1]
                mag = np.sqrt(dx**2 + dy**2)
                if mag > 1e-6:
                    ax2.annotate('', xy=(h_out[i, 0], h_out[i, 1]),
                                 xytext=(h_in[i, 0], h_in[i, 1]),
                                 arrowprops=dict(arrowstyle='->', color='#ffffff',
                                                 lw=0.5, alpha=0.4))

            ax2.scatter(h_in[:, 0], h_in[:, 1], s=15, facecolors='none',
                        edgecolors='#aaaaaa', linewidths=0.7, zorder=4, alpha=0.8)
            ax2.scatter(h_out[:, 0], h_out[:, 1], s=15, c='#ffcc44',
                        edgecolors='none', zorder=5, alpha=0.9)

            ax2.set_title('Input → Output Displacement', color='#c0d8e8', fontsize=10)
            ax2.tick_params(colors='#888888', labelsize=7)
            for spine in ax2.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle(f'Point Cloud — {prefix}', color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_pointcloud.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_warped_grid(self, data: Dict, results: Dict, prefix: str):
        """Plot the warped grid showing space deformation."""
        has_grid = ('grid_x_in' in data and 'grid_y_in' in data and
                    'grid_x_out' in data and 'grid_y_out' in data)
        if not has_grid:
            return

        gx_in = data['grid_x_in']
        gy_in = data['grid_y_in']
        gx_out = data['grid_x_out']
        gy_out = data['grid_y_out']
        grid_n = gx_in.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        # Input grid (gray)
        for i in range(grid_n):
            ax.plot(gx_in[i, :], gy_in[i, :], color='#555555', lw=0.4, alpha=0.5)
            ax.plot(gx_in[:, i], gy_in[:, i], color='#555555', lw=0.4, alpha=0.5)

        # Output grid (colored by local area change)
        grid_results = results.get('grid', {})
        if 'local_det' in grid_results:
            local_det = grid_results['local_det']
            log_det = np.log(np.clip(local_det, 1e-6, None))
            lv_absmax = max(np.abs(log_det).max(), 1e-6)

            for i in range(grid_n):
                for j in range(grid_n - 1):
                    lv = log_det[min(i, grid_n - 2), j]
                    t = np.clip(lv / lv_absmax, -1, 1)
                    if t > 0.05:
                        color = (0.3 + 0.7 * t, 0.25 * (1 - t), 0.2 * (1 - t))
                    elif t < -0.05:
                        at = abs(t)
                        color = (0.2 * (1 - at), 0.25 * (1 - at), 0.3 + 0.7 * at)
                    else:
                        color = (0.2, 0.55, 0.45)
                    ax.plot([gx_out[i, j], gx_out[i, j+1]],
                            [gy_out[i, j], gy_out[i, j+1]],
                            color=color, lw=0.8, alpha=0.85)
                for j in range(grid_n):
                    if i < grid_n - 1:
                        lv = log_det[i, min(j, grid_n - 2)]
                        t = np.clip(lv / lv_absmax, -1, 1)
                        if t > 0.05:
                            color = (0.3 + 0.7 * t, 0.25 * (1 - t), 0.2 * (1 - t))
                        elif t < -0.05:
                            at = abs(t)
                            color = (0.2 * (1 - at), 0.25 * (1 - at), 0.3 + 0.7 * at)
                        else:
                            color = (0.2, 0.55, 0.45)
                        ax.plot([gx_out[i, j], gx_out[i+1, j]],
                                [gy_out[i, j], gy_out[i+1, j]],
                                color=color, lw=0.8, alpha=0.85)
        else:
            for i in range(grid_n):
                ax.plot(gx_out[i, :], gy_out[i, :], color='#44aaff', lw=0.6, alpha=0.7)
                ax.plot(gx_out[:, i], gy_out[:, i], color='#44aaff', lw=0.6, alpha=0.7)

        if 'h_in_2d' in data and 'h_out_2d' in data:
            h_in = data['h_in_2d']
            h_out = data['h_out_2d']
            ax.scatter(h_in[:, 0], h_in[:, 1], s=18, facecolors='none',
                       edgecolors='#aaaaaa', linewidths=0.7, zorder=5)
            ax.scatter(h_out[:, 0], h_out[:, 1], s=14, c='#ffcc44',
                       edgecolors='none', zorder=6)

        ax.set_title(f'Warped Grid — {prefix}', color='#c0d8e8', fontsize=11)
        ax.set_aspect('auto')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_warped_grid.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_persistence(self, topo: Dict, prefix: str):
        """Plot persistence diagrams for H0, H1, H2."""
        dgms = topo['diagrams']

        fig, axes = plt.subplots(1, min(len(dgms), 3), figsize=(5 * min(len(dgms), 3), 5),
                                 facecolor='#0a0a1a')
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        colors = ['#44aaff', '#ff44aa', '#44ffaa']
        labels = ['H₀ (components)', 'H₁ (loops)', 'H₂ (voids)']

        for dim_idx, (ax, dgm) in enumerate(zip(axes, dgms)):
            if dim_idx >= 3:
                break
            ax.set_facecolor('#0d1117')

            finite = dgm[np.isfinite(dgm[:, 1])]
            infinite = dgm[~np.isfinite(dgm[:, 1])]

            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1], s=20,
                           c=colors[dim_idx], alpha=0.7, edgecolors='none')

            if len(infinite) > 0:
                max_death = finite[:, 1].max() if len(finite) > 0 else 1.0
                ax.scatter(infinite[:, 0],
                           np.full(len(infinite), max_death * 1.1),
                           s=30, c=colors[dim_idx], marker='^', alpha=0.7)

            all_vals = dgm[np.isfinite(dgm)].ravel()
            if len(all_vals) > 0:
                lo, hi = all_vals.min(), all_vals.max()
                ax.plot([lo, hi], [lo, hi], '--', color='#555555', lw=0.8)

            ax.set_title(labels[dim_idx], color=colors[dim_idx], fontsize=10)
            ax.set_xlabel('Birth', color='#888888', fontsize=8)
            ax.set_ylabel('Death', color='#888888', fontsize=8)
            ax.tick_params(colors='#888888', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle(f'Persistence Diagrams — {prefix}', color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_persistence.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_betti_curves(self, topo: Dict, prefix: str):
        """Plot Betti number curves as a function of filtration scale."""
        scales = topo['betti_scales']
        b0 = topo['betti_0']
        b1 = topo['betti_1']

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        ax.plot(scales, b0, color='#44aaff', lw=2, label='β₀ (components)')
        ax.plot(scales, b1, color='#ff44aa', lw=2, label='β₁ (loops)')
        ax.fill_between(scales, b1, alpha=0.15, color='#ff44aa')

        ax.set_xlabel('Filtration Scale', color='#888888')
        ax.set_ylabel('Betti Number', color='#888888')
        ax.set_title(f'Betti Curves — {prefix}', color='#c0d8e8', fontsize=11)
        ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#2a3a4a',
                  labelcolor='#cccccc')
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_betti.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_eigenvalues(self, jac_results: Dict, prefix: str):
        """Plot eigenvalues in the complex plane."""
        eigenvalues = jac_results['eigenvalues']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0a0a1a')

        ax = axes[0]
        ax.set_facecolor('#0d1117')
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), '--', color='#555555', lw=0.8)
        ax.scatter(eigenvalues.real, eigenvalues.imag, s=40, c='#ff6644',
                   edgecolors='#ffffff', linewidths=0.5, zorder=5)
        ax.axhline(0, color='#333333', lw=0.5)
        ax.axvline(0, color='#333333', lw=0.5)
        ax.set_xlabel('Real', color='#888888')
        ax.set_ylabel('Imaginary', color='#888888')
        ax.set_title('Eigenvalues (Complex Plane)', color='#c0d8e8', fontsize=10)
        ax.set_aspect('equal')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        ax2 = axes[1]
        ax2.set_facecolor('#0d1117')
        mags = np.abs(eigenvalues)
        phases = np.angle(eigenvalues)
        ax2.scatter(phases, mags, s=40, c=phases, cmap='hsv',
                    edgecolors='#ffffff', linewidths=0.5, zorder=5)
        ax2.axhline(1.0, color='#555555', lw=0.8, ls='--', label='|λ|=1')
        ax2.set_xlabel('Phase (rad)', color='#888888')
        ax2.set_ylabel('Magnitude', color='#888888')
        ax2.set_title('Eigenvalue Magnitude vs Phase', color='#c0d8e8', fontsize=10)
        ax2.legend(fontsize=8, facecolor='#0d1117', edgecolor='#2a3a4a',
                   labelcolor='#cccccc')
        ax2.tick_params(colors='#888888', labelsize=7)
        for spine in ax2.spines.values():
            spine.set_color('#2a3a4a')

        fig.suptitle(f'Eigenvalue Analysis — {prefix}', color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_eigenvalues.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_volume_histogram(self, data: Dict, prefix: str):
        """Plot histogram of per-token volume changes."""
        if 'per_token_volume' not in data:
            return
        vol = data['per_token_volume']

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')
        log_vol = np.log(np.clip(vol, 1e-6, None))
        ax.hist(log_vol, bins=30, color='#44aaff', alpha=0.7, edgecolor='#2a3a4a')
        ax.axvline(0, color='#ff4444', lw=1.5, ls='--', label='log(vol)=0 (preserving)')
        ax.set_xlabel('log(volume change)', color='#888888')
        ax.set_ylabel('Count', color='#888888')
        ax.set_title(f'Per-Token Volume Change — {prefix}', color='#c0d8e8', fontsize=11)
        ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#2a3a4a',
                  labelcolor='#cccccc')
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_volume_hist.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_area_change_heatmap(self, data: Dict, grid_results: Dict, prefix: str):
        """Plot heatmap of local area change on the grid."""
        local_det = grid_results['local_det']

        fig, ax = plt.subplots(1, 1, figsize=(7, 6), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')
        log_det = np.log(np.clip(local_det, 1e-6, None))
        vmax = max(abs(log_det.min()), abs(log_det.max()), 0.1)
        im = ax.imshow(log_det, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='auto')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('log(area ratio)', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)
        ax.set_title(f'Local Area Change — {prefix}', color='#c0d8e8', fontsize=11)
        ax.set_xlabel('Grid column', color='#888888')
        ax.set_ylabel('Grid row', color='#888888')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_area_heatmap.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    # ═══════════════════════════════════════════════════════════════════
    # NEW VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════

    def _plot_barcodes(self, topo: Dict, prefix: str):
        """
        Plot persistence BARCODES — horizontal bars showing birth→death
        for each topological feature in H0, H1, H2.
        """
        dgms = topo['diagrams']

        n_dims = min(len(dgms), 3)
        fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), facecolor='#0a0a1a')
        if n_dims == 1:
            axes = [axes]

        colors = ['#44aaff', '#ff44aa', '#44ffaa']
        labels = ['H₀ (components)', 'H₁ (loops)', 'H₂ (voids)']

        for dim_idx in range(n_dims):
            ax = axes[dim_idx]
            ax.set_facecolor('#0d1117')
            dgm = dgms[dim_idx]

            finite = dgm[np.isfinite(dgm[:, 1])]
            infinite = dgm[~np.isfinite(dgm[:, 1])]

            # Sort by lifetime (longest at top)
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                sort_idx = np.argsort(lifetimes)[::-1]
                finite_sorted = finite[sort_idx]

                # Limit to top 50 bars for readability
                max_bars = 50
                finite_sorted = finite_sorted[:max_bars]

                for bar_idx, (birth, death) in enumerate(finite_sorted):
                    lifetime = death - birth
                    # Color intensity by lifetime
                    alpha = 0.4 + 0.6 * (lifetime / (lifetimes.max() + 1e-10))
                    ax.barh(bar_idx, death - birth, left=birth, height=0.8,
                            color=colors[dim_idx], alpha=alpha, edgecolor='none')

            # Infinite features (draw to the right edge)
            if len(infinite) > 0:
                max_val = finite[:, 1].max() if len(finite) > 0 else 1.0
                n_finite_shown = min(len(finite), 50)
                for inf_idx, (birth, _) in enumerate(infinite[:10]):
                    y_pos = n_finite_shown + inf_idx
                    ax.barh(y_pos, max_val * 1.2 - birth, left=birth, height=0.8,
                            color=colors[dim_idx], alpha=0.3, edgecolor='none',
                            hatch='//')
                    ax.plot(max_val * 1.2, y_pos, '>', color=colors[dim_idx],
                            markersize=6, alpha=0.8)

            ax.set_xlabel('Filtration Scale', color='#888888', fontsize=9)
            ax.set_ylabel('Feature Index', color='#888888', fontsize=9)
            ax.set_title(f'{labels[dim_idx]} Barcode', color=colors[dim_idx], fontsize=10)
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.1, color='#333333', axis='x')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle(f'Persistence Barcodes — {prefix}', color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_barcodes.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_persistence_landscape(self, topo: Dict, prefix: str):
        """
        Plot persistence LANDSCAPES — functional summaries of persistence diagrams.
        Landscapes are piecewise-linear functions that capture the shape of the diagram.
        """
        dgms = topo['diagrams']

        fig, axes = plt.subplots(1, min(len(dgms), 2), figsize=(12, 5), facecolor='#0a0a1a')
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        colors_map = ['Blues', 'Reds']
        labels = ['H₀ Landscape', 'H₁ Landscape']

        for dim_idx in range(min(len(dgms), 2)):
            ax = axes[dim_idx]
            ax.set_facecolor('#0d1117')
            dgm = dgms[dim_idx]
            finite = dgm[np.isfinite(dgm[:, 1])]

            if len(finite) < 2:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                        ha='center', va='center', color='#888888')
                continue

            # Compute landscape manually (tent functions)
            births = finite[:, 0]
            deaths = finite[:, 1]
            lifetimes = deaths - births

            # Sort by lifetime descending
            sort_idx = np.argsort(lifetimes)[::-1]
            births_s = births[sort_idx]
            deaths_s = deaths[sort_idx]

            # Create landscape functions for top-k features
            n_landscapes = min(5, len(finite))
            x_range = np.linspace(births.min(), deaths.max(), 200)

            for k in range(n_landscapes):
                # The k-th landscape is the k-th largest tent function value at each x
                tent_values = np.zeros((len(finite), len(x_range)))
                for feat_idx in range(len(finite)):
                    b, d = births[feat_idx], deaths[feat_idx]
                    mid = (b + d) / 2
                    for xi, x in enumerate(x_range):
                        if b <= x <= mid:
                            tent_values[feat_idx, xi] = x - b
                        elif mid < x <= d:
                            tent_values[feat_idx, xi] = d - x
                        else:
                            tent_values[feat_idx, xi] = 0

                # k-th landscape: k-th largest value at each x
                tent_sorted = np.sort(tent_values, axis=0)[::-1]
                if k < tent_sorted.shape[0]:
                    landscape_k = tent_sorted[k]
                    alpha = 1.0 - 0.15 * k
                    ax.plot(x_range, landscape_k, lw=1.5, alpha=alpha,
                            label=f'λ_{k+1}')
                    ax.fill_between(x_range, landscape_k, alpha=0.05 * (n_landscapes - k))

            ax.set_xlabel('Filtration Scale', color='#888888', fontsize=9)
            ax.set_ylabel('Landscape Value', color='#888888', fontsize=9)
            ax.set_title(labels[dim_idx], color='#c0d8e8', fontsize=10)
            ax.legend(fontsize=7, facecolor='#0d1117', edgecolor='#2a3a4a',
                      labelcolor='#cccccc', loc='upper right')
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.1, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle(f'Persistence Landscapes — {prefix}', color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_landscapes.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_filtration_snapshots(self, data: Dict, topo: Dict, prefix: str):
        """
        Plot Vietoris-Rips filtration SNAPSHOTS at multiple scales,
        showing the simplicial complex growing.
        """
        if 'h_in_2d' not in data:
            return

        points = data['h_in_2d']
        if points.shape[0] > 100:
            idx = np.random.choice(points.shape[0], 100, replace=False)
            points = points[idx]

        thresh = topo.get('threshold', 1.0)
        n_snapshots = 6
        scales = np.linspace(0, thresh * 0.8, n_snapshots)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='#0a0a1a')
        axes_flat = axes.flatten()

        dists = squareform(pdist(points))

        for snap_idx, (ax, scale) in enumerate(zip(axes_flat, scales)):
            ax.set_facecolor('#0d1117')

            # Draw edges for pairs within scale
            n_pts = points.shape[0]
            n_edges = 0
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if dists[i, j] <= scale:
                        ax.plot([points[i, 0], points[j, 0]],
                                [points[i, 1], points[j, 1]],
                                color='#334466', lw=0.4, alpha=0.5)
                        n_edges += 1

            # Draw triangles (2-simplices) for triples all within scale
            n_triangles = 0
            if scale > 0:
                for i in range(min(n_pts, 50)):
                    for j in range(i + 1, min(n_pts, 50)):
                        if dists[i, j] > scale:
                            continue
                        for k in range(j + 1, min(n_pts, 50)):
                            if dists[i, k] <= scale and dists[j, k] <= scale:
                                triangle = plt.Polygon(
                                    [points[i], points[j], points[k]],
                                    alpha=0.08, color='#ff44aa', edgecolor='none')
                                ax.add_patch(triangle)
                                n_triangles += 1

            # Draw points
            ax.scatter(points[:, 0], points[:, 1], s=12, c='#44aaff',
                       edgecolors='none', zorder=5, alpha=0.9)

            ax.set_title(f'ε = {scale:.3f}\n{n_edges} edges, {n_triangles} triangles',
                         color='#c0d8e8', fontsize=9)
            ax.tick_params(colors='#888888', labelsize=6)
            ax.set_aspect('equal')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle(f'Vietoris-Rips Filtration Snapshots — {prefix}',
                     color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_filtration.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_displacement_quiver(self, data: Dict, prefix: str):
        """
        Plot a QUIVER field showing per-token displacement vectors.
        Arrows colored by magnitude.
        """
        if 'h_in_2d' not in data or 'h_out_2d' not in data:
            return

        h_in = data['h_in_2d']
        h_out = data['h_out_2d']
        dx = h_out[:, 0] - h_in[:, 0]
        dy = h_out[:, 1] - h_in[:, 1]
        magnitudes = np.sqrt(dx**2 + dy**2)

        fig, ax = plt.subplots(1, 1, figsize=(9, 8), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        # Normalize magnitudes for coloring
        mag_norm = magnitudes / (magnitudes.max() + 1e-10)

        quiv = ax.quiver(h_in[:, 0], h_in[:, 1], dx, dy,
                         mag_norm, cmap='plasma', scale=None,
                         width=0.004, headwidth=4, headlength=5,
                         alpha=0.85, zorder=4)

        ax.scatter(h_in[:, 0], h_in[:, 1], s=12, c='#aaaaaa',
                   edgecolors='none', zorder=5, alpha=0.6)

        cbar = fig.colorbar(quiv, ax=ax, shrink=0.8)
        cbar.set_label('Displacement Magnitude (normalized)', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)

        ax.set_title(f'Displacement Quiver Field — {prefix}', color='#c0d8e8', fontsize=11)
        ax.set_xlabel('x', color='#888888')
        ax.set_ylabel('y', color='#888888')
        ax.tick_params(colors='#888888', labelsize=7)
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_quiver.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_displacement_stream(self, data: Dict, prefix: str):
        """
        Plot a STREAM plot of the displacement field — continuous flow lines
        interpolated from the discrete token displacements.
        """
        if 'h_in_2d' not in data or 'h_out_2d' not in data:
            return

        h_in = data['h_in_2d']
        h_out = data['h_out_2d']

        if h_in.shape[0] < 6:
            return

        dx = h_out[:, 0] - h_in[:, 0]
        dy = h_out[:, 1] - h_in[:, 1]

        # Create a regular grid for interpolation
        margin = 0.05
        x_min, x_max = h_in[:, 0].min(), h_in[:, 0].max()
        y_min, y_max = h_in[:, 1].min(), h_in[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= margin * x_range
        x_max += margin * x_range
        y_min -= margin * y_range
        y_max += margin * y_range

        grid_res = 30
        xi = np.linspace(x_min, x_max, grid_res)
        yi = np.linspace(y_min, y_max, grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate displacement field onto grid using RBF
        try:
            grid_pts = np.column_stack([Xi.ravel(), Yi.ravel()])
            rbf_dx = RBFInterpolator(h_in, dx, kernel='thin_plate_spline', smoothing=0.1)
            rbf_dy = RBFInterpolator(h_in, dy, kernel='thin_plate_spline', smoothing=0.1)
            Dx = rbf_dx(grid_pts).reshape(grid_res, grid_res)
            Dy = rbf_dy(grid_pts).reshape(grid_res, grid_res)
        except Exception:
            return

        fig, ax = plt.subplots(1, 1, figsize=(9, 8), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        speed = np.sqrt(Dx**2 + Dy**2)
        lw = 1.5 * speed / (speed.max() + 1e-10)

        strm = ax.streamplot(xi, yi, Dx, Dy, color=speed, cmap='inferno',
                             linewidth=lw, density=1.5, arrowsize=1.2,
                             arrowstyle='->')

        ax.scatter(h_in[:, 0], h_in[:, 1], s=18, c='#44aaff',
                   edgecolors='#ffffff', linewidths=0.4, zorder=5, alpha=0.9)

        cbar = fig.colorbar(strm.lines, ax=ax, shrink=0.8)
        cbar.set_label('Flow Speed', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)

        ax.set_title(f'Displacement Stream Field — {prefix}', color='#c0d8e8', fontsize=11)
        ax.set_xlabel('x', color='#888888')
        ax.set_ylabel('y', color='#888888')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_streamplot.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_sv_spectrum(self, results: Dict, data: Dict, prefix: str):
        """
        Plot singular value SPECTRUM — waterfall plot + cumulative energy.
        Shows how many dimensions carry meaningful information.
        """
        sv = None
        if 'jacobian' in results and 'singular_values' in results['jacobian']:
            sv = results['jacobian']['singular_values']
        elif 'singular_values' in data:
            sv = data['singular_values']
            if isinstance(sv, np.ndarray) and sv.ndim == 0:
                return

        if sv is None or not isinstance(sv, np.ndarray) or len(sv) < 2:
            return

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0a0a1a')

        # Waterfall (log scale)
        ax = axes[0]
        ax.set_facecolor('#0d1117')
        indices = np.arange(len(sv))
        ax.bar(indices, sv, color='#44aaff', alpha=0.8, edgecolor='none', width=0.8)
        ax.set_yscale('log')
        ax.set_xlabel('Index', color='#888888')
        ax.set_ylabel('Singular Value (log)', color='#888888')
        ax.set_title('Singular Value Spectrum', color='#c0d8e8', fontsize=10)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333', axis='y')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        # Cumulative energy
        ax2 = axes[1]
        ax2.set_facecolor('#0d1117')
        sv_sq = sv**2
        cumulative = np.cumsum(sv_sq) / (sv_sq.sum() + 1e-10)
        ax2.plot(indices, cumulative, 'o-', color='#ffcc44', lw=2, markersize=4)
        ax2.axhline(0.9, color='#ff4444', lw=1, ls='--', alpha=0.7, label='90% energy')
        ax2.axhline(0.99, color='#44ff44', lw=1, ls='--', alpha=0.7, label='99% energy')

        # Mark effective rank
        eff_rank_90 = np.searchsorted(cumulative, 0.9) + 1
        eff_rank_99 = np.searchsorted(cumulative, 0.99) + 1
        ax2.axvline(eff_rank_90, color='#ff4444', lw=0.8, ls=':', alpha=0.6)
        ax2.axvline(eff_rank_99, color='#44ff44', lw=0.8, ls=':', alpha=0.6)

        ax2.set_xlabel('Index', color='#888888')
        ax2.set_ylabel('Cumulative Energy Fraction', color='#888888')
        ax2.set_title(f'Cumulative Energy (eff. rank: {eff_rank_90}@90%, {eff_rank_99}@99%)',
                      color='#c0d8e8', fontsize=10)
        ax2.legend(fontsize=8, facecolor='#0d1117', edgecolor='#2a3a4a',
                   labelcolor='#cccccc')
        ax2.tick_params(colors='#888888', labelsize=7)
        ax2.grid(True, alpha=0.15, color='#333333')
        for spine in ax2.spines.values():
            spine.set_color('#2a3a4a')

        fig.suptitle(f'Singular Value Analysis — {prefix}', color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_sv_spectrum.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_rotation_shear_polar(self, data: Dict, prefix: str):
        """
        Plot per-token rotation and shear in POLAR coordinates.
        Angle = rotation, radius = shear magnitude.
        """
        if 'per_token_rotation' not in data or 'per_token_shear_mag' not in data:
            return

        rotation = data['per_token_rotation']
        shear = data['per_token_shear_mag']

        fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                                 subplot_kw={}, facecolor='#0a0a1a')

        # Polar plot: rotation angle vs shear magnitude
        ax_polar = fig.add_subplot(121, polar=True, facecolor='#0d1117')
        colors = shear / (shear.max() + 1e-10)
        sc = ax_polar.scatter(rotation, shear, c=colors, cmap='magma',
                              s=25, alpha=0.8, edgecolors='none')
        ax_polar.set_title('Rotation (angle) vs Shear (radius)',
                           color='#c0d8e8', fontsize=10, pad=15)
        ax_polar.tick_params(colors='#888888', labelsize=7)
        ax_polar.set_facecolor('#0d1117')
        ax_polar.spines['polar'].set_color('#2a3a4a')

        # Remove the rectangular subplot we don't need
        axes[0].remove()
        axes[1].remove()

        # Histogram of rotation angles
        ax_hist = fig.add_subplot(122, facecolor='#0d1117')
        ax_hist.set_facecolor('#0d1117')
        ax_hist.hist(rotation, bins=36, color='#ff44aa', alpha=0.7,
                     edgecolor='#2a3a4a', density=True)
        ax_hist.axvline(0, color='#ffffff', lw=1, ls='--', alpha=0.5)
        ax_hist.set_xlabel('Rotation Angle (rad)', color='#888888')
        ax_hist.set_ylabel('Density', color='#888888')
        ax_hist.set_title('Rotation Angle Distribution', color='#c0d8e8', fontsize=10)
        ax_hist.tick_params(colors='#888888', labelsize=7)
        ax_hist.grid(True, alpha=0.15, color='#333333')
        for spine in ax_hist.spines.values():
            spine.set_color('#2a3a4a')

        fig.suptitle(f'Rotation & Shear Polar Analysis — {prefix}',
                     color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_polar_rot_shear.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_phase_portrait(self, jac_results: Dict, prefix: str):
        """
        Plot eigenvalue FLOW field / phase portrait.
        Shows the vector field implied by the Jacobian in 2D.
        """
        eigenvalues = jac_results.get('eigenvalues', None)
        if eigenvalues is None or len(eigenvalues) < 2:
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        # Use the first two eigenvalues to define a 2D linear system
        # dx/dt = A*x where A has these eigenvalues
        lam1 = eigenvalues[0]
        lam2 = eigenvalues[1] if len(eigenvalues) > 1 else eigenvalues[0]

        # Construct a 2D matrix with these eigenvalues
        # Simple diagonal case for real, rotation for complex
        if np.abs(lam1.imag) < 1e-8 and np.abs(lam2.imag) < 1e-8:
            A = np.diag([lam1.real, lam2.real])
        else:
            # Use real/imag parts of first complex eigenvalue
            a = lam1.real
            b = lam1.imag
            A = np.array([[a, -b], [b, a]])

        # Create grid for phase portrait
        lim = 2.0
        x = np.linspace(-lim, lim, 20)
        y = np.linspace(-lim, lim, 20)
        X, Y = np.meshgrid(x, y)

        # Compute vector field
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y

        speed = np.sqrt(U**2 + V**2)
        lw = 1.5 * speed / (speed.max() + 1e-10)

        strm = ax.streamplot(x, y, U, V, color=speed, cmap='coolwarm',
                             linewidth=lw, density=1.8, arrowsize=1.0)

        # Mark eigenvalues
        ax.scatter(eigenvalues.real[:5], eigenvalues.imag[:5],
                   s=80, c='#ffcc44', marker='*', zorder=10,
                   edgecolors='#ffffff', linewidths=0.5)

        # Mark origin
        ax.scatter([0], [0], s=60, c='#ffffff', marker='x', zorder=10, linewidths=2)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f'Phase Portrait (λ₁={lam1:.2f}, λ₂={lam2:.2f}) — {prefix}',
                     color='#c0d8e8', fontsize=10)
        ax.set_xlabel('x₁', color='#888888')
        ax.set_ylabel('x₂', color='#888888')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_phase_portrait.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_area_change_3d(self, grid_results: Dict, prefix: str):
        """
        Plot a 3D SURFACE of the local area change field.
        Height = log(area ratio), colored by value.
        """
        local_det = grid_results.get('local_det', None)
        if local_det is None:
            return

        log_det = np.log(np.clip(local_det, 1e-6, None))

        fig = plt.figure(figsize=(10, 8), facecolor='#0a0a1a')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0d1117')

        rows, cols = log_det.shape
        X = np.arange(cols)
        Y = np.arange(rows)
        X, Y = np.meshgrid(X, Y)

        vmax = max(abs(log_det.min()), abs(log_det.max()), 0.1)
        norm = Normalize(vmin=-vmax, vmax=vmax)

        surf = ax.plot_surface(X, Y, log_det, cmap='RdBu_r', norm=norm,
                               alpha=0.85, edgecolor='none', antialiased=True,
                               rstride=1, cstride=1)

        ax.set_xlabel('Grid Col', color='#888888', fontsize=8, labelpad=8)
        ax.set_ylabel('Grid Row', color='#888888', fontsize=8, labelpad=8)
        ax.set_zlabel('log(area ratio)', color='#888888', fontsize=8, labelpad=8)
        ax.set_title(f'3D Area Change Surface — {prefix}', color='#c0d8e8', fontsize=11)

        ax.tick_params(colors='#888888', labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#2a3a4a')
        ax.yaxis.pane.set_edgecolor('#2a3a4a')
        ax.zaxis.pane.set_edgecolor('#2a3a4a')

        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('log(area ratio)', color='#888888', fontsize=8)
        cbar.ax.tick_params(colors='#888888', labelsize=7)

        path = os.path.join(self.output_dir, f'{prefix}_area_3d.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_curvature_heatmap(self, data: Dict, grid_results: Dict, prefix: str):
        """
        Plot LOCAL CURVATURE heatmap on the warped grid.
        Curvature is estimated from the second derivatives of the displacement field.
        """
        has_grid = ('grid_x_out' in data and 'grid_y_out' in data)
        if not has_grid:
            return

        gx_out = data['grid_x_out']
        gy_out = data['grid_y_out']
        grid_n = gx_out.shape[0]

        if grid_n < 4:
            return

        # Estimate curvature via discrete Laplacian of the output grid
        # Laplacian of x-coordinates
        lap_x = np.zeros((grid_n - 2, grid_n - 2))
        lap_y = np.zeros((grid_n - 2, grid_n - 2))

        for i in range(1, grid_n - 1):
            for j in range(1, grid_n - 1):
                lap_x[i-1, j-1] = (gx_out[i+1, j] + gx_out[i-1, j] +
                                    gx_out[i, j+1] + gx_out[i, j-1] -
                                    4 * gx_out[i, j])
                lap_y[i-1, j-1] = (gy_out[i+1, j] + gy_out[i-1, j] +
                                    gy_out[i, j+1] + gy_out[i, j-1] -
                                    4 * gy_out[i, j])

        curvature = np.sqrt(lap_x**2 + lap_y**2)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        im = ax.imshow(curvature, cmap='hot', origin='lower', aspect='auto')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Local Curvature (|∇²|)', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)

        ax.set_title(f'Local Curvature Heatmap — {prefix}', color='#c0d8e8', fontsize=11)
        ax.set_xlabel('Grid column', color='#888888')
        ax.set_ylabel('Grid row', color='#888888')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_curvature.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_token_dendrogram(self, data: Dict, prefix: str):
        """
        Plot a DENDROGRAM of hierarchical clustering of output token positions.
        Shows which tokens are geometrically grouped.
        """
        if 'h_out_2d' not in data:
            return

        h_out = data['h_out_2d']
        if h_out.shape[0] < 4 or h_out.shape[0] > 200:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        # Compute linkage
        Z = linkage(h_out, method='ward')

        # Token labels
        labels = None
        if 'token_strings' in data:
            ts = data['token_strings']
            labels = [str(ts[i])[:10] if i < len(ts) else f't{i}'
                      for i in range(h_out.shape[0])]

        dendrogram(Z, ax=ax, labels=labels, leaf_rotation=90,
                   leaf_font_size=6, color_threshold=0.7 * Z[-1, 2],
                   above_threshold_color='#888888')

        ax.set_title(f'Token Hierarchical Clustering — {prefix}',
                     color='#c0d8e8', fontsize=11)
        ax.set_xlabel('Token', color='#888888')
        ax.set_ylabel('Ward Distance', color='#888888')
        ax.tick_params(colors='#888888', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        # Color the x-axis labels
        for lbl in ax.get_xticklabels():
            lbl.set_color('#cccccc')

        path = os.path.join(self.output_dir, f'{prefix}_dendrogram.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_token_strip(self, data: Dict, prefix: str):
        """
        Per-token STRIP plot — volume, rotation, and shear side by side
        for each token, showing the full per-token decomposition.
        """
        has_vol = 'per_token_volume' in data
        has_rot = 'per_token_rotation' in data
        has_shear = 'per_token_shear_mag' in data

        if not (has_vol or has_rot or has_shear):
            return

        n_panels = sum([has_vol, has_rot, has_shear])
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6),
                                 facecolor='#0a0a1a')
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0
        token_labels = None
        if 'token_strings' in data:
            ts = data['token_strings']
            n_tok = len(data.get('per_token_volume', data.get('per_token_rotation',
                        data.get('per_token_shear_mag', []))))
            token_labels = [str(ts[i])[:8] if i < len(ts) else f't{i}'
                            for i in range(n_tok)]

        if has_vol:
            ax = axes[panel_idx]
            ax.set_facecolor('#0d1117')
            vol = data['per_token_volume']
            n = len(vol)
            colors = ['#ff4444' if v < 0.9 else '#44ff44' if v > 1.1 else '#44aaff'
                      for v in vol]
            ax.barh(range(n), np.log(np.clip(vol, 1e-6, None)),
                    color=colors, alpha=0.8, edgecolor='none')
            ax.axvline(0, color='#ffffff', lw=0.8, ls='--', alpha=0.5)
            ax.set_xlabel('log(volume)', color='#888888')
            ax.set_ylabel('Token', color='#888888')
            ax.set_title('Per-Token Volume', color='#44aaff', fontsize=10)
            if token_labels and n <= 40:
                ax.set_yticks(range(n))
                ax.set_yticklabels(token_labels, fontsize=6, color='#cccccc')
            ax.tick_params(colors='#888888', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')
            panel_idx += 1

        if has_rot:
            ax = axes[panel_idx]
            ax.set_facecolor('#0d1117')
            rot = data['per_token_rotation']
            n = len(rot)
            colors_rot = plt.cm.hsv((rot - rot.min()) / (rot.max() - rot.min() + 1e-10))
            ax.barh(range(n), rot, color=colors_rot, alpha=0.8, edgecolor='none')
            ax.axvline(0, color='#ffffff', lw=0.8, ls='--', alpha=0.5)
            ax.set_xlabel('Rotation (rad)', color='#888888')
            ax.set_title('Per-Token Rotation', color='#ff44aa', fontsize=10)
            if token_labels and n <= 40:
                ax.set_yticks(range(n))
                ax.set_yticklabels(token_labels, fontsize=6, color='#cccccc')
            ax.tick_params(colors='#888888', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')
            panel_idx += 1

        if has_shear:
            ax = axes[panel_idx]
            ax.set_facecolor('#0d1117')
            shear = data['per_token_shear_mag']
            n = len(shear)
            ax.barh(range(n), shear, color='#ffaa44', alpha=0.8, edgecolor='none')
            ax.set_xlabel('Shear Magnitude', color='#888888')
            ax.set_title('Per-Token Shear', color='#ffaa44', fontsize=10)
            if token_labels and n <= 40:
                ax.set_yticks(range(n))
                ax.set_yticklabels(token_labels, fontsize=6, color='#cccccc')
            ax.tick_params(colors='#888888', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')
            panel_idx += 1

        fig.suptitle(f'Per-Token Decomposition — {prefix}', color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{prefix}_token_strip.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_radar(self, results: Dict, prefix: str):
        """
        Plot a RADAR/SPIDER chart summarizing multiple geometric metrics
        for this layer in a single glanceable fingerprint.
        """
        jac = results.get('jacobian', {})
        topo = results.get('topology', {})
        tokens = results.get('tokens', {})
        grid = results.get('grid', {})

        # Collect metrics (normalize to [0, 1] range for radar)
        metric_names = []
        metric_values = []

        candidates = [
            ('Divergence', abs(jac.get('divergence', 0))),
            ('Curl', jac.get('curl', 0)),
            ('Shear', jac.get('shear', 0)),
            ('Spectral Radius', jac.get('spectral_radius', 0)),
            ('H1 Loops', topo.get('h1_n_features', 0)),
            ('H1 Persistence', topo.get('h1_total_persistence', 0)),
            ('Volume Std', tokens.get('volume_std', 0)),
            ('Displacement', tokens.get('displacement_mean', 0)),
            ('Coherence', tokens.get('direction_coherence', 0)),
            ('Grid Disp', grid.get('mean_displacement', 0)),
        ]

        for name, val in candidates:
            if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                metric_names.append(name)
                metric_values.append(float(val))

        if len(metric_names) < 3:
            return

        # Normalize to [0, 1]
        values = np.array(metric_values)
        v_max = values.max()
        if v_max > 0:
            values_norm = values / v_max
        else:
            values_norm = values

        # Radar plot
        n_metrics = len(metric_names)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon
        values_plot = values_norm.tolist() + [values_norm[0]]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True),
                               facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        ax.plot(angles, values_plot, 'o-', color='#44aaff', lw=2, markersize=6)
        ax.fill(angles, values_plot, alpha=0.15, color='#44aaff')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=8, color='#cccccc')
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7, color='#888888')
        ax.spines['polar'].set_color('#2a3a4a')
        ax.grid(True, alpha=0.2, color='#444444')

        ax.set_title(f'Layer Fingerprint — {prefix}', color='#c0d8e8',
                     fontsize=11, pad=20)

        path = os.path.join(self.output_dir, f'{prefix}_radar.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    # ═══════════════════════════════════════════════════════════════════
    # CROSS-LAYER ANALYSIS (with new visualizations)
    # ═══════════════════════════════════════════════════════════════════

    def analyze_cross_layer(self, all_results: List[Dict]):
        """
        Analyze topological changes ACROSS layers at a fixed step.
        Now includes additional cross-layer visualizations.
        """
        if len(all_results) < 2:
            return

        self.log(f"\n[bold yellow]═══ Cross-Layer Analysis ═══[/]")

        step = all_results[0].get('step', '?')

        # Collect metrics
        layers = []
        h1_counts = []
        h1_total_persistence = []
        divergences = []
        curls = []
        shears = []
        spectral_radii = []
        condition_numbers = []
        persistence_entropies = []

        for r in all_results:
            layer = r.get('layer', '?')
            layers.append(layer)

            topo = r.get('topology', {})
            h1_counts.append(topo.get('h1_n_features', 0))
            h1_total_persistence.append(topo.get('h1_total_persistence', 0))
            persistence_entropies.append(topo.get('persistence_entropy', 0))

            jac = r.get('jacobian', {})
            divergences.append(jac.get('divergence', 0))
            curls.append(jac.get('curl', 0))
            shears.append(jac.get('shear', 0))
            spectral_radii.append(jac.get('spectral_radius', 1))
            condition_numbers.append(jac.get('condition_number', 1))

        # ── Original cross-layer plot ───────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0a0a1a')

        plot_configs = [
            (axes[0, 0], h1_counts, 'H₁ Loop Count', '#ff44aa'),
            (axes[0, 1], h1_total_persistence, 'H₁ Total Persistence', '#ff8844'),
            (axes[0, 2], divergences, 'Divergence (tr J)', '#44aaff'),
            (axes[1, 0], curls, 'Curl (||J_antisym||)', '#44ffaa'),
            (axes[1, 1], shears, 'Shear', '#ffaa44'),
            (axes[1, 2], spectral_radii, 'Spectral Radius', '#ff4444'),
        ]

        for ax, values, title, color in plot_configs:
            ax.set_facecolor('#0d1117')
            ax.plot(layers, values, 'o-', color=color, lw=2, markersize=6)
            ax.set_xlabel('Layer', color='#888888')
            ax.set_title(title, color=color, fontsize=10)
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.15, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle(f'Cross-Layer Geometric Analysis — Step {step}',
                     color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f'cross_layer_step{step}.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

        # ── NEW: Persistence entropy across layers ──────────────────
        self._plot_persistence_entropy_cross_layer(layers, persistence_entropies, step)

        # ── NEW: Cumulative Betti barcode (stacked area) ────────────
        self._plot_cumulative_betti(all_results, step)

        # ── NEW: Correlation matrix of all geometric features ───────
        self._plot_feature_correlation(all_results, step)

        # ── NEW: Bottleneck distance heatmap ────────────────────────
        if HAS_RIPSER and HAS_BOTTLENECK:
            self._plot_bottleneck_matrix(all_results, step)

        # ── Wasserstein distance matrix ─────────────────────────────
        if HAS_RIPSER and HAS_PERSIM:
            self._compute_wasserstein_matrix(all_results, step)

        # ── Phase transitions ───────────────────────────────────────
        self._detect_phase_transitions(layers, h1_counts, h1_total_persistence,
                                       divergences, curls, step)

        # ── Inner vs outer ──────────────────────────────────────────
        self._analyze_inner_outer(all_results)

    def _plot_persistence_entropy_cross_layer(self, layers, entropies, step):
        """Plot persistence entropy across layers — measures topological complexity."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        ax.bar(layers, entropies, color='#aa44ff', alpha=0.8, edgecolor='#2a3a4a')
        ax.plot(layers, entropies, 'o-', color='#ffcc44', lw=1.5, markersize=5, zorder=5)

        ax.set_xlabel('Layer', color='#888888')
        ax.set_ylabel('Persistence Entropy', color='#888888')
        ax.set_title(f'Persistence Entropy Across Layers — Step {step}\n'
                     f'(Higher = more complex topology)',
                     color='#c0d8e8', fontsize=11)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333', axis='y')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'persistence_entropy_step{step}.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_cumulative_betti(self, all_results: List[Dict], step):
        """
        Plot cumulative Betti numbers as a STACKED AREA chart across layers.
        Shows how topological features accumulate through the network.
        """
        layers = []
        b0_vals = []
        b1_vals = []

        for r in all_results:
            layers.append(r.get('layer', 0))
            topo = r.get('topology', {})
            b0_vals.append(topo.get('h0_n_features', 0))
            b1_vals.append(topo.get('h1_n_features', 0))

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        ax.fill_between(layers, 0, b0_vals, alpha=0.4, color='#44aaff',
                        label='β₀ (components)')
        ax.fill_between(layers, b0_vals,
                        [b0 + b1 for b0, b1 in zip(b0_vals, b1_vals)],
                        alpha=0.4, color='#ff44aa', label='β₁ (loops)')

        ax.plot(layers, b0_vals, 'o-', color='#44aaff', lw=1.5, markersize=4)
        ax.plot(layers, [b0 + b1 for b0, b1 in zip(b0_vals, b1_vals)],
                'o-', color='#ff44aa', lw=1.5, markersize=4)

        ax.set_xlabel('Layer', color='#888888')
        ax.set_ylabel('Feature Count', color='#888888')
        ax.set_title(f'Cumulative Betti Numbers — Step {step}',
                     color='#c0d8e8', fontsize=11)
        ax.legend(fontsize=9, facecolor='#0d1117', edgecolor='#2a3a4a',
                  labelcolor='#cccccc')
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'cumulative_betti_step{step}.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_feature_correlation(self, all_results: List[Dict], step):
        """
        Plot a CORRELATION MATRIX of all geometric features across layers.
        Reveals which geometric properties co-vary.
        """
        feature_names = ['divergence', 'curl', 'shear', 'spectral_radius',
                         'condition_number', 'h1_loops', 'h1_persistence',
                         'volume_mean', 'volume_std', 'displacement']

        feature_matrix = []
        for r in all_results:
            jac = r.get('jacobian', {})
            topo = r.get('topology', {})
            tokens = r.get('tokens', {})
            row = [
                jac.get('divergence', 0),
                jac.get('curl', 0),
                jac.get('shear', 0),
                jac.get('spectral_radius', 0),
                jac.get('condition_number', 0),
                topo.get('h1_n_features', 0),
                topo.get('h1_total_persistence', 0),
                tokens.get('volume_mean', 0),
                tokens.get('volume_std', 0),
                tokens.get('displacement_mean', 0),
            ]
            feature_matrix.append(row)

        if len(feature_matrix) < 3:
            return

        X = np.array(feature_matrix)
        # Compute correlation matrix
        # Handle constant columns
        stds = X.std(axis=0)
        valid_cols = stds > 1e-10
        if valid_cols.sum() < 3:
            return

        X_valid = X[:, valid_cols]
        names_valid = [n for n, v in zip(feature_names, valid_cols) if v]

        corr = np.corrcoef(X_valid.T)

        fig, ax = plt.subplots(1, 1, figsize=(9, 8), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pearson Correlation', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)

        n_feat = len(names_valid)
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(names_valid, fontsize=7, color='#cccccc', rotation=45, ha='right')
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(names_valid, fontsize=7, color='#cccccc')

        # Annotate cells
        for i in range(n_feat):
            for j in range(n_feat):
                val = corr[i, j]
                color = '#ffffff' if abs(val) > 0.5 else '#aaaaaa'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

        ax.set_title(f'Feature Correlation Matrix — Step {step}',
                     color='#c0d8e8', fontsize=11)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'correlation_matrix_step{step}.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _plot_bottleneck_matrix(self, all_results: List[Dict], step):
        """
        Plot BOTTLENECK distance heatmap between layer persistence diagrams.
        Bottleneck distance captures the single largest topological difference.
        """
        n = len(all_results)
        B = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                topo_i = all_results[i].get('topology', {})
                topo_j = all_results[j].get('topology', {})

                dgms_i = topo_i.get('diagrams', [])
                dgms_j = topo_j.get('diagrams', [])

                total_dist = 0.0
                for dim in range(min(len(dgms_i), len(dgms_j), 2)):
                    d_i = dgms_i[dim] if dim < len(dgms_i) else np.empty((0, 2))
                    d_j = dgms_j[dim] if dim < len(dgms_j) else np.empty((0, 2))

                    if len(d_i) > 0:
                        d_i = d_i[np.isfinite(d_i[:, 1])]
                    if len(d_j) > 0:
                        d_j = d_j[np.isfinite(d_j[:, 1])]

                    if len(d_i) > 0 and len(d_j) > 0:
                        try:
                            total_dist += bottleneck_distance(d_i, d_j)
                        except Exception:
                            pass

                B[i, j] = B[j, i] = total_dist

        fig, ax = plt.subplots(1, 1, figsize=(8, 7), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        im = ax.imshow(B, cmap='magma', interpolation='nearest', origin='lower')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Bottleneck Distance', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)

        layer_labels = [str(all_results[i].get('layer', i)) for i in range(n)]
        ax.set_xticks(range(n))
        ax.set_xticklabels(layer_labels, fontsize=7, color='#888888')
        ax.set_yticks(range(n))
        ax.set_yticklabels(layer_labels, fontsize=7, color='#888888')
        ax.set_xlabel('Layer', color='#888888')
        ax.set_ylabel('Layer', color='#888888')
        ax.set_title(f'Bottleneck Distance Between Layers — Step {step}',
                     color='#c0d8e8', fontsize=11)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'bottleneck_matrix_step{step}.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _compute_wasserstein_matrix(self, all_results: List[Dict], step):
        """Compute pairwise Wasserstein distances between layer persistence diagrams."""
        n = len(all_results)
        W = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                topo_i = all_results[i].get('topology', {})
                topo_j = all_results[j].get('topology', {})

                dgms_i = topo_i.get('diagrams', [])
                dgms_j = topo_j.get('diagrams', [])

                total_dist = 0.0
                for dim in range(min(len(dgms_i), len(dgms_j), 2)):
                    d_i = dgms_i[dim] if dim < len(dgms_i) else np.empty((0, 2))
                    d_j = dgms_j[dim] if dim < len(dgms_j) else np.empty((0, 2))

                    if len(d_i) > 0:
                        d_i = d_i[np.isfinite(d_i[:, 1])]
                    if len(d_j) > 0:
                        d_j = d_j[np.isfinite(d_j[:, 1])]

                    if len(d_i) > 0 and len(d_j) > 0:
                        try:
                            total_dist += wasserstein_distance(d_i, d_j)
                        except Exception:
                            pass

                W[i, j] = W[j, i] = total_dist

        fig, ax = plt.subplots(1, 1, figsize=(8, 7), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        im = ax.imshow(W, cmap='inferno', interpolation='nearest', origin='lower')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Wasserstein-1 Distance', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)

        layer_labels = [str(all_results[i].get('layer', i)) for i in range(n)]
        ax.set_xticks(range(n))
        ax.set_xticklabels(layer_labels, fontsize=7, color='#888888')
        ax.set_yticks(range(n))
        ax.set_yticklabels(layer_labels, fontsize=7, color='#888888')
        ax.set_xlabel('Layer', color='#888888')
        ax.set_ylabel('Layer', color='#888888')
        ax.set_title(f'Wasserstein Distance Between Layer Persistence Diagrams — Step {step}',
                     color='#c0d8e8', fontsize=11)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'wasserstein_matrix_step{step}.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    def _detect_phase_transitions(self, layers, h1_counts, h1_total_persistence,
                                   divergences, curls, step):
        """Detect topological phase transitions between layers."""
        self.log(f"\n[bold yellow]── Phase Transition Detection ──[/]")

        if len(h1_counts) < 3:
            self.log("  Too few layers for phase transition detection")
            return

        h1_diffs = [abs(h1_counts[i+1] - h1_counts[i]) for i in range(len(h1_counts)-1)]
        pers_diffs = [abs(h1_total_persistence[i+1] - h1_total_persistence[i])
                      for i in range(len(h1_total_persistence)-1)]

        if h1_diffs:
            max_h1_diff = max(h1_diffs)
            if max_h1_diff > 0:
                for i, d in enumerate(h1_diffs):
                    if d == max_h1_diff:
                        self.log(f"  [bold]Largest H1 change: layer {layers[i]}→{layers[i+1]} "
                                 f"(Δ loops = {d})[/]")

        if pers_diffs:
            max_pers_diff = max(pers_diffs)
            if max_pers_diff > 0:
                for i, d in enumerate(pers_diffs):
                    if d == max_pers_diff:
                        self.log(f"  [bold]Largest persistence change: "
                                 f"layer {layers[i]}→{layers[i+1]} "
                                 f"(Δ persistence = {d:.4f})[/]")

        if h1_total_persistence:
            mean_pers = np.mean(h1_total_persistence)
            active_layers = [l for l, p in zip(layers, h1_total_persistence)
                             if p > mean_pers * 1.5]
            if active_layers:
                self.log(f"  [green]Topologically active layers (persistence > 1.5× mean): "
                         f"{active_layers}[/]")

    def _analyze_inner_outer(self, all_results: List[Dict]):
        """Test Conjecture 4: inner layers compute, outer layers translate."""
        n = len(all_results)
        if n < 4:
            return

        self.log(f"\n[bold yellow]── Inner vs Outer Layer Analysis (Conjecture 4) ──[/]")

        third = max(1, n // 3)
        early = all_results[:third]
        middle = all_results[third:2*third]
        late = all_results[2*third:]

        def _avg_metric(results_list, metric_path):
            values = []
            for r in results_list:
                parts = metric_path.split('.')
                val = r
                for p in parts:
                    if isinstance(val, dict):
                        val = val.get(p, None)
                    else:
                        val = None
                    if val is None:
                        break
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)
            return np.mean(values) if values else 0

        metrics = [
            ('jacobian.divergence', 'Divergence'),
            ('jacobian.curl', 'Curl'),
            ('jacobian.shear', 'Shear'),
            ('jacobian.spectral_radius', 'Spectral Radius'),
            ('topology.h1_n_features', 'H1 Loop Count'),
            ('topology.h1_total_persistence', 'H1 Total Persistence'),
        ]

        table = Table(title="Inner vs Outer Layer Comparison",
                      box=box.ROUNDED, show_lines=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Early Layers", style="blue")
        table.add_column("Middle Layers", style="magenta")
        table.add_column("Late Layers", style="green")

        for metric_path, label in metrics:
            e = _avg_metric(early, metric_path)
            m = _avg_metric(middle, metric_path)
            l = _avg_metric(late, metric_path)
            table.add_row(label, f"{e:.4f}", f"{m:.4f}", f"{l:.4f}")

        console.print(table)

        mid_curl = _avg_metric(middle, 'jacobian.curl')
        early_curl = _avg_metric(early, 'jacobian.curl')
        late_curl = _avg_metric(late, 'jacobian.curl')

        if mid_curl > early_curl and mid_curl > late_curl:
            self.log(f"  [green]✓ Middle layers show highest curl — consistent with "
                     f"Conjecture 4 (inner layers compute)[/]")

        mid_h1 = _avg_metric(middle, 'topology.h1_n_features')
        early_h1 = _avg_metric(early, 'topology.h1_n_features')
        late_h1 = _avg_metric(late, 'topology.h1_n_features')

        if mid_h1 > early_h1 and mid_h1 > late_h1:
            self.log(f"  [green]✓ Middle layers have most H1 loops — consistent with "
                     f"topological computation hypothesis[/]")

    # ─── Topology tracking across training steps ────────────────────

    def track_topology_over_training(self, all_files: List[Dict]):
        """Track how topology evolves across training steps."""
        self.log(f"\n[bold yellow]═══ Topology Over Training ═══[/]")

        by_step = defaultdict(list)
        for info in all_files:
            step = info.get('step', 0)
            by_step[step].append(info)

        steps = sorted(by_step.keys())
        if len(steps) < 2:
            self.log("  Need at least 2 training steps for tracking")
            return

        self.log(f"  Found {len(steps)} training steps: "
                 f"{steps[:10]}{'...' if len(steps) > 10 else ''}")

        step_h1_total = []
        step_h1_max = []
        step_mean_curl = []
        step_mean_divergence = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Analyzing topology over training..."),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("topo_track", total=len(steps))

            for step in steps:
                files = by_step[step]
                step_results = []

                for info in files:
                    if 'data' not in info:
                        info['data'] = load_jacobi_npz(info['path'])
                    r = self.analyze_single(info)
                    step_results.append(r)

                # Aggregate
                h1_counts = [r.get('topology', {}).get('h1_n_features', 0)
                             for r in step_results]
                h1_pers = [r.get('topology', {}).get('h1_total_persistence', 0)
                           for r in step_results]
                curls = [r.get('jacobian', {}).get('curl', 0)
                         for r in step_results]
                divs = [r.get('jacobian', {}).get('divergence', 0)
                        for r in step_results]

                step_h1_total.append(sum(h1_counts))
                step_h1_max.append(max(h1_counts) if h1_counts else 0)
                step_mean_curl.append(np.mean(curls) if curls else 0)
                step_mean_divergence.append(np.mean(divs) if divs else 0)

                progress.update(task, advance=1)

        # Plot topology over training
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0a0a1a')

        plot_configs = [
            (axes[0, 0], step_h1_total, 'Total H1 Loops (all layers)', '#ff44aa'),
            (axes[0, 1], step_h1_max, 'Max H1 Loops (any single layer)', '#ff8844'),
            (axes[1, 0], step_mean_curl, 'Mean Curl (all layers)', '#44ffaa'),
            (axes[1, 1], step_mean_divergence, 'Mean Divergence (all layers)', '#44aaff'),
        ]

        for ax, values, title, color in plot_configs:
            ax.set_facecolor('#0d1117')
            ax.plot(steps, values, 'o-', color=color, lw=2, markersize=4)
            ax.set_xlabel('Training Step', color='#888888')
            ax.set_title(title, color=color, fontsize=10)
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.15, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle('Topological Evolution Over Training', color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'topology_over_training.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

        # NEW: Plot persistence entropy over training
        self._plot_persistence_entropy_over_training(steps, by_step)

        # Detect grokking-like transitions
        if len(step_h1_total) > 5:
            diffs = np.diff(step_h1_total)
            if len(diffs) > 0:
                max_jump_idx = np.argmax(np.abs(diffs))
                max_jump = diffs[max_jump_idx]
                if abs(max_jump) > 2:
                    self.log(f"  [bold green]⚡ Possible topological phase transition at "
                             f"step {steps[max_jump_idx]}→{steps[max_jump_idx+1]}: "
                             f"H1 loops changed by {max_jump}[/]")
                    self.log(f"  This could indicate grokking — the emergence of "
                             f"computational topological structures!")

    def _plot_persistence_entropy_over_training(self, steps, by_step):
        """Plot how persistence entropy evolves over training steps."""
        entropies = []

        for step in steps:
            files = by_step[step]
            step_entropies = []
            for info in files:
                data = info.get('data', {})
                if 'h_in_2d' not in data:
                    continue
                points = data['h_in_2d']
                if points.shape[0] < 4:
                    continue
                if points.shape[0] > 200:
                    idx = np.random.choice(points.shape[0], 200, replace=False)
                    points = points[idx]
                try:
                    dists = pdist(points)
                    thresh = np.percentile(dists[dists > 0], 95) if len(dists) > 0 and dists.max() > 0 else 1.0
                    result = ripser(points, maxdim=1, thresh=thresh)
                    dgms = result['dgms']
                    all_lifetimes = []
                    for dim_idx in range(min(len(dgms), 2)):
                        finite = dgms[dim_idx][np.isfinite(dgms[dim_idx][:, 1])]
                        if len(finite) > 0:
                            all_lifetimes.extend((finite[:, 1] - finite[:, 0]).tolist())
                    if all_lifetimes:
                        lt_arr = np.array(all_lifetimes)
                        lt_norm = lt_arr / (lt_arr.sum() + 1e-10)
                        step_entropies.append(scipy_entropy(lt_norm))
                except Exception:
                    pass

            entropies.append(np.mean(step_entropies) if step_entropies else 0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        ax.plot(steps, entropies, 'o-', color='#aa44ff', lw=2, markersize=5)
        ax.fill_between(steps, entropies, alpha=0.1, color='#aa44ff')

        ax.set_xlabel('Training Step', color='#888888')
        ax.set_ylabel('Mean Persistence Entropy', color='#888888')
        ax.set_title('Persistence Entropy Over Training\n'
                     '(Higher = more complex/distributed topology)',
                     color='#c0d8e8', fontsize=11)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, 'persistence_entropy_training.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    # ─── UMAP embedding of Jacobian fields ──────────────────────────

    def visualize_jacobian_umap(self, all_results: List[Dict]):
        """
        Embed the per-layer Jacobian fields into 2D using UMAP,
        colored by layer index.
        """
        if not HAS_UMAP:
            self.log("  [yellow]UMAP not available, skipping embedding[/]")
            return

        self.log(f"\n[bold yellow]── UMAP Embedding of Jacobian Fields ──[/]")

        features = []
        layer_ids = []
        step_ids = []

        for r in all_results:
            jac = r.get('jacobian', {})
            topo = r.get('topology', {})
            tokens = r.get('tokens', {})

            fv = []
            for key in ['divergence', 'curl', 'shear', 'spectral_radius',
                        'condition_number']:
                fv.append(jac.get(key, 0))

            for key in ['h1_n_features', 'h1_total_persistence', 'h0_n_features']:
                fv.append(topo.get(key, 0))

            for key in ['volume_mean', 'volume_std', 'displacement_mean',
                        'direction_coherence']:
                fv.append(tokens.get(key, 0))

            if fv:
                features.append(fv)
                layer_ids.append(r.get('layer', 0))
                step_ids.append(r.get('step', 0))

        if len(features) < 5:
            self.log("  Too few data points for UMAP")
            return

        X = np.array(features)
        X = StandardScaler().fit_transform(X)

        try:
            reducer = umap.UMAP(n_components=2, random_state=42,
                                n_neighbors=min(15, len(X) - 1))
            embedding = reducer.fit_transform(X)
        except Exception as e:
            self.log(f"  [red]UMAP failed: {e}[/]")
            return

        # Plot colored by layer
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a1a')

        ax = axes[0]
        ax.set_facecolor('#0d1117')
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                             c=layer_ids, cmap='viridis', s=40,
                             edgecolors='#ffffff', linewidths=0.5, zorder=5)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Layer Index', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)
        ax.set_title('Colored by Layer', color='#c0d8e8', fontsize=10)
        ax.set_xlabel('UMAP 1', color='#888888')
        ax.set_ylabel('UMAP 2', color='#888888')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        # Plot colored by step
        ax2 = axes[1]
        ax2.set_facecolor('#0d1117')
        scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1],
                               c=step_ids, cmap='plasma', s=40,
                               edgecolors='#ffffff', linewidths=0.5, zorder=5)
        cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label('Training Step', color='#888888', fontsize=9)
        cbar2.ax.tick_params(colors='#888888', labelsize=7)
        ax2.set_title('Colored by Training Step', color='#c0d8e8', fontsize=10)
        ax2.set_xlabel('UMAP 1', color='#888888')
        ax2.set_ylabel('UMAP 2', color='#888888')
        ax2.tick_params(colors='#888888', labelsize=7)
        for spine in ax2.spines.values():
            spine.set_color('#2a3a4a')

        fig.suptitle('UMAP Embedding of Jacobian Field Features',
                     color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'jacobian_umap.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")

    # ─── Network/graph analysis of token interactions ───────────────

    def analyze_token_graph(self, data: Dict, prefix: str):
        """
        Build a graph from token proximity in the output space
        and analyze its topological properties.
        """
        if not HAS_NETWORKX:
            return

        if 'h_out_2d' not in data:
            return

        h_out = data['h_out_2d']
        n = h_out.shape[0]
        if n < 4:
            return

        # Build epsilon-neighborhood graph
        dists = squareform(pdist(h_out))
        epsilon = np.percentile(dists[dists > 0], 15)

        G = nx.Graph()
        for i in range(n):
            G.add_node(i)

        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < epsilon:
                    G.add_edge(i, j, weight=dists[i, j])

        n_components = nx.number_connected_components(G)
        n_edges = G.number_of_edges()

        try:
            cycle_basis = nx.cycle_basis(G)
            n_cycles = len(cycle_basis)
        except Exception:
            n_cycles = 0

        self.log(f"  Token graph (ε={epsilon:.4f}): {n} nodes, {n_edges} edges, "
                 f"{n_components} components, {n_cycles} independent cycles")

        if n_cycles > 0:
            self.log(f"  [magenta]Graph cycles detected — potential loop structures "
                     f"in token space[/]")

        # Visualize the graph
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')

        pos = {i: (h_out[i, 0], h_out[i, 1]) for i in range(n)}

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#334455',
                               width=0.5, alpha=0.4)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=20,
                               node_color='#44aaff', edgecolors='none', alpha=0.8)

        # Highlight cycles
        if cycle_basis:
            for cycle in cycle_basis[:5]:
                cycle_edges = [(cycle[i], cycle[(i + 1) % len(cycle)])
                               for i in range(len(cycle))]
                nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, ax=ax,
                                       edge_color='#ff44aa', width=2.0, alpha=0.8)

        if 'token_strings' in data:
            ts = data['token_strings']
            labels = {}
            for i in range(min(n, 20)):
                if i % max(1, n // 15) == 0:
                    labels[i] = str(ts[i])[:8] if i < len(ts) else f"t{i}"
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6,
                                    font_color='#cccccc')

        ax.set_title(f'Token Proximity Graph — {prefix}\n'
                     f'{n_components} components, {n_cycles} cycles',
                     color='#c0d8e8', fontsize=10)
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        path = os.path.join(self.output_dir, f'{prefix}_token_graph.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        self.log(f"  📊 Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Jacobi field .npz files for topological structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 analyze_jacobi.py runs/0/jacobi_data/jacobi_step000100_layer00.npz
  python3 analyze_jacobi.py runs/0/jacobi_data/
  python3 analyze_jacobi.py runs/0/jacobi_data/ --step 100
  python3 analyze_jacobi.py runs/0/jacobi_data/ --compare-layers
  python3 analyze_jacobi.py runs/0/jacobi_data/ --track-topology
  python3 analyze_jacobi.py runs/0/jacobi_data/ --all
        """,
    )

    parser.add_argument("path", type=str,
                        help="Path to a .npz file or directory of .npz files")
    parser.add_argument("--step", type=int, default=None,
                        help="Analyze only this training step")
    parser.add_argument("--layer", type=int, default=None,
                        help="Analyze only this layer")
    parser.add_argument("--compare-layers", action="store_true",
                        help="Cross-layer comparison at a fixed step")
    parser.add_argument("--track-topology", action="store_true",
                        help="Track topology evolution across training steps")
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses")
    parser.add_argument("--output-dir", type=str, default="jacobi_analysis",
                        help="Output directory for plots and reports")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization (analysis only)")

    args = parser.parse_args()

    console.print(Panel(
        "[bold white]Jacobi Field Topology Analyzer (EXTENDED)[/]\n"
        "[dim]Searching for topological structures in transformer layer deformations[/]\n"
        "[dim]Now with barcodes, landscapes, quiver fields, stream plots, dendrograms,[/]\n"
        "[dim]3D surfaces, polar plots, radar charts, filtration snapshots, and more![/]",
        border_style="bold cyan",
        padding=(1, 4),
    ))

    # ── Discover files ──────────────────────────────────────────────────
    files = discover_npz_files(args.path, step=args.step, layer=args.layer)

    if not files:
        console.print("[bold red]No files found to analyze.[/]")
        sys.exit(1)

    console.print(f"[green]Found {len(files)} .npz files to analyze[/]")

    # ── Initialize analyzer ─────────────────────────────────────────────
    analyzer = JacobiAnalyzer(output_dir=args.output_dir)

    # ── Load data ───────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Loading Jacobi field data..."),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("load", total=len(files))
        for info in files:
            if 'data' not in info:
                info['data'] = load_jacobi_npz(info['path'])
            progress.update(task, advance=1)

    # ── Single-file analysis ────────────────────────────────────────────
    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Analyzing individual files..."),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("analyze", total=len(files))

        for info in files:
            results = analyzer.analyze_single(info)

            if not args.no_viz:
                analyzer.visualize_single(info, results)

                # Token graph analysis
                step = info.get('step', '?')
                layer = info.get('layer', '?')
                prefix = f"step{step}_layer{layer}"
                analyzer.analyze_token_graph(info['data'], prefix)

            all_results.append(results)
            progress.update(task, advance=1)

    # ── Cross-layer analysis ────────────────────────────────────────────
    if args.compare_layers or args.all:
        by_step = defaultdict(list)
        for r in all_results:
            by_step[r['step']].append(r)

        for step, step_results in sorted(by_step.items()):
            if len(step_results) > 1:
                step_results.sort(key=lambda r: r.get('layer', 0))
                analyzer.analyze_cross_layer(step_results)

    # ── Topology tracking over training ─────────────────────────────────
    if args.track_topology or args.all:
        analyzer.track_topology_over_training(files)

    # ── UMAP embedding ──────────────────────────────────────────────────
    if args.all and len(all_results) > 5:
        analyzer.visualize_jacobian_umap(all_results)

    # ── Summary statistics ──────────────────────────────────────────────
    analyzer.log(f"\n[bold cyan]═══ Summary ═══[/]")
    analyzer.log(f"  Files analyzed: {len(files)}")

    total_h1 = sum(r.get('topology', {}).get('h1_n_features', 0) for r in all_results)
    total_h1_sig = sum(r.get('topology', {}).get('h1_n_significant', 0) for r in all_results)
    total_h2 = sum(r.get('topology', {}).get('h2_n_features', 0) for r in all_results)

    analyzer.log(f"  Total H1 loops detected: {total_h1} ({total_h1_sig} significant)")
    analyzer.log(f"  Total H2 voids detected: {total_h2}")

    if total_h1_sig > 0:
        analyzer.log(f"\n  [bold green]⚡ SIGNIFICANT TOPOLOGICAL STRUCTURES FOUND![/]")
        analyzer.log(f"  These persistent loops may encode computational circuits")
        analyzer.log(f"  as predicted by Conjecture 3 (topological computation)")
    else:
        analyzer.log(f"\n  [yellow]No significant persistent loops detected.[/]")
        analyzer.log(f"  This could mean:")
        analyzer.log(f"    - The model hasn't grokked yet (try later training steps)")
        analyzer.log(f"    - The topological structures exist in higher dimensions")
        analyzer.log(f"    - The 2D projection loses the relevant topology")
        analyzer.log(f"    - The filtration threshold needs adjustment")

    # ── Visualization summary ───────────────────────────────────────────
    n_plots = len(glob.glob(os.path.join(args.output_dir, '*.png')))
    analyzer.log(f"\n  Total visualizations generated: {n_plots}")

    # ── Save report ─────────────────────────────────────────────────────
    analyzer.save_report()

    console.print(Panel(
        f"[bold green]Analysis complete![/]\n"
        f"Output directory: {args.output_dir}/\n"
        f"Files analyzed: {len(files)}\n"
        f"Visualizations generated: {n_plots}\n"
        f"H1 loops: {total_h1} ({total_h1_sig} significant)\n"
        f"H2 voids: {total_h2}",
        title="[bold]Results",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
