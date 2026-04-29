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
Jacobi Field Topology Analyzer
===============================

Analyzes .npz files produced by the Metric Space Explorer / train.py
Jacobi field computation, searching for topological structures predicted
by the fibre bundle hypothesis:

  - Persistent loops (H1 features) that may encode computational circuits
  - Connected components (H0) showing clustering/separation of token flows
  - Voids (H2) indicating higher-order topological structure
  - Cross-layer topological phase transitions
  - Eigenvalue spectra of the Jacobian (rotation, shear, expansion)
  - Divergence/curl/shear fields and their spatial coherence
  - Holonomy-like signatures (how much local frames rotate across layers)

Usage:
    python3 analyze_jacobi.py runs/0/jacobi_data/jacobi_step000100_layer00.npz
    python3 analyze_jacobi.py runs/0/jacobi_data/  # analyze all files in directory
    python3 analyze_jacobi.py runs/0/jacobi_data/ --step 100  # specific step
    python3 analyze_jacobi.py runs/0/jacobi_data/ --compare-layers
    python3 analyze_jacobi.py runs/0/jacobi_data/ --track-topology

Author: Auto-generated analyzer for the fibre bundle interpretability framework
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


# Must run BEFORE heavy imports
ensure_safe_env()

# ═══════════════════════════════════════════════════════════════════════════
# Heavy imports (after uv bootstrap)
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — always save to file
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, det, norm
from scipy.stats import entropy as scipy_entropy
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
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

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
        # Handle 0-d arrays (scalars saved as arrays)
        if arr.ndim == 0:
            result[key] = arr.item()
        else:
            result[key] = arr
    data.close()
    return result


def discover_npz_files(path: str, step: Optional[int] = None,
                       layer: Optional[int] = None) -> List[Dict]:
    """
    Discover all jacobi .npz files in a directory or load a single file.
    Returns list of dicts with parsed metadata.
    """
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
# ANALYSIS MODULES
# ═══════════════════════════════════════════════════════════════════════════

class JacobiAnalyzer:
    """Main analyzer class that orchestrates all analyses."""
    
    def __init__(self, output_dir: str = "jacobi_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.report_lines: List[str] = []
    
    def log(self, msg: str):
        """Add a line to the report."""
        self.report_lines.append(msg)
        console.print(f"  {msg}")
    
    def save_report(self, filename: str = "analysis_report.md"):
        """Save the accumulated report to a markdown file."""
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write("# Jacobi Field Topology Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            for line in self.report_lines:
                # Strip rich markup for the file
                import re
                clean = re.sub(r'\[.*?\]', '', line)
                f.write(clean + "\n")
        console.print(f"\n[bold green]📄 Report saved to: {path}[/]")
    
    # ─── Single-file analysis ───────────────────────────────────────────
    
    def analyze_single(self, info: Dict) -> Dict:
        """Full analysis of a single Jacobi field .npz file."""
        if 'data' not in info:
            info['data'] = load_jacobi_npz(info['path'])
        
        data = info['data']
        step = info.get('step', '?')
        layer = info.get('layer', '?')
        
        self.log(f"\n[bold cyan]═══ Step {step}, Layer {layer} ═══[/]")
        self.log(f"  File: {info['basename']}")
        self.log(f"  Keys: {list(data.keys())}")
        
        results = {
            'step': step,
            'layer': layer,
            'file': info['path'],
        }
        
        # ── Basic geometry ──────────────────────────────────────────
        results['geometry'] = self._analyze_geometry(data)
        
        # ── Jacobian decomposition ──────────────────────────────────
        results['jacobian'] = self._analyze_jacobian(data)
        
        # ── Token-level analysis ────────────────────────────────────
        results['tokens'] = self._analyze_tokens(data)
        
        # ── Persistent homology ─────────────────────────────────────
        if HAS_RIPSER:
            results['topology'] = self._analyze_topology(data)
        
        # ── Grid deformation analysis ───────────────────────────────
        results['grid'] = self._analyze_grid_deformation(data)
        
        # ── Eigenvalue analysis ─────────────────────────────────────
        results['eigenvalues'] = self._analyze_eigenvalues(data)
        
        return results
    
    def _analyze_geometry(self, data: Dict) -> Dict:
        """Analyze basic geometric properties."""
        results = {}
        
        if 'h_in_2d' in data:
            h_in = data['h_in_2d']
            results['n_tokens'] = h_in.shape[0]
            results['dim'] = h_in.shape[1] if h_in.ndim > 1 else 2
            results['input_centroid'] = h_in.mean(axis=0)
            results['input_spread'] = h_in.std(axis=0)
            results['input_bbox'] = (h_in.min(axis=0), h_in.max(axis=0))
            
            # Pairwise distances
            if h_in.shape[0] > 1:
                dists = pdist(h_in)
                results['mean_pairwise_dist'] = dists.mean()
                results['max_pairwise_dist'] = dists.max()
                results['min_pairwise_dist'] = dists[dists > 0].min() if (dists > 0).any() else 0
            
            self.log(f"  Tokens: {results['n_tokens']}, Dim: {results['dim']}")
            self.log(f"  Input spread: {results['input_spread']}")
            if 'mean_pairwise_dist' in results:
                self.log(f"  Mean pairwise dist: {results['mean_pairwise_dist']:.4f}")
        
        if 'h_out_2d' in data:
            h_out = data['h_out_2d']
            results['output_centroid'] = h_out.mean(axis=0)
            results['output_spread'] = h_out.std(axis=0)
            
            # Centroid shift
            if 'input_centroid' in results:
                shift = results['output_centroid'] - results['input_centroid']
                results['centroid_shift'] = shift
                results['centroid_shift_magnitude'] = np.linalg.norm(shift)
                self.log(f"  Centroid shift: {results['centroid_shift_magnitude']:.4f}")
        
        # Anisotropy and volume
        for key in ['anisotropy', 'log_det']:
            if key in data:
                results[key] = float(data[key]) if np.isscalar(data[key]) else float(data[key].item())
                self.log(f"  {key}: {results[key]:.4f}")
        
        return results
    
    def _analyze_jacobian(self, data: Dict) -> Dict:
        """Analyze the Jacobian matrix properties."""
        results = {}
        
        if 'jacobian' in data:
            J = data['jacobian']
            results['shape'] = J.shape
            D = J.shape[0]
            
            # Singular values
            try:
                U, S, Vh = svd(J)
                results['singular_values'] = S
                results['condition_number'] = S[0] / max(S[-1], 1e-10)
                results['determinant'] = np.prod(S)  # |det(J)| = product of singular values
                results['log_determinant'] = np.sum(np.log(np.clip(S, 1e-10, None)))
                
                self.log(f"  Jacobian shape: {J.shape}")
                self.log(f"  Singular values: [{S[0]:.3f}, ..., {S[-1]:.3f}]")
                self.log(f"  Condition number: {results['condition_number']:.2f}")
                self.log(f"  |det(J)|: {results['determinant']:.4f}")
            except Exception as e:
                self.log(f"  [yellow]SVD failed: {e}[/]")
            
            # Decomposition: symmetric + antisymmetric
            J_sym = (J + J.T) / 2
            J_antisym = (J - J.T) / 2
            
            # Divergence (trace)
            results['divergence'] = np.trace(J)
            
            # Curl (Frobenius norm of antisymmetric part)
            results['curl'] = norm(J_antisym, 'fro')
            
            # Shear (traceless symmetric part)
            trace_sym = np.trace(J_sym)
            shear_tensor = J_sym - (trace_sym / D) * np.eye(D)
            results['shear'] = norm(shear_tensor, 'fro')
            
            # Rotation angle (from antisymmetric part)
            if D == 2:
                results['rotation_angle'] = np.arctan2(J_antisym[1, 0], J_sym[0, 0])
            
            self.log(f"  Divergence (tr J): {results['divergence']:.4f}")
            self.log(f"  Curl (||J_antisym||): {results['curl']:.4f}")
            self.log(f"  Shear (||J_traceless_sym||): {results['shear']:.4f}")
            
            # Eigenvalue analysis
            try:
                eigenvalues = np.linalg.eigvals(J)
                results['eigenvalues'] = eigenvalues
                results['eigenvalue_magnitudes'] = np.abs(eigenvalues)
                results['eigenvalue_phases'] = np.angle(eigenvalues)
                
                # Check for complex eigenvalues (rotation)
                n_complex = np.sum(np.abs(eigenvalues.imag) > 1e-8)
                results['n_complex_eigenvalues'] = n_complex
                
                if n_complex > 0:
                    self.log(f"  [magenta]Complex eigenvalues: {n_complex}/{D} "
                             f"(indicates rotation/spiral)[/]")
                
                # Spectral radius
                results['spectral_radius'] = np.max(np.abs(eigenvalues))
                self.log(f"  Spectral radius: {results['spectral_radius']:.4f}")
                
            except Exception as e:
                self.log(f"  [yellow]Eigenvalue computation failed: {e}[/]")
        
        # Per-field divergence/curl/shear if available
        for key in ['divergence', 'curl', 'shear']:
            if key in data and key not in results:
                val = data[key]
                if np.isscalar(val) or (hasattr(val, 'ndim') and val.ndim == 0):
                    results[key] = float(val)
                else:
                    results[key] = float(val)
        
        return results
    
    def _analyze_tokens(self, data: Dict) -> Dict:
        """Analyze per-token properties."""
        results = {}
        
        # Per-token volume change
        if 'per_token_volume' in data:
            vol = data['per_token_volume']
            results['volume_mean'] = vol.mean()
            results['volume_std'] = vol.std()
            results['volume_min'] = vol.min()
            results['volume_max'] = vol.max()
            results['n_expanding'] = int(np.sum(vol > 1.1))
            results['n_contracting'] = int(np.sum(vol < 0.9))
            results['n_preserving'] = int(np.sum((vol >= 0.9) & (vol <= 1.1)))
            
            self.log(f"  Per-token volume: mean={vol.mean():.3f}, "
                     f"std={vol.std():.3f}, range=[{vol.min():.3f}, {vol.max():.3f}]")
            self.log(f"  Expanding: {results['n_expanding']}, "
                     f"Contracting: {results['n_contracting']}, "
                     f"Preserving: {results['n_preserving']}")
        
        # Per-token rotation
        if 'per_token_rotation' in data:
            rot = data['per_token_rotation']
            results['rotation_mean'] = rot.mean()
            results['rotation_std'] = rot.std()
            results['rotation_range'] = (rot.min(), rot.max())
            self.log(f"  Per-token rotation: mean={rot.mean():.4f} rad, "
                     f"std={rot.std():.4f}")
        
        # Per-token shear
        if 'per_token_shear_mag' in data:
            shear = data['per_token_shear_mag']
            results['shear_mean'] = shear.mean()
            results['shear_std'] = shear.std()
            results['shear_max'] = shear.max()
            self.log(f"  Per-token shear: mean={shear.mean():.4f}, max={shear.max():.4f}")
        
        # Per-token delta (displacement)
        if 'per_token_delta_2d' in data:
            delta = data['per_token_delta_2d']
            magnitudes = np.linalg.norm(delta, axis=1)
            results['displacement_mean'] = magnitudes.mean()
            results['displacement_std'] = magnitudes.std()
            results['displacement_max'] = magnitudes.max()
            
            # Direction coherence: how aligned are the displacements?
            if magnitudes.max() > 1e-8:
                unit_deltas = delta / (magnitudes[:, None] + 1e-10)
                mean_direction = unit_deltas.mean(axis=0)
                coherence = np.linalg.norm(mean_direction)  # 1 = all same direction, 0 = random
                results['direction_coherence'] = coherence
                self.log(f"  Displacement coherence: {coherence:.4f} "
                         f"(1=aligned, 0=random)")
        
        # Token strings
        if 'token_strings' in data:
            results['token_strings'] = data['token_strings']
        
        return results
    
    def _analyze_topology(self, data: Dict) -> Dict:
        """
        Compute persistent homology on the token point cloud.
        
        This is the KEY analysis for finding the topological structures
        predicted by Conjecture 3 (topological computation) in the paper.
        """
        results = {}
        
        # Determine which point cloud to use
        if 'h_in_2d' in data:
            points = data['h_in_2d']
        elif 'h_out_2d' in data:
            points = data['h_out_2d']
        else:
            self.log("  [yellow]No point cloud available for topology[/]")
            return results
        
        if points.shape[0] < 4:
            self.log("  [yellow]Too few points for persistent homology[/]")
            return results
        
        # Subsample if too many points
        max_pts = 300
        if points.shape[0] > max_pts:
            idx = np.random.choice(points.shape[0], max_pts, replace=False)
            points = points[idx]
        
        # Compute persistence
        try:
            # Normalize distances
            dists = pdist(points)
            if len(dists) > 0 and dists.max() > 0:
                thresh = np.percentile(dists[dists > 0], 95)
            else:
                thresh = 1.0
            
            result = ripser(points, maxdim=2, thresh=thresh)
            dgms = result['dgms']
            
            results['diagrams'] = dgms
            results['threshold'] = thresh
            
            # H0: Connected components
            h0 = dgms[0]
            h0_finite = h0[np.isfinite(h0[:, 1])]
            results['h0_n_features'] = len(h0_finite)
            if len(h0_finite) > 0:
                h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
                results['h0_max_lifetime'] = h0_lifetimes.max()
                results['h0_mean_lifetime'] = h0_lifetimes.mean()
                results['h0_total_persistence'] = h0_lifetimes.sum()
            
            self.log(f"  [bold]H0 (components):[/] {results['h0_n_features']} features")
            
            # H1: Loops — THE KEY FEATURE for topological computation
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
                    
                    # Significant loops (lifetime > median)
                    if len(h1_lifetimes) > 1:
                        median_lt = np.median(h1_lifetimes)
                        significant = h1_lifetimes > 2 * median_lt
                        results['h1_n_significant'] = int(significant.sum())
                    else:
                        results['h1_n_significant'] = len(h1_lifetimes)
                    
                    self.log(f"  [bold magenta]H1 (LOOPS):[/] {results['h1_n_features']} features, "
                             f"{results['h1_n_significant']} significant")
                    self.log(f"    Max lifetime: {results['h1_max_lifetime']:.4f}")
                    self.log(f"    Total persistence: {results['h1_total_persistence']:.4f}")
                    
                    if results['h1_n_significant'] > 0:
                        self.log(f"    [bold green]⚡ SIGNIFICANT LOOPS DETECTED — "
                                 f"potential topological computation structures![/]")
                else:
                    results['h1_n_features'] = 0
                    self.log(f"  H1 (loops): None detected")
            
            # H2: Voids
            if len(dgms) > 2:
                h2 = dgms[2]
                h2_finite = h2[np.isfinite(h2[:, 1])]
                results['h2_n_features'] = len(h2_finite)
                if len(h2_finite) > 0:
                    h2_lifetimes = h2_finite[:, 1] - h2_finite[:, 0]
                    results['h2_max_lifetime'] = h2_lifetimes.max()
                    results['h2_total_persistence'] = h2_lifetimes.sum()
                    self.log(f"  [bold blue]H2 (voids):[/] {results['h2_n_features']} features, "
                             f"max lifetime={results['h2_max_lifetime']:.4f}")
                else:
                    self.log(f"  H2 (voids): None detected")
            
            # Betti numbers at various scales
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
            
        except Exception as e:
            self.log(f"  [red]Persistent homology failed: {e}[/]")
        
        # ── Also analyze the OUTPUT point cloud topology ────────────
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
                    
                    # Compare input vs output topology
                    if 'h1_n_features' in results:
                        delta_h1 = len(h1_out_finite) - results['h1_n_features']
                        if delta_h1 > 0:
                            self.log(f"  [green]Layer CREATES {delta_h1} new loops "
                                     f"(input→output)[/]")
                        elif delta_h1 < 0:
                            self.log(f"  [red]Layer DESTROYS {-delta_h1} loops "
                                     f"(input→output)[/]")
                        else:
                            self.log(f"  Loop count preserved through layer")
            except Exception:
                pass
        
        return results
    
    def _analyze_grid_deformation(self, data: Dict) -> Dict:
        """Analyze the warped grid for deformation patterns."""
        results = {}
        
        has_grid_in = 'grid_x_in' in data and 'grid_y_in' in data
        has_grid_out = 'grid_x_out' in data and 'grid_y_out' in data
        has_grid_nd = 'grid_x' in data and 'grid_y' in data
        
        if has_grid_in and has_grid_out:
            gx_in = data['grid_x_in']
            gy_in = data['grid_y_in']
            gx_out = data['grid_x_out']
            gy_out = data['grid_y_out']
            
            # Displacement field
            dx = gx_out - gx_in
            dy = gy_out - gy_in
            displacement_mag = np.sqrt(dx**2 + dy**2)
            
            results['grid_shape'] = gx_in.shape
            results['mean_displacement'] = displacement_mag.mean()
            results['max_displacement'] = displacement_mag.max()
            results['displacement_std'] = displacement_mag.std()
            
            # Local area change (Jacobian determinant at each grid cell)
            grid_n = gx_in.shape[0]
            local_det = np.ones((grid_n - 1, grid_n - 1))
            
            for i in range(grid_n - 1):
                for j in range(grid_n - 1):
                    # Input parallelogram
                    v1_in = np.array([gx_in[i, j+1] - gx_in[i, j],
                                      gy_in[i, j+1] - gy_in[i, j]])
                    v2_in = np.array([gx_in[i+1, j] - gx_in[i, j],
                                      gy_in[i+1, j] - gy_in[i, j]])
                    area_in = abs(np.cross(v1_in, v2_in))
                    
                    # Output parallelogram
                    v1_out = np.array([gx_out[i, j+1] - gx_out[i, j],
                                      gy_out[i, j+1] - gy_out[i, j]])
                    v2_out = np.array([gx_out[i+1, j] - gx_out[i, j],
                                       gy_out[i+1, j] - gy_out[i, j]])
                    area_out = abs(np.cross(v1_out, v2_out))
                    
                    if area_in > 1e-12:
                        local_det[i, j] = area_out / area_in
            
            results['local_det'] = local_det
            results['mean_area_change'] = local_det.mean()
            results['max_area_change'] = local_det.max()
            results['min_area_change'] = local_det.min()
            
            # Regions of expansion vs contraction
            results['n_expanding_cells'] = int(np.sum(local_det > 1.1))
            results['n_contracting_cells'] = int(np.sum(local_det < 0.9))
            
            self.log(f"  Grid: {grid_n}x{grid_n}, mean displacement: "
                     f"{results['mean_displacement']:.4f}")
            self.log(f"  Area change: mean={results['mean_area_change']:.3f}, "
                     f"range=[{results['min_area_change']:.3f}, "
                     f"{results['max_area_change']:.3f}]")
        
        elif has_grid_nd:
            gx = data['grid_x']
            gy = data['grid_y']
            results['grid_shape'] = gx.shape
            self.log(f"  Grid shape: {gx.shape}")
        
        return results
    
    def _analyze_eigenvalues(self, data: Dict) -> Dict:
        """Analyze eigenvalue spectra for rotation/spiral signatures."""
        results = {}
        
        if 'singular_values' in data:
            sv = data['singular_values']
            if isinstance(sv, np.ndarray) and sv.ndim > 0:
                results['singular_values'] = sv
                results['condition_number'] = sv[0] / max(sv[-1], 1e-10)
                results['effective_rank'] = np.sum(sv > sv[0] * 0.01)
                
                # Entropy of normalized singular values (measure of isotropy)
                sv_norm = sv / (sv.sum() + 1e-10)
                results['sv_entropy'] = scipy_entropy(sv_norm)
                
                self.log(f"  Singular values: {sv[:5]}...")
                self.log(f"  Effective rank: {results['effective_rank']}/{len(sv)}")
                self.log(f"  SV entropy: {results['sv_entropy']:.4f}")
        
        return results
    
    # ─── Visualization ──────────────────────────────────────────────
    
    def visualize_single(self, info: Dict, results: Dict):
        """Generate all visualizations for a single file."""
        step = results.get('step', '?')
        layer = results.get('layer', '?')
        prefix = f"step{step}_layer{layer}"
        
        data = info['data'] if 'data' in info else load_jacobi_npz(info['path'])
        
        # 1. Point cloud with displacement
        self._plot_point_cloud(data, results, prefix)
        
        # 2. Warped grid
        self._plot_warped_grid(data, results, prefix)
        
        # 3. Persistence diagram
        if 'topology' in results and 'diagrams' in results['topology']:
            self._plot_persistence(results['topology'], prefix)
        
        # 4. Betti curves
        if 'topology' in results and 'betti_scales' in results['topology']:
            self._plot_betti_curves(results['topology'], prefix)
        
        # 5. Eigenvalue spectrum
        if 'jacobian' in results and 'eigenvalues' in results['jacobian']:
            self._plot_eigenvalues(results['jacobian'], prefix)
        
        # 6. Per-token volume histogram
        if 'tokens' in results and 'volume_mean' in results['tokens']:
            self._plot_volume_histogram(data, prefix)
        
        # 7. Grid deformation area change heatmap
        if 'grid' in results and 'local_det' in results['grid']:
            self._plot_area_change_heatmap(data, results['grid'], prefix)
    
    def _plot_point_cloud(self, data: Dict, results: Dict, prefix: str):
        """Plot input/output point clouds with displacement vectors."""
        if 'h_in_2d' not in data:
            return
        
        h_in = data['h_in_2d']
        fig, axes = plt.subplots(1, 2 if 'h_out_2d' in data else 1,
                                 figsize=(14, 6), facecolor='#0a0a1a')
        
        # Input cloud
        ax = axes[0] if isinstance(axes, np.ndarray) else axes
        ax.set_facecolor('#0d1117')
        ax.scatter(h_in[:, 0], h_in[:, 1], s=20, c='#44aaff', alpha=0.8,
                   edgecolors='none', zorder=3)
        
        # Token labels
        if 'token_strings' in data:
            ts = data['token_strings']
            for i in range(min(len(ts), h_in.shape[0])):
                if i % max(1, h_in.shape[0] // 15) == 0:
                    label = str(ts[i])[:12]
                    ax.annotate(label, (h_in[i, 0], h_in[i, 1]),
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
            
            # Draw displacement vectors
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
        if 'local_det' in results.get('grid', {}):
            local_det = results['grid']['local_det']
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
        
        # Token positions
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
            
            # Diagonal
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
        
        # Complex plane
        ax = axes[0]
        ax.set_facecolor('#0d1117')
        
        # Unit circle
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
        
        # Magnitude/phase
        ax2 = axes[1]
        ax2.set_facecolor('#0d1117')
        
        mags = np.abs(eigenvalues)
        phases = np.angle(eigenvalues)
        
        scatter = ax2.scatter(phases, mags, s=40, c=phases, cmap='hsv',
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
    
    # ─── Cross-layer analysis ───────────────────────────────────────
    
    def analyze_cross_layer(self, all_results: List[Dict]):
        """
        Analyze topological changes ACROSS layers at a fixed step.
        This is where we look for the phenomena predicted by the conjectures.
        """
        if len(all_results) < 2:
            return
        
        self.log(f"\n[bold yellow]═══ Cross-Layer Analysis ═══[/]")
        
        step = all_results[0].get('step', '?')
        
        # ── Track topology across layers ────────────────────────────
        layers = []
        h1_counts = []
        h1_total_persistence = []
        divergences = []
        curls = []
        shears = []
        spectral_radii = []
        condition_numbers = []
        
        for r in all_results:
            layer = r.get('layer', '?')
            layers.append(layer)
            
            topo = r.get('topology', {})
            h1_counts.append(topo.get('h1_n_features', 0))
            h1_total_persistence.append(topo.get('h1_total_persistence', 0))
            
            jac = r.get('jacobian', {})
            divergences.append(jac.get('divergence', 0))
            curls.append(jac.get('curl', 0))
            shears.append(jac.get('shear', 0))
            spectral_radii.append(jac.get('spectral_radius', 1))
            condition_numbers.append(jac.get('condition_number', 1))
        
        # ── Plot cross-layer metrics ────────────────────────────────
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
        
        # ── Wasserstein distance between persistence diagrams ───────
        if HAS_RIPSER and HAS_PERSIM:
            self._compute_wasserstein_matrix(all_results, step)
        
        # ── Detect topological phase transitions ────────────────────
        self._detect_phase_transitions(layers, h1_counts, h1_total_persistence,
                                       divergences, curls, step)
        
        # ── Inner vs outer layer analysis (Conjecture 4) ───────────
        self._analyze_inner_outer(all_results)
    
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
                
                # Compare H1 diagrams (loops — the key feature)
                for dim in range(min(len(dgms_i), len(dgms_j), 2)):
                    d_i = dgms_i[dim] if dim < len(dgms_i) else np.empty((0, 2))
                    d_j = dgms_j[dim] if dim < len(dgms_j) else np.empty((0, 2))
                    
                    # Filter to finite features
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
        
        # Plot the Wasserstein matrix
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
        """
        Detect topological phase transitions: layers where the topology
        changes dramatically.
        
        These are predicted by Conjecture 4 (inner layers compute,
        outer layers translate) — we expect the biggest topological
        changes at the boundaries between the "translation" and
        "computation" regimes.
        """
        self.log(f"\n[bold yellow]── Phase Transition Detection ──[/]")
        
        if len(h1_counts) < 3:
            self.log("  Too few layers for phase transition detection")
            return
        
        # Compute first differences of topological features
        h1_diffs = [abs(h1_counts[i+1] - h1_counts[i]) for i in range(len(h1_counts)-1)]
        pers_diffs = [abs(h1_total_persistence[i+1] - h1_total_persistence[i])
                      for i in range(len(h1_total_persistence)-1)]
        
        # Find layers with the largest topological changes
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
        
        # Identify the "computation zone" (layers with highest topological activity)
        if h1_total_persistence:
            mean_pers = np.mean(h1_total_persistence)
            active_layers = [l for l, p in zip(layers, h1_total_persistence)
                             if p > mean_pers * 1.5]
            if active_layers:
                self.log(f"  [green]Topologically active layers (persistence > 1.5× mean): "
                         f"{active_layers}[/]")
                self.log(f"  This is consistent with Conjecture 4: inner layers "
                         f"perform the bulk of geometric computation")
    
    def _analyze_inner_outer(self, all_results: List[Dict]):
        """
        Test Conjecture 4: inner layers compute, outer layers translate.
        
        Compare geometric complexity (divergence, curl, shear, topology)
        between inner and outer layers.
        """
        n = len(all_results)
        if n < 4:
            return
        
        self.log(f"\n[bold yellow]── Inner vs Outer Layer Analysis (Conjecture 4) ──[/]")
        
        # Split into thirds: early, middle, late
        third = max(1, n // 3)
        early = all_results[:third]
        middle = all_results[third:2*third]
        late = all_results[2*third:]
        
        def _avg_metric(results_list, metric_path):
            """Extract average of a nested metric."""
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
        
        # Check if middle layers are more active
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
        """
        Track how topology evolves across training steps.
        
        This is the key analysis for observing grokking:
        the emergence of topological structures during training.
        """
        self.log(f"\n[bold yellow]═══ Topology Over Training ═══[/]")
        
        # Group files by step
        by_step = defaultdict(list)
        for info in all_files:
            step = info.get('step', 0)
            by_step[step].append(info)
        
        steps = sorted(by_step.keys())
        if len(steps) < 2:
            self.log("  Need at least 2 training steps for tracking")
            return
        
        self.log(f"  Found {len(steps)} training steps: {steps[:10]}{'...' if len(steps) > 10 else ''}")
        
        # For each step, compute aggregate topology across all layers
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
        
        # Detect grokking-like transitions
        if len(step_h1_total) > 5:
            # Look for sudden jumps in H1 count
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
    
    # ─── UMAP embedding of Jacobian fields ──────────────────────────
    
    def visualize_jacobian_umap(self, all_results: List[Dict]):
        """
        Embed the per-layer Jacobian fields into 2D using UMAP,
        colored by layer index. This shows whether different layers
        occupy distinct regions of "deformation space."
        """
        if not HAS_UMAP:
            self.log("  [yellow]UMAP not available, skipping embedding[/]")
            return
        
        self.log(f"\n[bold yellow]── UMAP Embedding of Jacobian Fields ──[/]")
        
        # Collect feature vectors from each layer's Jacobian
        features = []
        layer_ids = []
        
        for r in all_results:
            jac = r.get('jacobian', {})
            topo = r.get('topology', {})
            tokens = r.get('tokens', {})
            
            # Build a feature vector from available metrics
            fv = []
            for key in ['divergence', 'curl', 'shear', 'spectral_radius',
                        'condition_number']:
                fv.append(jac.get(key, 0))
            
            for key in ['h1_n_features', 'h1_total_persistence',
                        'h0_n_features']:
                fv.append(topo.get(key, 0))
            
            for key in ['volume_mean', 'volume_std', 'displacement_mean',
                        'direction_coherence']:
                fv.append(tokens.get(key, 0))
            
            if fv:
                features.append(fv)
                layer_ids.append(r.get('layer', 0))
        
        if len(features) < 5:
            self.log("  Too few data points for UMAP")
            return
        
        X = np.array(features)
        X = StandardScaler().fit_transform(X)
        
        try:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X)-1))
            embedding = reducer.fit_transform(X)
        except Exception as e:
            self.log(f"  [red]UMAP failed: {e}[/]")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 7), facecolor='#0a0a1a')
        ax.set_facecolor('#0d1117')
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                             c=layer_ids, cmap='viridis', s=40,
                             edgecolors='#ffffff', linewidths=0.5, zorder=5)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Layer Index', color='#888888', fontsize=9)
        cbar.ax.tick_params(colors='#888888', labelsize=7)
        
        ax.set_title('UMAP Embedding of Jacobian Field Features',
                     color='#c0d8e8', fontsize=11)
        ax.set_xlabel('UMAP 1', color='#888888')
        ax.set_ylabel('UMAP 2', color='#888888')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')
        
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
        epsilon = np.percentile(dists[dists > 0], 15)  # connect nearest 15%
        
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        
        for i in range(n):
            for j in range(i+1, n):
                if dists[i, j] < epsilon:
                    G.add_edge(i, j, weight=dists[i, j])
        
        # Graph metrics
        n_components = nx.number_connected_components(G)
        n_edges = G.number_of_edges()
        
        # Cycle detection (proxy for H1 loops)
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
            for cycle in cycle_basis[:5]:  # show up to 5 cycles
                cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)])
                               for i in range(len(cycle))]
                nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, ax=ax,
                                       edge_color='#ff44aa', width=2.0, alpha=0.8)
        
        # Token labels
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
        "[bold white]Jacobi Field Topology Analyzer[/]\n"
        "[dim]Searching for topological structures in transformer layer deformations[/]\n"
        "[dim]Based on the fibre bundle interpretability framework[/]",
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
        # Group by step
        by_step = defaultdict(list)
        for r in all_results:
            by_step[r['step']].append(r)
        
        for step, step_results in sorted(by_step.items()):
            if len(step_results) > 1:
                # Sort by layer
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
    
    # ── Save report ─────────────────────────────────────────────────────
    analyzer.save_report()
    
    console.print(Panel(
        f"[bold green]Analysis complete![/]\n"
        f"Output directory: {args.output_dir}/\n"
        f"Files analyzed: {len(files)}\n"
        f"H1 loops: {total_h1} ({total_h1_sig} significant)\n"
        f"H2 voids: {total_h2}",
        title="[bold]Results",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
