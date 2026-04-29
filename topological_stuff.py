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
#   "kmapper",
#   "networkx",
#   "rich",
#   "PyWavelets",
# ]
# ///

"""
Topological Structure Extractor for Jacobi Fields
===================================================

Unified pipeline that extracts, tracks, and characterizes ALL topological
structures hidden in the Jacobi field data from transformer training.

Combines and extends:
  - Persistent homology (H0, H1, H2) with representative cocycles
  - Persistence images & landscapes as fixed-dimensional feature vectors
  - Mapper algorithm for topological skeletonization
  - Graph Laplacian spectrum & Fiedler partitioning
  - Helmholtz decomposition (irrotational vs solenoidal)
  - Symbolic dynamics of the Jacobian
  - NMF / ICA decomposition
  - Recurrence quantification analysis
  - Wasserstein & bottleneck distance tracking
  - Cross-correlation of topology with training loss
  - Phase transition / grokking detection

Usage:
    # Full extraction on a single run
    python3 extract_topology.py runs/0/jacobi_data/ --loss-csv runs/0/epoch_log.csv

    # Quick summary (skip slow probes)
    python3 extract_topology.py runs/0/jacobi_data/ --quick

    # Compare multiple runs
    python3 extract_topology.py runs/*/jacobi_data/ --compare
"""

import os
import sys
import glob
import re
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings

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
from matplotlib.colors import Normalize, LogNorm
from scipy.fft import fft, fft2, fftfreq, fftshift
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, eigvals, norm
from scipy.sparse.csgraph import laplacian
from scipy.stats import entropy as scipy_entropy, pearsonr, spearmanr
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pywt

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

try:
    from persim import wasserstein as wasserstein_distance
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

try:
    import gudhi
    from gudhi.representations import Landscape, PersistenceImage
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

try:
    import kmapper as km
    HAS_KMAPPER = True
except ImportError:
    HAS_KMAPPER = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)
from rich import box

console = Console()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def parse_filename(filepath: str) -> Tuple[Optional[int], Optional[int]]:
    basename = os.path.basename(filepath)
    step_match = re.search(r'step(\d+)', basename)
    layer_match = re.search(r'layer(\d+)', basename)
    step = int(step_match.group(1)) if step_match else None
    layer = int(layer_match.group(1)) if layer_match else None
    return step, layer


def load_npz(filepath: str) -> Dict:
    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.files:
        arr = data[key]
        result[key] = arr.item() if arr.ndim == 0 else arr
    data.close()
    return result


def load_all_files(paths: List[str]) -> Dict[Tuple[int, int], str]:
    """Load all jacobi npz files, organized by (step, layer) -> filepath."""
    data_map = {}
    for path in paths:
        if os.path.isdir(path):
            pattern = os.path.join(path, "jacobi_step*_layer*.npz")
            files = sorted(glob.glob(pattern))
        elif os.path.isfile(path):
            files = [path]
        else:
            files = sorted(glob.glob(path))

        for f in files:
            step, layer = parse_filename(f)
            if step is not None and layer is not None:
                data_map[(step, layer)] = f

    return data_map


def load_loss_csv(csv_path: str) -> Optional[Dict]:
    """Load training loss from epoch_log.csv or batch_log.csv."""
    if csv_path is None or not os.path.isfile(csv_path):
        return None

    import csv
    losses = {'steps': [], 'train_loss': [], 'val_loss': []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try different column name conventions
            step = None
            for key in ['epoch', 'step', 'global_step', 'batch']:
                if key in row:
                    try:
                        step = int(row[key])
                    except (ValueError, TypeError):
                        continue
                    break

            train_loss = None
            for key in ['train_loss', 'loss', 'total_loss', 'ce_loss']:
                if key in row:
                    try:
                        train_loss = float(row[key])
                    except (ValueError, TypeError):
                        continue
                    break

            val_loss = None
            for key in ['val_loss', 'validation_loss']:
                if key in row:
                    try:
                        val_loss = float(row[key])
                    except (ValueError, TypeError):
                        continue
                    break

            if step is not None and train_loss is not None:
                losses['steps'].append(step)
                losses['train_loss'].append(train_loss)
                losses['val_loss'].append(val_loss if val_loss is not None else np.nan)

    if not losses['steps']:
        return None

    for key in losses:
        losses[key] = np.array(losses[key])

    console.print(f"[green]✓ Loaded {len(losses['steps'])} loss entries from {csv_path}[/]")
    return losses


# ═══════════════════════════════════════════════════════════════════════════
# CORE TOPOLOGY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

class TopologyExtractor:
    """
    Extracts topological structures from a single Jacobi field .npz file.
    Returns a structured dict of all extracted features.
    """

    def __init__(self, max_pts: int = 300, max_dim: int = 2):
        self.max_pts = max_pts
        self.max_dim = max_dim

    def extract(self, data: Dict) -> Dict:
        """Run all extraction methods on a single file's data."""
        result = {}

        # ── 1. Persistent Homology (with cocycles) ─────────────────
        result['persistence'] = self._extract_persistence(data)

        # ── 2. Persistence Images (fixed-dim feature vectors) ──────
        result['persistence_image'] = self._extract_persistence_image(data)

        # ── 3. Persistence Landscapes ──────────────────────────────
        result['persistence_landscape'] = self._extract_persistence_landscape(data)

        # ── 4. Graph Laplacian Spectrum ────────────────────────────
        result['graph_spectrum'] = self._extract_graph_spectrum(data)

        # ── 5. Helmholtz Decomposition ─────────────────────────────
        result['helmholtz'] = self._extract_helmholtz(data)

        # ── 6. Jacobian Spectral Analysis ──────────────────────────
        result['jacobian_spectrum'] = self._extract_jacobian_spectrum(data)

        # ── 7. Per-token signal analysis ───────────────────────────
        result['token_signals'] = self._extract_token_signals(data)

        # ── 8. Mapper (topological skeleton) ───────────────────────
        result['mapper'] = self._extract_mapper(data)

        return result

    # ─── 1. Persistent Homology ─────────────────────────────────────

    def _extract_persistence(self, data: Dict) -> Dict:
        if not HAS_RIPSER:
            return {}

        points = data.get('h_out_2d', data.get('h_in_2d', None))
        if points is None or points.shape[0] < 4:
            return {}

        if points.shape[0] > self.max_pts:
            idx = np.random.choice(points.shape[0], self.max_pts, replace=False)
            points = points[idx]

        dists = pdist(points)
        if len(dists) == 0 or dists.max() == 0:
            return {}

        thresh = np.percentile(dists[dists > 0], 95)

        try:
            # do_cocycles=True gives us representative cocycles
            result = ripser(points, maxdim=self.max_dim, thresh=thresh,
                            do_cocycles=True)
            dgms = result['dgms']
            cocycles = result.get('cocycles', [])
        except Exception:
            return {}

        out = {
            'diagrams': dgms,
            'cocycles': cocycles,
            'threshold': thresh,
            'n_points': points.shape[0],
            'points_used': points,
        }

        # Per-dimension statistics
        for dim_idx, dim_name in enumerate(['h0', 'h1', 'h2']):
            if dim_idx >= len(dgms):
                break
            dgm = dgms[dim_idx]
            finite = dgm[np.isfinite(dgm[:, 1])]
            out[f'{dim_name}_count'] = len(finite)

            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                out[f'{dim_name}_total_persistence'] = lifetimes.sum()
                out[f'{dim_name}_max_lifetime'] = lifetimes.max()
                out[f'{dim_name}_mean_lifetime'] = lifetimes.mean()
                out[f'{dim_name}_lifetimes'] = lifetimes

                if len(lifetimes) > 1:
                    median_lt = np.median(lifetimes)
                    significant = lifetimes > 2 * median_lt
                    out[f'{dim_name}_significant'] = int(significant.sum())
                else:
                    out[f'{dim_name}_significant'] = len(lifetimes)
            else:
                out[f'{dim_name}_total_persistence'] = 0
                out[f'{dim_name}_max_lifetime'] = 0
                out[f'{dim_name}_mean_lifetime'] = 0
                out[f'{dim_name}_significant'] = 0

        # Persistence entropy
        all_lifetimes = []
        for dim_idx in range(min(len(dgms), 3)):
            finite = dgms[dim_idx][np.isfinite(dgms[dim_idx][:, 1])]
            if len(finite) > 0:
                all_lifetimes.extend((finite[:, 1] - finite[:, 0]).tolist())
        if all_lifetimes:
            lt_arr = np.array(all_lifetimes)
            lt_norm = lt_arr / (lt_arr.sum() + 1e-10)
            out['persistence_entropy'] = scipy_entropy(lt_norm)
        else:
            out['persistence_entropy'] = 0.0

        # Betti curves
        n_scales = 50
        scales = np.linspace(0, thresh, n_scales)
        betti = {}
        for dim_idx in range(min(len(dgms), 3)):
            b = np.zeros(n_scales)
            dgm = dgms[dim_idx]
            for si, s in enumerate(scales):
                b[si] = np.sum((dgm[:, 0] <= s) & (dgm[:, 1] > s))
            betti[f'betti_{dim_idx}'] = b
        out['betti_scales'] = scales
        out.update(betti)

        # Representative cocycle analysis (which points form loops?)
        if len(cocycles) > 1 and len(cocycles[1]) > 0:
            out['h1_cocycle_points'] = self._analyze_cocycles(
                cocycles[1], dgms[1], points
            )

        return out

    def _analyze_cocycles(self, cocycles_h1, dgm_h1, points) -> List[Dict]:
        """
        Analyze H1 representative cocycles to identify which points
        participate in each topological loop.
        """
        finite = dgm_h1[np.isfinite(dgm_h1[:, 1])]
        if len(finite) == 0:
            return []

        lifetimes = finite[:, 1] - finite[:, 0]
        # Sort by lifetime (most significant first)
        sort_idx = np.argsort(lifetimes)[::-1]

        loop_info = []
        for rank, feat_idx in enumerate(sort_idx[:10]):  # top 10 loops
            if feat_idx >= len(cocycles_h1):
                continue

            cocycle = cocycles_h1[feat_idx]
            if len(cocycle) == 0:
                continue

            # Extract the point indices involved in this cocycle
            # cocycle is an array of (simplex_vertex_1, simplex_vertex_2, coefficient)
            involved_points = set()
            for simplex in cocycle:
                if len(simplex) >= 2:
                    involved_points.add(int(simplex[0]))
                    involved_points.add(int(simplex[1]))

            involved_list = sorted(involved_points)

            # Compute geometric properties of this loop
            if involved_list and max(involved_list) < len(points):
                loop_pts = points[involved_list]
                centroid = loop_pts.mean(axis=0)
                radius = np.linalg.norm(loop_pts - centroid, axis=1).mean()

                loop_info.append({
                    'rank': rank,
                    'lifetime': lifetimes[feat_idx],
                    'birth': finite[feat_idx, 0],
                    'death': finite[feat_idx, 1],
                    'n_points': len(involved_list),
                    'point_indices': involved_list,
                    'centroid': centroid,
                    'mean_radius': radius,
                })

        return loop_info

    # ─── 2. Persistence Images ──────────────────────────────────────

    def _extract_persistence_image(self, data: Dict) -> Dict:
        if not HAS_GUDHI:
            return {}

        points = data.get('h_out_2d', data.get('h_in_2d', None))
        if points is None or points.shape[0] < 4:
            return {}

        if points.shape[0] > self.max_pts:
            idx = np.random.choice(points.shape[0], self.max_pts, replace=False)
            points = points[idx]

        try:
            dists = pdist(points)
            thresh = np.percentile(dists[dists > 0], 95) if dists.max() > 0 else 1.0
            result = ripser(points, maxdim=1, thresh=thresh)
            dgms = result['dgms']
        except Exception:
            return {}

        out = {}
        for dim_idx in range(min(len(dgms), 2)):
            finite = dgms[dim_idx][np.isfinite(dgms[dim_idx][:, 1])]
            if len(finite) < 2:
                continue

            try:
                pimgr = PersistenceImage(
                    bandwidth=0.1,
                    resolution=[20, 20],
                    weight=lambda x: x[1],  # weight by persistence
                )
                pi = pimgr.fit_transform([finite])
                out[f'h{dim_idx}_image'] = pi[0]  # (400,) vector
                out[f'h{dim_idx}_image_shape'] = (20, 20)
            except Exception:
                pass

        return out

    # ─── 3. Persistence Landscapes ──────────────────────────────────

    def _extract_persistence_landscape(self, data: Dict) -> Dict:
        if not HAS_GUDHI:
            return {}

        points = data.get('h_out_2d', data.get('h_in_2d', None))
        if points is None or points.shape[0] < 4:
            return {}

        if points.shape[0] > self.max_pts:
            idx = np.random.choice(points.shape[0], self.max_pts, replace=False)
            points = points[idx]

        try:
            dists = pdist(points)
            thresh = np.percentile(dists[dists > 0], 95) if dists.max() > 0 else 1.0
            result = ripser(points, maxdim=1, thresh=thresh)
            dgms = result['dgms']
        except Exception:
            return {}

        out = {}
        for dim_idx in range(min(len(dgms), 2)):
            finite = dgms[dim_idx][np.isfinite(dgms[dim_idx][:, 1])]
            if len(finite) < 2:
                continue

            try:
                ls = Landscape(
                    num_landscapes=5,
                    resolution=100,
                )
                landscape = ls.fit_transform([finite])
                out[f'h{dim_idx}_landscape'] = landscape[0]  # (500,) vector
            except Exception:
                pass

        return out

    # ─── 4. Graph Laplacian Spectrum ────────────────────────────────

    def _extract_graph_spectrum(self, data: Dict) -> Dict:
        points = data.get('h_out_2d', data.get('h_in_2d', None))
        if points is None or points.shape[0] < 5:
            return {}

        if points.shape[0] > self.max_pts:
            idx = np.random.choice(points.shape[0], self.max_pts, replace=False)
            points = points[idx]

        dists = squareform(pdist(points))
        k = min(10, points.shape[0] - 1)

        # k-NN adjacency
        A = np.zeros_like(dists)
        for i in range(len(dists)):
            neighbors = np.argsort(dists[i])[1:k + 1]
            A[i, neighbors] = 1
            A[neighbors, i] = 1

        L = laplacian(A, normed=True)
        L_dense = L.toarray() if hasattr(L, 'toarray') else L

        try:
            lap_eigs = np.sort(np.real(np.linalg.eigvalsh(L_dense)))
        except Exception:
            return {}

        n_components = int(np.sum(lap_eigs < 1e-6))
        spectral_gap = lap_eigs[n_components] if len(lap_eigs) > n_components else 0.0

        # Fiedler vector (natural partition)
        fiedler_vector = None
        try:
            _, eigvecs = np.linalg.eigh(L_dense)
            fiedler_vector = eigvecs[:, n_components]
        except Exception:
            pass

        return {
            'eigenvalues': lap_eigs,
            'n_components': n_components,
            'spectral_gap': spectral_gap,
            'fiedler_vector': fiedler_vector,
            'algebraic_connectivity': spectral_gap,
        }

    # ─── 5. Helmholtz Decomposition ────────────────────────────────

    def _extract_helmholtz(self, data: Dict) -> Dict:
        has_grid = all(k in data for k in
                       ['grid_x_in', 'grid_y_in', 'grid_x_out', 'grid_y_out'])
        if not has_grid:
            return {}

        gx_in, gy_in = data['grid_x_in'], data['grid_y_in']
        gx_out, gy_out = data['grid_x_out'], data['grid_y_out']

        dx = gx_out - gx_in
        dy = gy_out - gy_in

        dx_fft = fft2(dx)
        dy_fft = fft2(dy)

        N = dx.shape[0]
        kx = fftfreq(N)[:, np.newaxis]
        ky = fftfreq(N)[np.newaxis, :]
        k_sq = kx ** 2 + ky ** 2
        k_sq[0, 0] = 1  # avoid div by zero

        # Divergence (irrotational part)
        div_fft = 1j * kx * dx_fft + 1j * ky * dy_fft
        # Curl (solenoidal part)
        curl_fft = 1j * kx * dy_fft - 1j * ky * dx_fft

        irrot_energy = np.sum(np.abs(div_fft) ** 2)
        sol_energy = np.sum(np.abs(curl_fft) ** 2)
        total_energy = irrot_energy + sol_energy + 1e-10

        # Radial power spectrum
        k_mag = np.sqrt(kx ** 2 + ky ** 2)
        n_bins = N // 2
        radial_bins = np.linspace(0, k_mag.max(), n_bins)
        radial_power = np.zeros(n_bins - 1)
        total_power = np.abs(dx_fft) ** 2 + np.abs(dy_fft) ** 2

        for i in range(n_bins - 1):
            mask = (k_mag >= radial_bins[i]) & (k_mag < radial_bins[i + 1])
            if mask.any():
                radial_power[i] = total_power[mask].mean()

        # Power law fit
        power_law_exponent = None
        valid = radial_power > 0
        if valid.sum() > 5:
            log_k = np.log(radial_bins[:-1][valid] + 1e-10)
            log_p = np.log(radial_power[valid])
            try:
                slope, _ = np.polyfit(log_k[1:], log_p[1:], 1)
                power_law_exponent = slope
            except Exception:
                pass

        # Vorticity field
        curl_real = np.real(np.fft.ifft2(curl_fft))

        return {
            'irrotational_fraction': irrot_energy / total_energy,
            'solenoidal_fraction': sol_energy / total_energy,
            'radial_power': radial_power,
            'radial_bins': radial_bins[:-1],
            'power_law_exponent': power_law_exponent,
            'vorticity_field': curl_real,
            'vorticity_max': np.abs(curl_real).max(),
            'vorticity_mean': np.abs(curl_real).mean(),
        }

    # ─── 6. Jacobian Spectral Analysis ─────────────────────────────

    def _extract_jacobian_spectrum(self, data: Dict) -> Dict:
        if 'jacobian' not in data:
            return {}

        J = data['jacobian']
        D = J.shape[0]
        out = {}

        # SVD
        try:
            U, S, Vh = svd(J)
            out['singular_values'] = S
            out['condition_number'] = S[0] / max(S[-1], 1e-10)

            # Participation ratio
            sv_norm = S / (S.sum() + 1e-10)
            out['participation_ratio'] = 1.0 / (np.sum(sv_norm ** 2) + 1e-10)

            # SV entropy
            sv_entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
            out['sv_entropy'] = sv_entropy
            out['sv_entropy_ratio'] = sv_entropy / (np.log(len(S)) + 1e-10)

            # Effective dimensions
            cumulative = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-10)
            for thresh in [0.5, 0.9, 0.95, 0.99]:
                out[f'eff_dim_{int(thresh * 100)}'] = int(
                    np.searchsorted(cumulative, thresh) + 1
                )

            # Spectral gaps
            sv_diffs = np.abs(np.diff(S))
            gap_threshold = np.mean(sv_diffs) + 2 * np.std(sv_diffs)
            big_gaps = np.where(sv_diffs > gap_threshold)[0]
            out['n_spectral_modes'] = len(big_gaps) + 1
            out['spectral_gap_indices'] = big_gaps.tolist()
        except Exception:
            pass

        # Eigenvalues
        try:
            eigs = eigvals(J)
            out['eigenvalues'] = eigs
            out['spectral_radius'] = np.max(np.abs(eigs))
            out['n_complex_eigenvalues'] = int(np.sum(np.abs(eigs.imag) > 1e-8))
        except Exception:
            pass

        # Decomposition: divergence, curl, shear
        J_sym = (J + J.T) / 2
        J_antisym = (J - J.T) / 2
        out['divergence'] = float(np.trace(J))
        out['curl'] = float(norm(J_antisym, 'fro'))
        trace_sym = np.trace(J_sym)
        shear_tensor = J_sym - (trace_sym / D) * np.eye(D)
        out['shear'] = float(norm(shear_tensor, 'fro'))

        return out

    # ─── 7. Per-token Signal Analysis ──────────────────────────────

    def _extract_token_signals(self, data: Dict) -> Dict:
        out = {}

        for key in ['per_token_volume', 'per_token_rotation', 'per_token_shear_mag']:
            if key not in data:
                continue
            signal = data[key]
            if len(signal) < 5:
                continue

            sig_out = {
                'mean': float(signal.mean()),
                'std': float(signal.std()),
                'min': float(signal.min()),
                'max': float(signal.max()),
            }

            # FFT dominant frequencies
            N = len(signal)
            freqs = fftfreq(N)[:N // 2]
            fft_vals = fft(signal)
            power = np.abs(fft_vals[:N // 2]) ** 2

            if len(power) > 3:
                peak_freqs = np.argsort(power[1:])[::-1][:5] + 1
                valid_peaks = peak_freqs[peak_freqs < len(freqs)]
                if len(valid_peaks) > 0:
                    dominant_periods = 1.0 / (freqs[valid_peaks] + 1e-10)
                    sig_out['dominant_periods'] = dominant_periods[:3].tolist()

            # Autocorrelation
            sig_centered = signal - signal.mean()
            autocorr = np.correlate(sig_centered, sig_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr /= autocorr[0] + 1e-10

            threshold = 2 / np.sqrt(len(signal))
            significant_lags = np.where(
                np.abs(autocorr[1:len(signal) // 2]) > threshold
            )[0] + 1
            sig_out['significant_autocorr_lags'] = significant_lags[:10].tolist()
            sig_out['autocorrelation'] = autocorr[:min(50, len(autocorr))]

            # Wavelet scalogram summary
            if len(signal) >= 16:
                try:
                    scales = np.arange(1, min(len(signal) // 2, 32))
                    coeffs, freqs_cwt = pywt.cwt(signal, scales, 'morl')
                    # Summarize: energy per scale
                    scale_energy = np.sum(np.abs(coeffs) ** 2, axis=1)
                    sig_out['wavelet_scale_energy'] = scale_energy
                    sig_out['wavelet_dominant_scale'] = int(scales[np.argmax(scale_energy)])
                except Exception:
                    pass

            # Recurrence quantification
            if len(signal) >= 10:
                diff_mat = np.abs(signal[:, None] - signal[None, :])
                rec_thresh = np.percentile(diff_mat[diff_mat > 0], 10) if (diff_mat > 0).any() else 1.0
                recurrence = (diff_mat < rec_thresh).astype(float)
                sig_out['recurrence_rate'] = recurrence.mean()

                # Diagonal line lengths (determinism)
                diag_lengths = []
                N_sig = len(signal)
                for offset in range(1, N_sig):
                    diag = np.diag(recurrence, k=offset)
                    current_len = 0
                    for val in diag:
                        if val > 0.5:
                            current_len += 1
                        else:
                            if current_len > 1:
                                diag_lengths.append(current_len)
                            current_len = 0
                    if current_len > 1:
                        diag_lengths.append(current_len)

                if diag_lengths:
                    sig_out['determinism'] = sum(d for d in diag_lengths if d >= 2) / (
                        sum(diag_lengths) + 1e-10
                    )
                    sig_out['max_diagonal_line'] = max(diag_lengths)
                else:
                    sig_out['determinism'] = 0.0
                    sig_out['max_diagonal_line'] = 0

            out[key] = sig_out

        return out

    # ─── 8. Mapper (topological skeleton) ──────────────────────────

    def _extract_mapper(self, data: Dict) -> Dict:
        if not HAS_KMAPPER:
            return {}

        points = data.get('h_out_2d', data.get('h_in_2d', None))
        if points is None or points.shape[0] < 10:
            return {}

        if points.shape[0] > self.max_pts:
            idx = np.random.choice(points.shape[0], self.max_pts, replace=False)
            points = points[idx]

        try:
            mapper = km.KeplerMapper(verbose=0)

            # Use PCA lens (or eccentricity)
            lens = mapper.fit_transform(points, projection=[0])

            graph = mapper.map(
                lens, points,
                cover=km.Cover(n_cubes=10, perc_overlap=0.3),
                clusterer=DBSCAN(eps=0.5, min_samples=3),
            )

            # Extract graph statistics
            n_nodes = len(graph['nodes'])
            n_edges = len(graph['links'])

            # Compute connected components
            if HAS_NETWORKX and n_nodes > 0:
                G = nx.Graph()
                for node_id in graph['nodes']:
                    G.add_node(node_id)
                for source, targets in graph['links'].items():
                    for target in targets:
                        G.add_edge(source, target)

                n_components = nx.number_connected_components(G)

                # Detect cycles in the mapper graph
                try:
                    cycles = nx.cycle_basis(G)
                    n_cycles = len(cycles)
                except Exception:
                    n_cycles = 0
            else:
                n_components = 0
                n_cycles = 0

            return {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'n_components': n_components,
                'n_cycles': n_cycles,
                'graph': graph,
            }

        except Exception:
            return {}


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE: Orchestrates extraction across all files
# ═══════════════════════════════════════════════════════════════════════════

class TopologyPipeline:
    """
    Full pipeline that:
      1. Loads all Jacobi field files
      2. Extracts topology from each
      3. Tracks changes over training
      4. Correlates with training loss
      5. Detects phase transitions
      6. Generates comprehensive visualizations and reports
    """

    def __init__(self, output_dir: str = "topology_extraction",
                 max_pts: int = 300, quick: bool = False):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.extractor = TopologyExtractor(max_pts=max_pts)
        self.quick = quick

    def run(self, data_paths: List[str], loss_csv: Optional[str] = None):
        """Main entry point."""
        console.print(Panel(
            "[bold white]Topological Structure Extractor[/]\n"
            "[dim]Extracting all topological structures from Jacobi fields[/]",
            border_style="bold cyan",
            padding=(1, 4),
        ))

        # ── 1. Load all files ───────────────────────────────────────
        data_map = load_all_files(data_paths)
        if not data_map:
            console.print("[bold red]No Jacobi files found.[/]")
            return

        steps = sorted(set(s for s, l in data_map.keys()))
        layers = sorted(set(l for s, l in data_map.keys()))
        console.print(f"[green]Found {len(data_map)} files: "
                      f"{len(steps)} steps × {len(layers)} layers[/]")

        # ── 2. Load training loss ───────────────────────────────────
        loss_data = load_loss_csv(loss_csv)

        # ── 3. Extract topology from all files ──────────────────────
        all_extractions = {}  # (step, layer) -> extraction dict

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Extracting topology..."),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("extract", total=len(data_map))

            for (step, layer), filepath in data_map.items():
                data = load_npz(filepath)
                extraction = self.extractor.extract(data)
                all_extractions[(step, layer)] = extraction
                progress.update(task, advance=1)

        console.print(f"[green]✓ Extracted topology from {len(all_extractions)} files[/]")

        # ── 4. Build feature matrices ───────────────────────────────
        feature_matrices = self._build_feature_matrices(
            all_extractions, steps, layers
        )

        # ── 5. Generate visualizations ──────────────────────────────
        self._plot_topology_heatmaps(feature_matrices, steps, layers)
        self._plot_evolution_curves(feature_matrices, steps, layers)
        self._plot_persistence_image_gallery(all_extractions, steps, layers)

        if not self.quick:
            self._plot_mapper_summary(all_extractions, steps, layers)
            self._plot_helmholtz_summary(all_extractions, steps, layers)
            self._plot_cocycle_analysis(all_extractions, steps, layers)
            self._plot_graph_spectrum_summary(all_extractions, steps, layers)

        # ── 6. Cross-correlation with loss ──────────────────────────
        if loss_data is not None:
            self._correlate_with_loss(feature_matrices, steps, layers, loss_data)

        # ── 7. Phase transition detection ───────────────────────────
        self._detect_phase_transitions(feature_matrices, steps, layers)

        # ── 8. Save numerical results ───────────────────────────────
        self._save_results(all_extractions, feature_matrices, steps, layers)

        # ── 9. Print summary ────────────────────────────────────────
        self._print_summary(feature_matrices, steps, layers, all_extractions)

        n_plots = len(glob.glob(os.path.join(self.output_dir, '*.png')))
        console.print(Panel(
            f"[bold green]Topology extraction complete![/]\n"
            f"Output: {self.output_dir}/\n"
            f"Plots: {n_plots}\n"
            f"Files processed: {len(all_extractions)}",
            title="[bold]Results",
            border_style="green",
        ))

    # ─── Feature matrix construction ────────────────────────────────

    def _build_feature_matrices(self, all_extractions, steps, layers):
        """Build step×layer matrices for all extracted features."""
        n_steps = len(steps)
        n_layers = len(layers)
        step_idx = {s: i for i, s in enumerate(steps)}
        layer_idx = {l: i for i, l in enumerate(layers)}

        matrices = {
            'h0_count': np.zeros((n_steps, n_layers)),
            'h1_count': np.zeros((n_steps, n_layers)),
            'h1_significant': np.zeros((n_steps, n_layers)),
            'h1_total_persistence': np.zeros((n_steps, n_layers)),
            'h1_max_lifetime': np.zeros((n_steps, n_layers)),
            'h2_count': np.zeros((n_steps, n_layers)),
            'persistence_entropy': np.zeros((n_steps, n_layers)),
            'spectral_gap': np.zeros((n_steps, n_layers)),
            'irrotational_fraction': np.zeros((n_steps, n_layers)),
            'solenoidal_fraction': np.zeros((n_steps, n_layers)),
            'vorticity_max': np.zeros((n_steps, n_layers)),
            'condition_number': np.zeros((n_steps, n_layers)),
            'participation_ratio': np.zeros((n_steps, n_layers)),
            'sv_entropy': np.zeros((n_steps, n_layers)),
            'divergence': np.zeros((n_steps, n_layers)),
            'curl': np.zeros((n_steps, n_layers)),
            'shear': np.zeros((n_steps, n_layers)),
            'spectral_radius': np.zeros((n_steps, n_layers)),
            'mapper_nodes': np.zeros((n_steps, n_layers)),
            'mapper_cycles': np.zeros((n_steps, n_layers)),
            'n_cocycle_loops': np.zeros((n_steps, n_layers)),
        }

        for (step, layer), ext in all_extractions.items():
            si = step_idx[step]
            li = layer_idx[layer]

            pers = ext.get('persistence', {})
            for key in ['h0_count', 'h1_count', 'h1_significant',
                        'h1_total_persistence', 'h1_max_lifetime',
                        'h2_count', 'persistence_entropy']:
                if key in pers:
                    matrices[key][si, li] = pers[key]

            if 'h1_cocycle_points' in pers:
                matrices['n_cocycle_loops'][si, li] = len(pers['h1_cocycle_points'])

            gs = ext.get('graph_spectrum', {})
            if 'spectral_gap' in gs:
                matrices['spectral_gap'][si, li] = gs['spectral_gap']

            helm = ext.get('helmholtz', {})
            for key in ['irrotational_fraction', 'solenoidal_fraction', 'vorticity_max']:
                if key in helm:
                    matrices[key][si, li] = helm[key]

            jspec = ext.get('jacobian_spectrum', {})
            for key in ['condition_number', 'participation_ratio',
                        'sv_entropy', 'divergence', 'curl', 'shear',
                        'spectral_radius']:
                if key in jspec:
                    matrices[key][si, li] = jspec[key]

            mp = ext.get('mapper', {})
            if 'n_nodes' in mp:
                matrices['mapper_nodes'][si, li] = mp['n_nodes']
            if 'n_cycles' in mp:
                matrices['mapper_cycles'][si, li] = mp['n_cycles']

        return matrices

    # ─── Visualization methods ──────────────────────────────────────

    def _plot_topology_heatmaps(self, matrices, steps, layers):
        """Plot step×layer heatmaps for key topological features."""
        fig, axes = plt.subplots(3, 3, figsize=(22, 16), facecolor='#0a0a1a')

        configs = [
            (axes[0, 0], matrices['h1_count'], 'H₁ Loop Count', 'Reds'),
            (axes[0, 1], matrices['h1_total_persistence'], 'H₁ Total Persistence', 'OrRd'),
            (axes[0, 2], matrices['h1_significant'], 'H₁ Significant Loops', 'magma'),
            (axes[1, 0], matrices['persistence_entropy'], 'Persistence Entropy', 'viridis'),
            (axes[1, 1], matrices['spectral_gap'], 'Graph Spectral Gap', 'YlGnBu'),
            (axes[1, 2], matrices['irrotational_fraction'], 'Irrotational Fraction', 'RdYlBu_r'),
            (axes[2, 0], matrices['curl'], 'Curl ||J_antisym||', 'Purples'),
            (axes[2, 1], matrices['mapper_cycles'], 'Mapper Graph Cycles', 'Greens'),
            (axes[2, 2], matrices['n_cocycle_loops'], 'Representative Cocycle Loops', 'hot'),
        ]

        n_steps = len(steps)
        n_layers = len(layers)

        for ax, matrix, title, cmap in configs:
            ax.set_facecolor('#0d1117')
            im = ax.imshow(matrix.T, aspect='auto', cmap=cmap,
                           origin='lower', interpolation='nearest')
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(colors='#888888', labelsize=7)

            ax.set_xlabel('Training Step', color='#888888', fontsize=9)
            ax.set_ylabel('Layer', color='#888888', fontsize=9)
            ax.set_title(title, color='#c0d8e8', fontsize=11)

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

    def _plot_evolution_curves(self, matrices, steps, layers):
        """Plot per-layer evolution curves for key metrics."""
        n_layers = len(layers)
        cmap = plt.get_cmap('tab10')

        fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor='#0a0a1a')

        plot_configs = [
            (axes[0, 0], matrices['h1_count'], 'H₁ Loop Count'),
            (axes[0, 1], matrices['h1_total_persistence'], 'H₁ Total Persistence'),
            (axes[0, 2], matrices['persistence_entropy'], 'Persistence Entropy'),
            (axes[1, 0], matrices['spectral_gap'], 'Graph Spectral Gap'),
            (axes[1, 1], matrices['curl'], 'Curl'),
            (axes[1, 2], matrices['irrotational_fraction'], 'Irrotational Fraction'),
        ]

        for ax, matrix, title in plot_configs:
            ax.set_facecolor('#0d1117')
            for li in range(n_layers):
                color = cmap(li / max(n_layers - 1, 1))
                ax.plot(steps, matrix[:, li], 'o-', color=color, lw=1.8,
                        markersize=4, label=f'Layer {layers[li]}', alpha=0.85)

            ax.set_xlabel('Training Step', color='#888888', fontsize=9)
            ax.set_ylabel('Value', color='#888888', fontsize=9)
            ax.set_title(title, color='#c0d8e8', fontsize=11)
            ax.legend(fontsize=6, facecolor='#0d1117', edgecolor='#2a3a4a',
                      labelcolor='#cccccc', loc='best',
                      ncol=max(1, n_layers // 4))
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.15, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        fig.suptitle('Per-Layer Topological Evolution Over Training',
                     color='#c0d8e8', fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'evolution_curves.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    def _plot_persistence_image_gallery(self, all_extractions, steps, layers):
        """Plot persistence images for a selection of (step, layer) pairs."""
        if not HAS_GUDHI:
            return

        # Pick a representative subset
        sample_keys = []
        if len(steps) <= 5:
            sample_steps = steps
        else:
            idx = np.linspace(0, len(steps) - 1, 5, dtype=int)
            sample_steps = [steps[i] for i in idx]

        for s in sample_steps:
            for l in layers:
                if (s, l) in all_extractions:
                    ext = all_extractions[(s, l)]
                    pi = ext.get('persistence_image', {})
                    if 'h1_image' in pi:
                        sample_keys.append((s, l))

        if not sample_keys:
            return

        n_cols = min(len(sample_keys), 6)
        n_rows = (len(sample_keys) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows),
                                 facecolor='#0a0a1a')
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for idx, (step, layer) in enumerate(sample_keys[:n_rows * n_cols]):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            ax.set_facecolor('#0d1117')

            ext = all_extractions[(step, layer)]
            pi = ext.get('persistence_image', {})
            if 'h1_image' in pi:
                shape = pi.get('h1_image_shape', (20, 20))
                img = pi['h1_image'].reshape(shape)
                ax.imshow(img, cmap='hot', origin='lower', aspect='auto')

            ax.set_title(f'Step {step}, L{layer}', color='#c0d8e8', fontsize=8)
            ax.tick_params(labelsize=0, length=0)
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        # Hide unused axes
        for idx in range(len(sample_keys), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].set_visible(False)

        fig.suptitle('H₁ Persistence Images (fixed-dimensional topological fingerprints)',
                     color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'persistence_images.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    def _plot_mapper_summary(self, all_extractions, steps, layers):
        """Summarize Mapper graph statistics across training."""
        if not HAS_KMAPPER:
            return

        n_steps = len(steps)
        n_layers = len(layers)
        step_idx = {s: i for i, s in enumerate(steps)}

        # Aggregate mapper stats per step
        nodes_per_step = np.zeros(n_steps)
        cycles_per_step = np.zeros(n_steps)

        for (step, layer), ext in all_extractions.items():
            si = step_idx[step]
            mp = ext.get('mapper', {})
            nodes_per_step[si] += mp.get('n_nodes', 0)
            cycles_per_step[si] += mp.get('n_cycles', 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0a0a1a')

        ax = axes[0]
        ax.set_facecolor('#0d1117')
        ax.plot(steps, nodes_per_step, 'o-', color='#44aaff', lw=2, markersize=5)
        ax.set_xlabel('Training Step', color='#888888')
        ax.set_ylabel('Total Mapper Nodes', color='#888888')
        ax.set_title('Mapper Graph Complexity', color='#c0d8e8', fontsize=11)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        ax = axes[1]
        ax.set_facecolor('#0d1117')
        ax.plot(steps, cycles_per_step, 'o-', color='#ff44aa', lw=2, markersize=5)
        ax.fill_between(steps, cycles_per_step, alpha=0.1, color='#ff44aa')
        ax.set_xlabel('Training Step', color='#888888')
        ax.set_ylabel('Total Mapper Cycles', color='#888888')
        ax.set_title('Mapper Graph Loops (topological skeleton cycles)',
                     color='#c0d8e8', fontsize=11)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        fig.suptitle('Mapper Algorithm Summary', color='#c0d8e8', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'mapper_summary.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    def _plot_helmholtz_summary(self, all_extractions, steps, layers):
        """Summarize Helmholtz decomposition across training."""
        n_steps = len(steps)
        step_idx = {s: i for i, s in enumerate(steps)}

        irrot_per_step = np.zeros(n_steps)
        sol_per_step = np.zeros(n_steps)
        vort_per_step = np.zeros(n_steps)
        counts = np.zeros(n_steps)

        for (step, layer), ext in all_extractions.items():
            si = step_idx[step]
            helm = ext.get('helmholtz', {})
            if 'irrotational_fraction' in helm:
                irrot_per_step[si] += helm['irrotational_fraction']
                sol_per_step[si] += helm['solenoidal_fraction']
                vort_per_step[si] += helm.get('vorticity_max', 0)
                counts[si] += 1

        # Average
        mask = counts > 0
        irrot_per_step[mask] /= counts[mask]
        sol_per_step[mask] /= counts[mask]
        vort_per_step[mask] /= counts[mask]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0a0a1a')

        ax = axes[0]
        ax.set_facecolor('#0d1117')
        ax.fill_between(steps, 0, irrot_per_step, alpha=0.4, color='#44aaff',
                        label='Irrotational (∇φ)')
        ax.fill_between(steps, irrot_per_step,
                        [i + s for i, s in zip(irrot_per_step, sol_per_step)],
                        alpha=0.4, color='#ff44aa', label='Solenoidal (∇×ψ)')
        ax.set_xlabel('Training Step', color='#888888')
        ax.set_ylabel('Energy Fraction', color='#888888')
        ax.set_title('Helmholtz Decomposition Over Training', color='#c0d8e8', fontsize=11)
        ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#2a3a4a',
                  labelcolor='#cccccc')
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        ax = axes[1]
        ax.set_facecolor('#0d1117')
        ax.plot(steps, vort_per_step, 'o-', color='#ffaa44', lw=2, markersize=5)
        ax.set_xlabel('Training Step', color='#888888')
        ax.set_ylabel('Max Vorticity', color='#888888')
        ax.set_title('Peak Vorticity Over Training', color='#c0d8e8', fontsize=11)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'helmholtz_summary.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    def _plot_cocycle_analysis(self, all_extractions, steps, layers):
        """Analyze representative cocycles — which tokens form loops?"""
        # Collect all cocycle loop info
        all_loops = []
        for (step, layer), ext in all_extractions.items():
            pers = ext.get('persistence', {})
            loops = pers.get('h1_cocycle_points', [])
            for loop in loops:
                loop['step'] = step
                loop['layer'] = layer
                all_loops.append(loop)

        if not all_loops:
            console.print("  [dim]No representative cocycles found[/]")
            return

        console.print(f"  [green]Found {len(all_loops)} representative H₁ cocycle loops[/]")

        # Plot: loop size distribution
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0a0a1a')

        # 1. Loop size distribution
        ax = axes[0]
        ax.set_facecolor('#0d1117')
        sizes = [l['n_points'] for l in all_loops]
        ax.hist(sizes, bins=range(min(sizes), max(sizes) + 2),
                color='#ff44aa', alpha=0.8, edgecolor='#2a3a4a')
        ax.set_xlabel('Points in Loop', color='#888888')
        ax.set_ylabel('Count', color='#888888')
        ax.set_title('Cocycle Loop Size Distribution', color='#c0d8e8', fontsize=10)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        # 2. Loop lifetime vs size
        ax = axes[1]
        ax.set_facecolor('#0d1117')
        lifetimes = [l['lifetime'] for l in all_loops]
        n_points = [l['n_points'] for l in all_loops]
        layer_ids = [l['layer'] for l in all_loops]

        scatter = ax.scatter(n_points, lifetimes, c=layer_ids, cmap='viridis',
                             s=30, alpha=0.8, edgecolors='#ffffff', linewidths=0.3)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Layer', color='#888888', fontsize=8)
        cbar.ax.tick_params(colors='#888888', labelsize=7)
        ax.set_xlabel('Points in Loop', color='#888888')
        ax.set_ylabel('Lifetime', color='#888888')
        ax.set_title('Loop Lifetime vs Size (colored by layer)',
                     color='#c0d8e8', fontsize=10)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        # 3. Loops per step (timeline)
        ax = axes[2]
        ax.set_facecolor('#0d1117')
        loop_steps = [l['step'] for l in all_loops]
        unique_steps = sorted(set(loop_steps))
        loops_per_step = [loop_steps.count(s) for s in unique_steps]
        ax.bar(unique_steps, loops_per_step, color='#ff44aa', alpha=0.8,
               edgecolor='#2a3a4a')
        ax.set_xlabel('Training Step', color='#888888')
        ax.set_ylabel('Number of Cocycle Loops', color='#888888')
        ax.set_title('Representative Cocycle Loops Over Training',
                     color='#c0d8e8', fontsize=10)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333', axis='y')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        fig.suptitle('Representative H₁ Cocycle Analysis',
                     color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'cocycle_analysis.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    def _plot_graph_spectrum_summary(self, all_extractions, steps, layers):
        """Summarize graph Laplacian spectral properties across training."""
        n_steps = len(steps)
        step_idx = {s: i for i, s in enumerate(steps)}

        gap_per_step = np.zeros(n_steps)
        components_per_step = np.zeros(n_steps)
        counts = np.zeros(n_steps)

        for (step, layer), ext in all_extractions.items():
            si = step_idx[step]
            gs = ext.get('graph_spectrum', {})
            if 'spectral_gap' in gs:
                gap_per_step[si] += gs['spectral_gap']
                components_per_step[si] += gs.get('n_components', 0)
                counts[si] += 1

        mask = counts > 0
        gap_per_step[mask] /= counts[mask]
        components_per_step[mask] /= counts[mask]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0a0a1a')

        ax = axes[0]
        ax.set_facecolor('#0d1117')
        ax.plot(steps, gap_per_step, 'o-', color='#44ffaa', lw=2, markersize=5)
        ax.set_xlabel('Training Step', color='#888888')
        ax.set_ylabel('Mean Spectral Gap', color='#888888')
        ax.set_title('Graph Spectral Gap Over Training\n'
                     '(larger = more separated clusters)',
                     color='#c0d8e8', fontsize=11)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        ax = axes[1]
        ax.set_facecolor('#0d1117')
        ax.plot(steps, components_per_step, 'o-', color='#ffaa44', lw=2,
                markersize=5)
        ax.set_xlabel('Training Step', color='#888888')
        ax.set_ylabel('Mean Connected Components', color='#888888')
        ax.set_title('Graph Connected Components Over Training',
                     color='#c0d8e8', fontsize=11)
        ax.tick_params(colors='#888888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'graph_spectrum_summary.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── Cross-correlation with training loss ───────────────────────

    def _correlate_with_loss(self, matrices, steps, layers, loss_data):
        """
        Compute cross-correlation between topological features and
        training/validation loss. This is the key test: do topological
        changes PREDICT loss changes?
        """
        console.print("\n[bold yellow]═══ Topology ↔ Loss Correlation ═══[/]")

        loss_steps = loss_data['steps']
        train_loss = loss_data['train_loss']
        val_loss = loss_data['val_loss']

        # Interpolate loss to our topology steps
        from scipy.interpolate import interp1d

        try:
            train_interp = interp1d(loss_steps, train_loss, kind='linear',
                                    fill_value='extrapolate')
            train_at_topo = train_interp(steps)
        except Exception:
            console.print("  [yellow]Could not interpolate training loss[/]")
            return

        val_at_topo = None
        if not np.all(np.isnan(val_loss)):
            try:
                valid = ~np.isnan(val_loss)
                val_interp = interp1d(loss_steps[valid], val_loss[valid],
                                      kind='linear', fill_value='extrapolate')
                val_at_topo = val_interp(steps)
            except Exception:
                pass

        # Compute correlations for each topological feature
        topo_features = {
            'H₁ Count (total)': matrices['h1_count'].sum(axis=1),
            'H₁ Persistence (total)': matrices['h1_total_persistence'].sum(axis=1),
            'H₁ Significant (total)': matrices['h1_significant'].sum(axis=1),
            'Persistence Entropy (total)': matrices['persistence_entropy'].sum(axis=1),
            'Spectral Gap (mean)': matrices['spectral_gap'].mean(axis=1),
            'Irrotational Fraction (mean)': matrices['irrotational_fraction'].mean(axis=1),
            'Curl (total)': matrices['curl'].sum(axis=1),
            'Mapper Cycles (total)': matrices['mapper_cycles'].sum(axis=1),
            'Cocycle Loops (total)': matrices['n_cocycle_loops'].sum(axis=1),
        }

        from rich.table import Table as RichTable
        from rich import box as rich_box

        table = RichTable(title="Topology ↔ Training Loss Correlation",
                          box=rich_box.ROUNDED, show_lines=True)
        table.add_column("Topological Feature", style="bold cyan")
        table.add_column("Pearson r (train)", style="magenta")
        table.add_column("p-value", style="yellow")
        table.add_column("Spearman ρ", style="green")
        table.add_column("Interpretation", style="white")

        correlations = {}

        for name, feature in topo_features.items():
            if len(feature) < 3 or np.std(feature) < 1e-10:
                continue

            try:
                r, p = pearsonr(feature, train_at_topo)
                rho, _ = spearmanr(feature, train_at_topo)
            except Exception:
                continue

            correlations[name] = {'r': r, 'p': p, 'rho': rho}

            if abs(r) > 0.7:
                interp = "🔥 STRONG" + (" (loss ↓ as topo ↑)" if r < 0 else " (co-evolving)")
            elif abs(r) > 0.4:
                interp = "⚡ Moderate"
            elif abs(r) > 0.2:
                interp = "~ Weak"
            else:
                interp = "✗ None"

            table.add_row(
                name,
                f"{r:+.3f}",
                f"{p:.2e}" if p < 0.05 else f"[dim]{p:.2e}[/]",
                f"{rho:+.3f}",
                interp,
            )

        console.print(table)

        # Plot the correlations
        n_features = len(correlations)
        if n_features == 0:
            return

        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 4 * n_rows),
                                 facecolor='#0a0a1a')
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for idx, (name, corr) in enumerate(correlations.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            ax.set_facecolor('#0d1117')

            feature = topo_features[name]
            ax.scatter(feature, train_at_topo, s=30, c='#44aaff', alpha=0.8,
                       edgecolors='#ffffff', linewidths=0.3)

            # Trend line
            if len(feature) > 2:
                z = np.polyfit(feature, train_at_topo, 1)
                p_line = np.poly1d(z)
                x_range = np.linspace(feature.min(), feature.max(), 50)
                ax.plot(x_range, p_line(x_range), '--', color='#ff44aa', lw=1.5,
                        alpha=0.7)

            ax.set_xlabel(name, color='#888888', fontsize=8)
            ax.set_ylabel('Training Loss', color='#888888', fontsize=8)
            r_val = corr['r']
            p_val = corr['p']
            sig = "★" if p_val < 0.05 else ""
            ax.set_title(f'r={r_val:+.3f} {sig}', color='#c0d8e8', fontsize=9)
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.15, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        # Hide unused axes
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].set_visible(False)

        fig.suptitle('Topology ↔ Training Loss Scatter Plots',
                     color='#c0d8e8', fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'topology_loss_correlation.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

        # Lagged cross-correlation (does topology PREDICT loss?)
        if len(steps) > 5:
            self._compute_lagged_correlation(topo_features, train_at_topo, steps)

    def _compute_lagged_correlation(self, topo_features, train_loss, steps):
        """
        Compute lagged cross-correlation: does topology at step t
        predict loss at step t+lag?

        If topology changes BEFORE loss changes, this is evidence
        that topological structures are causally related to learning.
        """
        console.print("\n[bold]Lagged Cross-Correlation (does topology predict loss?):[/]")

        max_lag = min(len(steps) // 3, 10)
        if max_lag < 1:
            return

        fig, axes = plt.subplots(1, min(3, len(topo_features)),
                                 figsize=(6 * min(3, len(topo_features)), 4),
                                 facecolor='#0a0a1a')
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for idx, (name, feature) in enumerate(list(topo_features.items())[:3]):
            ax = axes[idx]
            ax.set_facecolor('#0d1117')

            lags = range(-max_lag, max_lag + 1)
            corrs = []

            for lag in lags:
                if lag >= 0:
                    f_slice = feature[:len(feature) - lag]
                    l_slice = train_loss[lag:]
                else:
                    f_slice = feature[-lag:]
                    l_slice = train_loss[:len(train_loss) + lag]

                if len(f_slice) < 3 or np.std(f_slice) < 1e-10:
                    corrs.append(0)
                    continue

                try:
                    r, _ = pearsonr(f_slice, l_slice)
                    corrs.append(r)
                except Exception:
                    corrs.append(0)

            ax.bar(list(lags), corrs, color='#44aaff', alpha=0.8,
                   edgecolor='#2a3a4a')
            ax.axhline(0, color='#555555', lw=0.5)
            ax.axvline(0, color='#ff44aa', lw=1, ls='--', alpha=0.5)

            # Highlight the peak
            peak_lag = list(lags)[np.argmax(np.abs(corrs))]
            peak_r = corrs[np.argmax(np.abs(corrs))]

            if peak_lag < 0:
                ax.set_title(f'{name}\nPeak at lag={peak_lag} (topo LEADS loss by '
                             f'{abs(peak_lag)} steps, r={peak_r:+.3f})',
                             color='#44ff44', fontsize=8)
                console.print(f"  [green]⚡ {name}: topology LEADS loss by "
                              f"{abs(peak_lag)} steps (r={peak_r:+.3f})[/]")
            elif peak_lag > 0:
                ax.set_title(f'{name}\nPeak at lag={peak_lag} (loss leads topo)',
                             color='#ffaa44', fontsize=8)
            else:
                ax.set_title(f'{name}\nPeak at lag=0 (simultaneous, r={peak_r:+.3f})',
                             color='#c0d8e8', fontsize=8)

            ax.set_xlabel('Lag (negative = topo leads)', color='#888888', fontsize=8)
            ax.set_ylabel('Pearson r', color='#888888', fontsize=8)
            ax.tick_params(colors='#888888', labelsize=7)
            ax.grid(True, alpha=0.15, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#2a3a4a')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'lagged_correlation.png')
        fig.savefig(path, dpi=150, facecolor='#0a0a1a', edgecolor='none',
                    bbox_inches='tight')
        plt.close(fig)
        console.print(f"  📊 Saved: {path}")

    # ─── Phase transition detection ─────────────────────────────────

    def _detect_phase_transitions(self, matrices, steps, layers):
        """
        Detect phase transitions: sudden jumps in topological complexity.
        Uses multiple signals and reports the most significant transitions.
        """
        if len(steps) < 3:
            return

        console.print("\n[bold yellow]═══ Phase Transition Detection ═══[/]")

        signals = {
            'H₁ Count': matrices['h1_count'].sum(axis=1),
            'H₁ Persistence': matrices['h1_total_persistence'].sum(axis=1),
            'Persistence Entropy': matrices['persistence_entropy'].sum(axis=1),
            'Curl': matrices['curl'].sum(axis=1),
            'Spectral Gap': matrices['spectral_gap'].mean(axis=1),
            'Mapper Cycles': matrices['mapper_cycles'].sum(axis=1),
        }

        all_transitions = []

        for name, signal in signals.items():
            diffs = np.diff(signal)
            if len(diffs) == 0:
                continue

            mean_diff = np.mean(np.abs(diffs))
            std_diff = np.std(np.abs(diffs))

            if std_diff < 1e-10:
                continue

            for idx, d in enumerate(diffs):
                z_score = (abs(d) - mean_diff) / (std_diff + 1e-10)
                if z_score > 2.0:
                    all_transitions.append({
                        'metric': name,
                        'step_from': steps[idx],
                        'step_to': steps[idx + 1],
                        'change': d,
                        'z_score': z_score,
                        'direction': 'formation' if d > 0 else 'disappearance',
                    })

        # Sort by z-score
        all_transitions.sort(key=lambda x: -x['z_score'])

        if all_transitions:
            from rich.table import Table as RichTable
            from rich import box as rich_box

            table = RichTable(title="Detected Phase Transitions",
                              box=rich_box.ROUNDED, show_lines=True)
            table.add_column("Metric", style="bold cyan")
            table.add_column("Transition", style="yellow")
            table.add_column("Change", style="magenta")
            table.add_column("Z-score", style="green")
            table.add_column("Type", style="white")

            for t in all_transitions[:15]:
                emoji = "🟢" if t['direction'] == 'formation' else "🔴"
                table.add_row(
                    t['metric'],
                    f"Step {t['step_from']} → {t['step_to']}",
                    f"{t['change']:+.4f}",
                    f"{t['z_score']:.2f}σ",
                    f"{emoji} {t['direction']}",
                )

            console.print(table)

            # Check for grokking: multiple metrics jumping at the same step
            from collections import Counter
            step_counts = Counter(
                (t['step_from'], t['step_to']) for t in all_transitions
            )
            for (s_from, s_to), count in step_counts.most_common(3):
                if count >= 3:
                    console.print(
                        f"\n  [bold green]⚡ POSSIBLE GROKKING EVENT at "
                        f"step {s_from}→{s_to}![/]"
                    )
                    console.print(
                        f"     {count} topological metrics changed simultaneously!"
                    )
                    console.print(
                        f"     This suggests sudden emergence of computational "
                        f"structures (Conjecture 3)"
                    )
        else:
            console.print("  [dim]No significant phase transitions detected[/]")

    # ─── Save results ───────────────────────────────────────────────

    def _save_results(self, all_extractions, feature_matrices, steps, layers):
        """Save numerical results to JSON and NPZ files."""
        # Save feature matrices as NPZ
        npz_path = os.path.join(self.output_dir, 'feature_matrices.npz')
        save_dict = {
            'steps': np.array(steps),
            'layers': np.array(layers),
        }
        for name, matrix in feature_matrices.items():
            save_dict[name] = matrix
        np.savez_compressed(npz_path, **save_dict)
        console.print(f"  💾 Saved: {npz_path}")

        # Save summary statistics as JSON
        summary = {
            'steps': [int(s) for s in steps],
            'layers': [int(l) for l in layers],
            'n_files': len(all_extractions),
            'feature_names': list(feature_matrices.keys()),
        }

        # Per-feature summary
        for name, matrix in feature_matrices.items():
            summary[f'{name}_mean'] = float(matrix.mean())
            summary[f'{name}_max'] = float(matrix.max())
            summary[f'{name}_std'] = float(matrix.std())

        json_path = os.path.join(self.output_dir, 'extraction_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        console.print(f"  💾 Saved: {json_path}")

    # ─── Print summary ──────────────────────────────────────────────

    def _print_summary(self, feature_matrices, steps, layers, all_extractions):
        """Print a comprehensive summary of all findings."""
        from rich.table import Table as RichTable
        from rich import box as rich_box

        console.print("\n")

        table = RichTable(title="Topological Extraction Summary",
                          box=rich_box.DOUBLE_EDGE, show_lines=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Total", style="yellow")
        table.add_column("Mean", style="green")
        table.add_column("Max", style="magenta")
        table.add_column("Interpretation", style="white")

        key_metrics = [
            ('h1_count', 'H₁ Loops', 'Topological loops in token space'),
            ('h1_significant', 'H₁ Significant', 'Long-lived loops (>2× median)'),
            ('h1_total_persistence', 'H₁ Persistence', 'Total loop lifetime'),
            ('persistence_entropy', 'Pers. Entropy', 'Topological complexity'),
            ('spectral_gap', 'Spectral Gap', 'Cluster separation'),
            ('curl', 'Curl', 'Rotational mixing'),
            ('mapper_cycles', 'Mapper Cycles', 'Skeleton graph loops'),
            ('n_cocycle_loops', 'Cocycle Loops', 'Representative H₁ loops'),
        ]

        for key, name, interp in key_metrics:
            matrix = feature_matrices.get(key, np.zeros((1, 1)))
            table.add_row(
                name,
                f"{matrix.sum():.2f}",
                f"{matrix.mean():.4f}",
                f"{matrix.max():.4f}",
                interp,
            )

        console.print(table)

        # Key findings
        console.print("\n[bold yellow]═══ Key Findings ═══[/]")

        total_sig = feature_matrices['h1_significant'].sum()
        if total_sig > 0:
            console.print(
                f"  [bold green]⚡ {int(total_sig)} SIGNIFICANT TOPOLOGICAL "
                f"STRUCTURES detected![/]"
            )
            console.print(
                f"     These persistent loops may encode computational "
                f"circuits (Conjecture 3 in paper.tex)"
            )
        else:
            console.print(
                f"  [yellow]No significant persistent loops detected.[/]"
            )

        # Helmholtz summary
        irrot = feature_matrices['irrotational_fraction']
        if irrot.sum() > 0:
            mean_irrot = irrot[irrot > 0].mean()
            console.print(
                f"\n  📐 Helmholtz: {mean_irrot:.1%} irrotational, "
                f"{1 - mean_irrot:.1%} solenoidal (average)"
            )
            if mean_irrot > 0.7:
                console.print(
                    f"     → Deformation is mostly gradient-like (expansion/contraction)"
                )
            elif mean_irrot < 0.3:
                console.print(
                    f"     → Deformation is mostly rotational (information mixing)"
                )

        # Mapper summary
        total_mapper_cycles = feature_matrices['mapper_cycles'].sum()
        if total_mapper_cycles > 0:
            console.print(
                f"\n  🕸️ Mapper: {int(total_mapper_cycles)} graph cycles detected "
                f"in the topological skeleton"
            )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract ALL topological structures from Jacobi field data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction on a single run
  python3 extract_topology.py runs/0/jacobi_data/ --loss-csv runs/0/epoch_log.csv

  # Quick summary (skip slow probes)
  python3 extract_topology.py runs/0/jacobi_data/ --quick

  # Compare multiple runs
  python3 extract_topology.py runs/*/jacobi_data/ --compare

  # Custom output directory
  python3 extract_topology.py runs/0/jacobi_data/ -o my_topology_results
        """,
    )

    parser.add_argument("paths", type=str, nargs='+',
                        help="Path(s) to jacobi_data directories or .npz files")
    parser.add_argument("--loss-csv", type=str, default=None,
                        help="Path to epoch_log.csv or batch_log.csv for "
                             "topology↔loss correlation")
    parser.add_argument("-o", "--output-dir", type=str,
                        default="topology_extraction",
                        help="Output directory for plots and data")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow probes (Mapper, cocycles, graph spectrum)")
    parser.add_argument("--max-pts", type=int, default=300,
                        help="Max points for persistent homology (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple runs (paths should be different runs)")

    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Dependency check ────────────────────────────────────────────────
    missing = []
    if not HAS_RIPSER:
        missing.append("ripser (pip install ripser)")
    if not HAS_PERSIM:
        missing.append("persim (pip install persim)")

    if missing:
        console.print(f"[bold red]Missing required packages: {', '.join(missing)}[/]")
        console.print("[yellow]Install with: pip install ripser persim[/]")
        sys.exit(1)

    optional_missing = []
    if not HAS_GUDHI:
        optional_missing.append("gudhi (persistence images/landscapes)")
    if not HAS_KMAPPER:
        optional_missing.append("kmapper (Mapper algorithm)")
    if not HAS_NETWORKX:
        optional_missing.append("networkx (graph analysis)")

    if optional_missing:
        console.print(f"[yellow]Optional packages not installed: "
                      f"{', '.join(optional_missing)}[/]")
        console.print("[dim]Some analyses will be skipped. Install with:[/]")
        console.print("[dim]  pip install gudhi kmapper networkx[/]")

    # ── Run the pipeline ────────────────────────────────────────────────
    pipeline = TopologyPipeline(
        output_dir=args.output_dir,
        max_pts=args.max_pts,
        quick=args.quick,
    )

    pipeline.run(args.paths, loss_csv=args.loss_csv)


if __name__ == "__main__":
    main()
