#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "scikit-learn",
#     "ripser",
#     "persim",
#     "tqdm",
# ]
# ///
#
# ============================================================
# SLURM DIRECTIVES (ignored by python, used by sbatch)
# ============================================================
#SBATCH --job-name=llm_fibre_analysis
#SBATCH --output=llm_fibre_%j.out
#SBATCH --error=llm_fibre_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#
# Usage:
#   python3 script.py --test-small
#   python3 script.py --test-all
#   python3 script.py --model gpt2-large --prompt "The meaning of life is"
#   sbatch script.py --test-small
#
# If using uv:
#   uv run script.py --test-small
#
"""
Transformer Layers as Fibre Bundle Morphisms: Empirical Analysis Tool

Analyzes real LLM internals to test the conjectures from:
"Transformer Layers as Fibre Bundle Morphisms: An Interpretability
 Conjecture and Visualization Tool" (Koch, 2026)

Extracts hidden states across all layers, computes Jacobian fields,
eigenvalue spectra, divergence/curl/shear decompositions, Ollivier-Ricci
curvature, persistent homology, and generates visualizations.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm

import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, orthogonal_procrustes
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Configuration
# ============================================================

SMALL_MODELS = ["gpt2"]
ALL_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "microsoft/phi-2",
]

DEFAULT_PROMPTS = [
    "The sky is blue because light scatters in the atmosphere",
    "The sky is red because the sunset filters long wavelengths",
    "Mathematics is the language of the universe and nature",
    "The cat sat on the mat and watched the birds outside",
]

OUTPUT_DIR = Path("fibre_analysis_output")


# ============================================================
# Utility Functions
# ============================================================


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  [GPU] Using CUDA: {torch.cuda.get_device_name(0)}")
        print(
            f"  [GPU] Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
        )
        return dev
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  [GPU] Using Apple MPS")
        return torch.device("mps")
    else:
        print("  [CPU] No GPU found, using CPU (will be slower)")
        return torch.device("cpu")


def safe_mkdir(path):
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# Model Loading and Hidden State Extraction
# ============================================================


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load a HuggingFace model with all hidden states output enabled."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
        )
    except Exception:
        model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
        )

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Parameters: {num_params / 1e6:.1f}M")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")

    return model, tokenizer, {"layers": num_layers, "hidden_dim": hidden_dim, "params": num_params}


def extract_hidden_states(model, tokenizer, prompt: str, device: torch.device):
    """Extract hidden states at every layer for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states: tuple of (num_layers+1) tensors, each [batch, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Stack into [num_layers+1, seq_len, hidden_dim]
    hs = torch.stack([h[0] for h in hidden_states]).cpu().numpy()

    return hs, tokens


# ============================================================
# Jacobian Estimation
# ============================================================


def estimate_jacobian_from_cloud(points_before, points_after, k_neighbors=None):
    """
    Estimate the Jacobian of the layer map from paired point clouds.

    For each point, use its k nearest neighbors to fit a local linear map
    via least squares: delta_after ≈ J @ delta_before

    Returns: array of Jacobians [n_points, d, d]
    """
    n, d = points_before.shape
    if k_neighbors is None:
        k_neighbors = min(max(d + 1, 8), n - 1)
    k_neighbors = min(k_neighbors, n - 1)

    # Compute pairwise distances in the 'before' space
    dists = squareform(pdist(points_before))

    jacobians = np.zeros((n, d, d))

    for i in range(n):
        # Find k nearest neighbors (excluding self)
        neighbors = np.argsort(dists[i])[1 : k_neighbors + 1]

        if len(neighbors) < d:
            # Not enough neighbors; use identity
            jacobians[i] = np.eye(d)
            continue

        # Deltas
        delta_before = points_before[neighbors] - points_before[i]  # [k, d]
        delta_after = points_after[neighbors] - points_after[i]  # [k, d]

        # Weighted least squares (weight by inverse distance)
        weights = 1.0 / (dists[i, neighbors] + 1e-10)
        W = np.diag(weights)

        # Solve: delta_after.T = J @ delta_before.T (weighted)
        # => (W @ delta_after) = (W @ delta_before) @ J.T
        A = np.sqrt(W) @ delta_before
        B = np.sqrt(W) @ delta_after

        try:
            J, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            jacobians[i] = J.T
        except np.linalg.LinAlgError:
            jacobians[i] = np.eye(d)

    return jacobians


def decompose_jacobian(J):
    """
    Decompose a Jacobian matrix into geometric invariants.

    Returns dict with: divergence, curl_norm, shear_norm, eigenvalues,
    singular_values, determinant, condition_number
    """
    d = J.shape[0]

    # Divergence = trace
    divergence = np.trace(J)

    # Symmetric and antisymmetric parts
    J_sym = (J + J.T) / 2
    J_antisym = (J - J.T) / 2

    # Curl norm (Frobenius norm of antisymmetric part)
    curl_norm = np.linalg.norm(J_antisym, "fro")

    # Shear norm (traceless symmetric part)
    shear = J_sym - (np.trace(J_sym) / d) * np.eye(d)
    shear_norm = np.linalg.norm(shear, "fro")

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(J)

    # Singular values
    try:
        _, singular_values, _ = svd(J)
    except np.linalg.LinAlgError:
        singular_values = np.abs(eigenvalues)

    # Determinant
    determinant = np.linalg.det(J)

    # Condition number
    if singular_values[-1] > 1e-12:
        condition_number = singular_values[0] / singular_values[-1]
    else:
        condition_number = np.inf

    return {
        "divergence": divergence,
        "curl_norm": curl_norm,
        "shear_norm": shear_norm,
        "eigenvalues": eigenvalues,
        "singular_values": singular_values,
        "determinant": determinant,
        "condition_number": condition_number,
    }


# ============================================================
# Curvature Computations
# ============================================================


def compute_ollivier_ricci_curvature(points, k=5):
    """
    Compute Ollivier-Ricci curvature for each pair of neighboring points.

    ORC(x,y) = 1 - W1(mu_x, mu_y) / d(x,y)

    Returns: mean ORC, std ORC, array of per-pair ORC values
    """
    from scipy.optimize import linear_sum_assignment

    n = points.shape[0]
    dists = squareform(pdist(points))
    k = min(k, n - 1)

    orc_values = []

    for i in range(n):
        neighbors_i = np.argsort(dists[i])[1 : k + 1]

        for j in neighbors_i:
            if j <= i:
                continue

            d_ij = dists[i, j]
            if d_ij < 1e-12:
                continue

            neighbors_j = np.argsort(dists[j])[1 : k + 1]

            # Local distributions: uniform over k neighbors
            # Compute W1 via optimal transport (assignment problem)
            cost_matrix = dists[np.ix_(neighbors_i, neighbors_j)]

            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                w1 = cost_matrix[row_ind, col_ind].mean()
            except ValueError:
                continue

            orc = 1.0 - w1 / d_ij
            orc_values.append(orc)

    orc_values = np.array(orc_values) if orc_values else np.array([0.0])
    return orc_values.mean(), orc_values.std(), orc_values


def compute_volume_change(points_before, points_after, k=5):
    """
    Compute log-volume change of local simplices between layers.
    Proxy for scalar curvature.
    """
    n = points_before.shape[0]
    k = min(k, n - 1)
    dists_before = squareform(pdist(points_before))

    volume_changes = []

    for i in range(n):
        neighbors = np.argsort(dists_before[i])[1 : k + 1]

        # Local simplex volume via distances
        local_before = points_before[neighbors] - points_before[i]
        local_after = points_after[neighbors] - points_after[i]

        # Use sum of squared distances as volume proxy
        vol_before = np.sum(np.linalg.norm(local_before, axis=1) ** 2) + 1e-12
        vol_after = np.sum(np.linalg.norm(local_after, axis=1) ** 2) + 1e-12

        volume_changes.append(np.log(vol_after / vol_before))

    return np.array(volume_changes)


def compute_procrustes_deviation(points_before, points_after, n_components=None):
    """
    Compute Procrustes deviation between local PCA frames across layers.
    Measures connection strength (how much the local frame rotates).
    """
    if n_components is None:
        n_components = min(points_before.shape[0] - 1, points_before.shape[1], 10)
    n_components = max(2, min(n_components, min(points_before.shape) - 1))

    pca_before = PCA(n_components=n_components)
    pca_after = PCA(n_components=n_components)

    pca_before.fit(points_before)
    pca_after.fit(points_after)

    # Align via Procrustes
    A = pca_before.components_.T  # [d, n_components]
    B = pca_after.components_.T

    R, _ = orthogonal_procrustes(A, B)
    deviation = np.linalg.norm(R - np.eye(R.shape[0]), "fro")

    return deviation, pca_before.explained_variance_ratio_, pca_after.explained_variance_ratio_


# ============================================================
# Persistent Homology
# ============================================================


def compute_persistent_homology(points, max_dim=2, n_points_subsample=100):
    """
    Compute persistent homology of a point cloud.
    Returns persistence diagrams for H0, H1, H2.
    """
    import ripser

    if points.shape[0] > n_points_subsample:
        idx = np.random.choice(points.shape[0], n_points_subsample, replace=False)
        points = points[idx]

    # Project to manageable dimension
    if points.shape[1] > 50:
        pca = PCA(n_components=50)
        points = pca.fit_transform(points)

    # Normalize
    med_dist = np.median(pdist(points))
    if med_dist > 1e-12:
        points = points / med_dist

    result = ripser.ripser(points, maxdim=max_dim, thresh=2.0)
    return result["dgms"]


def wasserstein_distance_diagrams(dgm1, dgm2):
    """Compute Wasserstein-1 distance between two persistence diagrams."""
    from persim import wasserstein

    try:
        return wasserstein(dgm1, dgm2)
    except Exception:
        return 0.0


# ============================================================
# Analysis Pipeline
# ============================================================


class FibreBundleAnalyzer:
    """Main analysis class that tests all conjectures from the paper."""

    def __init__(self, model_name: str, device: torch.device, output_dir: Path):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir / model_name.replace("/", "_")
        safe_mkdir(self.output_dir)

        self.model, self.tokenizer, self.model_info = load_model_and_tokenizer(
            model_name, device
        )
        self.results = {}

    def analyze_prompt(self, prompt: str, prompt_id: int = 0):
        """Run full analysis pipeline on a single prompt."""
        print(f"\n  Analyzing prompt: '{prompt[:60]}...'")

        hs, tokens = extract_hidden_states(self.model, self.tokenizer, prompt, self.device)
        n_layers, n_tokens, hidden_dim = hs.shape
        print(f"  Hidden states shape: {hs.shape} (layers×tokens×dim)")

        # PCA projection for visualization (project to manageable dims)
        pca_dim = min(50, hidden_dim, n_tokens - 1)
        all_points = hs.reshape(-1, hidden_dim)
        pca = PCA(n_components=pca_dim)
        all_projected = pca.fit_transform(all_points)
        hs_pca = all_projected.reshape(n_layers, n_tokens, pca_dim)

        result = {
            "prompt": prompt,
            "tokens": tokens,
            "hidden_states": hs,
            "hidden_states_pca": hs_pca,
            "pca_variance_explained": pca.explained_variance_ratio_,
            "n_layers": n_layers,
            "n_tokens": n_tokens,
            "hidden_dim": hidden_dim,
        }

        # --- Conjecture 1: Space-morphing (Jacobian analysis) ---
        print("  [1/6] Computing Jacobian fields...")
        result["jacobian_analysis"] = self._analyze_jacobians(hs_pca)

        # --- Conjecture 2: Holographic scrambling ---
        print("  [2/6] Testing holographic scrambling...")
        result["holographic_analysis"] = self._analyze_holographic(hs)

        # --- Conjecture 3: Topological computation ---
        print("  [3/6] Computing persistent homology...")
        result["topology_analysis"] = self._analyze_topology(hs_pca)

        # --- Conjecture 4: Inner layers compute, outer translate ---
        print("  [4/6] Analyzing layer-wise deformation magnitudes...")
        result["layer_deformation"] = self._analyze_layer_deformations(hs, hs_pca)

        # --- Conjecture 5: Jacobian as holographic carrier ---
        print("  [5/6] Analyzing Jacobian eigenvalue spectra...")
        result["eigenvalue_analysis"] = self._analyze_eigenvalue_spectra(hs_pca)

        # --- Curvature analysis ---
        print("  [6/6] Computing curvature measures...")
        result["curvature_analysis"] = self._analyze_curvature(hs_pca)

        self.results[prompt_id] = result
        return result

    def _analyze_jacobians(self, hs_pca):
        """Compute Jacobian decomposition at each layer transition."""
        n_layers, n_tokens, d = hs_pca.shape
        layer_stats = []

        for ell in range(n_layers - 1):
            before = hs_pca[ell]
            after = hs_pca[ell + 1]

            jacobians = estimate_jacobian_from_cloud(before, after)

            decomps = [decompose_jacobian(J) for J in jacobians]

            stats = {
                "layer": ell,
                "mean_divergence": np.mean([d_["divergence"] for d_ in decomps]),
                "std_divergence": np.std([d_["divergence"] for d_ in decomps]),
                "mean_curl": np.mean([d_["curl_norm"] for d_ in decomps]),
                "mean_shear": np.mean([d_["shear_norm"] for d_ in decomps]),
                "mean_det": np.mean([np.abs(d_["determinant"]) for d_ in decomps]),
                "mean_condition": np.mean(
                    [d_["condition_number"] for d_ in decomps if np.isfinite(d_["condition_number"])]
                ),
                "eigenvalue_magnitudes": np.mean(
                    [np.abs(d_["eigenvalues"]) for d_ in decomps], axis=0
                ),
                "eigenvalue_phases": np.mean(
                    [np.angle(d_["eigenvalues"]) for d_ in decomps], axis=0
                ),
            }
            layer_stats.append(stats)

        return layer_stats

    def _analyze_holographic(self, hs):
        """Test holographic scrambling: ablate dimensions and measure impact."""
        n_layers, n_tokens, hidden_dim = hs.shape

        # For each layer, measure how ablating random subsets of dimensions
        # affects ALL tokens (holographic) vs. specific tokens (local)
        ablation_fractions = [0.1, 0.25, 0.5]
        results = {"ablation_fractions": ablation_fractions, "per_layer": []}

        # Use middle layer for analysis
        test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        test_layers = [l for l in test_layers if l < n_layers]

        for ell in test_layers:
            layer_result = {"layer": ell, "ablation_results": []}
            h = hs[ell]  # [n_tokens, hidden_dim]

            # Compute pairwise distances (baseline)
            baseline_dists = pdist(h)

            for frac in ablation_fractions:
                n_ablate = max(1, int(frac * hidden_dim))
                # Random ablation (average over multiple trials)
                n_trials = 10
                dist_changes = []

                for _ in range(n_trials):
                    dims_to_ablate = np.random.choice(hidden_dim, n_ablate, replace=False)
                    h_ablated = h.copy()
                    h_ablated[:, dims_to_ablate] = 0

                    ablated_dists = pdist(h_ablated)
                    # Relative change in pairwise distances
                    rel_change = np.abs(ablated_dists - baseline_dists) / (
                        np.abs(baseline_dists) + 1e-10
                    )
                    dist_changes.append(rel_change.mean())

                layer_result["ablation_results"].append(
                    {
                        "fraction": frac,
                        "mean_distance_change": np.mean(dist_changes),
                        "std_distance_change": np.std(dist_changes),
                    }
                )

            results["per_layer"].append(layer_result)

        return results

    def _analyze_topology(self, hs_pca):
        """Compute persistent homology at each layer."""
        n_layers = hs_pca.shape[0]
        diagrams_per_layer = []

        # Subsample layers for efficiency
        layer_indices = np.linspace(0, n_layers - 1, min(n_layers, 13), dtype=int)
        layer_indices = np.unique(layer_indices)

        for ell in layer_indices:
            try:
                dgms = compute_persistent_homology(hs_pca[ell], max_dim=1)
                diagrams_per_layer.append({"layer": int(ell), "diagrams": dgms})
            except Exception as e:
                diagrams_per_layer.append({"layer": int(ell), "diagrams": None, "error": str(e)})

        # Compute Wasserstein distances between consecutive layers
        wasserstein_dists = []
        for i in range(len(diagrams_per_layer) - 1):
            d1 = diagrams_per_layer[i]
            d2 = diagrams_per_layer[i + 1]
            if d1["diagrams"] is not None and d2["diagrams"] is not None:
                total_dist = 0
                for dim in range(min(len(d1["diagrams"]), len(d2["diagrams"]))):
                    total_dist += wasserstein_distance_diagrams(
                        d1["diagrams"][dim], d2["diagrams"][dim]
                    )
                wasserstein_dists.append(
                    {
                        "from_layer": d1["layer"],
                        "to_layer": d2["layer"],
                        "distance": total_dist,
                    }
                )

        return {
            "diagrams_per_layer": diagrams_per_layer,
            "wasserstein_distances": wasserstein_dists,
        }

    def _analyze_layer_deformations(self, hs, hs_pca):
        """Measure deformation magnitude at each layer."""
        n_layers = hs.shape[0]

        deformations = []
        for ell in range(n_layers - 1):
            delta = hs[ell + 1] - hs[ell]  # [n_tokens, hidden_dim]
            delta_norms = np.linalg.norm(delta, axis=1)

            # Also measure in PCA space
            delta_pca = hs_pca[ell + 1] - hs_pca[ell]
            delta_pca_norms = np.linalg.norm(delta_pca, axis=1)

            # Cosine similarity between consecutive representations
            cos_sims = []
            for i in range(hs.shape[1]):
                norm_a = np.linalg.norm(hs[ell, i])
                norm_b = np.linalg.norm(hs[ell + 1, i])
                if norm_a > 1e-10 and norm_b > 1e-10:
                    cos_sims.append(np.dot(hs[ell, i], hs[ell + 1, i]) / (norm_a * norm_b))

            deformations.append(
                {
                    "layer": ell,
                    "mean_delta_norm": delta_norms.mean(),
                    "max_delta_norm": delta_norms.max(),
                    "mean_delta_pca_norm": delta_pca_norms.mean(),
                    "mean_cosine_sim": np.mean(cos_sims) if cos_sims else 0,
                    "per_token_delta": delta_norms.tolist(),
                }
            )

        # Check reversal: does the last layer undo deformation?
        # Compare embedding-layer geometry to final-layer geometry
        first_dists = pdist(hs[0])
        last_dists = pdist(hs[-1])
        middle_idx = n_layers // 2
        middle_dists = pdist(hs[middle_idx])

        corr_first_last, _ = pearsonr(first_dists, last_dists) if len(first_dists) > 2 else (0, 1)
        corr_first_middle, _ = pearsonr(first_dists, middle_dists) if len(first_dists) > 2 else (0, 1)

        return {
            "per_layer": deformations,
            "geometry_correlation_first_last": corr_first_last,
            "geometry_correlation_first_middle": corr_first_middle,
        }

    def _analyze_eigenvalue_spectra(self, hs_pca):
        """Analyze eigenvalue spectra of Jacobians across layers."""
        n_layers, n_tokens, d = hs_pca.shape
        spectra = []

        for ell in range(n_layers - 1):
            before = hs_pca[ell]
            after = hs_pca[ell + 1]

            jacobians = estimate_jacobian_from_cloud(before, after)

            all_eigs = []
            for J in jacobians:
                eigs = np.linalg.eigvals(J)
                all_eigs.append(eigs)

            all_eigs = np.array(all_eigs)

            spectra.append(
                {
                    "layer": ell,
                    "mean_magnitudes": np.mean(np.abs(all_eigs), axis=0),
                    "mean_phases": np.mean(np.angle(all_eigs), axis=0),
                    "spectral_radius": np.max(np.abs(all_eigs)),
                    "mean_spectral_gap": np.mean(
                        np.sort(np.abs(all_eigs), axis=1)[:, -1]
                        - np.sort(np.abs(all_eigs), axis=1)[:, -2]
                    )
                    if d > 1
                    else 0,
                }
            )

        return spectra

    def _analyze_curvature(self, hs_pca):
        """Compute curvature measures across layers."""
        n_layers = hs_pca.shape[0]
        curvature_results = []

        layer_indices = np.linspace(0, n_layers - 1, min(n_layers, 13), dtype=int)
        layer_indices = np.unique(layer_indices)

        for ell in layer_indices:
            points = hs_pca[ell]
            layer_curv = {"layer": int(ell)}

            # Ollivier-Ricci curvature
            try:
                mean_orc, std_orc, orc_vals = compute_ollivier_ricci_curvature(points, k=min(5, points.shape[0] - 1))
                layer_curv["orc_mean"] = mean_orc
                layer_curv["orc_std"] = std_orc
                layer_curv["orc_values"] = orc_vals
            except Exception as e:
                layer_curv["orc_mean"] = 0
                layer_curv["orc_std"] = 0
                layer_curv["orc_error"] = str(e)

            curvature_results.append(layer_curv)

        # Volume change between consecutive layers
        volume_changes = []
        for ell in range(hs_pca.shape[0] - 1):
            vc = compute_volume_change(hs_pca[ell], hs_pca[ell + 1], k=min(5, hs_pca.shape[1] - 1))
            volume_changes.append({"layer": ell, "mean_volume_change": vc.mean(), "std_volume_change": vc.std()})

        # Procrustes deviation
        procrustes_devs = []
        for ell in range(hs_pca.shape[0] - 1):
            try:
                dev, var_before, var_after = compute_procrustes_deviation(hs_pca[ell], hs_pca[ell + 1])
                procrustes_devs.append({"layer": ell, "deviation": dev})
            except Exception:
                procrustes_devs.append({"layer": ell, "deviation": 0})

        return {
            "per_layer_orc": curvature_results,
            "volume_changes": volume_changes,
            "procrustes_deviations": procrustes_devs,
        }

    # ============================================================
    # Visualization
    # ============================================================

    def generate_all_visualizations(self):
        """Generate all visualization figures for all analyzed prompts."""
        print(f"\n{'='*60}")
        print(f"Generating visualizations for {self.model_name}")
        print(f"{'='*60}")

        for pid, result in self.results.items():
            prefix = self.output_dir / f"prompt{pid}"
            self._plot_layer_deformation_profile(result, prefix)
            self._plot_jacobian_decomposition(result, prefix)
            self._plot_eigenvalue_spectra(result, prefix)
            self._plot_holographic_ablation(result, prefix)
            self._plot_topology(result, prefix)
            self._plot_curvature(result, prefix)
            self._plot_fibre_view(result, prefix)
            self._plot_pca_trajectories(result, prefix)
            self._plot_encoding_search(result, prefix)

        self._plot_cross_prompt_comparison()
        print(f"  All visualizations saved to: {self.output_dir}/")

    def _plot_layer_deformation_profile(self, result, prefix):
        """
        Conjecture 4: Inner layers compute, outer layers translate.
        Plot deformation magnitude across layers.
        """
        deformations = result["layer_deformation"]["per_layer"]
        layers = [d["layer"] for d in deformations]
        mean_deltas = [d["mean_delta_norm"] for d in deformations]
        cos_sims = [d["mean_cosine_sim"] for d in deformations]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Deformation magnitude
        ax = axes[0]
        bars = ax.bar(layers, mean_deltas, color="steelblue", alpha=0.8, edgecolor="navy")
        # Highlight middle layers
        n = len(layers)
        for i, bar in enumerate(bars):
            if n // 4 <= i <= 3 * n // 4:
                bar.set_color("firebrick")
                bar.set_alpha(0.8)
        ax.set_ylabel("Mean ‖Δh‖ (deformation magnitude)", fontsize=11)
        ax.set_title(
            f"Conjecture 4 Test: Layer Deformation Profile\n"
            f"Model: {self.model_name} | Red = middle layers (expected: highest deformation)",
            fontsize=12,
        )
        ax.grid(axis="y", alpha=0.3)

        # Cosine similarity
        ax = axes[1]
        ax.plot(layers, cos_sims, "o-", color="darkgreen", linewidth=2, markersize=5)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Identity (no change)")
        ax.set_ylabel("Mean cosine similarity (layer ℓ → ℓ+1)", fontsize=11)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylim(min(0.8, min(cos_sims) - 0.02), 1.01)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{prefix}_deformation_profile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_jacobian_decomposition(self, result, prefix):
        """
        Conjecture 1: Space-morphing. Plot Jacobian decomposition across layers.
        """
        jac_stats = result["jacobian_analysis"]
        layers = [s["layer"] for s in jac_stats]
        divs = [s["mean_divergence"] for s in jac_stats]
        curls = [s["mean_curl"] for s in jac_stats]
        shears = [s["mean_shear"] for s in jac_stats]
        dets = [s["mean_det"] for s in jac_stats]
        conds = [s["mean_condition"] for s in jac_stats]

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # Divergence
        ax = axes[0, 0]
        colors = ["firebrick" if d > 0 else "steelblue" for d in divs]
        ax.bar(layers, divs, color=colors, alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Divergence (tr J)")
        ax.set_title("Divergence: expansion (+) / contraction (−)")
        ax.grid(axis="y", alpha=0.3)

        # Curl
        ax = axes[0, 1]
        ax.bar(layers, curls, color="purple", alpha=0.7)
        ax.set_ylabel("‖Curl‖ (antisym part)")
        ax.set_title("Curl: rotational mixing between dimensions")
        ax.grid(axis="y", alpha=0.3)

        # Shear
        ax = axes[1, 0]
        ax.bar(layers, shears, color="darkorange", alpha=0.7)
        ax.set_ylabel("‖Shear‖ (traceless sym part)")
        ax.set_title("Shear: anisotropic distortion")
        ax.grid(axis="y", alpha=0.3)

        # Determinant
        ax = axes[1, 1]
        ax.bar(layers, dets, color="teal", alpha=0.7)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="|det|=1 (volume-preserving)")
        ax.set_ylabel("|det(J)|")
        ax.set_title("Determinant: signed volume change")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Condition number
        ax = axes[2, 0]
        ax.bar(layers, conds, color="brown", alpha=0.7)
        ax.set_ylabel("Condition number (σ₁/σ_d)")
        ax.set_title("Condition number: anisotropy of transformation")
        ax.grid(axis="y", alpha=0.3)

        # Combined normalized
        ax = axes[2, 1]
        max_div = max(np.abs(divs)) + 1e-10
        max_curl = max(curls) + 1e-10
        max_shear = max(shears) + 1e-10
        ax.plot(layers, np.array(np.abs(divs)) / max_div, "o-", label="|Div| (norm)", linewidth=2)
        ax.plot(layers, np.array(curls) / max_curl, "s-", label="Curl (norm)", linewidth=2)
        ax.plot(layers, np.array(shears) / max_shear, "^-", label="Shear (norm)", linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized magnitude")
        ax.set_title("Combined Jacobian decomposition (normalized)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Conjecture 1: Jacobian Field Decomposition — {self.model_name}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(f"{prefix}_jacobian_decomposition.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_eigenvalue_spectra(self, result, prefix):
        """
        Conjecture 5: Jacobian as holographic carrier.
        Plot eigenvalue magnitude and phase spectra across layers.
        """
        spectra = result["eigenvalue_analysis"]
        n_layers = len(spectra)

        # Build heatmap of eigenvalue magnitudes
        n_eigs = len(spectra[0]["mean_magnitudes"])
        mag_matrix = np.zeros((n_layers, n_eigs))
        phase_matrix = np.zeros((n_layers, n_eigs))

        for i, s in enumerate(spectra):
            mag_matrix[i] = s["mean_magnitudes"][:n_eigs]
            phase_matrix[i] = s["mean_phases"][:n_eigs]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Eigenvalue magnitudes heatmap
        ax = axes[0]
        im = ax.imshow(mag_matrix.T, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Eigenvalue index")
        ax.set_title("Eigenvalue magnitudes |λ_k|")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Eigenvalue phases heatmap
        ax = axes[1]
        im = ax.imshow(phase_matrix.T, aspect="auto", cmap="hsv", interpolation="nearest")
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Eigenvalue index")
        ax.set_title("Eigenvalue phases arg(λ_k)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Spectral radius across layers
        ax = axes[2]
        spectral_radii = [s["spectral_radius"] for s in spectra]
        spectral_gaps = [s["mean_spectral_gap"] for s in spectra]
        layers = [s["layer"] for s in spectra]
        ax.plot(layers, spectral_radii, "o-", color="firebrick", linewidth=2, label="Spectral radius")
        ax.plot(layers, spectral_gaps, "s-", color="steelblue", linewidth=2, label="Mean spectral gap")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Value")
        ax.set_title("Spectral radius & gap")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Conjecture 5: Eigenvalue Spectra — {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(f"{prefix}_eigenvalue_spectra.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_holographic_ablation(self, result, prefix):
        """
        Conjecture 2: Holographic scrambling.
        Plot ablation impact across layers and fractions.
        """
        holo = result["holographic_analysis"]
        per_layer = holo["per_layer"]

        fig, ax = plt.subplots(figsize=(10, 6))

        fractions = holo["ablation_fractions"]
        for frac_idx, frac in enumerate(fractions):
            layers = [pl["layer"] for pl in per_layer]
            means = [pl["ablation_results"][frac_idx]["mean_distance_change"] for pl in per_layer]
            stds = [pl["ablation_results"][frac_idx]["std_distance_change"] for pl in per_layer]
            ax.errorbar(
                layers,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=4,
                label=f"Ablate {int(frac*100)}% dims",
            )

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Mean relative distance change", fontsize=11)
        ax.set_title(
            f"Conjecture 2: Holographic Scrambling Test — {self.model_name}\n"
            f"(Holographic: ablation degrades uniformly, not selectively)",
            fontsize=12,
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Add interpretation
        ax.text(
            0.02,
            0.98,
            "If holographic: distance change ∝ ablation fraction\n"
            "(not concentrated in specific layers)",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        fig.savefig(f"{prefix}_holographic_ablation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_topology(self, result, prefix):
        """
        Conjecture 3: Topological computation.
        Plot persistence diagrams and Wasserstein distances.
        """
        topo = result["topology_analysis"]
        diagrams_per_layer = topo["diagrams_per_layer"]
        wass_dists = topo["wasserstein_distances"]

        # Plot persistence diagrams for selected layers
        valid_layers = [d for d in diagrams_per_layer if d["diagrams"] is not None]
        n_show = min(6, len(valid_layers))
        if n_show == 0:
            return

        fig, axes = plt.subplots(2, max(n_show, 1), figsize=(4 * max(n_show, 1), 8))
        if n_show == 1:
            axes = axes.reshape(2, 1)

        # Top row: persistence diagrams
        show_indices = np.linspace(0, len(valid_layers) - 1, n_show, dtype=int)
        for idx, si in enumerate(show_indices):
            ax = axes[0, idx]
            layer_data = valid_layers[si]
            dgms = layer_data["diagrams"]

            colors = ["blue", "red", "green"]
            labels = ["H₀ (components)", "H₁ (loops)", "H₂ (voids)"]

            for dim_idx, dgm in enumerate(dgms):
                if len(dgm) > 0:
                    finite_mask = np.isfinite(dgm[:, 1])
                    finite_pts = dgm[finite_mask]
                    if len(finite_pts) > 0:
                        ax.scatter(
                            finite_pts[:, 0],
                            finite_pts[:, 1],
                            c=colors[dim_idx],
                            s=20,
                            alpha=0.6,
                            label=labels[dim_idx],
                        )

            # Diagonal
            lims = ax.get_xlim()
            ax.plot([0, max(lims[1], 2)], [0, max(lims[1], 2)], "k--", alpha=0.3)
            ax.set_title(f"Layer {layer_data['layer']}", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Death")
                ax.legend(fontsize=7, loc="lower right")
            ax.set_xlabel("Birth")

        # Bottom row: Wasserstein distances
        if wass_dists:
            ax = axes[1, 0] if n_show > 1 else axes[1, 0]
            from_layers = [w["from_layer"] for w in wass_dists]
            distances = [w["distance"] for w in wass_dists]
            ax.bar(range(len(distances)), distances, color="darkviolet", alpha=0.7)
            ax.set_xticks(range(len(distances)))
            ax.set_xticklabels([f"{w['from_layer']}→{w['to_layer']}" for w in wass_dists], rotation=45, fontsize=7)
            ax.set_ylabel("Wasserstein distance")
            ax.set_title("Topological change between layers", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

        # Hide unused axes
        for idx in range(1, n_show):
            axes[1, idx].set_visible(False)

        fig.suptitle(
            f"Conjecture 3: Topological Computation — {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(f"{prefix}_topology.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_curvature(self, result, prefix):
        """Plot curvature measures across layers."""
        curv = result["curvature_analysis"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # ORC
        ax = axes[0]
        orc_layers = [c["layer"] for c in curv["per_layer_orc"]]
        orc_means = [c.get("orc_mean", 0) for c in curv["per_layer_orc"]]
        orc_stds = [c.get("orc_std", 0) for c in curv["per_layer_orc"]]
        ax.errorbar(orc_layers, orc_means, yerr=orc_stds, marker="o", linewidth=2, capsize=4, color="teal")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Ollivier-Ricci Curvature")
        ax.set_title("ORC: + = convergence, − = divergence")
        ax.grid(alpha=0.3)

        # Volume change
        ax = axes[1]
        vc = curv["volume_changes"]
        vc_layers = [v["layer"] for v in vc]
        vc_means = [v["mean_volume_change"] for v in vc]
        colors = ["firebrick" if v > 0 else "steelblue" for v in vc_means]
        ax.bar(vc_layers, vc_means, color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Mean Δlog(Volume)")
        ax.set_title("Volume change: red=expand, blue=contract")
        ax.grid(axis="y", alpha=0.3)

        # Procrustes deviation
        ax = axes[2]
        pd = curv["procrustes_deviations"]
        pd_layers = [p["layer"] for p in pd]
        pd_devs = [p["deviation"] for p in pd]
        ax.plot(pd_layers, pd_devs, "o-", color="darkred", linewidth=2, markersize=5)
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Procrustes deviation ‖R−I‖_F")
        ax.set_title("Connection strength (frame rotation)")
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Curvature Analysis — {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(f"{prefix}_curvature.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_fibre_view(self, result, prefix):
        """
        Fibre view: show how each token's representation evolves across layers
        in 2D PCA projection.
        """
        hs = result["hidden_states"]
        tokens = result["tokens"]
        n_layers, n_tokens, hidden_dim = hs.shape

        # Global PCA to 2D
        all_flat = hs.reshape(-1, hidden_dim)
        pca2d = PCA(n_components=2)
        projected = pca2d.fit_transform(all_flat).reshape(n_layers, n_tokens, 2)

        fig, ax = plt.subplots(figsize=(12, 8))

        cmap = cm.get_cmap("tab20", n_tokens)

        for tok_idx in range(n_tokens):
            trajectory = projected[:, tok_idx, :]
            color = cmap(tok_idx)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                "-",
                color=color,
                linewidth=1.5,
                alpha=0.7,
            )
            # Mark start and end
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color=color, marker="o", s=60, zorder=5)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, marker="*", s=100, zorder=5)

            # Label at midpoint
            mid = n_layers // 2
            label = tokens[tok_idx] if tok_idx < len(tokens) else f"t{tok_idx}"
            ax.annotate(
                label,
                (trajectory[mid, 0], trajectory[mid, 1]),
                fontsize=7,
                alpha=0.8,
                color=color,
            )

        ax.set_xlabel(f"PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
        ax.set_title(
            f"Fibre View: Token Trajectories Across Layers — {self.model_name}\n"
            f"○ = embedding layer, ★ = final layer",
            fontsize=12,
        )
        ax.grid(alpha=0.2)

        plt.tight_layout()
        fig.savefig(f"{prefix}_fibre_view.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_pca_trajectories(self, result, prefix):
        """
        Plot PCA variance explained across layers — tests whether
        information becomes more distributed (holographic) in middle layers.
        """
        hs = result["hidden_states"]
        n_layers, n_tokens, hidden_dim = hs.shape

        n_components = min(20, hidden_dim, n_tokens - 1)
        variance_ratios = []

        for ell in range(n_layers):
            pca = PCA(n_components=n_components)
            pca.fit(hs[ell])
            variance_ratios.append(pca.explained_variance_ratio_)

        variance_ratios = np.array(variance_ratios)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Heatmap
        ax = axes[0]
        im = ax.imshow(variance_ratios.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Layer")
        ax.set_ylabel("PC index")
        ax.set_title("PCA variance explained per layer")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Effective dimensionality (participation ratio)
        ax = axes[1]
        eff_dims = []
        for ell in range(n_layers):
            vr = variance_ratios[ell]
            # Participation ratio: (Σ p_i)^2 / Σ p_i^2
            pr = (np.sum(vr)) ** 2 / (np.sum(vr**2) + 1e-12)
            eff_dims.append(pr)

        ax.plot(range(n_layers), eff_dims, "o-", color="darkgreen", linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Effective dimensionality (participation ratio)")
        ax.set_title("Information distribution across dimensions")
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Holographic Encoding Analysis — {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(f"{prefix}_pca_trajectories.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_encoding_search(self, result, prefix):
        """
        Try to find the encoding itself: look for structure in the
        Jacobian eigenvalue spectra that might reveal the internal
        representation format used in later layers.

        This searches for:
        1. Dominant eigenvalue directions that persist across layers
        2. Phase structure in eigenvalues (rotational modes)
        3. Low-rank structure in the Jacobian (encoding subspaces)
        """
        spectra = result["eigenvalue_analysis"]
        n_layers = len(spectra)

        if n_layers < 3:
            return

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig)

        # 1. Eigenvalue magnitude evolution (top-k eigenvalues across layers)
        ax = fig.add_subplot(gs[0, 0])
        n_eigs = len(spectra[0]["mean_magnitudes"])
        top_k = min(10, n_eigs)
        for k in range(top_k):
            mags = [s["mean_magnitudes"][k] for s in spectra]
            ax.plot(range(n_layers), mags, linewidth=1.5, alpha=0.7, label=f"λ_{k}")
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Eigenvalue magnitude")
        ax.set_title("Top eigenvalue magnitudes across layers")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        # 2. Eigenvalue complex plane at different layers
        ax = fig.add_subplot(gs[0, 1])
        show_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        show_layers = [l for l in show_layers if l < n_layers]
        cmap_layers = cm.get_cmap("coolwarm", len(show_layers))

        for idx, ell in enumerate(show_layers):
            mags = spectra[ell]["mean_magnitudes"]
            phases = spectra[ell]["mean_phases"]
            x = mags * np.cos(phases)
            y = mags * np.sin(phases)
            ax.scatter(x, y, c=[cmap_layers(idx)] * len(x), s=30, alpha=0.7, label=f"Layer {ell}")

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title("Eigenvalues in complex plane")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        # 3. Spectral entropy across layers (encoding complexity)
        ax = fig.add_subplot(gs[1, 0])
        spectral_entropies = []
        for s in spectra:
            mags = s["mean_magnitudes"]
            mags_norm = mags / (mags.sum() + 1e-12)
            entropy = -np.sum(mags_norm * np.log(mags_norm + 1e-12))
            spectral_entropies.append(entropy)

        ax.plot(range(n_layers), spectral_entropies, "o-", color="darkviolet", linewidth=2)
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Spectral entropy")
        ax.set_title("Encoding complexity (spectral entropy of Jacobian)")
        ax.grid(alpha=0.3)

        # 4. Rank profile: how many eigenvalues are "significant"
        ax = fig.add_subplot(gs[1, 1])
        thresholds = [0.1, 0.5, 1.0]
        for thresh in thresholds:
            ranks = []
            for s in spectra:
                mags = s["mean_magnitudes"]
                rank = np.sum(mags > thresh)
                ranks.append(rank)
            ax.plot(range(n_layers), ranks, "o-", linewidth=2, label=f"|λ| > {thresh}")

        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Number of significant eigenvalues")
        ax.set_title("Effective rank of Jacobian (encoding dimensionality)")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Encoding Search: Internal Representation Structure — {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(f"{prefix}_encoding_search.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_cross_prompt_comparison(self):
        """Compare deformation profiles across different prompts."""
        if len(self.results) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Deformation magnitude comparison
        ax = axes[0, 0]
        for pid, result in self.results.items():
            deformations = result["layer_deformation"]["per_layer"]
            layers = [d["layer"] for d in deformations]
            deltas = [d["mean_delta_norm"] for d in deformations]
            label = result["prompt"][:40] + "..."
            ax.plot(layers, deltas, "o-", linewidth=2, markersize=4, label=f"P{pid}: {label}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean ‖Δh‖")
        ax.set_title("Deformation magnitude across prompts")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # 2. Cosine similarity comparison
        ax = axes[0, 1]
        for pid, result in self.results.items():
            deformations = result["layer_deformation"]["per_layer"]
            layers = [d["layer"] for d in deformations]
            cos_sims = [d["mean_cosine_sim"] for d in deformations]
            ax.plot(layers, cos_sims, "o-", linewidth=2, markersize=4, label=f"P{pid}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title("Layer-to-layer cosine similarity")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # 3. Geometry correlation (first-last vs first-middle)
        ax = axes[1, 0]
        pids = list(self.results.keys())
        fl_corrs = [self.results[p]["layer_deformation"]["geometry_correlation_first_last"] for p in pids]
        fm_corrs = [self.results[p]["layer_deformation"]["geometry_correlation_first_middle"] for p in pids]
        x = np.arange(len(pids))
        width = 0.35
        ax.bar(x - width / 2, fl_corrs, width, label="First↔Last", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, fm_corrs, width, label="First↔Middle", color="firebrick", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{p}" for p in pids])
        ax.set_ylabel("Pearson correlation of pairwise distances")
        ax.set_title("Conjecture 4: Does last layer return to embedding geometry?")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # 4. Topological change (Wasserstein distances)
        ax = axes[1, 1]
        for pid, result in self.results.items():
            wass = result["topology_analysis"]["wasserstein_distances"]
            if wass:
                layers = [w["from_layer"] for w in wass]
                dists = [w["distance"] for w in wass]
                ax.plot(layers, dists, "o-", linewidth=2, markersize=4, label=f"P{pid}")
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Wasserstein distance (topology change)")
        ax.set_title("Conjecture 3: Topological phase transitions")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Cross-Prompt Comparison — {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(self.output_dir / "cross_prompt_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ============================================================
    # Human-Readable Report
    # ============================================================

    def generate_report(self):
        """Generate a comprehensive human-readable report."""
        report_lines = []
        r = report_lines.append

        r("=" * 72)
        r(f"FIBRE BUNDLE ANALYSIS REPORT")
        r(f"Model: {self.model_name}")
        r(f"Parameters: {self.model_info['params'] / 1e6:.1f}M")
        r(f"Layers: {self.model_info['layers']}")
        r(f"Hidden dim: {self.model_info['hidden_dim']}")
        r(f"Prompts analyzed: {len(self.results)}")
        r(f"Output directory: {self.output_dir}")
        r("=" * 72)

        for pid, result in self.results.items():
            r(f"\n{'─'*72}")
            r(f"PROMPT {pid}: \"{result['prompt'][:80]}\"")
            r(f"Tokens: {result['n_tokens']} | Layers: {result['n_layers']} | Hidden dim: {result['hidden_dim']}")
            r(f"{'─'*72}")

            # --- Conjecture 1: Space-morphing ---
            r(f"\n  ┌─ CONJECTURE 1: Space-morphing (Jacobian analysis)")
            jac = result["jacobian_analysis"]
            divs = [s["mean_divergence"] for s in jac]
            curls = [s["mean_curl"] for s in jac]
            shears = [s["mean_shear"] for s in jac]

            max_div_layer = np.argmax(np.abs(divs))
            max_curl_layer = np.argmax(curls)
            max_shear_layer = np.argmax(shears)

            r(f"  │ Peak divergence at layer {max_div_layer}: {divs[max_div_layer]:.4f}")
            r(f"  │   (positive = expansion, negative = contraction)")
            r(f"  │ Peak curl at layer {max_curl_layer}: {curls[max_curl_layer]:.4f}")
            r(f"  │   (rotational mixing between dimensions)")
            r(f"  │ Peak shear at layer {max_shear_layer}: {shears[max_shear_layer]:.4f}")
            r(f"  │   (anisotropic distortion)")

            # Check if middle layers dominate
            n = len(divs)
            middle_range = range(n // 4, 3 * n // 4)
            outer_range = list(range(0, n // 4)) + list(range(3 * n // 4, n))

            middle_activity = np.mean([np.abs(divs[i]) + curls[i] + shears[i] for i in middle_range])
            outer_activity = np.mean([np.abs(divs[i]) + curls[i] + shears[i] for i in outer_range]) if outer_range else 0

            if middle_activity > outer_activity * 1.2:
                r(f"  │ ✓ Middle layers show {middle_activity/outer_activity:.1f}× more geometric activity")
                r(f"  │   SUPPORTS Conjecture 1: space-morphing is structured, not uniform")
            else:
                r(f"  │ ○ Middle/outer activity ratio: {middle_activity/(outer_activity+1e-10):.2f}")
                r(f"  │   Geometric activity is relatively uniform across layers")
            r(f"  └─")

            # --- Conjecture 2: Holographic scrambling ---
            r(f"\n  ┌─ CONJECTURE 2: Holographic scrambling")
            holo = result["holographic_analysis"]
            for pl in holo["per_layer"]:
                layer = pl["layer"]
                ablation_results = pl["ablation_results"]
                # Check proportionality: does 2× ablation → ~2× degradation?
                if len(ablation_results) >= 2:
                    frac1 = ablation_results[0]["fraction"]
                    frac2 = ablation_results[1]["fraction"]
                    change1 = ablation_results[0]["mean_distance_change"]
                    change2 = ablation_results[1]["mean_distance_change"]
                    ratio_frac = frac2 / frac1
                    ratio_change = change2 / (change1 + 1e-10)
                    r(f"  │ Layer {layer}: ablating {frac1*100:.0f}%→{frac2*100:.0f}% dims "
                      f"(ratio {ratio_frac:.1f}×) → distance change ratio {ratio_change:.2f}×")

            # Overall assessment
            all_changes = []
            for pl in holo["per_layer"]:
                for ar in pl["ablation_results"]:
                    all_changes.append(ar["mean_distance_change"])

            if all_changes:
                mean_change = np.mean(all_changes)
                std_change = np.std(all_changes)
                cv = std_change / (mean_change + 1e-10)
                if cv < 0.5:
                    r(f"  │ ✓ Ablation impact is relatively uniform (CV={cv:.2f})")
                    r(f"  │   SUPPORTS Conjecture 2: information is holographically distributed")
                else:
                    r(f"  │ ○ Ablation impact varies substantially (CV={cv:.2f})")
                    r(f"  │   Some layers may store information more locally")
            r(f"  └─")

            # --- Conjecture 3: Topological computation ---
            r(f"\n  ┌─ CONJECTURE 3: Topological computation")
            topo = result["topology_analysis"]
            wass = topo["wasserstein_distances"]
            if wass:
                dists = [w["distance"] for w in wass]
                max_wass_idx = np.argmax(dists)
                max_wass = wass[max_wass_idx]
                r(f"  │ Largest topological change: layer {max_wass['from_layer']}→{max_wass['to_layer']} "
                  f"(Wasserstein dist = {max_wass['distance']:.4f})")
                r(f"  │ Mean topological change: {np.mean(dists):.4f} ± {np.std(dists):.4f}")

                # Check for topological phase transitions (spikes)
                if len(dists) > 2:
                    mean_d = np.mean(dists)
                    std_d = np.std(dists)
                    spikes = [i for i, d in enumerate(dists) if d > mean_d + 2 * std_d]
                    if spikes:
                        spike_layers = [wass[i]["from_layer"] for i in spikes]
                        r(f"  │ ✓ Topological phase transitions detected at layers: {spike_layers}")
                        r(f"  │   SUPPORTS Conjecture 3: topology changes discretely, not smoothly")
                    else:
                        r(f"  │ ○ No sharp topological phase transitions detected")
                        r(f"  │   Topology changes gradually across layers")

            # Check persistence diagrams
            for dl in topo["diagrams_per_layer"]:
                if dl["diagrams"] is not None:
                    for dim_idx, dgm in enumerate(dl["diagrams"]):
                        if len(dgm) > 0:
                            finite_mask = np.isfinite(dgm[:, 1])
                            finite_pts = dgm[finite_mask]
                            if len(finite_pts) > 0:
                                lifetimes = finite_pts[:, 1] - finite_pts[:, 0]
                                long_lived = np.sum(lifetimes > np.median(lifetimes) * 2)
                                if long_lived > 0 and dim_idx > 0:
                                    r(f"  │ Layer {dl['layer']}: {long_lived} persistent H{dim_idx} features "
                                      f"(long-lived topological structures)")
                    break  # Just show first valid layer as example
            r(f"  └─")

            # --- Conjecture 4: Inner layers compute, outer translate ---
            r(f"\n  ┌─ CONJECTURE 4: Inner layers compute, outer layers translate")
            deform = result["layer_deformation"]
            per_layer = deform["per_layer"]
            deltas = [d["mean_delta_norm"] for d in per_layer]

            n = len(deltas)
            if n >= 4:
                first_quarter = np.mean(deltas[: n // 4])
                middle_half = np.mean(deltas[n // 4 : 3 * n // 4])
                last_quarter = np.mean(deltas[3 * n // 4 :])

                r(f"  │ Mean deformation — first quarter: {first_quarter:.4f}")
                r(f"  │ Mean deformation — middle half:   {middle_half:.4f}")
                r(f"  │ Mean deformation — last quarter:  {last_quarter:.4f}")

                if middle_half > first_quarter and middle_half > last_quarter:
                    r(f"  │ ✓ Middle layers show highest deformation ({middle_half/first_quarter:.1f}× first, "
                      f"{middle_half/last_quarter:.1f}× last)")
                    r(f"  │   SUPPORTS Conjecture 4: inner layers compute, outer layers translate")
                else:
                    r(f"  │ ○ Deformation pattern does not clearly peak in middle layers")

            fl_corr = deform["geometry_correlation_first_last"]
            fm_corr = deform["geometry_correlation_first_middle"]
            r(f"  │ Geometry correlation (first↔last):   {fl_corr:.4f}")
            r(f"  │ Geometry correlation (first↔middle): {fm_corr:.4f}")

            if fl_corr > fm_corr:
                r(f"  │ ✓ Last layer geometry is MORE similar to embedding than middle layer")
                r(f"  │   SUPPORTS Conjecture 4: last layer reverses toward embedding geometry")
            else:
                r(f"  │ ○ Last layer does not clearly reverse toward embedding geometry")
            r(f"  └─")

            # --- Conjecture 5: Jacobian as holographic carrier ---
            r(f"\n  ┌─ CONJECTURE 5: Jacobian eigenvalue spectra")
            eig = result["eigenvalue_analysis"]
            if eig:
                spectral_radii = [s["spectral_radius"] for s in eig]
                max_sr_layer = np.argmax(spectral_radii)
                r(f"  │ Peak spectral radius at layer {max_sr_layer}: {spectral_radii[max_sr_layer]:.4f}")
                r(f"  │ Mean spectral radius: {np.mean(spectral_radii):.4f} ± {np.std(spectral_radii):.4f}")

                # Check for complex eigenvalues (rotational modes)
                has_complex = False
                max_phase_layer = 0
                max_phase_val = 0
                for s in eig:
                    phases = s["mean_phases"]
                    max_abs_phase = np.max(np.abs(phases))
                    if max_abs_phase > 0.1:
                        has_complex = True
                    if max_abs_phase > max_phase_val:
                        max_phase_val = max_abs_phase
                        max_phase_layer = s["layer"]

                if has_complex:
                    r(f"  │ ✓ Complex eigenvalues detected (non-trivial phases)")
                    r(f"  │   Peak phase magnitude at layer {max_phase_layer}: {max_phase_val:.4f} rad")
                    r(f"  │   Indicates rotational modes in the Jacobian — information")
                    r(f"  │   is encoded in phase relationships, not just magnitudes")
                    r(f"  │   SUPPORTS Conjecture 5: Jacobian carries holographic info")
                else:
                    r(f"  │ ○ Eigenvalues are predominantly real (max phase: {max_phase_val:.4f} rad)")
                    r(f"  │   Transformations are mostly scaling/reflection, little rotation")

                # Check spectral gap structure
                spectral_gaps = [s["mean_spectral_gap"] for s in eig]
                max_gap_layer = np.argmax(spectral_gaps)
                r(f"  │ Peak spectral gap at layer {max_gap_layer}: {spectral_gaps[max_gap_layer]:.4f}")
                r(f"  │   (large gap = dominant mode separates from rest = low-rank structure)")

                # Check if spectral radius correlates with deformation
                if "layer_deformation" in result:
                    deform_deltas = [d["mean_delta_norm"] for d in result["layer_deformation"]["per_layer"]]
                    min_len = min(len(spectral_radii), len(deform_deltas))
                    if min_len > 3:
                        corr, pval = pearsonr(spectral_radii[:min_len], deform_deltas[:min_len])
                        r(f"  │ Correlation(spectral_radius, deformation): r={corr:.3f}, p={pval:.4f}")
                        if corr > 0.5 and pval < 0.05:
                            r(f"  │ ✓ Spectral radius tracks deformation magnitude")
                            r(f"  │   The Jacobian eigenvalues predict how much space is morphed")
                        elif corr < -0.5 and pval < 0.05:
                            r(f"  │ ○ Spectral radius anti-correlates with deformation")
                        else:
                            r(f"  │ ○ No strong correlation between spectral radius and deformation")
            r(f"  └─")

            # --- Conjecture 6: Dynamic map generation ---
            r(f"\n  ┌─ CONJECTURE 6: Dynamic map generation (layer-to-layer coupling)")
            # Test: does the Jacobian structure at layer ℓ predict structure at ℓ+1?
            if len(result["jacobian_analysis"]) >= 4:
                jac = result["jacobian_analysis"]
                divs = [s["mean_divergence"] for s in jac]
                curls = [s["mean_curl"] for s in jac]
                shears = [s["mean_shear"] for s in jac]

                # Autocorrelation of divergence
                if len(divs) > 3:
                    divs_arr = np.array(divs)
                    autocorr_div, p_div = pearsonr(divs_arr[:-1], divs_arr[1:])
                    autocorr_curl, p_curl = pearsonr(np.array(curls[:-1]), np.array(curls[1:]))
                    autocorr_shear, p_shear = pearsonr(np.array(shears[:-1]), np.array(shears[1:]))

                    r(f"  │ Autocorrelation of Jacobian properties (layer ℓ → ℓ+1):")
                    r(f"  │   Divergence: r={autocorr_div:.3f} (p={p_div:.4f})")
                    r(f"  │   Curl:       r={autocorr_curl:.3f} (p={p_curl:.4f})")
                    r(f"  │   Shear:      r={autocorr_shear:.3f} (p={p_shear:.4f})")

                    strong_coupling = sum(1 for ac, p in [
                        (autocorr_div, p_div), (autocorr_curl, p_curl), (autocorr_shear, p_shear)
                    ] if abs(ac) > 0.4 and p < 0.1)

                    if strong_coupling >= 2:
                        r(f"  │ ✓ Strong layer-to-layer coupling detected ({strong_coupling}/3 properties)")
                        r(f"  │   SUPPORTS Conjecture 6: each layer's geometry shapes the next")
                        r(f"  │   The propagating deformation is structured, not independent")
                    elif strong_coupling == 1:
                        r(f"  │ ○ Weak layer-to-layer coupling ({strong_coupling}/3 properties)")
                        r(f"  │   Some evidence for dynamic map generation")
                    else:
                        r(f"  │ ○ No significant layer-to-layer coupling detected")
                        r(f"  │   Jacobian properties appear independent across layers")

                    # Check for wave-like patterns (oscillation)
                    # Compute sign changes in divergence
                    sign_changes = np.sum(np.diff(np.sign(divs_arr)) != 0)
                    max_possible = len(divs_arr) - 1
                    oscillation_ratio = sign_changes / max_possible if max_possible > 0 else 0

                    if oscillation_ratio > 0.5:
                        r(f"  │ ✓ Oscillatory divergence pattern detected ({sign_changes}/{max_possible} sign changes)")
                        r(f"  │   Consistent with 'wave' of space-morphing propagating through layers")
                    elif oscillation_ratio > 0.3:
                        r(f"  │ ○ Moderate oscillation in divergence ({sign_changes}/{max_possible} sign changes)")
                    else:
                        r(f"  │ ○ Divergence is mostly monotonic ({sign_changes}/{max_possible} sign changes)")
            else:
                r(f"  │ (Insufficient layers for coupling analysis)")
            r(f"  └─")

            # --- Curvature ---
            r(f"\n  ┌─ CURVATURE ANALYSIS")
            curv = result["curvature_analysis"]

            # ORC
            orc_data = curv["per_layer_orc"]
            orc_means = [c.get("orc_mean", 0) for c in orc_data]
            orc_layers = [c["layer"] for c in orc_data]
            if orc_means:
                max_orc_idx = np.argmax(np.abs(orc_means))
                r(f"  │ ORC range: [{min(orc_means):.4f}, {max(orc_means):.4f}]")
                r(f"  │ Peak |ORC| at layer {orc_layers[max_orc_idx]}: {orc_means[max_orc_idx]:.4f}")
                r(f"  │   (positive = tokens converge, negative = tokens diverge)")

                # Check if ORC peaks in middle layers (supports Conjecture 4)
                n_orc = len(orc_means)
                if n_orc >= 4:
                    mid_orc = np.mean([abs(orc_means[i]) for i in range(n_orc // 4, 3 * n_orc // 4)])
                    out_orc_indices = list(range(0, n_orc // 4)) + list(range(3 * n_orc // 4, n_orc))
                    out_orc = np.mean([abs(orc_means[i]) for i in out_orc_indices]) if out_orc_indices else 0
                    if mid_orc > out_orc * 1.2:
                        r(f"  │ ✓ Curvature peaks in middle layers ({mid_orc:.4f} vs {out_orc:.4f})")
                    else:
                        r(f"  │ ○ Curvature does not clearly peak in middle layers")

            # Volume changes
            vc = curv["volume_changes"]
            if vc:
                vc_means = [v["mean_volume_change"] for v in vc]
                expanding = sum(1 for v in vc_means if v > 0.01)
                contracting = sum(1 for v in vc_means if v < -0.01)
                neutral = len(vc_means) - expanding - contracting
                r(f"  │ Volume: {expanding} expanding, {contracting} contracting, {neutral} neutral layers")
                r(f"  │ Net volume change: {sum(vc_means):.4f}")
                if sum(vc_means) < -0.1:
                    r(f"  │   Net contraction: model is 'deciding' — collapsing possibilities")
                elif sum(vc_means) > 0.1:
                    r(f"  │   Net expansion: model is 'exploring' — opening up possibilities")

            # Procrustes
            pd = curv["procrustes_deviations"]
            if pd:
                pd_devs = [p["deviation"] for p in pd]
                max_pd_layer = np.argmax(pd_devs)
                r(f"  │ Peak frame rotation at layer {max_pd_layer}: {pd_devs[max_pd_layer]:.4f}")
                r(f"  │ Mean frame rotation: {np.mean(pd_devs):.4f} ± {np.std(pd_devs):.4f}")
                r(f"  │   (high = large change in local coordinate frame = processing mode shift)")

                # Check if frame rotation correlates with deformation
                if "layer_deformation" in result:
                    deform_deltas = [d["mean_delta_norm"] for d in result["layer_deformation"]["per_layer"]]
                    min_len = min(len(pd_devs), len(deform_deltas))
                    if min_len > 3:
                        corr, pval = pearsonr(pd_devs[:min_len], deform_deltas[:min_len])
                        r(f"  │ Correlation(frame_rotation, deformation): r={corr:.3f}, p={pval:.4f}")
            r(f"  └─")

            # --- PCA variance / encoding structure ---
            r(f"\n  ┌─ ENCODING STRUCTURE")
            pca_var = result["pca_variance_explained"]
            top1 = pca_var[0] * 100
            top5 = sum(pca_var[:5]) * 100
            top10 = sum(pca_var[:min(10, len(pca_var))]) * 100
            r(f"  │ PCA variance explained (global): PC1={top1:.1f}%, top-5={top5:.1f}%, top-10={top10:.1f}%")

            # Participation ratio
            pr = (np.sum(pca_var)) ** 2 / (np.sum(pca_var ** 2) + 1e-12)
            r(f"  │ Participation ratio (effective dimensionality): {pr:.1f}")

            if top1 > 50:
                r(f"  │ ○ Representations are low-dimensional (PC1 dominates)")
                r(f"  │   May indicate simple encoding or insufficient prompt complexity")
            elif top10 < 50:
                r(f"  │ ✓ Information is distributed across many dimensions (top-10 < 50%)")
                r(f"  │   SUPPORTS holographic encoding hypothesis (Conjecture 2)")
            else:
                r(f"  │ ○ Moderate dimensionality — information partially distributed")

            if pr > 5:
                r(f"  │ ✓ High effective dimensionality ({pr:.1f}) — holographic-like encoding")
            elif pr > 2:
                r(f"  │ ○ Moderate effective dimensionality ({pr:.1f})")
            else:
                r(f"  │ ○ Low effective dimensionality ({pr:.1f}) — concentrated encoding")
            r(f"  └─")

        # --- Overall Summary ---
        r(f"\n{'='*72}")
        r(f"OVERALL SUMMARY FOR {self.model_name}")
        r(f"{'='*72}")

        conjecture_support = {
            "C1 (Space-morphing)": [],
            "C2 (Holographic)": [],
            "C3 (Topological)": [],
            "C4 (Inner compute, outer translate)": [],
            "C5 (Jacobian carrier)": [],
            "C6 (Dynamic map generation)": [],
        }

        for pid, result in self.results.items():
            # C1: Check structured Jacobian (non-uniform across layers)
            jac = result["jacobian_analysis"]
            divs = [s["mean_divergence"] for s in jac]
            curls = [s["mean_curl"] for s in jac]
            if np.std(divs) > 0.01 or np.std(curls) > 0.01:
                conjecture_support["C1 (Space-morphing)"].append(True)
            else:
                conjecture_support["C1 (Space-morphing)"].append(False)

            # C2: Check holographic (PCA distributed)
            pca_var = result["pca_variance_explained"]
            pr = (np.sum(pca_var)) ** 2 / (np.sum(pca_var ** 2) + 1e-12)
            conjecture_support["C2 (Holographic)"].append(pr > 3)

            # C3: Check topological phase transitions
            wass = result["topology_analysis"]["wasserstein_distances"]
            if wass:
                dists = [w["distance"] for w in wass]
                mean_d = np.mean(dists)
                std_d = np.std(dists)
                has_spike = any(d > mean_d + 1.5 * std_d for d in dists)
                conjecture_support["C3 (Topological)"].append(has_spike)

            # C4: Check middle > outer deformation
            deform = result["layer_deformation"]["per_layer"]
            deltas = [d["mean_delta_norm"] for d in deform]
            n = len(deltas)
            if n >= 4:
                mid = np.mean(deltas[n // 4: 3 * n // 4])
                out = np.mean(deltas[:n // 4] + deltas[3 * n // 4:])
                conjecture_support["C4 (Inner compute, outer translate)"].append(mid > out)

            # C4 reversal check
            fl = result["layer_deformation"]["geometry_correlation_first_last"]
            fm = result["layer_deformation"]["geometry_correlation_first_middle"]
            conjecture_support["C4 (Inner compute, outer translate)"].append(fl > fm)

            # C5: Complex eigenvalues
            eig = result["eigenvalue_analysis"]
            if eig:
                has_complex = any(
                    np.max(np.abs(s["mean_phases"])) > 0.1 for s in eig
                )
                conjecture_support["C5 (Jacobian carrier)"].append(has_complex)

            # C6: Layer-to-layer coupling
            if len(jac) >= 4:
                divs_arr = np.array(divs)
                if len(divs_arr) > 3:
                    ac, p = pearsonr(divs_arr[:-1], divs_arr[1:])
                    conjecture_support["C6 (Dynamic map generation)"].append(abs(ac) > 0.3 and p < 0.15)

        r(f"\n  Conjecture Assessment (across {len(self.results)} prompts):")
        r(f"  {'─'*60}")
        for conj, supports in conjecture_support.items():
            if supports:
                pct = sum(supports) / len(supports) * 100
                n_true = sum(supports)
                n_total = len(supports)
                if pct > 70:
                    status = "✓ SUPPORTED"
                elif pct > 40:
                    status = "○ MIXED"
                else:
                    status = "✗ NOT SUPPORTED"
                r(f"  {status:20s} | {conj}: {n_true}/{n_total} tests passed ({pct:.0f}%)")
            else:
                r(f"  {'? INSUFFICIENT DATA':20s} | {conj}")

        r(f"\n{'='*72}")
        r(f"INTERPRETATION GUIDE")
        r(f"{'='*72}")
        r(f"")
        r(f"  The paper (Koch, 2026) proposes that transformer layers morph")
        r(f"  embedding space via Lipschitz maps, not just move points through it.")
        r(f"  This tool tests that framework empirically by extracting hidden")
        r(f"  states and computing geometric invariants at each layer transition.")
        r(f"")
        r(f"  Key geometric quantities and what they mean:")
        r(f"  ┌────────────────────────┬──────────────────────────────────────────┐")
        r(f"  │ Divergence (tr J)      │ Local expansion (+) or contraction (−)  │")
        r(f"  │ Curl (‖J_antisym‖)     │ Rotational mixing between dimensions    │")
        r(f"  │ Shear (‖J_sym−tr/d·I‖) │ Anisotropic distortion                 │")
        r(f"  │ Spectral radius        │ Maximum stretching factor               │")
        r(f"  │ Spectral gap           │ Dominance of leading mode               │")
        r(f"  │ ORC                    │ Token convergence (+) / divergence (−)  │")
        r(f"  │ Volume change          │ Local simplex expansion/contraction     │")
        r(f"  │ Procrustes deviation   │ Frame rotation = processing mode shift  │")
        r(f"  │ Participation ratio    │ Effective dimensionality of encoding    │")
        r(f"  │ Wasserstein distance   │ Topological change between layers       │")
        r(f"  └────────────────────────┴──────────────────────────────────────────┘")
        r(f"")
        r(f"  What to look for in the visualizations:")
        r(f"  • deformation_profile.png: Does deformation peak in middle layers?")
        r(f"    (Red bars = middle layers, should be tallest for Conjecture 4)")
        r(f"  • jacobian_decomposition.png: Are div/curl/shear structured?")
        r(f"    (Non-uniform patterns support Conjecture 1)")
        r(f"  • eigenvalue_spectra.png: Are there complex eigenvalues?")
        r(f"    (Non-zero phases = rotational encoding, supports Conjecture 5)")
        r(f"  • holographic_ablation.png: Is ablation impact uniform?")
        r(f"    (Proportional degradation supports Conjecture 2)")
        r(f"  • topology.png: Are there persistent H1/H2 features?")
        r(f"    (Long-lived loops/voids support Conjecture 3)")
        r(f"  • curvature.png: Does ORC peak in middle layers?")
        r(f"    (Supports Conjecture 4)")
        r(f"  • fibre_view.png: Do token trajectories show structure?")
        r(f"    (Smooth, diverging paths = structured space-morphing)")
        r(f"  • encoding_search.png: Is there low-rank Jacobian structure?")
        r(f"    (Persistent eigenvalue modes = internal encoding format)")
        r(f"  • pca_trajectories.png: Does effective dim change across layers?")
        r(f"    (Peak in middle = holographic computation zone)")
        r(f"{'='*72}\n")

        report_text = "\n".join(report_lines)

        # Save to file
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        # Print to console
        print(report_text)

        return report_text

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# Main Entry Point
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Transformer Layers as Fibre Bundle Morphisms: Empirical Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 script.py --test-small                    # Quick test with GPT-2
  python3 script.py --test-all                      # Test all supported models
  python3 script.py --model gpt2-large              # Specific model
  python3 script.py --model gpt2 --prompt "Hello"   # Custom prompt
  sbatch script.py --test-all                       # Submit to SLURM

Based on: "Transformer Layers as Fibre Bundle Morphisms" (Koch, 2026)
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test-small",
        action="store_true",
        help="Quick test with small models (GPT-2 only)",
    )
    group.add_argument(
        "--test-all",
        action="store_true",
        help="Test all supported models (requires significant GPU memory)",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific HuggingFace model name (e.g., gpt2-large, EleutherAI/pythia-1b)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompt(s) to analyze (default: built-in test prompts)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fibre_analysis_output",
        help="Output directory for results and visualizations",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=4,
        help="Maximum number of prompts to analyze per model",
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="Skip persistent homology (faster, but misses Conjecture 3)",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="PCA projection dimension for Jacobian estimation (default: 50)",
    )

    args = parser.parse_args()

    # Determine models to test
    if args.test_small:
        models = SMALL_MODELS
    elif args.test_all:
        models = ALL_MODELS
    elif args.model:
        models = [args.model]
    else:
        # Default: test small
        models = SMALL_MODELS
        print("No mode specified, defaulting to --test-small (GPT-2)")

    # Determine prompts
    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    prompts = prompts[: args.max_prompts]

    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    print("╔" + "═" * 70 + "╗")
    print("║  Transformer Layers as Fibre Bundle Morphisms                       ║")
    print("║  Empirical Analysis Tool                                            ║")
    print("║  Based on Koch (2026)                                               ║")
    print("╚" + "═" * 70 + "╝")
    print(f"\n  Models to analyze: {models}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Output: {output_dir}/")
    print(f"  PCA dim: {args.pca_dim}")
    if args.skip_topology:
        print(f"  Topology: SKIPPED")

    device = get_device()

    all_reports = []
    start_time = time.time()

    for model_idx, model_name in enumerate(models):
        print(f"\n{'▓'*72}")
        print(f"  Model {model_idx+1}/{len(models)}: {model_name}")
        print(f"{'▓'*72}")

        try:
            analyzer = FibreBundleAnalyzer(model_name, device, output_dir)

            for pid, prompt in enumerate(prompts):
                analyzer.analyze_prompt(prompt, prompt_id=pid)

            analyzer.generate_all_visualizations()
            report = analyzer.generate_report()
            all_reports.append((model_name, report))

            analyzer.cleanup()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  ✗ OUT OF GPU MEMORY for {model_name}")
            print(f"    Try a smaller model or use --pca-dim to reduce dimensionality")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"\n  ✗ ERROR analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    elapsed = time.time() - start_time

    # Final summary across all models
    print(f"\n{'═'*72}")
    print(f"FINAL CROSS-MODEL SUMMARY")
    print(f"{'═'*72}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Models completed: {len(all_reports)}/{len(models)}")

    for model_name, _ in all_reports:
        print(f"  ✓ {model_name}")

    if len(all_reports) < len(models):
        failed = set(models) - set(m for m, _ in all_reports)
        for model_name in failed:
            print(f"  ✗ {model_name} (failed)")

    print(f"\n  All results saved to: {output_dir}/")
    print(f"  Each model subdirectory contains:")
    print(f"    analysis_report.txt            Human-readable findings")
    print(f"    prompt*_deformation_profile.png  Conjecture 4: layer deformation")
    print(f"    prompt*_jacobian_decomposition.png  Conjecture 1: Jacobian field")
    print(f"    prompt*_eigenvalue_spectra.png  Conjecture 5: eigenvalue spectra")
    print(f"    prompt*_holographic_ablation.png  Conjecture 2: holographic test")
    print(f"    prompt*_topology.png           Conjecture 3: persistent homology")
    print(f"    prompt*_curvature.png          Curvature measures (ORC, volume, Procrustes)")
    print(f"    prompt*_fibre_view.png         Token trajectories across layers")
    print(f"    prompt*_pca_trajectories.png   Dimensionality / encoding analysis")
    print(f"    prompt*_encoding_search.png    Internal encoding structure search")
    print(f"    cross_prompt_comparison.png    Cross-prompt comparison")

    print(f"\n  Done! 🎉\n")


if __name__ == "__main__":
    main()
