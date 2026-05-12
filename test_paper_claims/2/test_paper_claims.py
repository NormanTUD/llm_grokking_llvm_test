#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "scipy",
#   "scikit-learn",
#   "ripser",
#   "persim",
#   "matplotlib",
#   "seaborn",
#   "rich",
#   "pandas",
# ]
# ///

"""
LLM Space-Deformation Explorer
===============================

Tests Koch's fibre bundle hypothesis by:
1. Running multiple LLMs on algorithmic tasks
2. Extracting hidden states at every layer
3. Computing Jacobian fields, FFT decompositions, topological features
4. Visualizing space deformations, waves, and algorithm execution traces
5. Comparing how different architectures "compute" internally

Usage:
    python3 explore_deformations.py
    python3 explore_deformations.py --models gpt2 EleutherAI/pythia-70m
    python3 explore_deformations.py --device cuda --tasks all

Auto-bootstraps with uv if dependencies are missing.
"""

import os
import sys
from datetime import datetime, timedelta

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc


# ════════════════════════════════════════════════════════════════════════════
# AUTO-BOOTSTRAP WITH UV
# ════════════════════════════════════════════════════════════════════════════

def compute_exclude_newer_date(days_back=8):
    return (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")


def should_set_exclude_newer():
    return not os.environ.get("UV_EXCLUDE_NEWER")


def restart_with_uv(script_path, args, env):
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", script_path] + args, env)
    except FileNotFoundError:
        print("uv is not installed. Try:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def ensure_safe_env():
    if not should_set_exclude_newer():
        return
    past_date = compute_exclude_newer_date(8)
    os.environ["UV_EXCLUDE_NEWER"] = past_date
    restart_with_uv(sys.argv[0], sys.argv[1:], os.environ)


# Must run BEFORE heavy imports
ensure_safe_env()

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS (after bootstrap)
# ════════════════════════════════════════════════════════════════════════════

import argparse
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm
import seaborn as sns

from scipy.linalg import svdvals, orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kurtosis
from scipy.fft import fft, fft2, fftfreq, fftshift
from scipy.interpolate import RBFInterpolator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn, TimeRemainingColumn,
)
from rich.text import Text
from rich.tree import Tree
from rich.layout import Layout
from rich import box

try:
    from ripser import ripser
    from persim import wasserstein as wasserstein_distance
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")
warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MODELS = [
    "gpt2",
    "EleutherAI/pythia-70m",
    "facebook/opt-125m",
]

# Algorithmic tasks the LLM must "execute" — we trace HOW it does so
ALGORITHMIC_TASKS = {
    "addition": {
        "prompts": [
            "2 + 3 =",
            "15 + 27 =",
            "100 + 200 =",
            "-5 + 8 =",
            "0 + 0 =",
        ],
        "description": "Integer addition — tests if the model uses structured geometric operations",
    },
    "comparison": {
        "prompts": [
            "Is 5 greater than 3? Answer: ",
            "Is 2 greater than 9? Answer: ",
            "Is 100 greater than 100? Answer: ",
            "Is -1 greater than 0? Answer: ",
        ],
        "description": "Number comparison — tests directional deformation in representation space",
    },
    "negation_parity": {
        "prompts": [
            "not not True is ",
            "not not not True is ",
            "not not not not True is ",
            "not True is ",
            "not not not not not True is ",
        ],
        "description": "Negation parity (Z/2Z) — tests for rotational/reflective Jacobian structure",
    },
    "pattern_completion": {
        "prompts": [
            "1, 2, 3, 4, ",
            "2, 4, 6, 8, ",
            "1, 1, 2, 3, 5, ",
            "10, 20, 30, 40, ",
        ],
        "description": "Sequence continuation — tests for spiral/helical topological structures",
    },
    "semantic_substitution": {
        "prompts": [
            "The sky is blue",
            "The sky is red",
            "The sky is green",
            "The ocean is blue",
            "The ocean is red",
        ],
        "description": "Semantic substitution — tests selective dimensional response (Koch Fig. 8)",
    },
}

OUTPUT_DIR = Path("deformation_results")


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerAnalysis:
    """Analysis results for a single layer transition."""
    layer_idx: int
    jacobian_approx: np.ndarray          # (D, D) approximate Jacobian
    eigenvalues: np.ndarray              # complex eigenvalues
    singular_values: np.ndarray          # singular values
    divergence: float                    # tr(J) — expansion/contraction
    curl_magnitude: float                # ||J_antisym||_F — rotational mixing
    shear_magnitude: float               # ||J_sym - (tr/d)I||_F — anisotropic distortion
    determinant: float                   # det(J) — signed volume change
    condition_number: float              # σ_max / σ_min
    deformation_magnitude: float         # ||delta||_F
    fft_spectrum: np.ndarray             # FFT of the delta
    fft_dominant_freqs: np.ndarray       # dominant frequency components


@dataclass
class PromptAnalysis:
    """Full analysis of one prompt through one model."""
    model_name: str
    prompt: str
    tokens: List[str]
    hidden_states: np.ndarray            # (n_layers+1, seq_len, hidden_dim)
    layer_analyses: List[LayerAnalysis]
    persistence_diagrams: Optional[List] = None
    fibre_trajectories: Optional[np.ndarray] = None  # (n_layers+1, seq_len, 2) PCA-projected


@dataclass
class ModelReport:
    """Aggregated report for one model across all tasks."""
    model_name: str
    n_params: int
    n_layers: int
    hidden_dim: int
    prompt_analyses: List[PromptAnalysis]
    wave_propagation: Optional[np.ndarray] = None  # deformation magnitudes per layer
    topological_transitions: Optional[np.ndarray] = None


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & HIDDEN STATE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name: str, device: str = "cpu"):
    """Load a HuggingFace model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"  [cyan]Loading {model_name}...[/]")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    console.print(
        f"  [green]✓[/] {model_name}: {n_params/1e6:.1f}M params, "
        f"{n_layers} layers, d={hidden_dim}"
    )

    return model, tokenizer, n_params, n_layers, hidden_dim


def extract_hidden_states(
    model, tokenizer, prompt: str, device: str = "cpu"
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract hidden states at every layer.
    Returns: (states array of shape (n_layers+1, seq_len, hidden_dim), token strings)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]
    states = np.stack(hidden, axis=0)  # (n_layers+1, seq_len, hidden_dim)

    return states, tokens


# ════════════════════════════════════════════════════════════════════════════
# JACOBIAN FIELD COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_jacobian_from_deltas(
    h_in: np.ndarray, h_out: np.ndarray, n_components: int = 50
) -> np.ndarray:
    """
    Approximate the Jacobian of the layer map Φ: R^d -> R^d
    from the token cloud using least-squares in a PCA subspace.

    Koch (Section 4): "In practice, we estimate J^(ℓ) from the token cloud
    via weighted least-squares regression in a projected subspace."

    h_in:  (seq_len, hidden_dim) — representations before the layer
    h_out: (seq_len, hidden_dim) — representations after the layer

    Returns: (n_components, n_components) approximate Jacobian in PCA space
    """
    seq_len, hidden_dim = h_in.shape
    n_comp = min(n_components, seq_len - 1, hidden_dim)

    if n_comp < 2:
        return np.eye(min(hidden_dim, 2))

    # Center the data
    h_in_c = h_in - h_in.mean(axis=0)
    delta = h_out - h_in
    delta_c = delta - delta.mean(axis=0)

    # Project to PCA subspace for tractability
    pca = PCA(n_components=n_comp)
    h_in_proj = pca.fit_transform(h_in_c)
    delta_proj = pca.transform(delta_c)

    # Least-squares: delta ≈ J @ h_in
    # J = delta^T @ h_in @ (h_in^T @ h_in)^{-1}
    try:
        J, residuals, rank, sv = np.linalg.lstsq(h_in_proj, delta_proj, rcond=None)
    except np.linalg.LinAlgError:
        J = np.eye(n_comp)

    return J  # (n_comp, n_comp)


def analyze_jacobian(J: np.ndarray) -> dict:
    """
    Extract geometric invariants from the Jacobian field.

    Koch (Section 4): divergence, curl, shear, eigenvalue spectrum,
    determinant, singular values, condition number.
    """
    d = J.shape[0]

    # Divergence: tr(J) — local expansion/contraction
    divergence = np.trace(J)

    # Curl: ||J_antisym||_F — rotational mixing
    J_antisym = (J - J.T) / 2
    curl = np.linalg.norm(J_antisym, 'fro')

    # Shear: ||J_sym - (tr(J_sym)/d) * I||_F — anisotropic distortion
    J_sym = (J + J.T) / 2
    shear_mat = J_sym - (np.trace(J_sym) / d) * np.eye(d)
    shear = np.linalg.norm(shear_mat, 'fro')

    # Eigenvalues (complex — phases indicate rotation)
    eigenvalues = np.linalg.eigvals(J)

    # Singular values
    singular_values = np.linalg.svd(J, compute_uv=False)

    # Determinant — signed volume change
    try:
        det = np.linalg.det(J)
    except Exception:
        det = 0.0

    # Condition number
    sv_pos = singular_values[singular_values > 1e-10]
    condition = sv_pos[0] / sv_pos[-1] if len(sv_pos) >= 2 else 1.0

    return {
        "divergence": float(divergence),
        "curl": float(curl),
        "shear": float(shear),
        "eigenvalues": eigenvalues,
        "singular_values": singular_values,
        "determinant": float(det),
        "condition_number": float(condition),
    }


# ════════════════════════════════════════════════════════════════════════════
# FFT ANALYSIS — HOLOGRAPHIC SCRAMBLING TEST
# ════════════════════════════════════════════════════════════════════════════

def compute_fft_analysis(delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply FFT to the layer delta to test Koch's holographic scrambling conjecture.

    Koch (Conjecture 2): Information is holographically distributed like a Fourier
    transform — deleting frequency components degrades everything uniformly.

    delta: (seq_len, hidden_dim) — the residual stream delta for one layer

    Returns:
        spectrum: (seq_len, hidden_dim) — magnitude of 2D FFT
        dominant_freqs: top-k frequency components
    """
    # 2D FFT of the delta matrix
    fft_result = fft2(delta)
    spectrum = np.abs(fftshift(fft_result))

    # Find dominant frequencies
    flat_spectrum = spectrum.flatten()
    top_k = min(20, len(flat_spectrum))
    top_indices = np.argpartition(flat_spectrum, -top_k)[-top_k:]
    dominant_freqs = flat_spectrum[top_indices]

    return spectrum, dominant_freqs


def compute_1d_fft_per_dimension(delta: np.ndarray) -> np.ndarray:
    """
    Compute 1D FFT along the token axis for each hidden dimension.
    This reveals the "wave" structure Koch describes — how deformations
    propagate across the token sequence.

    Returns: (seq_len//2+1, hidden_dim) magnitude spectrum
    """
    seq_len, hidden_dim = delta.shape
    spectra = np.zeros((seq_len // 2 + 1, hidden_dim))

    for d in range(hidden_dim):
        fft_d = fft(delta[:, d])
        spectra[:, d] = np.abs(fft_d[:seq_len // 2 + 1])

    return spectra


# ════════════════════════════════════════════════════════════════════════════
# TOPOLOGICAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def compute_persistence(
    points: np.ndarray, max_dim: int = 1, max_points: int = 150
) -> Optional[dict]:
    """Compute persistent homology of a point cloud."""
    if not HAS_RIPSER:
        return None

    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]

    n_comp = min(15, points.shape[0] - 1, points.shape[1])
    if n_comp < 2:
        return None

    if points.shape[1] > n_comp:
        pca = PCA(n_components=n_comp)
        points = pca.fit_transform(StandardScaler().fit_transform(points))

    try:
        result = ripser(points, maxdim=max_dim)
        return result
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM EXECUTION TRACING
# ════════════════════════════════════════════════════════════════════════════

def trace_algorithm_execution(
    states: np.ndarray, tokens: List[str], task_type: str
) -> dict:
    """
    Trace how the model "executes" an algorithm by analyzing the
    geometric transformations at each layer.

    Koch (Conjecture 3): "A topological spiral in the representation space
    can be thought of as a sequence of states on a Turing machine tape,
    with attention providing the 'jump' (head movement) operation and
    the FFN providing the 'write' operation."

    This function tracks:
    1. Which tokens' representations change most at each layer
    2. The direction and magnitude of change (the "instruction")
    3. Whether the changes form coherent patterns (spirals, reflections, etc.)
    4. The effective "algorithm" the model appears to execute

    Returns a dict with execution trace information.
    """
    n_layers, seq_len, hidden_dim = states.shape
    trace = {
        "task_type": task_type,
        "n_layers": n_layers - 1,
        "tokens": tokens,
        "per_layer": [],
        "token_importance_evolution": np.zeros((n_layers - 1, seq_len)),
        "cumulative_displacement": np.zeros((n_layers, seq_len)),
    }

    # Track cumulative displacement from embedding
    for ell in range(1, n_layers):
        for tok in range(seq_len):
            trace["cumulative_displacement"][ell, tok] = np.linalg.norm(
                states[ell, tok] - states[0, tok]
            )

    for ell in range(n_layers - 1):
        h_in = states[ell]
        h_out = states[ell + 1]
        delta = h_out - h_in

        # Per-token deformation magnitude
        per_token_magnitude = np.linalg.norm(delta, axis=1)
        trace["token_importance_evolution"][ell] = per_token_magnitude

        # Which token changed most?
        most_active_token = int(np.argmax(per_token_magnitude))

        # Direction analysis: are tokens moving in similar directions?
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        delta_normed = delta / norms
        cos_matrix = delta_normed @ delta_normed.T
        triu_idx = np.triu_indices(seq_len, k=1)
        mean_coherence = float(np.mean(cos_matrix[triu_idx])) if len(triu_idx[0]) > 0 else 0.0

        # Compute the Jacobian invariants
        J = compute_jacobian_from_deltas(h_in, h_out, n_components=min(30, seq_len - 1, hidden_dim))
        invariants = analyze_jacobian(J)

        # Check for rotational structure (relevant for negation parity)
        eig_phases = np.angle(invariants["eigenvalues"])
        has_rotation = bool(np.any(np.abs(eig_phases) > 0.1))

        # Check for reflective structure (Z/2Z for parity)
        neg_eigenvalues = np.sum(np.real(invariants["eigenvalues"]) < -0.1)
        has_reflection = neg_eigenvalues > 0

        layer_info = {
            "layer": ell,
            "most_active_token_idx": most_active_token,
            "most_active_token": tokens[most_active_token] if most_active_token < len(tokens) else "?",
            "max_deformation": float(np.max(per_token_magnitude)),
            "mean_deformation": float(np.mean(per_token_magnitude)),
            "coherence": mean_coherence,
            "divergence": invariants["divergence"],
            "curl": invariants["curl"],
            "shear": invariants["shear"],
            "has_rotation": has_rotation,
            "has_reflection": has_reflection,
            "n_negative_eigenvalues": int(neg_eigenvalues),
            "condition_number": invariants["condition_number"],
            "interpretation": _interpret_layer_operation(
                invariants, mean_coherence, per_token_magnitude, task_type, ell, n_layers - 1
            ),
        }
        trace["per_layer"].append(layer_info)

    return trace


def _interpret_layer_operation(
    invariants: dict, coherence: float, per_token_mag: np.ndarray,
    task_type: str, layer_idx: int, total_layers: int
) -> str:
    """
    Generate a human-readable interpretation of what a layer is doing.

    Koch (Conjecture 4): "Inner layers compute, outer layers translate."
    Koch (Conjecture 5): "The Jacobian field acts as a discrete Jacobi field."
    """
    parts = []

    # Position in the network
    relative_pos = layer_idx / max(total_layers - 1, 1)
    if relative_pos < 0.2:
        parts.append("EARLY (routing/embedding)")
    elif relative_pos > 0.8:
        parts.append("LATE (output translation)")
    else:
        parts.append("MIDDLE (computation)")

    # Expansion vs contraction
    div = invariants["divergence"]
    if div > 0.5:
        parts.append(f"EXPANDING (div={div:.2f})")
    elif div < -0.5:
        parts.append(f"CONTRACTING (div={div:.2f})")
    else:
        parts.append(f"~isometric (div={div:.2f})")

    # Rotation
    if invariants["curl"] > 0.5:
        parts.append(f"ROTATING (curl={invariants['curl']:.2f})")

    # Shear
    if invariants["shear"] > 1.0:
        parts.append(f"SHEARING (shear={invariants['shear']:.2f})")

    # Coherence
    if coherence > 0.5:
        parts.append("tokens move TOGETHER (global shift)")
    elif coherence < -0.1:
        parts.append("tokens move APART (separation)")
    else:
        parts.append("tokens move INDEPENDENTLY (local ops)")

    # Task-specific interpretation
    if task_type == "negation_parity":
        neg_eig = sum(1 for e in invariants["eigenvalues"] if np.real(e) < -0.1)
        if neg_eig > 0:
            parts.append(f"⚡ {neg_eig} reflective eigenvalues (parity flip!)")

    return " | ".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# FULL ANALYSIS PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def analyze_prompt(
    model, tokenizer, prompt: str, model_name: str, device: str = "cpu"
) -> PromptAnalysis:
    """Run the full analysis pipeline on a single prompt."""
    states, tokens = extract_hidden_states(model, tokenizer, prompt, device)
    n_layers, seq_len, hidden_dim = states.shape

    layer_analyses = []
    for ell in range(n_layers - 1):
        h_in = states[ell]
        h_out = states[ell + 1]
        delta = h_out - h_in

        # Jacobian
        n_comp = min(30, seq_len - 1, hidden_dim)
        J = compute_jacobian_from_deltas(h_in, h_out, n_components=n_comp)
        inv = analyze_jacobian(J)

        # FFT
        fft_spectrum, fft_dominant = compute_fft_analysis(delta)

        # Deformation magnitude
        deformation_mag = float(np.linalg.norm(delta, 'fro'))

        la = LayerAnalysis(
            layer_idx=ell,
            jacobian_approx=J,
            eigenvalues=inv["eigenvalues"],
            singular_values=inv["singular_values"],
            divergence=inv["divergence"],
            curl_magnitude=inv["curl"],
            shear_magnitude=inv["shear"],
            determinant=inv["determinant"],
            condition_number=inv["condition_number"],
            deformation_magnitude=deformation_mag,
            fft_spectrum=fft_spectrum,
            fft_dominant_freqs=fft_dominant,
        )
        layer_analyses.append(la)

    # Persistence diagrams per layer
    persistence_diagrams = []
    if HAS_RIPSER:
        for ell in range(n_layers):
            pd = compute_persistence(states[ell])
            persistence_diagrams.append(pd)

    # Fibre trajectories (PCA projection for visualization)
    all_points = states.reshape(-1, hidden_dim)
    n_comp_vis = min(2, all_points.shape[0] - 1, hidden_dim)
    if n_comp_vis >= 2:
        pca_vis = PCA(n_components=2)
        projected = pca_vis.fit_transform(all_points)
        fibre_trajectories = projected.reshape(n_layers, seq_len, 2)
    else:
        fibre_trajectories = None

    return PromptAnalysis(
        model_name=model_name,
        prompt=prompt,
        tokens=tokens,
        hidden_states=states,
        layer_analyses=layer_analyses,
        persistence_diagrams=persistence_diagrams if persistence_diagrams else None,
        fibre_trajectories=fibre_trajectories,
    )


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def plot_deformed_grid(analysis: PromptAnalysis, layer_idx: int, save_path: Path):
    """
    Visualize the space deformation at a specific layer as a deformed grid.

    Koch (Section 6, Fig. 2): "A regular grid in the embedding space is deformed
    by the cumulative layer maps, visualized via RBF interpolation."
    """
    states = analysis.hidden_states
    n_layers, seq_len, hidden_dim = states.shape

    if seq_len < 3:
        return

    # Project to 2D
    h_in = states[layer_idx]
    h_out = states[layer_idx + 1]

    all_pts = np.vstack([h_in, h_out])
    n_comp = min(2, all_pts.shape[0] - 1, hidden_dim)
    if n_comp < 2:
        return

    pca = PCA(n_components=2)
    all_proj = pca.fit_transform(all_pts)
    pts_in = all_proj[:seq_len]
    pts_out = all_proj[seq_len:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original grid with token positions
    ax = axes[0]
    ax.set_title(f"Layer {layer_idx} (input)", fontsize=12)

    # Create a regular grid around the data
    x_range = pts_in[:, 0].max() - pts_in[:, 0].min()
    y_range = pts_in[:, 1].max() - pts_in[:, 1].min()
    margin = max(x_range, y_range) * 0.3
    x_min, x_max = pts_in[:, 0].min() - margin, pts_in[:, 0].max() + margin
    y_min, y_max = pts_in[:, 1].min() - margin, pts_in[:, 1].max() + margin

    grid_n = 15
    gx = np.linspace(x_min, x_max, grid_n)
    gy = np.linspace(y_min, y_max, grid_n)

    # Draw grid
    for x in gx:
        ax.plot([x, x], [y_min, y_max], 'k-', alpha=0.2, linewidth=0.5)
    for y in gy:
        ax.plot([x_min, x_max], [y, y], 'k-', alpha=0.2, linewidth=0.5)

    # Plot tokens
    
    for i, tok in enumerate(analysis.tokens):
        if i < seq_len:
            ax.scatter(pts_in[i, 0], pts_in[i, 1], c='blue', s=60, zorder=5)
            ax.annotate(tok.strip(), (pts_in[i, 0], pts_in[i, 1]),
                       fontsize=7, ha='center', va='bottom', color='blue')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    # Panel 2: Deformed grid (output)
    ax = axes[1]
    ax.set_title(f"Layer {layer_idx+1} (output)", fontsize=12)

    # Use RBF interpolation to deform the grid
    # Koch (Section 6): "A regular grid in the embedding space is deformed
    # by the cumulative layer maps, visualized via RBF interpolation."
    try:
        # Build RBF from input points -> output points
        rbf_x = RBFInterpolator(pts_in, pts_out[:, 0], kernel='thin_plate_spline', smoothing=0.1)
        rbf_y = RBFInterpolator(pts_in, pts_out[:, 1], kernel='thin_plate_spline', smoothing=0.1)

        # Deform grid lines
        for x in gx:
            line_pts = np.column_stack([np.full(50, x), np.linspace(y_min, y_max, 50)])
            dx = rbf_x(line_pts)
            dy = rbf_y(line_pts)
            ax.plot(dx, dy, 'k-', alpha=0.3, linewidth=0.5)
        for y in gy:
            line_pts = np.column_stack([np.linspace(x_min, x_max, 50), np.full(50, y)])
            dx = rbf_x(line_pts)
            dy = rbf_y(line_pts)
            ax.plot(dx, dy, 'k-', alpha=0.3, linewidth=0.5)
    except Exception:
        # Fallback: just draw straight lines between output points
        pass

    # Plot output tokens
    for i, tok in enumerate(analysis.tokens):
        if i < seq_len:
            ax.scatter(pts_out[i, 0], pts_out[i, 1], c='red', s=60, zorder=5)
            ax.annotate(tok.strip(), (pts_out[i, 0], pts_out[i, 1]),
                       fontsize=7, ha='center', va='bottom', color='red')
    ax.set_aspect('equal')

    # Panel 3: Displacement vectors
    ax = axes[2]
    ax.set_title(f"Displacement (Layer {layer_idx} → {layer_idx+1})", fontsize=12)

    for i in range(seq_len):
        dx = pts_out[i, 0] - pts_in[i, 0]
        dy = pts_out[i, 1] - pts_in[i, 1]
        mag = np.sqrt(dx**2 + dy**2)
        color = cm.hot(min(mag / (np.max([np.sqrt((pts_out[j,0]-pts_in[j,0])**2 +
                (pts_out[j,1]-pts_in[j,1])**2) for j in range(seq_len)]) + 1e-10), 1.0))
        ax.arrow(pts_in[i, 0], pts_in[i, 1], dx, dy,
                head_width=0.02 * max(x_range, y_range),
                head_length=0.01 * max(x_range, y_range),
                fc=color, ec=color, alpha=0.7)
        ax.annotate(analysis.tokens[i].strip(), (pts_in[i, 0], pts_in[i, 1]),
                   fontsize=6, ha='center', va='bottom', alpha=0.6)
    ax.set_aspect('equal')

    fig.suptitle(
        f"Space Deformation: \"{analysis.prompt[:50]}...\" — {analysis.model_name}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fibre_trajectories(analysis: PromptAnalysis, save_path: Path):
    """
    Visualize fibre trajectories — how each token's representation
    moves through the PCA-projected space across layers.

    Koch (Section 6, Fig. 4): "Each vertical strand is a fibre
    (one token position), stacked from the embedding layer (bottom)
    to the final layer (top)."
    """
    if analysis.fibre_trajectories is None:
        return

    traj = analysis.fibre_trajectories  # (n_layers, seq_len, 2)
    n_layers, seq_len, _ = traj.shape

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: 2D trajectory plot
    ax = axes[0]
    ax.set_title("Fibre Trajectories (PCA projection)", fontsize=12)
    cmap = cm.get_cmap('tab20', seq_len)

    for tok_idx in range(seq_len):
        color = cmap(tok_idx)
        x = traj[:, tok_idx, 0]
        y = traj[:, tok_idx, 1]
        ax.plot(x, y, '-o', color=color, markersize=3, linewidth=1.5, alpha=0.7)
        # Label start and end
        ax.annotate(f"L0", (x[0], y[0]), fontsize=5, color=color, alpha=0.5)
        ax.annotate(f"L{n_layers-1}", (x[-1], y[-1]), fontsize=5, color=color)
        # Label token
        tok_label = analysis.tokens[tok_idx].strip() if tok_idx < len(analysis.tokens) else "?"
        ax.annotate(tok_label, (x[0], y[0]), fontsize=7, fontweight='bold',
                   color=color, ha='right', va='bottom')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    # Panel 2: Layer-by-layer displacement heatmap
    ax = axes[1]
    ax.set_title("Per-token displacement magnitude per layer", fontsize=12)

    displacement = np.zeros((n_layers - 1, seq_len))
    for ell in range(n_layers - 1):
        for tok in range(seq_len):
            displacement[ell, tok] = np.linalg.norm(
                analysis.hidden_states[ell + 1, tok] - analysis.hidden_states[ell, tok]
            )

    tok_labels = [t.strip()[:8] for t in analysis.tokens[:seq_len]]
    sns.heatmap(displacement, ax=ax, cmap='YlOrRd',
                xticklabels=tok_labels,
                yticklabels=[f"L{i}→{i+1}" for i in range(n_layers - 1)])
    ax.set_xlabel("Token")
    ax.set_ylabel("Layer transition")

    fig.suptitle(
        f"Fibre View: \"{analysis.prompt[:50]}...\" — {analysis.model_name}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_jacobian_invariants(analysis: PromptAnalysis, save_path: Path):
    """
    Visualize the Jacobian field invariants across layers.

    Koch (Section 4): "divergence, curl, shear, eigenvalue spectrum,
    determinant, singular values, condition number."
    """
    n_layers = len(analysis.layer_analyses)
    if n_layers == 0:
        return

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    layers = list(range(n_layers))

    # 1. Divergence, Curl, Shear
    ax = fig.add_subplot(gs[0, 0])
    divs = [la.divergence for la in analysis.layer_analyses]
    curls = [la.curl_magnitude for la in analysis.layer_analyses]
    shears = [la.shear_magnitude for la in analysis.layer_analyses]
    ax.plot(layers, divs, 'r-o', label='Divergence', markersize=4)
    ax.plot(layers, curls, 'b-s', label='Curl', markersize=4)
    ax.plot(layers, shears, 'g-^', label='Shear', markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Magnitude")
    ax.set_title("Jacobian Invariants")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Deformation magnitude
    ax = fig.add_subplot(gs[0, 1])
    mags = [la.deformation_magnitude for la in analysis.layer_analyses]
    ax.bar(layers, mags, color='steelblue', alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("||Δ||_F")
    ax.set_title("Deformation Magnitude per Layer")
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Condition number
    ax = fig.add_subplot(gs[0, 2])
    conds = [la.condition_number for la in analysis.layer_analyses]
    ax.semilogy(layers, conds, 'k-o', markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("σ_max / σ_min")
    ax.set_title("Condition Number (anisotropy)")
    ax.grid(True, alpha=0.3)

    # 4. Eigenvalue spectrum (complex plane)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Eigenvalue Spectrum (all layers)")
    cmap_layers = cm.get_cmap('viridis', n_layers)
    for ell, la in enumerate(analysis.layer_analyses):
        eigs = la.eigenvalues
        ax.scatter(np.real(eigs), np.imag(eigs), c=[cmap_layers(ell)] * len(eigs),
                  s=15, alpha=0.6, label=f"L{ell}" if ell % max(1, n_layers // 5) == 0 else "")
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_aspect('equal')

    # 5. Singular value distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Singular Value Distributions")
    for ell, la in enumerate(analysis.layer_analyses):
        svs = la.singular_values
        if ell % max(1, n_layers // 6) == 0:
            ax.semilogy(range(len(svs)), svs, '-', alpha=0.7, label=f"L{ell}")
    ax.set_xlabel("Index")
    ax.set_ylabel("σ_k")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 6. Determinant
    ax = fig.add_subplot(gs[1, 2])
    dets = [la.determinant for la in analysis.layer_analyses]
    colors = ['red' if d < 0 else 'blue' for d in dets]
    ax.bar(layers, dets, color=colors, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("det(J)")
    ax.set_title("Determinant (signed volume change)")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # 7. FFT dominant frequencies
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("FFT Dominant Frequencies per Layer")
    for ell, la in enumerate(analysis.layer_analyses):
        freqs = np.sort(la.fft_dominant_freqs)[::-1][:10]
        if ell % max(1, n_layers // 6) == 0:
            ax.plot(range(len(freqs)), freqs, '-o', markersize=3, alpha=0.7, label=f"L{ell}")
    ax.set_xlabel("Frequency rank")
    ax.set_ylabel("Magnitude")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 8. FFT spectrum heatmap (last layer)
    ax = fig.add_subplot(gs[2, 1])
    last_la = analysis.layer_analyses[-1]
    spectrum = last_la.fft_spectrum
    if spectrum.shape[0] > 1 and spectrum.shape[1] > 1:
        ax.imshow(np.log1p(spectrum), aspect='auto', cmap='inferno')
        ax.set_title(f"FFT Spectrum (Layer {n_layers-1})")
        ax.set_xlabel("Hidden dim frequency")
        ax.set_ylabel("Token frequency")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')

    # 9. Wave propagation summary
    ax = fig.add_subplot(gs[2, 2])
    ax.set_title("Wave Propagation (cumulative deformation)")
    cumulative = np.cumsum(mags)
    ax.fill_between(layers, 0, cumulative, alpha=0.3, color='steelblue')
    ax.plot(layers, cumulative, 'b-o', markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative ||Δ||_F")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Jacobian Field Analysis: \"{analysis.prompt[:40]}...\" — {analysis.model_name}",
        fontsize=14, fontweight='bold'
    )
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_algorithm_trace(trace: dict, save_path: Path):
    """
    Visualize the algorithm execution trace — how the model
    "executes" an algorithm by deforming space at each layer.
    """
    n_layers = trace["n_layers"]
    tokens = trace["tokens"]
    seq_len = len(tokens)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Token importance evolution heatmap
    ax = fig.add_subplot(gs[0, 0:2])
    importance = trace["token_importance_evolution"]
    tok_labels = [t.strip()[:8] for t in tokens]
    sns.heatmap(importance, ax=ax, cmap='YlOrRd',
                xticklabels=tok_labels,
                yticklabels=[f"L{i}" for i in range(n_layers)])
    ax.set_title("Token Importance per Layer (deformation magnitude)")
    ax.set_xlabel("Token")
    ax.set_ylabel("Layer")

    # 2. Per-layer operation summary
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.set_title("Layer Operations", fontsize=11, fontweight='bold')

    text_lines = []
    for info in trace["per_layer"][:12]:  # Show first 12 layers
        layer = info["layer"]
        interp = info["interpretation"]
        # Truncate interpretation
        if len(interp) > 60:
            interp = interp[:57] + "..."
        text_lines.append(f"L{layer}: {interp}")

    text = "\n".join(text_lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 3. Coherence across layers
    ax = fig.add_subplot(gs[1, 0])
    coherences = [info["coherence"] for info in trace["per_layer"]]
    ax.plot(range(n_layers), coherences, 'b-o', markersize=4)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cosine coherence")
    ax.set_title("Token Movement Coherence")
    ax.grid(True, alpha=0.3)

    # 4. Rotation and reflection indicators
    ax = fig.add_subplot(gs[1, 1])
    rotations = [1 if info["has_rotation"] else 0 for info in trace["per_layer"]]
    reflections = [info["n_negative_eigenvalues"] for info in trace["per_layer"]]
    ax.bar(range(n_layers), rotations, alpha=0.5, color='blue', label='Has rotation')
    ax.bar(range(n_layers), [r / max(max(reflections), 1) for r in reflections],
           alpha=0.5, color='red', label='Neg eigenvalues (norm)')
    ax.set_xlabel("Layer")
    ax.set_title(f"Geometric Operations ({trace['task_type']})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Cumulative displacement from embedding
    ax = fig.add_subplot(gs[1, 2])
    cumdisp = trace["cumulative_displacement"]
    for tok_idx in range(min(seq_len, 10)):
        ax.plot(range(cumdisp.shape[0]), cumdisp[:, tok_idx],
                '-', alpha=0.6, linewidth=1.5,
                label=tokens[tok_idx].strip()[:8] if tok_idx < 5 else "")
    ax.set_xlabel("Layer")
    ax.set_ylabel("||h^(ℓ) - h^(0)||")
    ax.set_title("Cumulative Displacement from Embedding")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Algorithm Execution Trace: {trace['task_type']}",
        fontsize=14, fontweight='bold'
    )
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cross_model_comparison(reports: List[ModelReport], save_path: Path):
    """
    Compare deformation profiles across multiple models.
    """
    if len(reports) < 2:
        return

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Deformation magnitude profiles
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Deformation Magnitude Profile")
    for report in reports:
        if report.wave_propagation is not None:
            layers = range(len(report.wave_propagation))
            ax.plot(layers, report.wave_propagation, '-o', markersize=3,
                    label=f"{report.model_name.split('/')[-1]} ({report.n_layers}L)")
    ax.set_xlabel("Layer (normalized)")
    ax.set_ylabel("Mean ||Δ||_F")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Divergence profiles
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Divergence Profile")
    for report in reports:
        divs = []
        for pa in report.prompt_analyses:
            for la in pa.layer_analyses:
                divs.append(la.divergence)
        if divs:
            n_layers = report.n_layers
            per_layer = [[] for _ in range(n_layers)]
            for pa in report.prompt_analyses:
                for i, la in enumerate(pa.layer_analyses):
                    if i < n_layers:
                        per_layer[i].append(la.divergence)
            means = [np.mean(pl) if pl else 0 for pl in per_layer]
            ax.plot(range(n_layers), means, '-o', markersize=3,
                    label=report.model_name.split('/')[-1])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean divergence")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Curl profiles
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Curl (Rotational Mixing) Profile")
    for report in reports:
        n_layers = report.n_layers
        per_layer = [[] for _ in range(n_layers)]
        for pa in report.prompt_analyses:
            for i, la in enumerate(pa.layer_analyses):
                if i < n_layers:
                    per_layer[i].append(la.curl_magnitude)
        means = [np.mean(pl) if pl else 0 for pl in per_layer]
        ax.plot(range(n_layers), means, '-o', markersize=3,
                label=report.model_name.split('/')[-1])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean curl")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. Condition number profiles
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Condition Number Profile")
    for report in reports:
        n_layers = report.n_layers
        per_layer = [[] for _ in range(n_layers)]
        for pa in report.prompt_analyses:
            for i, la in enumerate(pa.layer_analyses):
                if i < n_layers:
                    per_layer[i].append(la.condition_number)
        means = [np.mean(pl) if pl else 1 for pl in per_layer]
        ax.semilogy(range(n_layers), means, '-o', markersize=3,
                    label=report.model_name.split('/')[-1])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean condition number")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. Model comparison table
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis('off')
    table_data = []
    for report in reports:
        all_divs = [la.divergence for pa in report.prompt_analyses for la in pa.layer_analyses]
        all_curls = [la.curl_magnitude for pa in report.prompt_analyses for la in pa.layer_analyses]
        all_shears = [la.shear_magnitude for pa in report.prompt_analyses for la in pa.layer_analyses]
        all_mags = [la.deformation_magnitude for pa in report.prompt_analyses for la in pa.layer_analyses]
        table_data.append([
            report.model_name.split('/')[-1],
            f"{report.n_params / 1e6:.1f}M",
            str(report.n_layers),
            str(report.hidden_dim),
            f"{np.mean(all_divs):.3f}" if all_divs else "N/A",
            f"{np.mean(all_curls):.3f}" if all_curls else "N/A",
            f"{np.mean(all_shears):.3f}" if all_shears else "N/A",
            f"{np.mean(all_mags):.2f}" if all_mags else "N/A",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Model", "Params", "Layers", "d", "Div", "Curl", "Shear", "||Δ||"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title("Model Comparison Summary", fontsize=12, fontweight='bold', pad=20)

    fig.suptitle("Cross-Model Deformation Comparison", fontsize=14, fontweight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_holographic_test(analysis: PromptAnalysis, save_path: Path):
    """
    Visualize the holographic scrambling test — FFT analysis of layer deltas.

    Koch (Conjecture 2): "Information is holographically distributed like a
    Fourier transform — deleting frequency components degrades everything uniformly."
    """
    n_layers = len(analysis.layer_analyses)
    if n_layers == 0:
        return

    states = analysis.hidden_states
    n_total_layers, seq_len, hidden_dim = states.shape

    # Pick a few representative layers
    layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layer_indices = sorted(set([min(i, n_layers - 1) for i in layer_indices]))

    fig, axes = plt.subplots(2, len(layer_indices), figsize=(4 * len(layer_indices), 8))
    if len(layer_indices) == 1:
        axes = axes.reshape(2, 1)

    for col, ell in enumerate(layer_indices):
        delta = states[ell + 1] - states[ell]

        # Row 1: 2D FFT spectrum
        ax = axes[0, col]
        spectrum = np.abs(fftshift(fft2(delta)))
        ax.imshow(np.log1p(spectrum), aspect='auto', cmap='inferno')
        ax.set_title(f"Layer {ell} FFT", fontsize=10)
        if col == 0:
            ax.set_ylabel("Token freq")
        ax.set_xlabel("Dim freq")

        # Row 2: 1D FFT per dimension (averaged)
        ax = axes[1, col]
        spectra_1d = compute_1d_fft_per_dimension(delta)
        mean_spectrum = spectra_1d.mean(axis=1)
        freqs = fftfreq(seq_len)[:seq_len // 2 + 1]
        ax.plot(freqs, mean_spectrum, 'b-', linewidth=1.5)
        ax.fill_between(freqs, 0, mean_spectrum, alpha=0.3)
        ax.set_title(f"Layer {ell} Avg 1D FFT", fontsize=10)
        if col == 0:
            ax.set_ylabel("Magnitude")
        ax.set_xlabel("Frequency")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Holographic Scrambling Test (FFT): \"{analysis.prompt[:40]}...\"",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# RICH CONSOLE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def print_analysis_summary(analysis: PromptAnalysis):
    """Print a Rich summary of the analysis for one prompt."""
    table = Table(
        title=f"[bold]{analysis.model_name}[/] — \"{analysis.prompt[:60]}...\"",
        box=box.ROUNDED, show_lines=True
    )
    table.add_column("Layer", style="bold cyan", justify="center")
    table.add_column("||Δ||_F", justify="right")
    table.add_column("Div", justify="right")
    table.add_column("Curl", justify="right")
    table.add_column("Shear", justify="right")
    table.add_column("det(J)", justify="right")
    table.add_column("κ(J)", justify="right")

    for la in analysis.layer_analyses:
        div_color = "red" if la.divergence > 0.5 else ("blue" if la.divergence < -0.5 else "white")
        table.add_row(
            str(la.layer_idx),
            f"{la.deformation_magnitude:.3f}",
            f"[{div_color}]{la.divergence:.3f}[/]",
            f"{la.curl_magnitude:.3f}",
            f"{la.shear_magnitude:.3f}",
            f"{la.determinant:.4f}",
            f"{la.condition_number:.1f}",
        )

    console.print(table)


def print_trace_summary(trace: dict):
    """Print a Rich tree of the algorithm execution trace."""
    tree = Tree(f"[bold]Algorithm Trace: {trace['task_type']}[/]")

    for info in trace["per_layer"]:
        layer_node = tree.add(
            f"[cyan]Layer {info['layer']}[/] — "
            f"max_deform={info['max_deformation']:.3f}, "
            f"active_token=\"{info['most_active_token']}\""
        )
        layer_node.add(f"[dim]{info['interpretation']}[/]")

    console.print(tree)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM Space-Deformation Explorer — Test Koch's fibre bundle hypothesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: test 3 small models on all tasks
  python3 explore_deformations.py

  # Specific models
  python3 explore_deformations.py --models gpt2 EleutherAI/pythia-70m

  # GPU with specific tasks
  python3 explore_deformations.py --device cuda --tasks addition negation_parity

  # All tasks, all default models
  python3 explore_deformations.py --tasks all --device cpu
        """
    )
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="HuggingFace model names (default: gpt2, pythia-70m, opt-125m)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"],
                        help="Tasks to run (default: all). Options: " +
                             ", ".join(ALGORITHMIC_TASKS.keys()) + ", all")
    parser.add_argument("--output-dir", type=str, default="deformation_results",
                        help="Output directory for plots and data (default: deformation_results)")
    parser.add_argument("--max-layers-to-plot", type=int, default=5,
                        help="Max number of layers to generate deformed grid plots for")

    args = parser.parse_args()

    # Resolve models
    model_names = args.models if args.models else DEFAULT_MODELS
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve tasks
    if "all" in args.tasks:
        task_names = list(ALGORITHMIC_TASKS.keys())
    else:
        task_names = [t for t in args.tasks if t in ALGORITHMIC_TASKS]
        if not task_names:
            console.print("[red]No valid tasks specified. Available:[/]")
            for t in ALGORITHMIC_TASKS:
                console.print(f"  - {t}")
            sys.exit(1)

    # ════════════════════════════════════════════════════════════════════
    # MAIN EXECUTION
    # ════════════════════════════════════════════════════════════════════

    console.print(Panel(
        "[bold white]LLM Space-Deformation Explorer[/]\n"
        "[dim]Testing Koch's fibre bundle hypothesis via geometric analysis[/]\n\n"
        f"Models: {', '.join(model_names)}\n"
        f"Tasks:  {', '.join(task_names)}\n"
        f"Device: {args.device}\n"
        f"Output: {OUTPUT_DIR}/",
        title="[bold cyan]🔬 Configuration",
        border_style="cyan",
    ))

    all_reports: List[ModelReport] = []

    for model_name in model_names:
        console.print(f"\n{'═' * 70}")
        console.print(f"[bold cyan]  Model: {model_name}[/]")
        console.print(f"{'═' * 70}")

        try:
            model, tokenizer, n_params, n_layers, hidden_dim = load_model_and_tokenizer(
                model_name, args.device
            )
        except Exception as e:
            console.print(f"  [red]Failed to load {model_name}: {e}[/]")
            continue

        model_dir = OUTPUT_DIR / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        prompt_analyses: List[PromptAnalysis] = []
        all_traces: List[dict] = []

        for task_name in task_names:
            task_info = ALGORITHMIC_TASKS[task_name]
            console.print(f"\n  [yellow]Task: {task_name}[/] — {task_info['description']}")

            task_dir = model_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Analyzing prompts..."),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                ptask = progress.add_task("analyze", total=len(task_info["prompts"]))

                for prompt_idx, prompt in enumerate(task_info["prompts"]):
                    try:
                        analysis = analyze_prompt(
                            model, tokenizer, prompt, model_name, args.device
                        )
                        prompt_analyses.append(analysis)

                        # Print summary table
                        print_analysis_summary(analysis)

                        # Algorithm execution trace
                        trace = trace_algorithm_execution(
                            analysis.hidden_states, analysis.tokens, task_name
                        )
                        all_traces.append(trace)
                        print_trace_summary(trace)

                        # ── Generate visualizations ──────────────────────────

                        # 1. Jacobian invariants
                        plot_jacobian_invariants(
                            analysis,
                            task_dir / f"jacobian_invariants_p{prompt_idx}.png"
                        )

                        # 2. Fibre trajectories
                        plot_fibre_trajectories(
                            analysis,
                            task_dir / f"fibre_trajectories_p{prompt_idx}.png"
                        )

                        # 3. Holographic FFT test
                        plot_holographic_test(
                            analysis,
                            task_dir / f"holographic_fft_p{prompt_idx}.png"
                        )

                        # 4. Deformed grids (for selected layers)
                        n_layer_transitions = len(analysis.layer_analyses)
                        layer_indices_to_plot = [
                            0,
                            n_layer_transitions // 4,
                            n_layer_transitions // 2,
                            3 * n_layer_transitions // 4,
                            n_layer_transitions - 1,
                        ]
                        layer_indices_to_plot = sorted(set([
                            min(i, n_layer_transitions - 1)
                            for i in layer_indices_to_plot
                        ]))[:args.max_layers_to_plot]

                        for layer_idx in layer_indices_to_plot:
                            plot_deformed_grid(
                                analysis, layer_idx,
                                task_dir / f"deformed_grid_p{prompt_idx}_L{layer_idx}.png"
                            )

                        # 5. Algorithm trace visualization
                        plot_algorithm_trace(
                            trace,
                            task_dir / f"algorithm_trace_p{prompt_idx}.png"
                        )

                    except Exception as e:
                        console.print(f"    [red]Error analyzing prompt '{prompt[:40]}...': {e}[/]")
                        import traceback
                        traceback.print_exc()

                    progress.update(ptask, advance=1)

        # ── Compute model-level aggregates ──────────────────────────────

        # Wave propagation profile (mean deformation per layer)
        if prompt_analyses:
            max_layers_seen = max(len(pa.layer_analyses) for pa in prompt_analyses)
            wave_prop = np.zeros(max_layers_seen)
            counts = np.zeros(max_layers_seen)
            for pa in prompt_analyses:
                for i, la in enumerate(pa.layer_analyses):
                    wave_prop[i] += la.deformation_magnitude
                    counts[i] += 1
            counts = np.maximum(counts, 1)
            wave_prop /= counts
        else:
            wave_prop = None

        report = ModelReport(
            model_name=model_name,
            n_params=n_params,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            prompt_analyses=prompt_analyses,
            wave_propagation=wave_prop,
        )
        all_reports.append(report)

        # ── Print model summary ─────────────────────────────────────────
        console.print(Panel(
            f"[bold]{model_name}[/]\n"
            f"  Params: {n_params/1e6:.1f}M | Layers: {n_layers} | d={hidden_dim}\n"
            f"  Prompts analyzed: {len(prompt_analyses)}\n"
            f"  Traces computed: {len(all_traces)}\n"
            f"  Plots saved to: {model_dir}/",
            title="[bold green]✓ Model Complete",
            border_style="green",
        ))

        # Free memory
        del model
        import gc
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # CROSS-MODEL COMPARISON
    # ════════════════════════════════════════════════════════════════════

    if len(all_reports) >= 2:
        console.print(f"\n{'═' * 70}")
        console.print("[bold cyan]  Cross-Model Comparison[/]")
        console.print(f"{'═' * 70}")

        plot_cross_model_comparison(
            all_reports,
            OUTPUT_DIR / "cross_model_comparison.png"
        )
        console.print(f"  [green]✓ Cross-model comparison saved to {OUTPUT_DIR}/cross_model_comparison.png[/]")

    # ════════════════════════════════════════════════════════════════════
    # SAVE SUMMARY JSON
    # ════════════════════════════════════════════════════════════════════

    summary = {
        "models": [],
        "tasks": task_names,
        "device": args.device,
    }

    for report in all_reports:
        model_summary = {
            "name": report.model_name,
            "n_params": report.n_params,
            "n_layers": report.n_layers,
            "hidden_dim": report.hidden_dim,
            "n_prompts_analyzed": len(report.prompt_analyses),
            "wave_propagation": report.wave_propagation.tolist() if report.wave_propagation is not None else None,
            "per_layer_stats": [],
        }

        # Aggregate per-layer stats across all prompts
        if report.prompt_analyses:
            max_layers = max(len(pa.layer_analyses) for pa in report.prompt_analyses)
            for ell in range(max_layers):
                layer_divs = []
                layer_curls = []
                layer_shears = []
                layer_mags = []
                layer_conds = []
                for pa in report.prompt_analyses:
                    if ell < len(pa.layer_analyses):
                        la = pa.layer_analyses[ell]
                        layer_divs.append(la.divergence)
                        layer_curls.append(la.curl_magnitude)
                        layer_shears.append(la.shear_magnitude)
                        layer_mags.append(la.deformation_magnitude)
                        layer_conds.append(la.condition_number)

                model_summary["per_layer_stats"].append({
                    "layer": ell,
                    "mean_divergence": float(np.mean(layer_divs)) if layer_divs else 0,
                    "mean_curl": float(np.mean(layer_curls)) if layer_curls else 0,
                    "mean_shear": float(np.mean(layer_shears)) if layer_shears else 0,
                    "mean_deformation": float(np.mean(layer_mags)) if layer_mags else 0,
                    "mean_condition": float(np.mean(layer_conds)) if layer_conds else 0,
                })

        summary["models"].append(model_summary)

    summary_path = OUTPUT_DIR / "exploration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════════════════════

    console.print(f"\n{'═' * 70}")
    console.print("[bold green]  EXPLORATION COMPLETE[/]")
    console.print(f"{'═' * 70}")

    # Summary table
    final_table = Table(
        title="[bold]Exploration Results Summary[/]",
        box=box.ROUNDED, show_lines=True
    )
    final_table.add_column("Model", style="bold cyan")
    final_table.add_column("Params", justify="right")
    final_table.add_column("Layers", justify="center")
    final_table.add_column("d", justify="center")
    final_table.add_column("Prompts", justify="center")
    final_table.add_column("Mean Div", justify="right")
    final_table.add_column("Mean Curl", justify="right")
    final_table.add_column("Mean ||Δ||", justify="right")

    for report in all_reports:
        all_divs = [la.divergence for pa in report.prompt_analyses for la in pa.layer_analyses]
        all_curls = [la.curl_magnitude for pa in report.prompt_analyses for la in pa.layer_analyses]
        all_mags = [la.deformation_magnitude for pa in report.prompt_analyses for la in pa.layer_analyses]

        final_table.add_row(
            report.model_name.split("/")[-1],
            f"{report.n_params/1e6:.1f}M",
            str(report.n_layers),
            str(report.hidden_dim),
            str(len(report.prompt_analyses)),
            f"{np.mean(all_divs):.3f}" if all_divs else "N/A",
            f"{np.mean(all_curls):.3f}" if all_curls else "N/A",
            f"{np.mean(all_mags):.2f}" if all_mags else "N/A",
        )

    console.print(final_table)

    # Key observations
    console.print(Panel(
        "[bold]Key things to look for in the generated plots:[/]\n\n"
        "1. [cyan]Jacobian Invariants[/]: Do divergence/curl/shear peak at middle layers?\n"
        "   Koch (C4) predicts inner layers compute, outer layers translate.\n\n"
        "2. [cyan]Deformed Grids[/]: Are deformations structured and prompt-dependent?\n"
        "   Koch (C1) predicts space-morphing encodes semantic content.\n\n"
        "3. [cyan]FFT Spectra[/]: Is information distributed across frequencies?\n"
        "   Koch (C2) predicts holographic scrambling like a Fourier transform.\n\n"
        "4. [cyan]Fibre Trajectories[/]: Do tokens follow coherent paths through layers?\n"
        "   Koch (C5) predicts the Jacobian field carries holographic information.\n\n"
        "5. [cyan]Algorithm Traces[/]: Do negation-parity tasks show reflective eigenvalues?\n"
        "   Koch (C3) predicts topological structures perform computation.\n\n"
        "6. [cyan]Cross-Model Comparison[/]: Are deformation profiles consistent across\n"
        "   architectures? If so, Koch's framework describes something fundamental.\n\n"
        f"[bold green]All results saved to: {OUTPUT_DIR}/[/]",
        title="[bold yellow]🔍 Interpretation Guide",
        border_style="yellow",
    ))

    console.print(f"\n[bold]Summary JSON: {summary_path}[/]")
    console.print(f"[bold]Total plots generated: {sum(len(list((OUTPUT_DIR / m.model_name.replace('/', '_')).rglob('*.png'))) for m in all_reports if (OUTPUT_DIR / m.model_name.replace('/', '_')).exists())}[/]")
    console.print(f"\n[dim]Run 'python3 {sys.argv[0]} --help' for more options.[/]\n")


if __name__ == "__main__":
    main()
