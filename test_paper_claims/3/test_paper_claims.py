#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "scipy",
#   "matplotlib",
#   "seaborn",
#   "rich",
# ]
# ///

"""
LLM Space-Deformation Explorer v2
==================================

Traces how LLMs actually compute by watching what happens to
real hidden-state dimensions at every layer.

NO PCA. NO DIMENSIONALITY REDUCTION. Only real data.

Usage:
    python3 explore_deformations.py
    python3 explore_deformations.py --models gpt2 EleutherAI/pythia-70m
    python3 explore_deformations.py --device cuda

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


ensure_safe_env()

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════

import argparse
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm
import seaborn as sns

from scipy.linalg import svdvals
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kurtosis

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.tree import Tree
from rich import box

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

# We pick specific dimensions to watch — the TOP MOVERS
# (dimensions that change the most). This is NOT dimensionality reduction.
# We are looking at REAL dimensions, just choosing which ones to show.
N_DIMS_TO_TRACK = 8  # Show the top 8 most active real dimensions

# Algorithmic tasks — we trace HOW the model executes these
TASKS = {
    "addition": {
        "prompts": [
            "2 + 3 =",
            "15 + 27 =",
            "100 + 200 =",
        ],
        "description": "Integer addition",
    },
    "negation_parity": {
        "prompts": [
            "not not True is ",
            "not not not True is ",
            "not not not not True is ",
        ],
        "description": "Negation parity (Z/2Z group operation)",
    },
    "comparison": {
        "prompts": [
            "Is 5 greater than 3? Answer: ",
            "Is 2 greater than 9? Answer: ",
        ],
        "description": "Number comparison",
    },
    "pattern": {
        "prompts": [
            "1, 2, 3, 4, ",
            "2, 4, 6, 8, ",
        ],
        "description": "Sequence continuation",
    },
    "semantic": {
        "prompts": [
            "The sky is blue",
            "The sky is red",
            "The ocean is blue",
        ],
        "description": "Semantic substitution — which dimensions respond?",
    },
}

OUTPUT_DIR = Path("deformation_results_v2")


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & HIDDEN STATE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, device: str = "cpu"):
    """Load model and tokenizer."""
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


def extract_hidden_states(model, tokenizer, prompt: str, device: str = "cpu"):
    """
    Extract hidden states at every layer. Returns REAL data, no projection.
    Shape: (n_layers+1, seq_len, hidden_dim)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]
    states = np.stack(hidden, axis=0)
    return states, tokens


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS: Find the most interesting real dimensions
# ════════════════════════════════════════════════════════════════════════════

def find_top_moving_dimensions(states: np.ndarray, n_dims: int = 8) -> np.ndarray:
    """
    Find the real hidden dimensions that change the most across layers.
    NO PCA. These are actual dimension indices in the model's hidden space.

    states: (n_layers, seq_len, hidden_dim)
    Returns: array of dimension indices, shape (n_dims,)
    """
    n_layers, seq_len, hidden_dim = states.shape

    # Total absolute change per dimension across all layers and tokens
    total_change = np.zeros(hidden_dim)
    for ell in range(n_layers - 1):
        delta = states[ell + 1] - states[ell]  # (seq_len, hidden_dim)
        total_change += np.sum(np.abs(delta), axis=0)  # sum over tokens

    # Pick the top-N dimensions by total movement
    top_dims = np.argsort(total_change)[-n_dims:][::-1]
    return top_dims


def find_dimensions_that_differ(states_a: np.ndarray, states_b: np.ndarray,
                                 n_dims: int = 8) -> np.ndarray:
    """
    Find real dimensions where two prompts diverge the most.
    Koch (Fig. 8): "Selective dimensional response to semantic substitution."

    states_a, states_b: (n_layers, seq_len, hidden_dim)
    Returns: dimension indices where the two prompts differ most
    """
    n_layers = min(states_a.shape[0], states_b.shape[0])
    seq_len = min(states_a.shape[1], states_b.shape[1])
    hidden_dim = states_a.shape[2]

    # Per-dimension total difference across all layers
    dim_diff = np.zeros(hidden_dim)
    for ell in range(n_layers):
        diff = states_a[ell, :seq_len, :] - states_b[ell, :seq_len, :]
        dim_diff += np.sum(np.abs(diff), axis=0)

    top_dims = np.argsort(dim_diff)[-n_dims:][::-1]
    return top_dims


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS: Jacobian invariants on REAL dimensions
# ════════════════════════════════════════════════════════════════════════════

def compute_per_token_deformation(states: np.ndarray):
    """
    For each layer transition, compute how much each token moved
    in REAL space (L2 norm of the delta vector).

    Returns: (n_layers-1, seq_len) array of deformation magnitudes
    """
    n_layers, seq_len, hidden_dim = states.shape
    deformations = np.zeros((n_layers - 1, seq_len))
    for ell in range(n_layers - 1):
        delta = states[ell + 1] - states[ell]
        deformations[ell] = np.linalg.norm(delta, axis=1)
    return deformations


def compute_per_dimension_delta(states: np.ndarray, dim_idx: int):
    """
    Track how a SPECIFIC real dimension changes across layers for each token.

    Returns: (n_layers-1, seq_len) — the signed change in dimension dim_idx
    """
    n_layers, seq_len, _ = states.shape
    deltas = np.zeros((n_layers - 1, seq_len))
    for ell in range(n_layers - 1):
        deltas[ell] = states[ell + 1, :, dim_idx] - states[ell, :, dim_idx]
    return deltas


def compute_direction_coherence(states: np.ndarray):
    """
    Koch (Conjecture 1): Space-morphing means tokens move TOGETHER as a field,
    not independently. Measure: mean pairwise cosine similarity of per-token
    delta vectors at each layer.

    Returns: (n_layers-1,) array of coherence values in [-1, 1]
    """
    n_layers, seq_len, hidden_dim = states.shape
    coherences = np.zeros(n_layers - 1)

    for ell in range(n_layers - 1):
        delta = states[ell + 1] - states[ell]  # (seq_len, hidden_dim)
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        delta_normed = delta / norms

        if seq_len < 2:
            coherences[ell] = 0.0
            continue

        cos_matrix = delta_normed @ delta_normed.T
        triu_idx = np.triu_indices(seq_len, k=1)
        coherences[ell] = float(np.mean(cos_matrix[triu_idx]))

    return coherences


def compute_expansion_contraction(states: np.ndarray):
    """
    Koch (Section 4): Divergence = tr(J) = local expansion/contraction.
    We approximate this by tracking how the pairwise distances between
    tokens change at each layer. If distances grow → expansion. Shrink → contraction.

    Returns: (n_layers-1,) array of mean distance ratios (>1 = expansion, <1 = contraction)
    """
    n_layers, seq_len, hidden_dim = states.shape
    if seq_len < 2:
        return np.ones(n_layers - 1)

    ratios = np.zeros(n_layers - 1)
    for ell in range(n_layers - 1):
        d_before = pdist(states[ell])
        d_after = pdist(states[ell + 1])
        # Avoid division by zero
        d_before = np.maximum(d_before, 1e-10)
        ratios[ell] = float(np.mean(d_after / d_before))

    return ratios


def compute_singular_value_spectrum(states: np.ndarray):
    """
    At each layer, compute the singular values of the token-by-dimension matrix.
    This shows how "spread out" the representations are in real space.

    Returns: list of (min(seq_len, hidden_dim),) arrays, one per layer
    """
    n_layers, seq_len, hidden_dim = states.shape
    spectra = []
    for ell in range(n_layers):
        svs = svdvals(states[ell])
        spectra.append(svs)
    return spectra


def compute_layer_delta_svd(states: np.ndarray):
    """
    SVD of the delta matrix at each layer transition.
    This reveals the RANK of the deformation — how many independent
    directions the model uses to transform the space.

    Koch (Meta-test): If rank ≈ 1, all tokens move the same way (just a bias).
    If rank >> 1, the model is doing genuine per-token space morphing.

    Returns: list of singular value arrays, one per layer transition
    """
    n_layers, seq_len, hidden_dim = states.shape
    delta_spectra = []
    for ell in range(n_layers - 1):
        delta = states[ell + 1] - states[ell]
        svs = svdvals(delta)
        delta_spectra.append(svs)
    return delta_spectra


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: Clear, labeled, traceable plots
# ════════════════════════════════════════════════════════════════════════════

def plot_dimension_trajectories(states, tokens, top_dims, model_name, prompt, save_path):
    """
    PLOT 1: "Fibre View" — Track specific REAL dimensions across layers.

    Each subplot shows one real dimension (e.g., "Dimension 47").
    Each colored line is one token. You can trace exactly how
    token "the" changes in dimension 47 from layer 0 to layer 12.

    Koch (Fig. 4): "Each vertical strand is a fibre (one token position)."
    """
    n_layers, seq_len, hidden_dim = states.shape
    n_dims = len(top_dims)

    cols = min(4, n_dims)
    rows = math.ceil(n_dims / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    cmap = cm.get_cmap('tab10', seq_len)
    layer_indices = np.arange(n_layers)

    for idx, dim in enumerate(top_dims):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        for tok_idx in range(seq_len):
            values = states[:, tok_idx, dim]
            label = tokens[tok_idx].strip()[:10] if tok_idx < 6 else None
            ax.plot(layer_indices, values, '-o', color=cmap(tok_idx),
                    markersize=2, linewidth=1.2, alpha=0.8, label=label)

        ax.set_title(f"Real Dimension {dim}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Value in dim {dim}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, loc='best', ncol=2)

    # Hide unused subplots
    for idx in range(n_dims, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Token Trajectories in Real Dimensions\n"
        f"Model: {model_name} | Prompt: \"{prompt[:50]}\"",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_deformation_heatmap(states, tokens, model_name, prompt, save_path):
    """
    PLOT 2: "Who moves where?" — Heatmap of per-token deformation at each layer.

    X-axis: tokens (labeled with actual text)
    Y-axis: layer transitions (Layer 0→1, Layer 1→2, ...)
    Color: how much that token moved (L2 norm of delta vector)

    Koch (Observation 3): "Inner layers have the highest amount of space morphing."
    You can see this directly: middle rows should be brighter.
    """
    deformations = compute_per_token_deformation(states)
    n_transitions, seq_len = deformations.shape

    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.8), max(6, n_transitions * 0.4)))

    tok_labels = [t.strip()[:12] for t in tokens[:seq_len]]
    layer_labels = [f"L{i}→{i+1}" for i in range(n_transitions)]

    sns.heatmap(
        deformations, ax=ax, cmap='YlOrRd',
        xticklabels=tok_labels, yticklabels=layer_labels,
        annot=n_transitions <= 15 and seq_len <= 15,
        fmt='.1f' if n_transitions <= 15 and seq_len <= 15 else '',
        linewidths=0.5, linecolor='white',
    )

    ax.set_xlabel("Token", fontsize=12)
    ax.set_ylabel("Layer Transition", fontsize=12)
    ax.set_title(
        f"Per-Token Deformation Magnitude (||Δh||₂)\n"
        f"Model: {model_name} | Prompt: \"{prompt[:50]}\"",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_dimension_delta_heatmap(states, tokens, top_dims, model_name, prompt, save_path):
    """
    PLOT 3: "What changes in each dimension?" — Signed delta per dimension.

    For each of the top-moving dimensions, show a heatmap:
    X-axis: tokens, Y-axis: layer transitions
    Color: SIGNED change (blue = decrease, red = increase, white = no change)

    This lets you trace the exact computation: "In dimension 47,
    token 'not' increases by 0.3 at layer 5, then decreases by 0.5 at layer 6."
    """
    n_dims = len(top_dims)
    cols = min(4, n_dims)
    rows = math.ceil(n_dims / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    tok_labels = [t.strip()[:10] for t in tokens]
    n_transitions = states.shape[0] - 1
    layer_labels = [f"L{i}→{i+1}" for i in range(n_transitions)]

    for idx, dim in enumerate(top_dims):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        deltas = compute_per_dimension_delta(states, dim)

        # Symmetric colormap centered at 0
        vmax = max(abs(deltas.min()), abs(deltas.max()), 1e-6)
        sns.heatmap(
            deltas, ax=ax, cmap='RdBu_r', center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=tok_labels, yticklabels=layer_labels if idx % cols == 0 else False,
            linewidths=0.3, linecolor='gray',
        )
        ax.set_title(f"Dim {dim}: Signed Δ per layer", fontsize=10, fontweight='bold')
        if idx >= (rows - 1) * cols:
            ax.set_xlabel("Token", fontsize=9)

    for idx in range(n_dims, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Per-Dimension Signed Changes (blue=decrease, red=increase)\n"
        f"Model: {model_name} | Prompt: \"{prompt[:50]}\"",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_geometric_invariants(states, tokens, model_name, prompt, save_path):
    """
    PLOT 4: "The geometry of the morphing" — Coherence, expansion, and deformation.

    Three panels showing layer-by-layer geometric properties:
    1. Direction coherence: Do tokens move together (field) or independently?
    2. Expansion/contraction: Does the space expand or shrink?
    3. Total deformation: How much does the space change overall?

    Koch (Conjecture 1): "The geometry of the map — its Jacobian field,
    eigenvalue spectrum, divergence, curl, and shear — provides indirect
    access to the model's reasoning."
    """
    coherences = compute_direction_coherence(states)
    expansion = compute_expansion_contraction(states)
    deformations = compute_per_token_deformation(states)
    mean_deformation = deformations.mean(axis=1)

    n_transitions = len(coherences)
    layers = np.arange(n_transitions)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Coherence
    ax = axes[0]
    colors = ['green' if c > 0.1 else ('red' if c < -0.1 else 'gray') for c in coherences]
    ax.bar(layers, coherences, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Layer Transition", fontsize=11)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax.set_title("Direction Coherence\n(Do tokens move together?)", fontsize=12, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate
    ax.text(0.02, 0.98, "Green = tokens move TOGETHER\n(space deformation)",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 2: Expansion/Contraction
    ax = axes[1]
    colors = ['red' if r > 1.01 else ('blue' if r < 0.99 else 'gray') for r in expansion]
    ax.bar(layers, expansion, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='black', linewidth=1, linestyle='--', label='Isometric (no change)')
    ax.set_xlabel("Layer Transition", fontsize=11)
    ax.set_ylabel("Mean Distance Ratio (after/before)", fontsize=11)
    ax.set_title("Expansion / Contraction\n(Does the space stretch or shrink?)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax.text(0.02, 0.98, "Red = EXPANSION (>1)\nBlue = CONTRACTION (<1)",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 3: Total deformation
    ax = axes[2]
    ax.bar(layers, mean_deformation, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Layer Transition", fontsize=11)
    ax.set_ylabel("Mean ||Δh||₂ across tokens", fontsize=11)
    ax.set_title("Total Deformation per Layer\n(How much does the space change?)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Mark the peak
    peak_layer = np.argmax(mean_deformation)
    ax.annotate(f"Peak: Layer {peak_layer}→{peak_layer+1}",
                xy=(peak_layer, mean_deformation[peak_layer]),
                xytext=(peak_layer + 1, mean_deformation[peak_layer] * 1.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red', fontweight='bold')

    fig.suptitle(
        f"Geometric Invariants of the Space Morphing\n"
        f"Model: {model_name} | Prompt: \"{prompt[:50]}\"",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_delta_rank_analysis(states, model_name, prompt, save_path):
    """
    PLOT 5: "Is this real space-morphing or just a global shift?"

    Koch (Meta-test): If the delta matrix at each layer has rank ≈ 1,
    all tokens move the same way and there's no real "morphing."
    If rank >> 1, different tokens are deformed differently.

    Shows:
    1. Singular values of the delta matrix at each layer
    2. Effective rank (participation ratio)
    3. Fraction of variance NOT explained by the mean shift
    """
    delta_spectra = compute_layer_delta_svd(states)
    n_transitions = len(delta_spectra)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Singular value curves
    ax = axes[0]
    cmap = cm.get_cmap('viridis', n_transitions)
    for ell, svs in enumerate(delta_spectra):
        n_show = min(20, len(svs))
        if ell % max(1, n_transitions // 6) == 0 or ell == n_transitions - 1:
            ax.semilogy(range(n_show), svs[:n_show], '-o', color=cmap(ell),
                       markersize=3, linewidth=1.5, label=f"L{ell}→{ell+1}")
    ax.set_xlabel("Singular Value Index", fontsize=11)
    ax.set_ylabel("Singular Value (log scale)", fontsize=11)
    ax.set_title("Singular Values of Layer Deltas\n(Steep drop = low rank = simple shift)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Effective rank per layer
    ax = axes[1]
    eff_ranks = []
    for svs in delta_spectra:
        svs_pos = svs[svs > 1e-10]
        if len(svs_pos) >= 2:
            pr = (np.sum(svs_pos) ** 2) / (np.sum(svs_pos ** 2) + 1e-10)
        else:
            pr = 1.0
        eff_ranks.append(pr)

    colors = ['green' if r > 3 else ('orange' if r > 1.5 else 'red') for r in eff_ranks]
    ax.bar(range(n_transitions), eff_ranks, color=colors, alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='red', linewidth=1, linestyle='--', label='Rank 1 (global shift)')
    ax.set_xlabel("Layer Transition", fontsize=11)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Effective Rank of Deformation\n(>1 = genuine per-token morphing)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax.text(0.02, 0.98, "Green = rich morphing (rank>3)\nRed = simple shift (rank≈1)",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 3: Variance not explained by mean shift
    ax = axes[2]
    diversities = []
    for ell in range(states.shape[0] - 1):
        delta = states[ell + 1] - states[ell]  # (seq_len, hidden_dim)
        mean_delta = delta.mean(axis=0, keepdims=True)  # (1, hidden_dim)
        residual = delta - mean_delta  # per-token deviation from mean shift

        total_var = np.sum(delta ** 2)
        residual_var = np.sum(residual ** 2)

        if total_var > 1e-10:
            diversity = residual_var / total_var
        else:
            diversity = 0.0
        diversities.append(diversity)

    colors = ['green' if d > 0.3 else ('orange' if d > 0.1 else 'red') for d in diversities]
    ax.bar(range(len(diversities)), diversities, color=colors, alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.axhline(0.5, color='green', linewidth=1, linestyle='--', label='50% (rich morphing)')
    ax.axhline(0.1, color='red', linewidth=1, linestyle='--', label='10% (mostly global shift)')
    ax.set_xlabel("Layer Transition", fontsize=11)
    ax.set_ylabel("Fraction of variance NOT from mean shift", fontsize=11)
    ax.set_title("Deformation Diversity\n(Is each token deformed differently?)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.98, "Green = genuine per-token morphing\nRed = just a global shift",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(
        f"Is This Real Space-Morphing or Just a Global Shift?\n"
        f"Model: {model_name} | Prompt: \"{prompt[:50]}\"",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_token_pair_distance_evolution(states, tokens, model_name, prompt, save_path):
    """
    PLOT 6: "How do token relationships change across layers?"

    For every pair of tokens, track their REAL Euclidean distance
    across layers. This directly shows Koch's "space morphing":
    if the space is being deformed, distances between tokens change.

    Koch (Observation 1): "The deformations are prompt-dependent:
    different inputs produce qualitatively different morphings."

    Koch (Section 5): "Positive ORC indicates that neighboring tokens
    converge (a gravitational source); negative ORC indicates divergence."
    """
    n_layers, seq_len, hidden_dim = states.shape

    if seq_len < 2 or seq_len > 12:
        # Too few or too many tokens to plot pairwise
        return

    # Compute pairwise distances at each layer
    n_pairs = seq_len * (seq_len - 1) // 2
    pair_distances = np.zeros((n_layers, n_pairs))
    pair_labels = []

    pair_idx = 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            for ell in range(n_layers):
                pair_distances[ell, pair_idx] = np.linalg.norm(
                    states[ell, i] - states[ell, j]
                )
            tok_i = tokens[i].strip()[:6]
            tok_j = tokens[j].strip()[:6]
            pair_labels.append(f"{tok_i}↔{tok_j}")
            pair_idx += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Distance evolution lines
    ax = axes[0]
    cmap = cm.get_cmap('tab20', n_pairs)
    for p in range(min(n_pairs, 15)):  # Show at most 15 pairs
        ax.plot(range(n_layers), pair_distances[:, p], '-o',
                color=cmap(p), markersize=2, linewidth=1.2, alpha=0.7,
                label=pair_labels[p] if p < 8 else None)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Euclidean Distance (real space)", fontsize=11)
    ax.set_title("Token-Pair Distance Evolution\n(How relationships change across layers)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Distance matrix at first vs last layer
    ax = axes[1]
    dist_first = squareform(pdist(states[0]))
    dist_last = squareform(pdist(states[-1]))

    # Ratio: how much did each distance change?
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = dist_last / (dist_first + 1e-10)

    tok_labels = [t.strip()[:8] for t in tokens[:seq_len]]
    vmax = min(np.nanpercentile(ratio[ratio > 0], 95), 3.0)
    sns.heatmap(
        ratio, ax=ax, cmap='RdBu_r', center=1.0,
        vmin=0, vmax=vmax,
        xticklabels=tok_labels, yticklabels=tok_labels,
        linewidths=0.3, linecolor='gray',
        annot=seq_len <= 10, fmt='.2f',
    )
    ax.set_title("Distance Ratio (Last Layer / First Layer)\n"
                 "(Blue=tokens got closer, Red=tokens got farther)",
                 fontsize=10, fontweight='bold')

    fig.suptitle(
        f"Token Relationship Evolution\n"
        f"Model: {model_name} | Prompt: \"{prompt[:50]}\"",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_semantic_comparison(states_a, states_b, tokens_a, tokens_b,
                             prompt_a, prompt_b, model_name, save_path):
    """
    PLOT 7: "Which dimensions respond to a semantic change?"

    Koch (Fig. 8, Observation 2): "Comparing two prompts that differ in a
    single concept reveals that the change does not affect all dimensions
    equally: some dimensions show large shifts while others remain unchanged."

    This plot shows REAL dimensions, not PCA. For each dimension, we compute
    the total absolute difference between the two prompts across all layers.
    Then we show the top-N most responsive dimensions.
    """
    n_layers = min(states_a.shape[0], states_b.shape[0])
    seq_len = min(states_a.shape[1], states_b.shape[1])
    hidden_dim = states_a.shape[2]

    # Find dimensions that differ most
    top_dims = find_dimensions_that_differ(states_a, states_b, n_dims=N_DIMS_TO_TRACK)

    fig, axes = plt.subplots(2, len(top_dims), figsize=(4 * len(top_dims), 8),
                             squeeze=False)

    for col, dim in enumerate(top_dims):
        # Row 1: Prompt A trajectory in this dimension
        ax = axes[0][col]
        for tok_idx in range(min(seq_len, 6)):
            values = states_a[:n_layers, tok_idx, dim]
            label = tokens_a[tok_idx].strip()[:8] if tok_idx < 4 else None
            ax.plot(range(n_layers), values, '-o', markersize=2, linewidth=1.2,
                    alpha=0.7, label=label)
        ax.set_title(f"Dim {dim}: \"{prompt_a[:20]}...\"", fontsize=9, fontweight='bold')
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Value in dim {dim}")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=5, ncol=2)

        # Row 2: Prompt B trajectory in this dimension
        ax = axes[1][col]
        for tok_idx in range(min(seq_len, 6)):
            values = states_b[:n_layers, tok_idx, dim]
            label = tokens_b[tok_idx].strip()[:8] if tok_idx < 4 else None
            ax.plot(range(n_layers), values, '-o', markersize=2, linewidth=1.2,
                    alpha=0.7, label=label)
        ax.set_title(f"Dim {dim}: \"{prompt_b[:20]}...\"", fontsize=9, fontweight='bold')
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Value in dim {dim}")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=5, ncol=2)

    fig.suptitle(
        f"Selective Dimensional Response to Semantic Change\n"
        f"Model: {model_name} | Top {len(top_dims)} most-different real dimensions",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_wave_propagation(states, tokens, model_name, prompt, save_path):
    """
    PLOT 8: "The wave of space-morphing"

    Koch (Section 2.3): "The cumulative effect h_i^(L) = h_i^(0) + sum(delta)
    is a superposition of deformations — a 'wave' of space-morphing that
    propagates from layer 0 to layer L."

    This shows the cumulative deformation from the embedding layer,
    broken down by token. You can see the "wave" build up.
    """
    n_layers, seq_len, hidden_dim = states.shape

    # Cumulative displacement from embedding for each token
    cumulative = np.zeros((n_layers, seq_len))
    for ell in range(n_layers):
        for tok in range(seq_len):
            cumulative[ell, tok] = np.linalg.norm(states[ell, tok] - states[0, tok])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Cumulative displacement lines per token
    ax = axes[0]
    cmap = cm.get_cmap('tab10', seq_len)
    for tok in range(min(seq_len, 10)):
        label = tokens[tok].strip()[:10]
        ax.plot(range(n_layers), cumulative[:, tok], '-o',
                color=cmap(tok), markersize=3, linewidth=1.5, alpha=0.8,
                label=label)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("||h^(ℓ) - h^(0)||₂ (cumulative displacement)", fontsize=11)
    ax.set_title("Wave Propagation: Cumulative Displacement from Embedding",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Heatmap of cumulative displacement
    ax = axes[1]
    tok_labels = [t.strip()[:10] for t in tokens[:seq_len]]
    layer_labels = [f"L{i}" for i in range(n_layers)]
    sns.heatmap(
        cumulative, ax=ax, cmap='magma',
        xticklabels=tok_labels, yticklabels=layer_labels,
        linewidths=0.3, linecolor='gray',
    )
    ax.set_xlabel("Token", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Cumulative Displacement Heatmap\n(Brighter = farther from embedding)",
                 fontsize=10, fontweight='bold')

    fig.suptitle(
        f"Wave of Space-Morphing\n"
        f"Model: {model_name} | Prompt: \"{prompt[:50]}\"",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_last_layer_reversal(states, tokens, model_name, prompt, save_path):
    """
    PLOT 9: "Does the last layer reverse the deformation?"

    Koch (Conjecture 4, Observation 3): "The last layer often exhibits a
    'reversal' pattern where the deformation partially undoes the inner-layer
    morphing... returning the space to a configuration readable by the
    language-modeling head."

    We measure: cosine similarity between each layer's delta and the
    first layer's delta. If the last layer reverses, its cosine should
    be NEGATIVE (pointing in the opposite direction).
    """
    n_layers, seq_len, hidden_dim = states.shape
    if n_layers < 4:
        return

    first_delta = (states[1] - states[0]).flatten()
    first_norm = np.linalg.norm(first_delta)
    if first_norm < 1e-10:
        return

    cosines = []
    for ell in range(n_layers - 1):
        delta = (states[ell + 1] - states[ell]).flatten()
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-10:
            cosines.append(0.0)
        else:
            cosines.append(float(np.dot(first_delta, delta) / (first_norm * delta_norm)))

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['green' if c > 0.1 else ('red' if c < -0.1 else 'gray') for c in cosines]
    bars = ax.bar(range(len(cosines)), cosines, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel("Layer Transition", fontsize=12)
    ax.set_ylabel("Cosine Similarity with First Layer's Delta", fontsize=12)
    ax.set_title(
        f"Last-Layer Reversal Test\n"
        f"(Negative = layer pushes OPPOSITE to first layer = reversal)\n"
        f"Model: {model_name} | Prompt: \"{prompt[:50]}\"",
        fontsize=12, fontweight='bold'
    )
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate last bar
    last_cos = cosines[-1]
    if last_cos < -0.1:
        ax.annotate(f"REVERSAL\n(cos={last_cos:.2f})",
                    xy=(len(cosines) - 1, last_cos),
                    xytext=(len(cosines) - 2, last_cos - 0.3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')
    else:
        ax.annotate(f"No reversal\n(cos={last_cos:.2f})",
                    xy=(len(cosines) - 1, last_cos),
                    xytext=(len(cosines) - 2, last_cos + 0.2),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=9, color='gray')

    ax.text(0.02, 0.02,
            "Green = same direction as first layer\n"
            "Red = opposite direction (reversal)\n"
            "Gray = orthogonal (independent)",
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# RICH CONSOLE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def print_analysis_summary(states, tokens, model_name, prompt):
    """Print a Rich summary of the analysis."""
    n_layers, seq_len, hidden_dim = states.shape

    deformations = compute_per_token_deformation(states)
    coherences = compute_direction_coherence(states)
    expansion = compute_expansion_contraction(states)

    table = Table(
        title=f"[bold]{model_name}[/] — \"{prompt[:60]}\"",
        box=box.ROUNDED, show_lines=True
    )
    table.add_column("Layer", style="bold cyan", justify="center")
    table.add_column("Mean ||Δ||₂", justify="right")
    table.add_column("Max ||Δ||₂", justify="right")
    table.add_column("Coherence", justify="right")
    table.add_column("Expansion", justify="right")
    table.add_column("Most Active Token", justify="left")

    for ell in range(n_layers - 1):
        mean_def = deformations[ell].mean()
        max_def = deformations[ell].max()
        most_active = int(np.argmax(deformations[ell]))
        tok_name = tokens[most_active].strip()[:12] if most_active < len(tokens) else "?"

        coh = coherences[ell]
        coh_color = "green" if coh > 0.1 else ("red" if coh < -0.1 else "white")

        exp = expansion[ell]
        exp_color = "red" if exp > 1.01 else ("blue" if exp < 0.99 else "white")

        table.add_row(
            f"{ell}→{ell+1}",
            f"{mean_def:.3f}",
            f"{max_def:.3f}",
            f"[{coh_color}]{coh:.3f}[/]",
            f"[{exp_color}]{exp:.3f}[/]",
            tok_name,
        )

    console.print(table)


def print_algorithm_trace(states, tokens, task_type):
    """Print a Rich tree showing how the model 'executes' the algorithm."""
    n_layers, seq_len, hidden_dim = states.shape
    deformations = compute_per_token_deformation(states)
    coherences = compute_direction_coherence(states)

    tree = Tree(f"[bold]Algorithm Trace: {task_type}[/]")

    for ell in range(n_layers - 1):
        most_active = int(np.argmax(deformations[ell]))
        tok_name = tokens[most_active].strip()[:12] if most_active < len(tokens) else "?"
        mean_def = deformations[ell].mean()
        coh = coherences[ell]

        # Interpret
        if coh > 0.3:
            movement = "tokens move TOGETHER (global field deformation)"
        elif coh < -0.1:
            movement = "tokens move APART (separation)"
        else:
            movement = "tokens move INDEPENDENTLY (local operations)"

        layer_node = tree.add(
            f"[cyan]Layer {ell}→{ell+1}[/] — "
            f"mean_Δ={mean_def:.3f}, active=\"{tok_name}\""
        )
        layer_node.add(f"[dim]{movement}[/]")

    console.print(tree)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM Space-Deformation Explorer v2 — Test Koch's fibre bundle hypothesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 explore_deformations.py
  python3 explore_deformations.py --models gpt2 EleutherAI/pythia-70m
  python3 explore_deformations.py --device cuda --tasks all
        """
    )
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="HuggingFace model names (default: gpt2, pythia-70m, opt-125m)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"],
                        help="Tasks to run (default: all)")
    parser.add_argument("--output-dir", type=str, default="deformation_results_v2",
                        help="Output directory")
    parser.add_argument("--n-dims", type=int, default=8,
                        help="Number of real dimensions to track (top movers)")

    args = parser.parse_args()

    global N_DIMS_TO_TRACK
    N_DIMS_TO_TRACK = args.n_dims

    model_names = args.models if args.models else DEFAULT_MODELS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "all" in args.tasks:
        task_names = list(TASKS.keys())
    else:
        task_names = [t for t in args.tasks if t in TASKS]
        if not task_names:
            console.print("[red]No valid tasks. Available:[/]")
            for t in TASKS:
                console.print(f"  - {t}")
            sys.exit(1)

    console.print(Panel(
        "[bold white]LLM Space-Deformation Explorer v2[/]\n"
        "[dim]NO PCA. NO DIMENSIONALITY REDUCTION. Only real data.[/]\n\n"
        f"Models: {', '.join(model_names)}\n"
        f"Tasks:  {', '.join(task_names)}\n"
        f"Device: {args.device}\n"
        f"Dims to track: {N_DIMS_TO_TRACK} (top movers)\n"
        f"Output: {output_dir}/",
        title="[bold cyan]🔬 Configuration",
        border_style="cyan",
    ))

    for model_name in model_names:
        console.print(f"\n{'═' * 70}")
        console.print(f"[bold cyan]  Model: {model_name}[/]")
        console.print(f"{'═' * 70}")

        try:
            model, tokenizer, n_params, n_layers, hidden_dim = load_model(
                model_name, args.device
            )
        except Exception as e:
            console.print(f"  [red]Failed to load {model_name}: {e}[/]")
            continue

        model_dir = output_dir / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        for task_name in task_names:
            task_info = TASKS[task_name]
            console.print(f"\n  [yellow]Task: {task_name}[/] — {task_info['description']}")

            task_dir = model_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            # For semantic comparison, we need pairs
            semantic_pairs = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Analyzing..."),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                ptask = progress.add_task("analyze", total=len(task_info["prompts"]))

                for p_idx, prompt in enumerate(task_info["prompts"]):
                    try:
                        states, tokens = extract_hidden_states(
                            model, tokenizer, prompt, args.device
                        )

                        # Find top-moving real dimensions for this prompt
                        top_dims = find_top_moving_dimensions(states, N_DIMS_TO_TRACK)

                        console.print(f"\n    [dim]Prompt: \"{prompt}\"[/]")
                        console.print(f"    [dim]Tokens: {tokens}[/]")
                        console.print(f"    [dim]Top moving dims: {top_dims.tolist()}[/]")

                        # Print Rich summary
                        print_analysis_summary(states, tokens, model_name, prompt)
                        print_algorithm_trace(states, tokens, task_name)

                        # ── Generate all plots ──────────────────────────

                        # Plot 1: Real dimension trajectories
                        plot_dimension_trajectories(
                            states, tokens, top_dims, model_name, prompt,
                            task_dir / f"p{p_idx}_dim_trajectories.png"
                        )

                        # Plot 2: Deformation heatmap
                        plot_deformation_heatmap(
                            states, tokens, model_name, prompt,
                            task_dir / f"p{p_idx}_deformation_heatmap.png"
                        )

                        # Plot 3: Per-dimension signed delta heatmap
                        plot_per_dimension_delta_heatmap(
                            states, tokens, top_dims, model_name, prompt,
                            task_dir / f"p{p_idx}_dim_deltas.png"
                        )

                        # Plot 4: Geometric invariants
                        plot_geometric_invariants(
                            states, tokens, model_name, prompt,
                            task_dir / f"p{p_idx}_geometric_invariants.png"
                        )

                        # Plot 5: Delta rank analysis
                        plot_delta_rank_analysis(
                            states, model_name, prompt,
                            task_dir / f"p{p_idx}_delta_rank.png"
                        )

                        # Plot 6: Token pair distance evolution
                        plot_token_pair_distance_evolution(
                            states, tokens, model_name, prompt,
                            task_dir / f"p{p_idx}_pair_distances.png"
                        )

                        # Plot 8: Wave propagation
                        plot_wave_propagation(
                            states, tokens, model_name, prompt,
                            task_dir / f"p{p_idx}_wave.png"
                        )

                        # Plot 9: Last layer reversal
                        plot_last_layer_reversal(
                            states, tokens, model_name, prompt,
                            task_dir / f"p{p_idx}_reversal.png"
                        )

                        # Store for semantic comparison
                        semantic_pairs.append((states, tokens, prompt))

                    except Exception as e:
                        console.print(f"    [red]Error: {e}[/]")
                        import traceback
                        traceback.print_exc()

                    progress.update(ptask, advance=1)

            # ── Plot 7: Semantic comparison (if we have pairs) ──────────
            if task_name == "semantic" and len(semantic_pairs) >= 2:
                for i in range(len(semantic_pairs)):
                    for j in range(i + 1, min(i + 3, len(semantic_pairs))):
                        states_a, tokens_a, prompt_a = semantic_pairs[i]
                        states_b, tokens_b, prompt_b = semantic_pairs[j]
                        try:
                            plot_semantic_comparison(
                                states_a, states_b, tokens_a, tokens_b,
                                prompt_a, prompt_b, model_name,
                                task_dir / f"semantic_compare_{i}_vs_{j}.png"
                            )
                        except Exception as e:
                            console.print(f"    [red]Semantic comparison error: {e}[/]")

        # Free memory
        del model
        import gc
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════════════════════

    console.print(Panel(
        "[bold]What to look for in the generated plots:[/]\n\n"
        "1. [cyan]Dimension Trajectories (Plot 1)[/]: Each line is one token in one\n"
        "   REAL dimension. Can you see tokens converge, diverge, or cross?\n"
        "   Koch (C1): The geometry encodes semantic content.\n\n"
        "2. [cyan]Deformation Heatmap (Plot 2)[/]: Which tokens move most at which layers?\n"
        "   Koch (C4): Middle rows should be brightest (inner layers compute).\n\n"
        "3. [cyan]Signed Delta Heatmap (Plot 3)[/]: Blue=decrease, Red=increase.\n"
        "   Can you trace the exact computation? E.g., 'not' flips a dimension.\n\n"
        "4. [cyan]Geometric Invariants (Plot 4)[/]: Coherence, expansion, deformation.\n"
        "   Koch (C1): Tokens should move together (coherence > 0) = field deformation.\n\n"
        "5. [cyan]Delta Rank (Plot 5)[/]: Is this real morphing or just a global shift?\n"
        "   Green bars = genuine per-token morphing. Red = just a bias.\n\n"
        "6. [cyan]Pair Distances (Plot 6)[/]: How do token relationships change?\n"
        "   Koch (C5): The Jacobian field carries holographic information.\n\n"
        "7. [cyan]Semantic Comparison (Plot 7)[/]: Which REAL dimensions respond\n"
        "   to changing 'blue' to 'red'? Koch (Obs 2): Selective response.\n\n"
        "8. [cyan]Wave Propagation (Plot 8)[/]: The cumulative 'wave' of deformation.\n"
        "   Koch (Section 2.3): Superposition of deformations.\n\n"
        "9. [cyan]Last Layer Reversal (Plot 9)[/]: Does the last layer undo the morphing?\n"
        "   Koch (C4): Last layer morphs space back toward embedding geometry.\n\n"
        f"[bold green]All results saved to: {output_dir}/[/]",
        title="[bold yellow]🔍 Interpretation Guide",
        border_style="yellow",
    ))

    # Count total plots
    total_plots = 0
    for dirpath, dirnames, filenames in os.walk(output_dir):
        total_plots += sum(1 for f in filenames if f.endswith('.png'))

    console.print(f"\n[bold]Total plots generated: {total_plots}[/]")
    console.print(f"[bold]Output directory: {output_dir}/[/]")
    console.print(f"\n[dim]Run 'python3 {sys.argv[0]} --help' for more options.[/]\n")


if __name__ == "__main__":
    main()
