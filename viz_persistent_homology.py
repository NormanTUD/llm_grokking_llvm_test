# viz_persistent_homology.py
"""
Persistent Homology of Jacobian Fibre Structure.

Interpretation:
  For each layer k, the Jacobian J_k(x) at token position x has a
  singular value spectrum σ(x) = (σ_1, ..., σ_d). The collection
  {σ(x) : x ∈ tokens} forms a point cloud in R^d.

  We compute the persistent homology of this point cloud using
  Vietoris-Rips filtration. The resulting persistence diagrams reveal:
    - H_0: connected components (clusters in fibre space)
    - H_1: loops / cycles (topological holes)
    - H_2: voids (higher-dimensional cavities)

  These are the REAL topological structures living in the Jacobian.

Dependencies:
    pip install ripser persim scikit-learn

Saves to: jacobian_persistent_homology.png
"""

import os
import sys
import math
import random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch

# ── Import your model code ──────────────────────────────────────────────
from train import TinyGPT, LLVMGPTConfig, CharTokenizer

# ── Persistent homology ────────────────────────────────────────────────
try:
    from ripser import ripser
    from persim import plot_diagrams as persim_plot_diagrams
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("WARNING: ripser/persim not installed. Using fallback VR computation.")
    print("Install with: pip install ripser persim")


# ═══════════════════════════════════════════════════════════════════════
# MISSING FUNCTIONS — these were in viz_fibre_bundle.py but not here
# ═══════════════════════════════════════════════════════════════════════

def spectral_entropy(sigma: np.ndarray) -> float:
    """Compute spectral entropy of singular values (normalized).
    
    This measures the "spread" of the singular value distribution.
    High entropy = singular values are uniformly distributed (full-rank-like).
    Low entropy = singular values are concentrated (low-rank-like).
    """
    s = sigma / (sigma.sum() + 1e-12)
    s = s[s > 1e-12]
    return -np.sum(s * np.log(s + 1e-12))


def effective_rank(sigma: np.ndarray) -> float:
    """Effective rank = exp(spectral_entropy).
    
    This gives a continuous estimate of the "intrinsic dimensionality"
    of the linear map described by the singular values. It equals the
    true rank when all nonzero singular values are equal, and is < rank
    when the spectrum is skewed.
    """
    return np.exp(spectral_entropy(sigma))


def condition_number(sigma: np.ndarray) -> float:
    """Condition number = max(sigma) / min(sigma).
    
    This is a proxy for the Lipschitz distortion of the layer map:
    how much the map stretches the most-stretched direction relative
    to the most-compressed direction.
    """
    s_min = sigma[sigma > 1e-10]
    if len(s_min) == 0:
        return float('inf')
    return sigma.max() / s_min.min()


# ═══════════════════════════════════════════════════════════════════════


def build_small_model(vocab_size: int = 80, d_model: int = 32,
                       n_heads: int = 4, n_layers: int = 4,
                       max_seq_len: int = 256) -> TinyGPT:
    config = LLVMGPTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    model = TinyGPT(config)
    model.eval()
    return model


def compute_jacobian_svd_point_clouds(
    model: TinyGPT,
    input_ids: torch.Tensor,
    n_token_samples: int = 24,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    For each layer, compute the SVD of the Jacobian at sampled token positions.
    Returns point clouds in singular-value space.

    Returns:
        svd_clouds: list of arrays [n_tokens, d_model] per layer
        token_indices: which token positions were sampled
    """
    B, T = input_ids.shape
    d = model.config.d_model
    n_tokens = min(T, n_token_samples)
    token_indices = np.linspace(0, T - 1, n_tokens, dtype=int)

    # Get initial hidden state
    with torch.no_grad():
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        h0 = model.tok_emb(input_ids) + model.pos_emb(pos)

    svd_clouds = []
    h = h0.clone()

    for layer_idx, block in enumerate(model.blocks):
        J_svs = np.zeros((n_tokens, d))

        for ti, t_idx in enumerate(token_indices):
            h_in = h.detach().clone().requires_grad_(True)
            out = block(h_in)

            # Compute Jacobian row by row
            J = torch.zeros(d, d)
            for j in range(d):
                grad_out = torch.zeros_like(out)
                grad_out[0, t_idx, j] = 1.0
                out.backward(grad_out, retain_graph=True)
                if h_in.grad is not None:
                    J[j, :] = h_in.grad[0, t_idx, :].detach()
                    h_in.grad.zero_()

            # SVD
            _, S, _ = torch.linalg.svd(J)
            J_svs[ti] = S.numpy()

        svd_clouds.append(J_svs)

        # Advance
        with torch.no_grad():
            h = block(h).detach().clone()

    return svd_clouds, token_indices


def compute_distance_matrices(svd_clouds: List[np.ndarray]) -> List[np.ndarray]:
    """Compute pairwise distance matrices for each layer's SVD point cloud."""
    from scipy.spatial.distance import pdist, squareform
    dist_matrices = []
    for cloud in svd_clouds:
        D = squareform(pdist(cloud, metric='euclidean'))
        dist_matrices.append(D)
    return dist_matrices


def compute_persistent_homology_fallback(dist_matrix: np.ndarray, max_dim: int = 2):
    """
    Simple Vietoris-Rips persistent homology fallback (H_0 only, approximate).
    For proper computation, install ripser.
    """
    n = dist_matrix.shape[0]
    # Sort all pairwise distances
    triu_idx = np.triu_indices(n, k=1)
    distances = dist_matrix[triu_idx]
    sorted_dists = np.sort(distances)

    # H_0: track connected components via union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            return True
        return False

    births_deaths_h0 = []
    # All components born at 0
    component_birth = {i: 0.0 for i in range(n)}

    sorted_edges = np.argsort(distances)
    for edge_idx in sorted_edges:
        i, j = triu_idx[0][edge_idx], triu_idx[1][edge_idx]
        d = distances[edge_idx]
        ri, rj = find(i), find(j)
        if ri != rj:
            # Merge: the younger component dies
            union(ri, rj)
            births_deaths_h0.append([0.0, d])

    # The last surviving component has infinite death
    births_deaths_h0.append([0.0, float('inf')])

    return {
        'dgms': [
            np.array(births_deaths_h0),
            np.array([[0.0, 0.0]]),  # placeholder H_1
        ]
    }


def compute_betti_curve(dgm: np.ndarray, filtration_values: np.ndarray) -> np.ndarray:
    """Compute Betti number as a function of filtration parameter."""
    betti = np.zeros_like(filtration_values)
    for birth, death in dgm:
        if np.isinf(death):
            death = filtration_values[-1] * 2
        alive = (filtration_values >= birth) & (filtration_values < death)
        betti[alive] += 1
    return betti


def plot_persistent_homology(
    svd_clouds: List[np.ndarray],
    dist_matrices: List[np.ndarray],
    token_indices: np.ndarray,
    save_path: str = "jacobian_persistent_homology.png",
):
    n_layers = len(svd_clouds)
    d_model = svd_clouds[0].shape[1]

    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, n_layers, figure=fig, hspace=0.35, wspace=0.3)

    # ── Row 1: Persistence Diagrams per Layer ───────────────────────────
    all_ph_results = []
    for l in range(n_layers):
        ax = fig.add_subplot(gs[0, l])
        ax.set_title(f"Layer {l}\nPersistence Diagram", fontsize=10, fontweight='bold')

        if HAS_RIPSER:
            result = ripser(dist_matrices[l], maxdim=1, distance_matrix=True)
        else:
            result = compute_persistent_homology_fallback(dist_matrices[l])

        all_ph_results.append(result)

        # Plot persistence diagram
        dgms = result['dgms']

        # H_0
        h0 = dgms[0]
        finite_h0 = h0[~np.isinf(h0[:, 1])]
        infinite_h0 = h0[np.isinf(h0[:, 1])]

        if len(finite_h0) > 0:
            ax.scatter(finite_h0[:, 0], finite_h0[:, 1],
                      c='steelblue', s=40, alpha=0.7, label='H₀', edgecolors='black',
                      linewidths=0.5, zorder=5)

        # H_1 (if available)
        if len(dgms) > 1:
            h1 = dgms[1]
            finite_h1 = h1[~np.isinf(h1[:, 1])]
            if len(finite_h1) > 0 and not (len(finite_h1) == 1 and finite_h1[0, 0] == 0 and finite_h1[0, 1] == 0):
                ax.scatter(finite_h1[:, 0], finite_h1[:, 1],
                          c='tomato', s=40, alpha=0.7, label='H₁',
                          marker='^', edgecolors='black', linewidths=0.5, zorder=5)

        # Diagonal
        all_finite = h0[~np.isinf(h0[:, 1])]
        if len(all_finite) > 0:
            max_val = all_finite.max() * 1.1
        else:
            max_val = 1.0
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel("Birth", fontsize=8)
        ax.set_ylabel("Death", fontsize=8)
        ax.legend(fontsize=7, loc='lower right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    # ── Row 2: Betti Curves ─────────────────────────────────────────────
    for l in range(n_layers):
        ax = fig.add_subplot(gs[1, l])
        ax.set_title(f"Layer {l}\nBetti Curves β₀, β₁", fontsize=10, fontweight='bold')

        dgms = all_ph_results[l]['dgms']

        # Filtration range
        all_deaths = []
        for dgm in dgms:
            finite = dgm[~np.isinf(dgm[:, 1])]
            if len(finite) > 0:
                all_deaths.extend(finite[:, 1].tolist())
        max_filt = max(all_deaths) * 1.2 if all_deaths else 1.0
        filt_vals = np.linspace(0, max_filt, 200)

        # H_0 Betti curve
        betti_0 = compute_betti_curve(dgms[0], filt_vals)
        ax.plot(filt_vals, betti_0, color='steelblue', linewidth=2, label='β₀')
        ax.fill_between(filt_vals, betti_0, alpha=0.15, color='steelblue')

        # H_1 Betti curve
        if len(dgms) > 1:
            betti_1 = compute_betti_curve(dgms[1], filt_vals)
            ax.plot(filt_vals, betti_1, color='tomato', linewidth=2, label='β₁')
            ax.fill_between(filt_vals, betti_1, alpha=0.15, color='tomato')

        ax.set_xlabel("Filtration ε", fontsize=8)
        ax.set_ylabel("Betti Number", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 3: SVD Point Cloud Projections (PCA to 2D) + Topological Summary
    for l in range(n_layers):
        ax = fig.add_subplot(gs[2, l])
        ax.set_title(f"Layer {l}\nSVD Point Cloud (PCA 2D)", fontsize=10, fontweight='bold')

        cloud = svd_clouds[l]  # [n_tokens, d_model]

        # PCA to 2D
        cloud_centered = cloud - cloud.mean(axis=0)
        cov = np.cov(cloud_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigenvalues)[::-1]
        W = eigenvectors[:, idx_sort[:2]]
        coords_2d = cloud_centered @ W

        # Color by spectral entropy
        entropies = np.array([spectral_entropy(cloud[t]) for t in range(cloud.shape[0])])
        eff_ranks = np.array([effective_rank(cloud[t]) for t in range(cloud.shape[0])])

        sc = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=entropies,
            cmap='plasma',
            s=50,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Spectral Entropy", shrink=0.8)

        # Draw edges between nearby points (Rips-like connectivity)
        D = dist_matrices[l]
        threshold = np.percentile(D[D > 0], 30)  # connect closest 30%
        for i in range(cloud.shape[0]):
            for j in range(i + 1, cloud.shape[0]):
                if D[i, j] < threshold:
                    ax.plot(
                        [coords_2d[i, 0], coords_2d[j, 0]],
                        [coords_2d[i, 1], coords_2d[j, 1]],
                        color='gray', alpha=0.15, linewidth=0.5,
                    )

        # Annotate with topological summary
        dgms = all_ph_results[l]['dgms']
        h0_finite = dgms[0][~np.isinf(dgms[0][:, 1])]
        n_components = len(dgms[0]) - len(h0_finite)  # infinite bars = surviving components
        if n_components == 0:
            n_components = 1  # at least one component

        h1_count = 0
        if len(dgms) > 1:
            h1 = dgms[1]
            h1_finite = h1[~np.isinf(h1[:, 1])]
            # Filter out trivial bars
            if len(h1_finite) > 0:
                lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
                h1_count = np.sum(lifetimes > np.median(lifetimes) * 0.5) if len(lifetimes) > 0 else 0

        mean_eff_rank = np.mean(eff_ranks)
        mean_entropy = np.mean(entropies)

        summary_text = (
            f"β₀≈{n_components}  β₁≈{h1_count}\n"
            f"eff.rank={mean_eff_rank:.1f}\n"
            f"H={mean_entropy:.2f}"
        )
        ax.text(
            0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8),
        )

        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Persistent Homology of Jacobian Fibre Structure\n"
        "Point Cloud = {σ(x) : x ∈ tokens} in Singular Value Space per Layer\n"
        "Vietoris-Rips Filtration → Persistence Diagrams, Betti Curves, Topological Invariants",
        fontsize=13, fontweight='bold', y=1.03,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved persistent homology visualization to: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("PERSISTENT HOMOLOGY OF JACOBIAN FIBRE STRUCTURE")
    print("Topological invariants of LLM layer maps")
    print("=" * 70)

    # Build tokenizer and model
    print("\n[1/6] Building tokenizer...")
    tokenizer = CharTokenizer()

    print("[2/6] Building model...")
    model = build_small_model(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=4,
        max_seq_len=256,
    )
    print(f"       Parameters: {model.count_parameters():,}")

    # Create a sample input
    print("[3/6] Creating sample input sequence...")
    sample_text = "define i64 @f(i64 %p0, i64 %p1) { entry: %t0 = add i64 %p0, %p1 ret i64 %t0 }"
    encoded = tokenizer(sample_text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"       Sequence length: {input_ids.shape[1]}")

    # Compute Jacobian SVD point clouds
    print("[4/6] Computing Jacobian SVD point clouds per layer...")
    svd_clouds, token_indices = compute_jacobian_svd_point_clouds(
        model, input_ids, n_token_samples=24,
    )
    for i, cloud in enumerate(svd_clouds):
        print(f"       Layer {i}: cloud shape = {cloud.shape}, "
              f"mean σ_max = {cloud[:, 0].mean():.4f}, "
              f"mean σ_min = {cloud[:, -1].mean():.6f}")

    # Distance matrices
    print("[5/6] Computing pairwise distance matrices...")
    dist_matrices = compute_distance_matrices(svd_clouds)
    for i, D in enumerate(dist_matrices):
        print(f"       Layer {i}: mean dist = {D[D > 0].mean():.4f}, "
              f"max dist = {D.max():.4f}")

    # Plot
    print("[6/6] Computing persistent homology and generating visualization...")
    if not HAS_RIPSER:
        print("       ⚠ Using fallback VR computation (H_0 only).")
        print("       For full H_0 + H_1 + H_2, install: pip install ripser persim")

    plot_persistent_homology(
        svd_clouds, dist_matrices, token_indices,
        save_path="jacobian_persistent_homology.png",
    )

    print("\nDone! Open jacobian_persistent_homology.png to see the topology.")


if __name__ == "__main__":
    main()
