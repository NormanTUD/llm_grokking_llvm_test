# viz_fibre_bundle.py
"""
Fibre Bundle Visualization of TinyGPT.

Interpretation:
  - Base manifold B: token embedding space (projected to 2D via PCA)
  - Fibre F_x over each token x: the singular value spectrum of the
    Jacobian d(layer_k)/d(h) evaluated at that token's hidden state.
  - Each layer is a section of the fibre bundle, i.e., a Lipschitz map
    between tangent spaces. The Jacobian encodes the local linear
    approximation of that map.

The visualization shows fibres as vertical "stalks" rising from the
base manifold, colored by topological invariants (effective rank,
condition number, spectral entropy).

Saves to: fibre_bundle_topology.png
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
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

# ── Import your model code ──────────────────────────────────────────────
from train import TinyGPT, LLVMGPTConfig, CharTokenizer, build_tokenizer_from_samples


def build_small_model(vocab_size: int = 80, d_model: int = 32,
                       n_heads: int = 4, n_layers: int = 4,
                       max_seq_len: int = 256) -> TinyGPT:
    """Build a small model for visualization (or load a pretrained one)."""
    config = LLVMGPTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=0.0,  # no dropout for clean Jacobians
    )
    model = TinyGPT(config)
    model.eval()
    return model


def compute_layer_jacobians(
    model: TinyGPT,
    input_ids: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Compute the Jacobian of each transformer block's output w.r.t. its input.

    For block k: J_k = d(block_k(h)) / d(h)
    where h is the hidden state entering block k.

    Returns: list of Jacobians, one per layer, shape [T, d_model, d_model]
             (averaged over batch if batch > 1, computed per-token position)
    """
    model.eval()
    B, T = input_ids.shape
    d = model.config.d_model

    # Forward pass to get intermediate hidden states
    # We need to hook into each block
    hidden_states = []

    def make_hook(storage):
        def hook(module, input, output):
            storage.append(output.detach())
        return hook

    # Get input to first block
    with torch.no_grad():
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        h0 = model.tok_emb(input_ids) + model.pos_emb(pos)
        # no dropout since we set it to 0

    # Compute Jacobian for each block
    jacobians = []

    h = h0.clone().requires_grad_(True)
    for layer_idx, block in enumerate(model.blocks):
        # We compute the Jacobian by backpropagating from each output dimension
        # For efficiency, we pick a subset of token positions
        n_tokens_sample = min(T, 16)
        token_indices = np.linspace(0, T - 1, n_tokens_sample, dtype=int)

        J_layer = torch.zeros(n_tokens_sample, d, d)

        for ti, t_idx in enumerate(token_indices):
            for j in range(d):
                h_in = h.detach().clone().requires_grad_(True)
                out = block(h_in)
                # Backprop from output[0, t_idx, j]
                grad_output = torch.zeros_like(out)
                grad_output[0, t_idx, j] = 1.0
                out.backward(grad_output, retain_graph=True)

                if h_in.grad is not None:
                    J_layer[ti, j, :] = h_in.grad[0, t_idx, :].detach()
                    h_in.grad.zero_()

        jacobians.append(J_layer)

        # Advance hidden state
        with torch.no_grad():
            h = block(h).detach().clone().requires_grad_(True)

    return jacobians, token_indices


def singular_value_spectra(jacobians: List[torch.Tensor]) -> List[np.ndarray]:
    """Compute SVD of each Jacobian. Returns list of arrays [n_tokens, d_model]."""
    spectra = []
    for J in jacobians:
        n_tok, d, _ = J.shape
        svs = np.zeros((n_tok, d))
        for t in range(n_tok):
            U, S, Vh = torch.linalg.svd(J[t])
            svs[t] = S.numpy()
        spectra.append(svs)
    return spectra


def spectral_entropy(sigma: np.ndarray) -> float:
    """Compute spectral entropy of singular values (normalized)."""
    s = sigma / (sigma.sum() + 1e-12)
    s = s[s > 1e-12]
    return -np.sum(s * np.log(s + 1e-12))


def effective_rank(sigma: np.ndarray) -> float:
    """Effective rank = exp(spectral_entropy)."""
    return np.exp(spectral_entropy(sigma))


def condition_number(sigma: np.ndarray) -> float:
    """Condition number = max(sigma) / min(sigma)."""
    s_min = sigma[sigma > 1e-10]
    if len(s_min) == 0:
        return float('inf')
    return sigma.max() / s_min.min()


def compute_base_manifold_coords(model: TinyGPT, input_ids: torch.Tensor,
                                  token_indices: np.ndarray) -> np.ndarray:
    """Project token hidden states to 2D using PCA for the base manifold."""
    with torch.no_grad():
        T = input_ids.shape[1]
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        h = model.tok_emb(input_ids) + model.pos_emb(pos)
        h = h[0, token_indices, :].numpy()  # [n_tokens, d_model]

    # PCA to 2D
    h_centered = h - h.mean(axis=0)
    cov = np.cov(h_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top 2
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx[:2]]
    coords_2d = h_centered @ W
    return coords_2d


def plot_fibre_bundle(
    base_coords: np.ndarray,
    spectra: List[np.ndarray],
    token_indices: np.ndarray,
    save_path: str = "fibre_bundle_topology.png",
):
    """
    3D visualization: base manifold in XY plane, fibres extend in Z.

    Each fibre is a vertical stalk showing the singular value spectrum
    of the Jacobian at that token position. Color encodes effective rank.
    Different layers are shown as different "rings" of fibres.
    """
    n_layers = len(spectra)
    n_tokens = base_coords.shape[0]
    d_model = spectra[0].shape[1]

    fig = plt.figure(figsize=(20, 16))

    # ── Main 3D fibre bundle plot ───────────────────────────────────────
    ax3d = fig.add_subplot(221, projection='3d')
    ax3d.set_title("Fibre Bundle: Jacobian SVD Spectra over Token Space",
                    fontsize=11, fontweight='bold', pad=15)

    # Color map for layers
    layer_colors = cm.viridis(np.linspace(0.2, 0.9, n_layers))

    for layer_idx in range(n_layers):
        svs = spectra[layer_idx]  # [n_tokens, d_model]
        # Offset in the Z direction per layer
        z_offset = layer_idx * 1.5

        for t in range(n_tokens):
            x_base = base_coords[t, 0]
            y_base = base_coords[t, 1]

            # The fibre: singular values as heights
            sigma = svs[t]
            sigma_norm = sigma / (sigma.max() + 1e-10)

            # Draw the fibre as a vertical line with varying thickness
            z_vals = np.linspace(z_offset, z_offset + 1.2, len(sigma))

            # Color by effective rank
            eff_r = effective_rank(sigma)
            color_intensity = min(eff_r / d_model, 1.0)

            # Draw stalk
            for i in range(len(sigma) - 1):
                alpha = max(0.15, sigma_norm[i])
                ax3d.plot(
                    [x_base, x_base],
                    [y_base, y_base],
                    [z_vals[i], z_vals[i + 1]],
                    color=layer_colors[layer_idx],
                    alpha=alpha,
                    linewidth=0.5 + 2.0 * sigma_norm[i],
                )

            # Mark the base point
            ax3d.scatter(
                [x_base], [y_base], [z_offset],
                c=[color_intensity],
                cmap='plasma',
                s=20 + 30 * color_intensity,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.3,
                vmin=0, vmax=1,
                zorder=5,
            )

    # Layer labels
    for layer_idx in range(n_layers):
        z_offset = layer_idx * 1.5
        ax3d.text(
            base_coords[:, 0].max() * 1.3,
            base_coords[:, 1].mean(),
            z_offset + 0.6,
            f"Layer {layer_idx}",
            fontsize=8,
            color=layer_colors[layer_idx],
            fontweight='bold',
        )

    ax3d.set_xlabel("Base Manifold PC1", fontsize=9)
    ax3d.set_ylabel("Base Manifold PC2", fontsize=9)
    ax3d.set_zlabel("Fibre (Layer / SVD spectrum)", fontsize=9)
    ax3d.view_init(elev=25, azim=45)

    # ── Spectral entropy heatmap across layers and tokens ───────────────
    ax_heat = fig.add_subplot(222)
    ax_heat.set_title("Spectral Entropy of Jacobians\n(Topological Complexity per Layer×Token)",
                       fontsize=10, fontweight='bold')

    entropy_matrix = np.zeros((n_layers, n_tokens))
    for l in range(n_layers):
        for t in range(n_tokens):
            entropy_matrix[l, t] = spectral_entropy(spectra[l][t])

    im = ax_heat.imshow(entropy_matrix, aspect='auto', cmap='inferno',
                         interpolation='bilinear')
    ax_heat.set_xlabel("Token Position", fontsize=9)
    ax_heat.set_ylabel("Layer", fontsize=9)
    ax_heat.set_yticks(range(n_layers))
    ax_heat.set_yticklabels([f"Layer {i}" for i in range(n_layers)])
    plt.colorbar(im, ax=ax_heat, label="Spectral Entropy")

    # ── Effective rank evolution ────────────────────────────────────────
    ax_rank = fig.add_subplot(223)
    ax_rank.set_title("Effective Rank of Jacobian per Layer\n(Fibre Dimension Estimate)",
                       fontsize=10, fontweight='bold')

    for t in range(n_tokens):
        ranks = [effective_rank(spectra[l][t]) for l in range(n_layers)]
        ax_rank.plot(range(n_layers), ranks, alpha=0.3, color='steelblue', linewidth=0.8)

    # Mean
    mean_ranks = [np.mean([effective_rank(spectra[l][t]) for t in range(n_tokens)])
                  for l in range(n_layers)]
    ax_rank.plot(range(n_layers), mean_ranks, color='red', linewidth=2.5,
                 label='Mean eff. rank', marker='o', markersize=6)
    ax_rank.set_xlabel("Layer", fontsize=9)
    ax_rank.set_ylabel("Effective Rank", fontsize=9)
    ax_rank.legend()
    ax_rank.grid(True, alpha=0.3)
    ax_rank.set_xticks(range(n_layers))

    # ── Condition number (Lipschitz bound proxy) ────────────────────────
    ax_cond = fig.add_subplot(224)
    ax_cond.set_title("Condition Number of Jacobian\n(Lipschitz Distortion of Fibre Maps)",
                       fontsize=10, fontweight='bold')

    cond_matrix = np.zeros((n_layers, n_tokens))
    for l in range(n_layers):
        for t in range(n_tokens):
            cond_matrix[l, t] = min(condition_number(spectra[l][t]), 1000)

    for l in range(n_layers):
        ax_cond.scatter(
            [l] * n_tokens,
            cond_matrix[l],
            c=layer_colors[l],
            alpha=0.5,
            s=25,
            edgecolors='black',
            linewidths=0.3,
        )

    medians = [np.median(cond_matrix[l]) for l in range(n_layers)]
    ax_cond.plot(range(n_layers), medians, color='red', linewidth=2,
                 marker='D', markersize=6, label='Median κ(J)')
    ax_cond.set_xlabel("Layer", fontsize=9)
    ax_cond.set_ylabel("Condition Number κ(J)", fontsize=9)
    ax_cond.set_yscale('log')
    ax_cond.legend()
    ax_cond.grid(True, alpha=0.3)
    ax_cond.set_xticks(range(n_layers))

    fig.suptitle(
        "LLM as Fibre Bundle: π: E → B\n"
        "B = Token Embedding Manifold, F = Jacobian Singular Value Spectrum,\n"
        "Layer Maps = Lipschitz Sections of the Bundle",
        fontsize=13, fontweight='bold', y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved fibre bundle visualization to: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("FIBRE BUNDLE TOPOLOGY VISUALIZATION")
    print("Interpreting LLM layers as Lipschitz maps on fibre bundles")
    print("=" * 70)

    # Build tokenizer and model
    print("\n[1/5] Building tokenizer...")
    tokenizer = CharTokenizer()

    print("[2/5] Building model...")
    model = build_small_model(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=4,
        max_seq_len=256,
    )
    print(f"       Parameters: {model.count_parameters():,}")

    # Create a sample input
    print("[3/5] Creating sample input sequence...")
    sample_text = "define i64 @f(i64 %p0, i64 %p1) { entry: %t0 = add i64 %p0, %p1 ret i64 %t0 }"
    encoded = tokenizer(sample_text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"       Sequence length: {input_ids.shape[1]}")

    # Compute Jacobians
    print("[4/5] Computing layer Jacobians (this may take a moment)...")
    jacobians, token_indices = compute_layer_jacobians(model, input_ids)
    print(f"       Computed {len(jacobians)} layer Jacobians")
    for i, J in enumerate(jacobians):
        print(f"       Layer {i}: J shape = {J.shape}, "
              f"mean |J| = {J.abs().mean():.4f}")

    # SVD
    spectra = singular_value_spectra(jacobians)

    # Base manifold coordinates
    base_coords = compute_base_manifold_coords(model, input_ids, token_indices)

    # Plot
    print("[5/5] Generating fibre bundle visualization...")
    plot_fibre_bundle(base_coords, spectra, token_indices,
                      save_path="fibre_bundle_topology.png")

    print("\nDone! Open fibre_bundle_topology.png to see the topology.")


if __name__ == "__main__":
    main()
