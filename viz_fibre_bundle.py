# viz_fibre_bundle.py
"""
Fibre Bundle Visualization of TinyGPT.

Usage:
    python3 viz_fibre_bundle.py runs/my_run_folder/

The script will:
  1. Auto-discover the checkpoint (.pt/.pth) and config in the run folder
  2. Load the TRAINED model weights
  3. Compute Jacobian SVD spectra per layer per token
  4. Visualize the fibre bundle structure
  5. Save visualization to <run_folder>/fibre_bundle_topology.png

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

Saves to: <run_folder>/fibre_bundle_topology.png
"""

import os
import sys
import glob
import json
import math
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path

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
from train import TinyGPT, LLVMGPTConfig, BPETokenizer


# ═══════════════════════════════════════════════════════════════════════
# Topological helper functions
# ═══════════════════════════════════════════════════════════════════════

def spectral_entropy(sigma: np.ndarray) -> float:
    """Compute spectral entropy of singular values (normalized).

    Measures the "spread" of the singular value distribution.
    High entropy = singular values are uniformly distributed (full-rank-like).
    Low entropy = singular values are concentrated (low-rank-like).
    """
    s = sigma / (sigma.sum() + 1e-12)
    s = s[s > 1e-12]
    return -np.sum(s * np.log(s + 1e-12))


def effective_rank(sigma: np.ndarray) -> float:
    """Effective rank = exp(spectral_entropy).

    Continuous estimate of the "intrinsic dimensionality" of the linear
    map described by the singular values.
    """
    return np.exp(spectral_entropy(sigma))


def condition_number(sigma: np.ndarray) -> float:
    """Condition number = max(sigma) / min(sigma).

    Proxy for the Lipschitz distortion of the layer map.
    """
    s_min = sigma[sigma > 1e-10]
    if len(s_min) == 0:
        return float('inf')
    return sigma.max() / s_min.min()


# ═══════════════════════════════════════════════════════════════════════
# Run folder discovery and model loading
# ═══════════════════════════════════════════════════════════════════════

def discover_run_folder(run_dir: str) -> Dict:
    """
    Auto-discover checkpoint, config, and metadata from a run folder
    OR a direct checkpoint file path.
    """
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        print(f"ERROR: Path does not exist: {run_path}")
        sys.exit(1)

    # ── If the user passed a file, use it as the checkpoint directly ──
    if run_path.is_file() and run_path.suffix in ('.pt', '.pth'):
        print(f"  Detected direct checkpoint file: {run_path.name}")
        parent_dir = run_path.parent
        discovered = {
            "run_dir": str(parent_dir),
            "checkpoint": str(run_path),
            "config": None,
            "config_data": {},
        }
        # Still try to find a config in the parent directory
        config_patterns = ["config.json", "hparams.json", "params.json", "*.json"]
        for pattern in config_patterns:
            found = sorted(parent_dir.rglob(pattern))
            if found:
                discovered["config"] = str(found[0])
                print(f"  Found config: {found[0].name}")
                try:
                    with open(found[0], "r") as f:
                        discovered["config_data"] = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"    WARNING: Could not parse config: {e}")
                break
        return discovered

    # ── Otherwise, treat it as a directory (existing behavior) ────────
    print(f"  Scanning run folder: {run_path}")

    discovered = {
        "run_dir": str(run_path),
        "checkpoint": None,
        "config": None,
        "config_data": {},
    }

    # ── Find checkpoint ─────────────────────────────────────────────
    checkpoint_patterns = [
        "best_model*.pt", "best_model*.pth",
        "best*.pt", "best*.pth",
        "checkpoint*.pt", "checkpoint*.pth",
        "model*.pt", "model*.pth",
        "*.pt", "*.pth",
    ]

    all_checkpoints = []
    for pattern in checkpoint_patterns:
        found = sorted(run_path.rglob(pattern))
        all_checkpoints.extend(found)

    # Deduplicate preserving order
    seen = set()
    unique_checkpoints = []
    for cp in all_checkpoints:
        if cp not in seen:
            seen.add(cp)
            unique_checkpoints.append(cp)

    if unique_checkpoints:
        def checkpoint_priority(p: Path) -> Tuple[int, int]:
            name = p.stem.lower()
            if "best" in name:
                priority = 0
            elif "checkpoint" in name:
                priority = 1
            elif "model" in name:
                priority = 2
            else:
                priority = 3
            size = p.stat().st_size
            return (priority, -size)

        unique_checkpoints.sort(key=checkpoint_priority)
        discovered["checkpoint"] = str(unique_checkpoints[0])
        print(f"  Found checkpoint: {unique_checkpoints[0].name}")
        if len(unique_checkpoints) > 1:
            print(f"    (also found: {', '.join(c.name for c in unique_checkpoints[1:5])})")
    else:
        print("  WARNING: No checkpoint (.pt/.pth) found in run folder!")

    # ── Find config ─────────────────────────────────────────────────
    config_patterns = ["config.json", "hparams.json", "params.json", "*.json"]
    for pattern in config_patterns:
        found = sorted(run_path.rglob(pattern))
        if found:
            discovered["config"] = str(found[0])
            print(f"  Found config: {found[0].name}")
            try:
                with open(found[0], "r") as f:
                    discovered["config_data"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"    WARNING: Could not parse config: {e}")
            break

    return discovered


def extract_model_config(discovered: Dict, tokenizer: BPETokenizer) -> LLVMGPTConfig:
    """
    Extract model hyperparameters from the discovered config or checkpoint.
    Falls back to inspecting the checkpoint's state_dict shapes.
    """
    config_data = discovered.get("config_data", {})

    d_model = config_data.get("d_model", None)
    n_heads = config_data.get("n_heads", None)
    n_layers = config_data.get("n_layers", None)
    max_seq_len = config_data.get("max_seq_len", None)
    vocab_size = config_data.get("vocab_size", None)

    # If config didn't have everything, infer from checkpoint
    if discovered["checkpoint"] and (d_model is None or n_layers is None):
        print("  Inferring model config from checkpoint state_dict shapes...")
        checkpoint = torch.load(discovered["checkpoint"], map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict",
                         checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            # Infer d_model from token embedding shape
            for key in state_dict:
                if "tok_emb" in key and "weight" in key:
                    shape = state_dict[key].shape
                    if vocab_size is None:
                        vocab_size = shape[0]
                    if d_model is None:
                        d_model = shape[1]
                    print(f"    tok_emb.weight: {shape} -> vocab_size={vocab_size}, d_model={d_model}")
                    break

            # Infer n_layers by counting block indices
            block_indices = set()
            for key in state_dict:
                if "blocks." in key:
                    parts = key.split(".")
                    try:
                        idx = int(parts[parts.index("blocks") + 1])
                        block_indices.add(idx)
                    except (ValueError, IndexError):
                        pass
            if block_indices and n_layers is None:
                n_layers = max(block_indices) + 1
                print(f"    Found {n_layers} transformer blocks")

            # Infer n_heads
            if n_heads is None and d_model is not None:
                for key in state_dict:
                    if "attn" in key and "qkv" in key and "weight" in key:
                        for hd in [64, 32, 16, 8]:
                            if d_model % hd == 0:
                                n_heads = d_model // hd
                                print(f"    Inferred n_heads={n_heads} (head_dim={hd})")
                                break
                        break

            # Infer max_seq_len from positional embedding
            if max_seq_len is None:
                for key in state_dict:
                    if "pos_emb" in key and "weight" in key:
                        max_seq_len = state_dict[key].shape[1]
                        print(f"    pos_emb: max_seq_len={max_seq_len}")
                        break

    # Final defaults
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    if d_model is None:
        d_model = 32
    if n_heads is None:
        n_heads = 4
    if n_layers is None:
        n_layers = 4
    if max_seq_len is None:
        max_seq_len = 256

    print(f"  Model config: vocab_size={vocab_size}, d_model={d_model}, "
          f"n_heads={n_heads}, n_layers={n_layers}, max_seq_len={max_seq_len}")

    return LLVMGPTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )


def load_trained_model(discovered: Dict, config: LLVMGPTConfig) -> TinyGPT:
    """Load trained model from checkpoint."""
    model = TinyGPT(config)

    if discovered["checkpoint"] is None:
        print("  WARNING: No checkpoint found -- using random initialization!")
        model.eval()
        return model

    print(f"  Loading weights from: {discovered['checkpoint']}")
    checkpoint = torch.load(discovered["checkpoint"], map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if "epoch" in checkpoint:
                print(f"    Epoch: {checkpoint['epoch']}")
            if "step" in checkpoint:
                print(f"    Step: {checkpoint['step']}")
            if "train_loss" in checkpoint:
                print(f"    Train loss: {checkpoint['train_loss']:.4f}")
            if "val_loss" in checkpoint:
                print(f"    Val loss: {checkpoint['val_loss']:.4f}")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            sample_key = next(iter(checkpoint.keys()), "")
            if "." in sample_key and ("weight" in sample_key or "bias" in sample_key):
                state_dict = checkpoint
            else:
                print(f"    Checkpoint keys: {list(checkpoint.keys())[:10]}")
                state_dict = checkpoint
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, strict=True)
        print("    ✅ Loaded weights (strict match)")
    except RuntimeError as e:
        print(f"    ⚠ Strict load failed: {e}")
        print("    Trying non-strict load...")
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"    Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"    Unexpected keys: {result.unexpected_keys[:5]}...")
        print("    ✅ Loaded weights (non-strict)")

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# Jacobian computation
# ═══════════════════════════════════════════════════════════════════════

def compute_layer_jacobians(
    model: TinyGPT,
    input_ids: torch.Tensor,
    n_token_samples: int = 16,
) -> Tuple[List[torch.Tensor], np.ndarray]:
    """
    Compute the Jacobian of each transformer block's output w.r.t. its input.

    For block k: J_k = d(block_k(h)) / d(h)
    where h is the hidden state entering block k.

    Returns: list of Jacobians, one per layer, shape [n_tokens_sample, d_model, d_model]
             and the token indices that were sampled.
    """
    model.eval()
    B, T = input_ids.shape
    d = model.config.d_model

    n_tokens_sample = min(T, n_token_samples)
    token_indices = np.linspace(0, T - 1, n_tokens_sample, dtype=int)

    # Get input to first block
    with torch.no_grad():
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        h0 = model.tok_emb(input_ids) + model.pos_emb(pos)

    # Compute Jacobian for each block
    jacobians = []
    h = h0.clone()

    for layer_idx, block in enumerate(model.blocks):
        J_layer = torch.zeros(n_tokens_sample, d, d)

        for ti, t_idx in enumerate(token_indices):
            for j in range(d):
                h_in = h.detach().clone().requires_grad_(True)
                out = block(h_in)
                grad_output = torch.zeros_like(out)
                grad_output[0, t_idx, j] = 1.0
                out.backward(grad_output, retain_graph=True)

                if h_in.grad is not None:
                    J_layer[ti, j, :] = h_in.grad[0, t_idx, :].detach()
                    h_in.grad.zero_()

        jacobians.append(J_layer)

        # Advance hidden state
        with torch.no_grad():
            h = block(h).detach().clone()

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
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx[:2]]
    coords_2d = h_centered @ W
    return coords_2d


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_fibre_bundle(
    base_coords: np.ndarray,
    spectra: List[np.ndarray],
    token_indices: np.ndarray,
    run_info: Dict,
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
        z_offset = layer_idx * 1.5

        for t in range(n_tokens):
            x_base = base_coords[t, 0]
            y_base = base_coords[t, 1]

            sigma = svs[t]
            sigma_norm = sigma / (sigma.max() + 1e-10)

            z_vals = np.linspace(z_offset, z_offset + 1.2, len(sigma))

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

    # ── Title with run info ─────────────────────────────────────────────
    run_name = os.path.basename(run_info.get("run_dir", "unknown"))
    checkpoint_name = os.path.basename(run_info.get("checkpoint", "none") or "none")
    source_label = "TRAINED" if run_info.get("checkpoint") else "RANDOM INIT"

    fig.suptitle(
        f"LLM as Fibre Bundle: π: E → B [{source_label}]\n"
        f"Run: {run_name} | Checkpoint: {checkpoint_name}\n"
        f"B = Token Embedding Manifold, F = Jacobian Singular Value Spectrum,\n"
        f"Layer Maps = Lipschitz Sections of the Bundle",
        fontsize=13, fontweight='bold', y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved fibre bundle visualization to: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── Parse CLI argument ──────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python3 viz_fibre_bundle.py <run_folder>")
        print("")
        print("Examples:")
        print("  python3 viz_fibre_bundle.py runs/run_20250418_143022/")
        print("  python3 viz_fibre_bundle.py runs/latest/")
        print("  python3 viz_fibre_bundle.py ./my_checkpoint_dir/")
        print("")
        print("The script will auto-discover the checkpoint (.pt/.pth) and config")
        print("inside the given folder, load the TRAINED model, and generate the")
        print("fibre bundle visualization of the Jacobian structure.")
        sys.exit(1)

    run_dir = sys.argv[1].rstrip("/")

    print("=" * 70)
    print("FIBRE BUNDLE TOPOLOGY VISUALIZATION")
    print("Interpreting LLM layers as Lipschitz maps on fibre bundles")
    print("=" * 70)

    # ── Discover run folder contents ────────────────────────────────
    print(f"\n[1/6] Discovering run folder: {run_dir}")
    discovered = discover_run_folder(run_dir)

    # ── Build tokenizer ─────────────────────────────────────────────
    print("\n[2/6] Building tokenizer...")
    tok_dir = discovered["run_dir"]
    tok_file = os.path.join(tok_dir, "tokenizer.json")
    if os.path.exists(tok_file):
        tokenizer = BPETokenizer.from_pretrained(tok_dir)
        print(f"  Loaded BPE tokenizer from {tok_dir} (vocab_size={tokenizer.vocab_size})")
    else:
        print(f"  ERROR: No tokenizer.json found in {tok_dir}")
        print(f"  The BPE tokenizer must exist alongside the checkpoint.")
        print(f"  (It is saved automatically during training.)")
        sys.exit(1)

    # ── Extract config and load model ───────────────────────────────
    print("\n[3/6] Extracting model config...")
    config = extract_model_config(discovered, tokenizer)

    print("\n[4/6] Loading trained model...")
    model = load_trained_model(discovered, config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"       Parameters: {param_count:,}")

    # ── Create sample input ─────────────────────────────────────────
    print("\n[5/6] Creating sample input sequence...")
    sample_text = "define i64 @f(i64 %p0, i64 %p1) { entry: %t0 = add i64 %p0, %p1 ret i64 %t0 }"
    encoded = tokenizer(sample_text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"       Sequence length: {input_ids.shape[1]}")

    # ── Compute Jacobians and visualize ─────────────────────────────
    print("\n[6/6] Computing layer Jacobians (this may take a moment)...")
    jacobians, token_indices = compute_layer_jacobians(model, input_ids)
    print(f"       Computed {len(jacobians)} layer Jacobians")
    for i, J in enumerate(jacobians):
        print(f"       Layer {i}: J shape = {J.shape}, "
              f"mean |J| = {J.abs().mean():.4f}")

    # SVD
    spectra = singular_value_spectra(jacobians)

    # Base manifold coordinates
    base_coords = compute_base_manifold_coords(model, input_ids, token_indices)

    # Save into the run folder
    save_path = os.path.join(discovered["run_dir"], "fibre_bundle_topology.png")

    print("       Generating fibre bundle visualization...")
    plot_fibre_bundle(base_coords, spectra, token_indices,
                      run_info=discovered,
                      save_path=save_path)

    print(f"\nDone! Open {save_path} to see the topology.")


if __name__ == "__main__":
    main()
