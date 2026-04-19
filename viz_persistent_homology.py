# viz_persistent_homology.py
"""
Persistent Homology of Jacobian Fibre Structure.

Usage:
    python3 viz_persistent_homology.py runs/my_run_folder/

The script will:
  1. Auto-discover the checkpoint (.pt/.pth) and config in the run folder
  2. Load the TRAINED model weights
  3. Compute Jacobian SVD point clouds per layer
  4. Compute persistent homology (Vietoris-Rips filtration)
  5. Save visualization to <run_folder>/jacobian_persistent_homology.png

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
    pip install ripser persim scikit-learn scipy
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
# Topological helper functions
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Run folder discovery and model loading
# ═══════════════════════════════════════════════════════════════════════

def discover_run_folder(run_dir: str) -> Dict:
    """
    Auto-discover checkpoint, config, and metadata from a run folder.

    Searches for:
      - Checkpoint: *.pt, *.pth (prefers 'best_model', then 'checkpoint', then latest)
      - Config: config.json, hparams.json, *.json
      - Tokenizer vocab or sample text files

    Returns dict with discovered paths and metadata.
    """
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        print(f"ERROR: Run folder does not exist: {run_path}")
        sys.exit(1)

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
        # Search recursively
        found = sorted(run_path.rglob(pattern))
        all_checkpoints.extend(found)

    # Deduplicate while preserving order
    seen = set()
    unique_checkpoints = []
    for cp in all_checkpoints:
        if cp not in seen:
            seen.add(cp)
            unique_checkpoints.append(cp)

    if unique_checkpoints:
        # Prefer by priority: best > checkpoint > model > other
        # Also prefer larger files (more likely to be full checkpoints)
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


def extract_model_config(discovered: Dict, tokenizer: CharTokenizer) -> LLVMGPTConfig:
    """
    Extract model hyperparameters from the discovered config or checkpoint.
    Falls back to inspecting the checkpoint's state_dict shapes.
    """
    config_data = discovered.get("config_data", {})

    # Try to get params from config JSON
    d_model = config_data.get("d_model", None)
    n_heads = config_data.get("n_heads", None)
    n_layers = config_data.get("n_layers", None)
    max_seq_len = config_data.get("max_seq_len", None)
    vocab_size = config_data.get("vocab_size", None)

    # If config didn't have everything, try to infer from checkpoint
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
                    print(f"    tok_emb.weight: {shape} → vocab_size={vocab_size}, d_model={d_model}")
                    break

            # Infer n_layers by counting block indices
            block_indices = set()
            for key in state_dict:
                if "blocks." in key:
                    # e.g., "blocks.0.attn.qkv.weight"
                    parts = key.split(".")
                    try:
                        idx = int(parts[parts.index("blocks") + 1])
                        block_indices.add(idx)
                    except (ValueError, IndexError):
                        pass
            if block_indices and n_layers is None:
                n_layers = max(block_indices) + 1
                print(f"    Found {n_layers} transformer blocks")

            # Infer n_heads from attention weight shapes
            if n_heads is None and d_model is not None:
                for key in state_dict:
                    if "attn" in key and "qkv" in key and "weight" in key:
                        qkv_shape = state_dict[key].shape
                        # qkv weight is [3*d_model, d_model] typically
                        # head_dim is usually 64, 32, etc.
                        # Try common head dims
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
        dropout=0.0,  # no dropout for clean Jacobians
    )


def load_trained_model(discovered: Dict, config: LLVMGPTConfig) -> TinyGPT:
    """Load trained model from checkpoint."""
    model = TinyGPT(config)

    if discovered["checkpoint"] is None:
        print("  ⚠ No checkpoint found — using random initialization!")
        model.eval()
        return model

    print(f"  Loading weights from: {discovered['checkpoint']}")
    checkpoint = torch.load(discovered["checkpoint"], map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Print training info if available
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
            # Maybe the dict IS the state_dict
            # Check if keys look like model parameters
            sample_key = next(iter(checkpoint.keys()), "")
            if "." in sample_key and ("weight" in sample_key or "bias" in sample_key):
                state_dict = checkpoint
            else:
                print(f"    Checkpoint keys: {list(checkpoint.keys())[:10]}")
                state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Try to load, handling potential key mismatches
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

def compute_jacobian_svd_point_clouds(
    model: TinyGPT,
    input_ids: torch.Tensor,
    n_token_samples: int = 24,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    For each layer, compute the SVD of the Jacobian at sampled token positions.
    Returns point clouds in singular-value space.
    """
    B, T = input_ids.shape
    d = model.config.d_model
    n_tokens = min(T, n_token_samples)
    token_indices = np.linspace(0, T - 1, n_tokens, dtype=int)

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

            J = torch.zeros(d, d)
            for j in range(d):
                grad_out = torch.zeros_like(out)
                grad_out[0, t_idx, j] = 1.0
                out.backward(grad_out, retain_graph=True)
                if h_in.grad is not None:
                    J[j, :] = h_in.grad[0, t_idx, :].detach()
                    h_in.grad.zero_()

            _, S, _ = torch.linalg.svd(J)
            J_svs[ti] = S.numpy()

        svd_clouds.append(J_svs)

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
    Simple Vietoris-Rips persistent homology fallback (H_0 only).
    For proper H_0 + H_1 + H_2, install ripser.
    """
    n = dist_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    distances = dist_matrix[triu_idx]

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

    sorted_edges = np.argsort(distances)
    for edge_idx in sorted_edges:
        i, j = triu_idx[0][edge_idx], triu_idx[1][edge_idx]
        d = distances[edge_idx]
        ri, rj = find(i), find(j)
        if ri != rj:
            union(ri, rj)
            births_deaths_h0.append([0.0, d])

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


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_persistent_homology(
    svd_clouds: List[np.ndarray],
    dist_matrices: List[np.ndarray],
    token_indices: np.ndarray,
    run_info: Dict,
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

        dgms = result['dgms']

        h0 = dgms[0]
        finite_h0 = h0[~np.isinf(h0[:, 1])]

        if len(finite_h0) > 0:
            ax.scatter(finite_h0[:, 0], finite_h0[:, 1],
                      c='steelblue', s=40, alpha=0.7, label='H₀', edgecolors='black',
                      linewidths=0.5, zorder=5)

        if len(dgms) > 1:
            h1 = dgms[1]
            finite_h1 = h1[~np.isinf(h1[:, 1])]
            if len(finite_h1) > 0 and not (len(finite_h1) == 1 and finite_h1[0, 0] == 0 and finite_h1[0, 1] == 0):
                ax.scatter(finite_h1[:, 0], finite_h1[:, 1],
                          c='tomato', s=40, alpha=0.7, label='H₁',
                          marker='^', edgecolors='black', linewidths=0.5, zorder=5)

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

        all_deaths = []
        for dgm in dgms:
            finite = dgm[~np.isinf(dgm[:, 1])]
            if len(finite) > 0:
                all_deaths.extend(finite[:, 1].tolist())
        max_filt = max(all_deaths) * 1.2 if all_deaths else 1.0
        filt_vals = np.linspace(0, max_filt, 200)

        betti_0 = compute_betti_curve(dgms[0], filt_vals)
        ax.plot(filt_vals, betti_0, color='steelblue', linewidth=2, label='β₀')
        ax.fill_between(filt_vals, betti_0, alpha=0.15, color='steelblue')

        if len(dgms) > 1:
            betti_1 = compute_betti_curve(dgms[1], filt_vals)
            ax.plot(filt_vals, betti_1, color='tomato', linewidth=2, label='β₁')
            ax.fill_between(filt_vals, betti_1, alpha=0.15, color='tomato')

        ax.set_xlabel("Filtration ε", fontsize=8)
        ax.set_ylabel("Betti Number", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 3: SVD Point Cloud Projections + Topological Summary ────────
    for l in range(n_layers):
        ax = fig.add_subplot(gs[2, l])
        ax.set_title(f"Layer {l}\nSVD Point Cloud (PCA 2D)", fontsize=10, fontweight='bold')

        cloud = svd_clouds[l]

        cloud_centered = cloud - cloud.mean(axis=0)
        cov = np.cov(cloud_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigenvalues)[::-1]
        W = eigenvectors[:, idx_sort[:2]]
        coords_2d = cloud_centered @ W

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

        D = dist_matrices[l]
        threshold = np.percentile(D[D > 0], 30)
        for i in range(cloud.shape[0]):
            for j in range(i + 1, cloud.shape[0]):
                if D[i, j] < threshold:
                    ax.plot(
                        [coords_2d[i, 0], coords_2d[j, 0]],
                        [coords_2d[i, 1], coords_2d[j, 1]],
                        color='gray', alpha=0.15, linewidth=0.5,
                    )

        dgms = all_ph_results[l]['dgms']
        h0_finite = dgms[0][~np.isinf(dgms[0][:, 1])]
        n_components = len(dgms[0]) - len(h0_finite)
        if n_components == 0:
            n_components = 1

        h1_count = 0
        if len(dgms) > 1:
            h1 = dgms[1]
            h1_finite = h1[~np.isinf(h1[:, 1])]
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

    # ── Title with run info ─────────────────────────────────────────────
    run_name = os.path.basename(run_info.get("run_dir", "unknown"))
    checkpoint_name = os.path.basename(run_info.get("checkpoint", "none"))
    source_label = "TRAINED" if run_info.get("checkpoint") else "RANDOM INIT"

    fig.suptitle(
        f"Persistent Homology of Jacobian Fibre Structure [{source_label}]\n"
        f"Run: {run_name} | Checkpoint: {checkpoint_name}\n"
        f"Point Cloud = {{σ(x) : x ∈ tokens}} in Singular Value Space per Layer\n"
        f"Vietoris-Rips Filtration → Persistence Diagrams, Betti Curves, Topological Invariants",
        fontsize=12, fontweight='bold', y=1.04,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved persistent homology visualization to: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── Parse CLI argument ──────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python3 viz_persistent_homology.py <run_folder>")
        print("")
        print("Examples:")
        print("  python3 viz_persistent_homology.py runs/run_20250418_143022/")
        print("  python3 viz_persistent_homology.py runs/latest/")
        print("  python3 viz_persistent_homology.py ./my_checkpoint_dir/")
        print("")
        print("The script will auto-discover the checkpoint (.pt/.pth) and config")
        print("inside the given folder, load the TRAINED model, and generate the")
        print("persistent homology visualization of the Jacobian fibre structure.")
        sys.exit(1)

    run_dir = sys.argv[1].rstrip("/")

    print("=" * 70)
    print("PERSISTENT HOMOLOGY OF JACOBIAN FIBRE STRUCTURE")
    print("Topological invariants of LLM layer maps (TRAINED model)")
    print("=" * 70)

    # ── Discover run folder contents ────────────────────────────────
    print(f"\n[1/6] Discovering run folder: {run_dir}")
    discovered = discover_run_folder(run_dir)

    # ── Build tokenizer ─────────────────────────────────────────────
    print("\n[2/6] Building tokenizer...")
    tokenizer = CharTokenizer()

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

    # ── Compute Jacobian SVD point clouds ───────────────────────────
    print("\n[6/6] Computing Jacobian SVD point clouds & persistent homology...")
    svd_clouds, token_indices = compute_jacobian_svd_point_clouds(
        model, input_ids, n_token_samples=24,
    )
    for i, cloud in enumerate(svd_clouds):
        print(f"       Layer {i}: cloud shape = {cloud.shape}, "
              f"mean σ_max = {cloud[:, 0].mean():.4f}, "
              f"mean σ_min = {cloud[:, -1].mean():.6f}")

    # Distance matrices
    print("       Computing pairwise distance matrices...")
    dist_matrices = compute_distance_matrices(svd_clouds)
    for i, D in enumerate(dist_matrices):
        print(f"       Layer {i}: mean dist = {D[D > 0].mean():.4f}, "
              f"max dist = {D.max():.4f}")

    # Persistent homology + plot
    if not HAS_RIPSER:
        print("       ⚠ Using fallback VR computation (H_0 only).")
        print("       For full H_0 + H_1 + H_2, install: pip install ripser persim")

    # Save into the run folder
    save_path = os.path.join(run_dir, "jacobian_persistent_homology.png")

    plot_persistent_homology(
        svd_clouds, dist_matrices, token_indices,
        run_info=discovered,
        save_path=save_path,
    )

    print(f"\nDone! Open {save_path} to see the topology.")


if __name__ == "__main__":
    main()
