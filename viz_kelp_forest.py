# viz_kelp_forest.py
"""
Kelp-Forest Fibre Bundle Visualization of TinyGPT.

Base manifold: token index (1-D sequence position).
Fibres: per-layer Jacobian singular-value stalks that LEAN in the
        dominant gradient-flow direction, like kelp bending in a current.

Usage:
    python3 viz_kelp_forest.py runs/my_run_folder/
    python3 viz_kelp_forest.py runs/my_run_folder/model_best.pt

Saves to: <run_folder>/kelp_forest_fibres.png
"""

import os
import sys
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

# ── Import model code from train.py ────────────────────────────────────
from train import TinyGPT, LLVMGPTConfig, BPETokenizer


# ═══════════════════════════════════════════════════════════════════════
# Topological / spectral helpers
# ═══════════════════════════════════════════════════════════════════════

def spectral_entropy(sigma: np.ndarray) -> float:
    s = sigma / (sigma.sum() + 1e-12)
    s = s[s > 1e-12]
    return -np.sum(s * np.log(s + 1e-12))


def effective_rank(sigma: np.ndarray) -> float:
    return np.exp(spectral_entropy(sigma))


def condition_number(sigma: np.ndarray) -> float:
    s_min = sigma[sigma > 1e-10]
    if len(s_min) == 0:
        return float("inf")
    return sigma.max() / s_min.min()


# ═══════════════════════════════════════════════════════════════════════
# Run-folder discovery  (reused from viz_fibre_bundle.py)
# ═══════════════════════════════════════════════════════════════════════

def discover_run_folder(run_dir: str) -> Dict:
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        print(f"ERROR: Path does not exist: {run_path}")
        sys.exit(1)

    if run_path.is_file() and run_path.suffix in (".pt", ".pth"):
        parent_dir = run_path.parent
        discovered = {
            "run_dir": str(parent_dir),
            "checkpoint": str(run_path),
            "config": None,
            "config_data": {},
        }
        for pattern in ["config.json", "hparams.json", "params.json", "*.json"]:
            found = sorted(parent_dir.rglob(pattern))
            if found:
                discovered["config"] = str(found[0])
                try:
                    with open(found[0], "r") as f:
                        discovered["config_data"] = json.load(f)
                except Exception:
                    pass
                break
        return discovered

    discovered = {
        "run_dir": str(run_path),
        "checkpoint": None,
        "config": None,
        "config_data": {},
    }

    all_checkpoints = []
    for pat in [
        "best_model*.pt", "best_model*.pth", "best*.pt", "best*.pth",
        "checkpoint*.pt", "checkpoint*.pth", "model*.pt", "model*.pth",
        "*.pt", "*.pth",
    ]:
        all_checkpoints.extend(sorted(run_path.rglob(pat)))

    seen = set()
    unique = []
    for cp in all_checkpoints:
        if cp not in seen:
            seen.add(cp)
            unique.append(cp)

    if unique:
        def _prio(p):
            n = p.stem.lower()
            if "best" in n:
                return (0, -p.stat().st_size)
            if "checkpoint" in n:
                return (1, -p.stat().st_size)
            if "model" in n:
                return (2, -p.stat().st_size)
            return (3, -p.stat().st_size)
        unique.sort(key=_prio)
        discovered["checkpoint"] = str(unique[0])
        print(f"  Found checkpoint: {unique[0].name}")

    for pat in ["config.json", "hparams.json", "params.json", "*.json"]:
        found = sorted(run_path.rglob(pat))
        if found:
            discovered["config"] = str(found[0])
            try:
                with open(found[0], "r") as f:
                    discovered["config_data"] = json.load(f)
            except Exception:
                pass
            break

    return discovered


def extract_model_config(discovered: Dict) -> LLVMGPTConfig:
    cd = discovered.get("config_data", {})
    d_model = cd.get("d_model")
    n_heads = cd.get("n_heads")
    n_layers = cd.get("n_layers")
    max_seq_len = cd.get("max_seq_len")
    vocab_size = cd.get("vocab_size")

    if discovered["checkpoint"] and (d_model is None or n_layers is None):
        ckpt = torch.load(discovered["checkpoint"], map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if isinstance(sd, dict):
            for k in sd:
                if "tok_emb" in k and "weight" in k:
                    vocab_size = vocab_size or sd[k].shape[0]
                    d_model = d_model or sd[k].shape[1]
                    break
            block_ids = set()
            for k in sd:
                if "blocks." in k:
                    parts = k.split(".")
                    try:
                        block_ids.add(int(parts[parts.index("blocks") + 1]))
                    except Exception:
                        pass
            if block_ids and n_layers is None:
                n_layers = max(block_ids) + 1
            if n_heads is None and d_model:
                for hd in [64, 32, 16, 8, 4, 2]:
                    if d_model % hd == 0:
                        n_heads = d_model // hd
                        break
            if max_seq_len is None:
                for k in sd:
                    if "pos_emb" in k and "weight" in k:
                        max_seq_len = sd[k].shape[0]
                        break

    return LLVMGPTConfig(
        vocab_size=vocab_size or 512,
        d_model=d_model or 32,
        n_heads=n_heads or 4,
        n_layers=n_layers or 4,
        max_seq_len=max_seq_len or 2048,
        dropout=0.0,
    )


def load_trained_model(discovered: Dict, config: LLVMGPTConfig) -> TinyGPT:
    model = TinyGPT(config)
    if discovered["checkpoint"] is None:
        print("  WARNING: no checkpoint — random init")
        model.eval()
        return model

    ckpt = torch.load(discovered["checkpoint"], map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    try:
        model.load_state_dict(sd, strict=True)
        print("  ✅ Loaded weights (strict)")
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
        print("  ✅ Loaded weights (non-strict)")
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# Jacobian computation — returns SVD spectra AND dominant directions
# ═══════════════════════════════════════════════════════════════════════

def compute_layer_jacobians(
    model: TinyGPT,
    input_ids: torch.Tensor,
    n_token_samples: int = 24,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    For each transformer block, compute the Jacobian of the block output
    w.r.t. its input at sampled token positions.

    Returns
    -------
    spectra : list[ndarray]   – [n_layers] each (n_tokens, d_model)  singular values
    directions : list[ndarray] – [n_layers] each (n_tokens, 2)       dominant 2-D direction
    token_indices : ndarray    – which token positions were sampled
    """
    model.eval()
    B, T = input_ids.shape
    d = model.config.d_model

    n_tok = min(T, n_token_samples)
    token_indices = np.linspace(0, T - 1, n_tok, dtype=int)

    # Initial hidden state
    with torch.no_grad():
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        h0 = model.tok_emb(input_ids) + model.pos_emb(pos)

    spectra: List[np.ndarray] = []
    directions: List[np.ndarray] = []
    h = h0.clone()

    for layer_idx, block in enumerate(model.blocks):
        print(f"    Layer {layer_idx}: computing Jacobian ({n_tok} tokens × {d} dims)...")
        J_layer = torch.zeros(n_tok, d, d)

        for ti, t_idx in enumerate(token_indices):
            for j in range(d):
                h_in = h.detach().clone().requires_grad_(True)
                out = block(h_in)
                grad_out = torch.zeros_like(out)
                grad_out[0, t_idx, j] = 1.0
                out.backward(grad_out, retain_graph=True)
                if h_in.grad is not None:
                    J_layer[ti, j, :] = h_in.grad[0, t_idx, :].detach()
                    h_in.grad.zero_()

        # SVD per token
        svs = np.zeros((n_tok, d))
        dirs_2d = np.zeros((n_tok, 2))

        for t in range(n_tok):
            U, S, Vh = torch.linalg.svd(J_layer[t])
            svs[t] = S.numpy()
            # Dominant right-singular vector (input direction most amplified)
            v1 = Vh[0].numpy()  # shape (d,)
            # Project onto first two PCA-like axes (just take dims 0,1 of v1)
            dirs_2d[t, 0] = v1[0]
            dirs_2d[t, 1] = v1[1] if d > 1 else 0.0

        spectra.append(svs)
        directions.append(dirs_2d)

        with torch.no_grad():
            h = block(h).detach().clone()

    return spectra, directions, token_indices


# ═══════════════════════════════════════════════════════════════════════
# Kelp-forest plotting
# ═══════════════════════════════════════════════════════════════════════

def _bezier_stalk(
    x_base: float,
    y_base: float,
    height: float,
    lean_dx: float,
    n_points: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a smooth Bézier-like curve from (x_base, y_base) upward
    by `height`, leaning horizontally by `lean_dx`.

    Returns arrays (xs, ys) of shape (n_points,).
    """
    t = np.linspace(0, 1, n_points)
    # Quadratic Bézier: P0 = base, P1 = control, P2 = tip
    p0 = np.array([x_base, y_base])
    p2 = np.array([x_base + lean_dx, y_base + height])
    # Control point: halfway up, leaning 60% of the way
    p1 = np.array([x_base + lean_dx * 0.6, y_base + height * 0.55])

    xs = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    ys = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return xs, ys


def plot_kelp_forest(
    spectra: List[np.ndarray],
    directions: List[np.ndarray],
    token_indices: np.ndarray,
    run_info: Dict,
    save_path: str = "kelp_forest_fibres.png",
):
    """
    Main kelp-forest visualization.

    Layout:
      - Top-left  (large): the kelp forest itself
      - Top-right        : spectral entropy heatmap
      - Bottom-left      : effective rank per layer
      - Bottom-right     : condition number per layer
    """
    n_layers = len(spectra)
    n_tokens = len(token_indices)
    d_model = spectra[0].shape[1]

    # ── Precompute lean angles and magnitudes ───────────────────────────
    # For each (layer, token) we have a 2-D direction vector.
    # The angle encodes "which way the kelp leans"; magnitude = top singular value.
    angles = np.zeros((n_layers, n_tokens))
    magnitudes = np.zeros((n_layers, n_tokens))
    eff_ranks = np.zeros((n_layers, n_tokens))

    for l in range(n_layers):
        for t in range(n_tokens):
            dx, dy = directions[l][t]
            angles[l, t] = np.arctan2(dy, dx)  # radians, [-π, π]
            magnitudes[l, t] = spectra[l][t, 0]  # top singular value
            eff_ranks[l, t] = effective_rank(spectra[l][t])

    # Normalize magnitudes globally for lean scaling
    mag_max = magnitudes.max() + 1e-12

    # ── Figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 18), facecolor="#0a0e1a")
    fig.patch.set_facecolor("#0a0e1a")

    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[2.2, 1, 1],
        height_ratios=[1.2, 1],
        hspace=0.32, wspace=0.30,
    )

    # ════════════════════════════════════════════════════════════════════
    # Panel 1: Kelp Forest  (spans left two columns, full height)
    # ════════════════════════════════════════════════════════════════════
    ax_kelp = fig.add_subplot(gs[:, 0])
    ax_kelp.set_facecolor("#060a14")

    layer_height = 1.4  # vertical space per layer
    stalk_height = layer_height * 0.85
    max_lean = 0.6  # max horizontal lean in index-space units

    # Angle → color via a cyclic colormap
    angle_cmap = cm.get_cmap("hsv")
    angle_norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)

    # Draw from bottom layer (layer 0) upward
    for l in range(n_layers):
        y_base = l * layer_height

        for ti in range(n_tokens):
            x_base = token_indices[ti]
            angle = angles[l, ti]
            mag = magnitudes[l, ti]
            er = eff_ranks[l, ti]

            # Lean proportional to magnitude, direction from angle
            lean_scale = (mag / mag_max) * max_lean
            lean_dx = lean_scale * np.cos(angle)

            # Stalk color from angle
            stalk_color = angle_cmap(angle_norm(angle))

            # Thickness from effective rank (thicker = higher rank)
            base_lw = 0.4 + 2.5 * min(er / d_model, 1.0)

            # Generate the curved stalk
            xs, ys = _bezier_stalk(x_base, y_base, stalk_height, lean_dx, n_points=50)

            # Draw with tapering linewidth (thick at base, thin at tip)
            n_seg = len(xs) - 1
            for si in range(n_seg):
                frac = si / n_seg
                lw = base_lw * (1.0 - 0.7 * frac)  # taper
                alpha = 0.25 + 0.65 * (1.0 - 0.5 * frac)

                # Slight color shift along the stalk (lighter toward tip)
                r, g, b, _ = stalk_color
                lighten = 0.3 * frac
                seg_color = (
                    min(r + lighten, 1.0),
                    min(g + lighten, 1.0),
                    min(b + lighten, 1.0),
                    alpha,
                )

                ax_kelp.plot(
                    xs[si : si + 2], ys[si : si + 2],
                    color=seg_color,
                    linewidth=lw,
                    solid_capstyle="round",
                )

            # Small dot at the tip
            tip_alpha = 0.5 + 0.5 * (mag / mag_max)
            ax_kelp.plot(
                xs[-1], ys[-1], "o",
                color=stalk_color,
                markersize=2.0 + 2.0 * (mag / mag_max),
                alpha=tip_alpha,
                markeredgewidth=0,
            )

        # Layer label on the right
        ax_kelp.text(
            token_indices[-1] + 2.5,
            y_base + stalk_height * 0.45,
            f"Layer {l}",
            fontsize=9, fontweight="bold",
            color="white", alpha=0.7,
            verticalalignment="center",
        )

        # Faint horizontal baseline
        ax_kelp.axhline(
            y=y_base, color="white", alpha=0.08,
            linewidth=0.5, linestyle="--",
        )

    # Base manifold markers (token indices along x)
    for ti in range(n_tokens):
        ax_kelp.plot(
            token_indices[ti], -0.15, "^",
            color="#4fc3f7", markersize=5, alpha=0.6,
            markeredgewidth=0,
        )

    ax_kelp.set_xlim(token_indices[0] - 3, token_indices[-1] + 6)
    ax_kelp.set_ylim(-0.4, n_layers * layer_height + 0.3)
    ax_kelp.set_xlabel("Token Index (Base Manifold)", fontsize=11, color="white")
    ax_kelp.set_ylabel("Layer (Fibre Height)", fontsize=11, color="white")
    ax_kelp.set_title(
        "Kelp Forest: Jacobian Fibre Directions over Token Space",
        fontsize=13, fontweight="bold", color="white", pad=12,
    )
    ax_kelp.tick_params(colors="white", labelsize=8)
    for spine in ax_kelp.spines.values():
        spine.set_color("#333")

    # ── Colorbar for angle ──────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=angle_cmap, norm=angle_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_kelp, fraction=0.02, pad=0.02)
    cbar.set_label("Dominant Direction (angle, rad)", fontsize=9, color="white")
    cbar.ax.tick_params(colors="white", labelsize=7)

    # ════════════════════════════════════════════════════════════════════
    # Panel 2: Spectral Entropy Heatmap
    # ════════════════════════════════════════════════════════════════════
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_heat.set_facecolor("#0d1117")

    entropy_matrix = np.zeros((n_layers, n_tokens))
    for l in range(n_layers):
        for t in range(n_tokens):
            entropy_matrix[l, t] = spectral_entropy(spectra[l][t])

    im = ax_heat.imshow(
        entropy_matrix, aspect="auto", cmap="inferno",
        interpolation="bilinear", origin="lower",
    )
    ax_heat.set_xlabel("Token Sample", fontsize=9, color="white")
    ax_heat.set_ylabel("Layer", fontsize=9, color="white")
    ax_heat.set_yticks(range(n_layers))
    ax_heat.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
    ax_heat.set_title(
        "Spectral Entropy\n(Topological Complexity)",
        fontsize=10, fontweight="bold", color="white",
    )
    ax_heat.tick_params(colors="white", labelsize=7)
    cb2 = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="white", labelsize=7)
    cb2.set_label("Entropy", fontsize=8, color="white")

    # ════════════════════════════════════════════════════════════════════
    # Panel 3: Dominant Direction Quiver (top-right)
    # ════════════════════════════════════════════════════════════════════
    ax_quiver = fig.add_subplot(gs[0, 2])
    ax_quiver.set_facecolor("#0d1117")

    # For each layer, show arrows at each token position
    for l in range(n_layers):
        for t in range(n_tokens):
            dx, dy = directions[l][t]
            mag = magnitudes[l, t]
            angle = angles[l, t]
            color = angle_cmap(angle_norm(angle))
            arrow_len = 0.3 * (mag / mag_max)
            ax_quiver.arrow(
                token_indices[t], l,
                arrow_len * np.cos(angle), arrow_len * np.sin(angle) * 0.3,
                head_width=0.12, head_length=0.05,
                fc=color, ec="none", alpha=0.7,
                linewidth=0.8,
            )

    ax_quiver.set_xlim(token_indices[0] - 2, token_indices[-1] + 2)
    ax_quiver.set_ylim(-0.5, n_layers - 0.3)
    ax_quiver.set_xlabel("Token Index", fontsize=9, color="white")
    ax_quiver.set_ylabel("Layer", fontsize=9, color="white")
    ax_quiver.set_yticks(range(n_layers))
    ax_quiver.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
    ax_quiver.set_title(
        "Dominant Jacobian Direction\n(Quiver Field)",
        fontsize=10, fontweight="bold", color="white",
    )
    ax_quiver.tick_params(colors="white", labelsize=7)
    for spine in ax_quiver.spines.values():
        spine.set_color("#333")

    # ════════════════════════════════════════════════════════════════════
    # Panel 4: Effective Rank Evolution (bottom-center)
    # ════════════════════════════════════════════════════════════════════
    ax_rank = fig.add_subplot(gs[1, 1])
    ax_rank.set_facecolor("#0d1117")

    for t in range(n_tokens):
        ranks = [eff_ranks[l, t] for l in range(n_layers)]
        ax_rank.plot(range(n_layers), ranks, alpha=0.2, color="#64b5f6", linewidth=0.7)

    mean_ranks = [eff_ranks[l].mean() for l in range(n_layers)]
    ax_rank.plot(
        range(n_layers), mean_ranks,
        color="#ff5252", linewidth=2.5, marker="o", markersize=6,
        label="Mean eff. rank", zorder=5,
    )
    ax_rank.set_xlabel("Layer", fontsize=9, color="white")
    ax_rank.set_ylabel("Effective Rank", fontsize=9, color="white")
    ax_rank.set_title(
        "Effective Rank per Layer\n(Fibre Intrinsic Dimension)",
        fontsize=10, fontweight="bold", color="white",
    )
    ax_rank.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax_rank.grid(True, alpha=0.15, color="white")
    ax_rank.set_xticks(range(n_layers))
    ax_rank.tick_params(colors="white", labelsize=7)
    for spine in ax_rank.spines.values():
        spine.set_color("#333")

    # ════════════════════════════════════════════════════════════════════
    # Panel 5: Condition Number (bottom-right)
    # ════════════════════════════════════════════════════════════════════
    ax_cond = fig.add_subplot(gs[1, 2])
    ax_cond.set_facecolor("#0d1117")

    layer_colors = cm.viridis(np.linspace(0.2, 0.9, n_layers))
    cond_matrix = np.zeros((n_layers, n_tokens))
    for l in range(n_layers):
        for t in range(n_tokens):
            cond_matrix[l, t] = min(condition_number(spectra[l][t]), 1e4)

    for l in range(n_layers):
        ax_cond.scatter(
            [l] * n_tokens, cond_matrix[l],
            c=[layer_colors[l]], alpha=0.5, s=20,
            edgecolors="white", linewidths=0.2, zorder=3,
        )

    medians = [np.median(cond_matrix[l]) for l in range(n_layers)]
    ax_cond.plot(
        range(n_layers), medians,
        color="#ff5252", linewidth=2, marker="D", markersize=5,
        label="Median κ(J)", zorder=5,
    )
    ax_cond.set_xlabel("Layer", fontsize=9, color="white")
    ax_cond.set_ylabel("Condition Number κ(J)", fontsize=9, color="white")
    ax_cond.set_yscale("log")
    ax_cond.set_title(
        "Condition Number\n(Lipschitz Distortion)",
        fontsize=10, fontweight="bold", color="white",
    )
    ax_cond.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax_cond.grid(True, alpha=0.15, color="white")
    ax_cond.set_xticks(range(n_layers))
    ax_cond.tick_params(colors="white", labelsize=7)
    for spine in ax_cond.spines.values():
        spine.set_color("#333")

    # ════════════════════════════════════════════════════════════════════
    # Suptitle
    # ════════════════════════════════════════════════════════════════════
    run_name = os.path.basename(run_info.get("run_dir", "unknown"))
    ckpt_name = os.path.basename(run_info.get("checkpoint", "none") or "none")
    source = "TRAINED" if run_info.get("checkpoint") else "RANDOM INIT"

    fig.suptitle(
        f"Kelp-Forest Fibre Bundle  [{source}]\n"
        f"Run: {run_name}  |  Checkpoint: {ckpt_name}\n"
        f"Base = Token Index,  Fibres = Jacobian SVD stalks leaning in dominant gradient direction",
        fontsize=14, fontweight="bold", color="white", y=0.99,
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Saved kelp-forest visualization to: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 viz_kelp_forest.py <run_folder_or_checkpoint>")
        print()
        print("Examples:")
        print("  python3 viz_kelp_forest.py runs/run_20250418_143022/")
        print("  python3 viz_kelp_forest.py runs/run_20250418_143022/model_best.pt")
        sys.exit(1)

    run_dir = sys.argv[1].rstrip("/")

    print("=" * 70)
    print("KELP-FOREST FIBRE BUNDLE VISUALIZATION")
    print("Jacobian fibres leaning in dominant gradient-flow directions")
    print("=" * 70)

    # ── Discover ────────────────────────────────────────────────────────
    print(f"\n[1/5] Discovering run folder: {run_dir}")
    discovered = discover_run_folder(run_dir)

    # ── Config + model ──────────────────────────────────────────────────
    print("\n[2/5] Extracting model config...")
    config = extract_model_config(discovered)

    print("\n[3/5] Loading trained model...")
    model = load_trained_model(discovered, config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"       Parameters: {param_count:,}")
    print(f"       d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, max_seq_len={config.max_seq_len}")

    # ── Create sample input ─────────────────────────────────────────────
    print("\n[4/5] Creating sample input sequence...")

    # Try to load the tokenizer from the run folder so we can encode a
    # realistic LLVM IR snippet.  Fall back to random token IDs.
    tokenizer = None
    for tok_dir in [discovered["run_dir"],
                    discovered["run_dir"] + "_best",
                    str(Path(discovered["run_dir"]).parent)]:
        tok_path = os.path.join(tok_dir, "tokenizer.json")
        if os.path.isfile(tok_path):
            try:
                tokenizer = BPETokenizer.from_pretrained(tok_dir)
                print(f"       Loaded tokenizer from: {tok_dir}")
                break
            except Exception as e:
                print(f"       WARNING: tokenizer load failed ({e}), trying next...")

    if tokenizer is not None:
        sample_text = (
            "define i64 @f(i64 %p0, i64 %p1) {\n"
            "entry:\n"
            "  %t0 = add i64 %p0, %p1\n"
            "  %t1 = mul i64 %t0, %p1\n"
            "  ret i64 %t1\n"
            "}<sep>5,3<sep>24"
        )
        ids = [tokenizer.bos_token_id] + tokenizer.encode(sample_text) + [tokenizer.eos_token_id]
        # Clamp to max_seq_len
        ids = ids[: config.max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long)
        print(f"       Encoded sample: {len(ids)} tokens")
    else:
        # No tokenizer available — use random token IDs
        seq_len = min(64, config.max_seq_len)
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long)
        print(f"       No tokenizer found — using {seq_len} random token IDs")

    print(f"       Sequence shape: {input_ids.shape}")

    # ── Compute Jacobians and visualize ─────────────────────────────────
    print("\n[5/5] Computing layer Jacobians + dominant directions...")
    print(f"       (this may take a while for {config.n_layers} layers × "
          f"{config.d_model}-dim hidden states)")

    spectra, directions, token_indices = compute_layer_jacobians(
        model, input_ids, n_token_samples=24,
    )

    print(f"       ✅ Computed {len(spectra)} layer Jacobians")
    for i in range(len(spectra)):
        svs = spectra[i]
        dirs = directions[i]
        mean_sv1 = svs[:, 0].mean()
        mean_angle = np.mean(np.arctan2(dirs[:, 1], dirs[:, 0]))
        er = np.mean([effective_rank(svs[t]) for t in range(svs.shape[0])])
        print(f"       Layer {i}: mean σ₁={mean_sv1:.4f}, "
              f"mean angle={np.degrees(mean_angle):.1f}°, "
              f"mean eff.rank={er:.2f}")

    # ── Generate the kelp-forest plot ───────────────────────────────────
    save_path = os.path.join(discovered["run_dir"], "kelp_forest_fibres.png")
    print(f"\n       Generating kelp-forest visualization...")

    plot_kelp_forest(
        spectra=spectra,
        directions=directions,
        token_indices=token_indices,
        run_info=discovered,
        save_path=save_path,
    )

    print(f"\nDone! Open {save_path} to see the kelp forest.")


if __name__ == "__main__":
    main()
