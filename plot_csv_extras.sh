#!/bin/bash
# plot_csv_extras.sh — Generate additional plots from CSV data
# Usage: bash plot_csv_extras.sh runs/0/

set -euo pipefail

RUN_DIR="${1:?Usage: $0 <run_folder>}"
RUN_DIR="${RUN_DIR%/}/"
PLOTS_DIR="${RUN_DIR}plots/"
mkdir -p "$PLOTS_DIR"

echo "📊 Generating extra CSV-based plots for ${RUN_DIR}..."

# ═══════════════════════════════════════════════════════════════
# We use a single Python invocation with heredoc for all plots
# ═══════════════════════════════════════════════════════════════
python3 - "$RUN_DIR" "$PLOTS_DIR" <<'PYEOF'
import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RUN_DIR  = sys.argv[1]
PLOT_DIR = sys.argv[2]

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "#08090d",
    "axes.facecolor":     "#12152a",
    "axes.edgecolor":     "#2a2f55",
    "axes.labelcolor":    "#e8eaf6",
    "text.color":         "#e8eaf6",
    "xtick.color":        "#6b70a0",
    "ytick.color":        "#6b70a0",
    "grid.color":         "#1e2340",
    "grid.alpha":         0.6,
    "legend.facecolor":   "#181c35",
    "legend.edgecolor":   "#2a2f55",
    "legend.fontsize":    8,
    "font.family":        "monospace",
    "font.size":          9,
})

ACCENT  = "#7c5cfc"
GREEN   = "#00d4aa"
PINK    = "#f472b6"
YELLOW  = "#f0c040"
RED     = "#ff5c72"
CYAN    = "#22d3ee"
ORANGE  = "#fb923c"

def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  ✅ Saved {path}")


# ═══════════════════════════════════════════════════════════════
# 1. TRAIN vs VAL LOSS CURVE  (from epoch_losses.csv)
# ═══════════════════════════════════════════════════════════════
epoch_loss_path = os.path.join(RUN_DIR, "epoch_losses.csv")
if os.path.isfile(epoch_loss_path):
    df = pd.read_csv(epoch_loss_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["epoch"], df["train_loss"], color=ACCENT, linewidth=2,
            marker="o", markersize=5, label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"], color=GREEN, linewidth=2,
            marker="s", markersize=5, label="Val Loss")
    ax.fill_between(df["epoch"], df["train_loss"], df["val_loss"],
                     alpha=0.08, color=ACCENT)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "loss_curve.png")
else:
    print("  ⚠ epoch_losses.csv not found — skipping loss curve.")


# ═══════════════════════════════════════════════════════════════
# 2. ACCURACY OVER EPOCHS  (from epoch_log.csv)
# ═══════════════════════════════════════════════════════════════
epoch_log_path = os.path.join(RUN_DIR, "epoch_log.csv")
if os.path.isfile(epoch_log_path):
    el = pd.read_csv(epoch_log_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(el["epoch"], el["train_accuracy_pct"], color=PINK, linewidth=2,
            marker="D", markersize=5, label="Train Accuracy %")
    ax.plot(el["epoch"], el["val_accuracy_pct"], color=CYAN, linewidth=2,
            marker="^", markersize=5, label="Val Accuracy %")
    ax.axhline(y=50, color=YELLOW, linestyle=":", alpha=0.5, label="50% line")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-2, 105)
    ax.set_title("Accuracy Over Epochs (Grokking Tracker)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "accuracy_curve.png")
else:
    print("  ⚠ epoch_log.csv not found — skipping accuracy curve.")


# ═══════════════════════════════════════════════════════════════
# 3. LOSS COMPONENT BREAKDOWN  (CE, Value, Structure)
# ═══════════════════════════════════════════════════════════════
if os.path.isfile(epoch_log_path):
    el = pd.read_csv(epoch_log_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(el["epoch"], el["train_ce_loss_mean"], color=ACCENT, linewidth=2,
            marker="o", markersize=4, label="CE Loss")
    ax.plot(el["epoch"], el["train_value_loss_mean"], color=GREEN, linewidth=2,
            marker="s", markersize=4, label="Value Loss")
    ax.plot(el["epoch"], el["train_structure_loss_mean"], color=PINK, linewidth=2,
            marker="^", markersize=4, label="Structure Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Component")
    ax.set_title("Loss Component Breakdown (Train)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "loss_components.png")


# ═══════════════════════════════════════════════════════════════
# 4. PARSE RATE & UNPARSEABLE FRACTION
# ═══════════════════════════════════════════════════════════════
if os.path.isfile(epoch_log_path):
    el = pd.read_csv(epoch_log_path)
    total_train = el["train_correct"] + el["train_wrong"] + el["train_unparseable"]
    total_val   = el["val_correct"]   + el["val_wrong"]   + el["val_unparseable"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Train composition
    ax1.bar(el["epoch"], el["train_correct"],     color=GREEN,  label="Correct")
    ax1.bar(el["epoch"], el["train_wrong"],        color=RED,
            bottom=el["train_correct"], label="Wrong")
    ax1.bar(el["epoch"], el["train_unparseable"],  color="#4a4f78",
            bottom=el["train_correct"]+el["train_wrong"], label="Unparseable")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Samples")
    ax1.set_title("Train: Correct / Wrong / Unparseable", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7); ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Val composition
    ax2.bar(el["epoch"], el["val_correct"],     color=GREEN,  label="Correct")
    ax2.bar(el["epoch"], el["val_wrong"],        color=RED,
            bottom=el["val_correct"], label="Wrong")
    ax2.bar(el["epoch"], el["val_unparseable"],  color="#4a4f78",
            bottom=el["val_correct"]+el["val_wrong"], label="Unparseable")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Samples")
    ax2.set_title("Val: Correct / Wrong / Unparseable", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7); ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Sample Parse & Correctness Breakdown", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "parse_breakdown.png")


# ═══════════════════════════════════════════════════════════════
# 5. GRADIENT NORM OVER EPOCHS
# ═══════════════════════════════════════════════════════════════
if os.path.isfile(epoch_log_path):
    el = pd.read_csv(epoch_log_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(el["epoch"], el["mean_grad_norm"], color=ORANGE, linewidth=2,
            marker="o", markersize=5, label="Mean Grad Norm")
    ax.plot(el["epoch"], el["max_grad_norm"], color=RED, linewidth=1.5,
            marker="x", markersize=5, linestyle="--", label="Max Grad Norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms Over Epochs", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "grad_norms.png")


# ═══════════════════════════════════════════════════════════════
# 6. BATCH-LEVEL LOSS (train) — fine-grained view
# ═══════════════════════════════════════════════════════════════
batch_train_path = os.path.join(RUN_DIR, "batch_losses_train.csv")
if os.path.isfile(batch_train_path):
    bt = pd.read_csv(batch_train_path)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(bt))
    ax.plot(x, bt["loss"], color="#4a4f78", linewidth=0.6, alpha=0.6, label="Batch Loss")
    ax.plot(x, bt["ema_loss"], color=ACCENT, linewidth=2, label="EMA Loss")

    # Mark epoch boundaries
    epoch_starts = bt.index[bt["batch"] == 0].tolist()
    for es in epoch_starts:
        ax.axvline(x=es, color=YELLOW, alpha=0.2, linewidth=0.8)

    ax.set_xlabel("Global Batch Step")
    ax.set_ylabel("Loss")
    ax.set_title("Batch-Level Training Loss (with EMA)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    save(fig, "batch_loss_train.png")
else:
    print("  ⚠ batch_losses_train.csv not found — skipping batch loss plot.")


# ═══════════════════════════════════════════════════════════════
# 7. BATCH-LEVEL VAL LOSS
# ═══════════════════════════════════════════════════════════════
batch_val_path = os.path.join(RUN_DIR, "batch_losses_val.csv")
if os.path.isfile(batch_val_path):
    bv = pd.read_csv(batch_val_path)
    fig, ax = plt.subplots(figsize=(12, 5))
    # Group by epoch, plot each epoch's batches
    for ep, grp in bv.groupby("epoch"):
        ax.plot(grp["batch"].values, grp["loss"].values,
                marker=".", markersize=3, linewidth=1.2, alpha=0.8,
                label=f"Epoch {ep}")
    ax.set_xlabel("Batch Index (within epoch)")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss per Batch (by Epoch)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)
    save(fig, "batch_loss_val.png")


# ═══════════════════════════════════════════════════════════════
# 8. EPOCH TIMING — wall-clock seconds per epoch
# ═══════════════════════════════════════════════════════════════
if os.path.isfile(epoch_loss_path):
    df = pd.read_csv(epoch_loss_path)
    if "elapsed_secs" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(df["epoch"], df["elapsed_secs"], color=ACCENT, alpha=0.8,
                       edgecolor="#2a2f55", linewidth=0.5)
        avg = df["elapsed_secs"].mean()
        ax.axhline(y=avg, color=YELLOW, linestyle="--", linewidth=1.2,
                    label=f"Mean: {avg:.1f}s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds")
        ax.set_title("Wall-Clock Time per Epoch", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        save(fig, "epoch_timing.png")


# ═══════════════════════════════════════════════════════════════
# 9. OFF-BY TOLERANCE HEATMAP (train & val)
# ═══════════════════════════════════════════════════════════════
if os.path.isfile(epoch_log_path):
    el = pd.read_csv(epoch_log_path)
    tol_cols_train = ["train_pct_off_by_0","train_pct_off_by_le_1",
                      "train_pct_off_by_le_5","train_pct_off_by_le_10",
                      "train_pct_off_by_le_100"]
    tol_cols_val   = ["val_pct_off_by_0","val_pct_off_by_le_1",
                      "val_pct_off_by_le_5","val_pct_off_by_le_10",
                      "val_pct_off_by_le_100"]

    if all(c in el.columns for c in tol_cols_train):
        tol_labels = ["Exact", "≤1", "≤5", "≤10", "≤100"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        train_mat = el[tol_cols_train].values.T
        im1 = ax1.imshow(train_mat, aspect="auto", cmap="inferno",
                          vmin=0, vmax=100, interpolation="nearest")
        ax1.set_yticks(range(len(tol_labels))); ax1.set_yticklabels(tol_labels)
        ax1.set_xticks(range(len(el))); ax1.set_xticklabels(el["epoch"].astype(int))
        ax1.set_xlabel("Epoch"); ax1.set_title("Train: % within tolerance", fontsize=11, fontweight="bold")
        # Annotate cells
        for i in range(train_mat.shape[0]):
            for j in range(train_mat.shape[1]):
                v = train_mat[i, j]
                c = "white" if v < 60 else "black"
                ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=c)

        val_mat = el[tol_cols_val].values.T
        im2 = ax2.imshow(val_mat, aspect="auto", cmap="inferno",
                          vmin=0, vmax=100, interpolation="nearest")
        ax2.set_yticks(range(len(tol_labels))); ax2.set_yticklabels(tol_labels)
        ax2.set_xticks(range(len(el))); ax2.set_xticklabels(el["epoch"].astype(int))
        ax2.set_xlabel("Epoch"); ax2.set_title("Val: % within tolerance", fontsize=11, fontweight="bold")
        for i in range(val_mat.shape[0]):
            for j in range(val_mat.shape[1]):
                v = val_mat[i, j]
                c = "white" if v < 60 else "black"
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=c)

        fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="% of samples")
        fig.suptitle("Prediction Tolerance Heatmap", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        save(fig, "tolerance_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# 10. GENERALIZATION GAP (train_loss - val_loss)
# ═══════════════════════════════════════════════════════════════
if os.path.isfile(epoch_loss_path):
    df = pd.read_csv(epoch_loss_path)
    gap = df["train_loss"] - df["val_loss"]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [GREEN if g > 0 else RED for g in gap]
    ax.bar(df["epoch"], gap, color=colors, alpha=0.85, edgecolor="#2a2f55", linewidth=0.5)
    ax.axhline(y=0, color="#6b70a0", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss − Val Loss")
    ax.set_title("Generalization Gap (positive = val better than train)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "generalization_gap.png")


# ═══════════════════════════════════════════════════════════════
# 11. COMBINED DASHBOARD SUMMARY (2x2 subplot)
# ═══════════════════════════════════════════════════════════════
if os.path.isfile(epoch_loss_path) and os.path.isfile(epoch_log_path):
    df = pd.read_csv(epoch_loss_path)
    el = pd.read_csv(epoch_log_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Loss
    axes[0,0].plot(df["epoch"], df["train_loss"], color=ACCENT, lw=2, marker="o", ms=4, label="Train")
    axes[0,0].plot(df["epoch"], df["val_loss"], color=GREEN, lw=2, marker="s", ms=4, label="Val")
    axes[0,0].set_title("Loss", fontweight="bold"); axes[0,0].legend(fontsize=7)
    axes[0,0].grid(True, ls="--", alpha=0.3)
    axes[0,0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # (0,1) Accuracy
    axes[0,1].plot(el["epoch"], el["train_accuracy_pct"], color=PINK, lw=2, marker="D", ms=4, label="Train")
    axes[0,1].plot(el["epoch"], el["val_accuracy_pct"], color=CYAN, lw=2, marker="^", ms=4, label="Val")
    axes[0,1].set_ylim(-2, 105)
    axes[0,1].set_title("Accuracy %", fontweight="bold"); axes[0,1].legend(fontsize=7)
    axes[0,1].grid(True, ls="--", alpha=0.3)
    axes[0,1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # (1,0) Grad norms
    axes[1,0].plot(el["epoch"], el["mean_grad_norm"], color=ORANGE, lw=2, marker="o", ms=4, label="Mean")
    axes[1,0].plot(el["epoch"], el["max_grad_norm"], color=RED, lw=1.5, marker="x", ms=4, ls="--", label="Max")
    axes[1,0].set_title("Gradient Norms", fontweight="bold"); axes[1,0].legend(fontsize=7)
    axes[1,0].grid(True, ls="--", alpha=0.3)
    axes[1,0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # (1,1) Parse rate
    axes[1,1].plot(el["epoch"], el["train_parse_rate_mean"]*100, color=YELLOW, lw=2, marker="o", ms=4, label="Train Parse %")
    axes[1,1].plot(el["epoch"], el["val_parse_rate_mean"]*100, color=GREEN, lw=2, marker="s", ms=4, label="Val Parse %")
    axes[1,1].set_ylim(-2, 105)
    axes[1,1].set_title("Parse Rate %", fontweight="bold"); axes[1,1].legend(fontsize=7)
    axes[1,1].grid(True, ls="--", alpha=0.3)
    axes[1,1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for ax in axes.flat:
        ax.set_xlabel("Epoch", fontsize=8)

    fig.suptitle("Training Overview Dashboard", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "overview_dashboard.png")


print("\n🎉 All extra plots generated!")
PYEOF

echo "✅ plot_csv_extras.sh finished."
