#!/bin/bash
# plot_csv_extras.sh — Generate additional plots from CSV data
# NO PANDAS REQUIRED — uses only csv module + matplotlib
# Usage: bash plot_csv_extras.sh runs/0/

set -euo pipefail

RUN_DIR="${1:?Usage: $0 <run_folder>}"
RUN_DIR="${RUN_DIR%/}/"
PLOTS_DIR="${RUN_DIR}plots/"
mkdir -p "$PLOTS_DIR"

echo "📊 Generating extra CSV-based plots for ${RUN_DIR}..."

python3 - "$RUN_DIR" "$PLOTS_DIR" <<'PYEOF'
import sys, os, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    print(f"  ✅ {name}")

def read_csv(filepath):
    """Read CSV into list of dicts using stdlib csv module."""
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def col_float(rows, key):
    """Extract a column as list of floats, skipping empty/missing."""
    vals = []
    for r in rows:
        v = r.get(key, "").strip()
        if v == "" or v == "None":
            vals.append(None)
        else:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(None)
    return vals

def col_int(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "").strip()
        if v == "" or v == "None":
            vals.append(None)
        else:
            try:
                vals.append(int(float(v)))
            except ValueError:
                vals.append(None)
    return vals

def clean_pairs(xs, ys):
    """Remove entries where either is None."""
    px, py = [], []
    for x, y in zip(xs, ys):
        if x is not None and y is not None:
            px.append(x)
            py.append(y)
    return px, py


# ═══════════════════════════════════════════════════════════════
# Load CSVs
# ═══════════════════════════════════════════════════════════════
epoch_loss_path = os.path.join(RUN_DIR, "epoch_losses.csv")
epoch_log_path  = os.path.join(RUN_DIR, "epoch_log.csv")
batch_train_path = os.path.join(RUN_DIR, "batch_losses_train.csv")
batch_val_path   = os.path.join(RUN_DIR, "batch_losses_val.csv")

has_epoch_loss = os.path.isfile(epoch_loss_path)
has_epoch_log  = os.path.isfile(epoch_log_path)
has_batch_train = os.path.isfile(batch_train_path)
has_batch_val   = os.path.isfile(batch_val_path)

if has_epoch_loss:
    el_rows = read_csv(epoch_loss_path)
if has_epoch_log:
    log_rows = read_csv(epoch_log_path)
if has_batch_train:
    bt_rows = read_csv(batch_train_path)
if has_batch_val:
    bv_rows = read_csv(batch_val_path)


# ═══════════════════════════════════════════════════════════════
# 1. TRAIN vs VAL LOSS CURVE
# ═══════════════════════════════════════════════════════════════
if has_epoch_loss:
    epochs = col_int(el_rows, "epoch")
    train_loss = col_float(el_rows, "train_loss")
    val_loss = col_float(el_rows, "val_loss")

    ex, ty = clean_pairs(epochs, train_loss)
    ex2, vy = clean_pairs(epochs, val_loss)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ex, ty, color=ACCENT, linewidth=2, marker="o", markersize=5, label="Train Loss")
    ax.plot(ex2, vy, color=GREEN, linewidth=2, marker="s", markersize=5, label="Val Loss")
    if len(ex) == len(ex2):
        ax.fill_between(ex, ty, vy, alpha=0.08, color=ACCENT)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "loss_curve.png")


# ═══════════════════════════════════════════════════════════════
# 2. ACCURACY OVER EPOCHS
# ═══════════════════════════════════════════════════════════════
if has_epoch_log:
    epochs = col_int(log_rows, "epoch")
    train_acc = col_float(log_rows, "train_accuracy_pct")
    val_acc = col_float(log_rows, "val_accuracy_pct")

    fig, ax = plt.subplots(figsize=(10, 5))
    ex, ty = clean_pairs(epochs, train_acc)
    ex2, vy = clean_pairs(epochs, val_acc)
    ax.plot(ex, ty, color=PINK, linewidth=2, marker="D", markersize=5, label="Train Accuracy %")
    ax.plot(ex2, vy, color=CYAN, linewidth=2, marker="^", markersize=5, label="Val Accuracy %")
    ax.axhline(y=50, color=YELLOW, linestyle=":", alpha=0.5, label="50% line")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-2, 105)
    ax.set_title("Accuracy Over Epochs (Grokking Tracker)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "accuracy_curve.png")


# ═══════════════════════════════════════════════════════════════
# 3. LOSS COMPONENT BREAKDOWN (CE, Value, Structure)
# ═══════════════════════════════════════════════════════════════
if has_epoch_log:
    epochs = col_int(log_rows, "epoch")
    ce = col_float(log_rows, "train_ce_loss_mean")
    val_l = col_float(log_rows, "train_value_loss_mean")
    struct = col_float(log_rows, "train_structure_loss_mean")

    fig, ax = plt.subplots(figsize=(10, 5))
    ex, cy = clean_pairs(epochs, ce)
    ax.plot(ex, cy, color=ACCENT, linewidth=2, marker="o", markersize=4, label="CE Loss")
    ex2, vy2 = clean_pairs(epochs, val_l)
    ax.plot(ex2, vy2, color=GREEN, linewidth=2, marker="s", markersize=4, label="Value Loss")
    ex3, sy = clean_pairs(epochs, struct)
    ax.plot(ex3, sy, color=PINK, linewidth=2, marker="^", markersize=4, label="Structure Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Component")
    ax.set_title("Loss Component Breakdown (Train)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "loss_components.png")


# ═══════════════════════════════════════════════════════════════
# 4. PARSE RATE & SAMPLE BREAKDOWN (stacked bars)
# ═══════════════════════════════════════════════════════════════
if has_epoch_log:
    epochs = col_int(log_rows, "epoch")
    t_correct = col_int(log_rows, "train_correct")
    t_wrong = col_int(log_rows, "train_wrong")
    t_unparse = col_int(log_rows, "train_unparseable")
    v_correct = col_int(log_rows, "val_correct")
    v_wrong = col_int(log_rows, "val_wrong")
    v_unparse = col_int(log_rows, "val_unparseable")

    # Filter out None
    valid = [(e, c, w, u) for e, c, w, u in zip(epochs, t_correct, t_wrong, t_unparse)
             if all(x is not None for x in (e, c, w, u))]

    if valid:
        ep = [v[0] for v in valid]
        tc = [v[1] for v in valid]
        tw = [v[2] for v in valid]
        tu = [v[3] for v in valid]

        valid_v = [(e, c, w, u) for e, c, w, u in zip(epochs, v_correct, v_wrong, v_unparse)
                   if all(x is not None for x in (e, c, w, u))]
        ep_v = [v[0] for v in valid_v]
        vc = [v[1] for v in valid_v]
        vw = [v[2] for v in valid_v]
        vu = [v[3] for v in valid_v]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Train
        ax1.bar(ep, tc, color=GREEN, label="Correct")
        ax1.bar(ep, tw, bottom=tc, color=RED, label="Wrong")
        bot2 = [a+b for a, b in zip(tc, tw)]
        ax1.bar(ep, tu, bottom=bot2, color="#4a4f78", label="Unparseable")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Samples")
        ax1.set_title("Train: Correct / Wrong / Unparseable", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=7); ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Val
        if valid_v:
            ax2.bar(ep_v, vc, color=GREEN, label="Correct")
            ax2.bar(ep_v, vw, bottom=vc, color=RED, label="Wrong")
            bot2v = [a+b for a, b in zip(vc, vw)]
            ax2.bar(ep_v, vu, bottom=bot2v, color="#4a4f78", label="Unparseable")
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
if has_epoch_log:
    epochs = col_int(log_rows, "epoch")
    mean_gn = col_float(log_rows, "mean_grad_norm")
    max_gn = col_float(log_rows, "max_grad_norm")

    fig, ax = plt.subplots(figsize=(10, 5))
    ex, my = clean_pairs(epochs, mean_gn)
    ax.plot(ex, my, color=ORANGE, linewidth=2, marker="o", markersize=5, label="Mean Grad Norm")
    ex2, xy = clean_pairs(epochs, max_gn)
    ax.plot(ex2, xy, color=RED, linewidth=1.5, marker="x", markersize=5, linestyle="--", label="Max Grad Norm")
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
if has_batch_train:
    loss_vals = col_float(bt_rows, "loss")
    ema_vals = col_float(bt_rows, "ema_loss")
    batch_idx_col = col_int(bt_rows, "batch")

    x = list(range(len(loss_vals)))
    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw loss
    clean_x_l = [i for i, v in zip(x, loss_vals) if v is not None]
    clean_l = [v for v in loss_vals if v is not None]
    ax.plot(clean_x_l, clean_l, color="#4a4f78", linewidth=0.6, alpha=0.6, label="Batch Loss")

    # EMA
    clean_x_e = [i for i, v in zip(x, ema_vals) if v is not None]
    clean_e = [v for v in ema_vals if v is not None]
    ax.plot(clean_x_e, clean_e, color=ACCENT, linewidth=2, label="EMA Loss")

    # Epoch boundaries
    if batch_idx_col:
        for i, b in enumerate(batch_idx_col):
            if b == 0:
                ax.axvline(x=i, color=YELLOW, alpha=0.2, linewidth=0.8)

    ax.set_xlabel("Global Batch Step")
    ax.set_ylabel("Loss")
    ax.set_title("Batch-Level Training Loss (with EMA)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    save(fig, "batch_loss_train.png")


# ═══════════════════════════════════════════════════════════════
# 7. BATCH-LEVEL VAL LOSS (by epoch)
# ═══════════════════════════════════════════════════════════════
if has_batch_val:
    bv_epochs = col_int(bv_rows, "epoch")
    bv_batch = col_int(bv_rows, "batch")
    bv_loss = col_float(bv_rows, "loss")

    # Group by epoch
    epoch_groups = {}
    for e, b, l in zip(bv_epochs, bv_batch, bv_loss):
        if e is not None and b is not None and l is not None:
            epoch_groups.setdefault(e, ([], []))
            epoch_groups[e][0].append(b)
            epoch_groups[e][1].append(l)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [ACCENT, GREEN, PINK, CYAN, YELLOW, ORANGE, RED, "#a78bfa", "#34d399", "#fbbf24"]
    for i, (ep, (batches, losses)) in enumerate(sorted(epoch_groups.items())):
        c = colors[i % len(colors)]
        ax.plot(batches, losses, marker=".", markersize=3, linewidth=1.2, alpha=0.8,
                color=c, label=f"Epoch {ep}")
    ax.set_xlabel("Batch Index (within epoch)")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss per Batch (by Epoch)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)
    save(fig, "batch_loss_val.png")


# ═══════════════════════════════════════════════════════════════
# 8. EPOCH TIMING — wall-clock seconds per epoch
# ═══════════════════════════════════════════════════════════════
if has_epoch_loss:
    epochs = col_int(el_rows, "epoch")
    elapsed = col_float(el_rows, "elapsed_secs")

    ex, ey = clean_pairs(epochs, elapsed)
    if ex:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(ex, ey, color=ACCENT, alpha=0.8, edgecolor="#2a2f55", linewidth=0.5)
        avg = sum(ey) / len(ey)
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
# 9. TOLERANCE HEATMAP (off-by-0/1/5/10/100)
# ═══════════════════════════════════════════════════════════════
if has_epoch_log:
    tol_keys_train = ["train_pct_off_by_0","train_pct_off_by_le_1",
                      "train_pct_off_by_le_5","train_pct_off_by_le_10",
                      "train_pct_off_by_le_100"]
    tol_keys_val   = ["val_pct_off_by_0","val_pct_off_by_le_1",
                      "val_pct_off_by_le_5","val_pct_off_by_le_10",
                      "val_pct_off_by_le_100"]
    tol_labels = ["Exact", "≤1", "≤5", "≤10", "≤100"]

    # Check columns exist
    if all(k in log_rows[0] for k in tol_keys_train):
        epochs = col_int(log_rows, "epoch")
        n_epochs = len(epochs)

        train_mat = []
        for k in tol_keys_train:
            train_mat.append(col_float(log_rows, k))
        val_mat = []
        for k in tol_keys_val:
            val_mat.append(col_float(log_rows, k))

        # Replace None with 0
        for row in train_mat:
            for i in range(len(row)):
                if row[i] is None: row[i] = 0.0
        for row in val_mat:
            for i in range(len(row)):
                if row[i] is None: row[i] = 0.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        im1 = ax1.imshow(train_mat, aspect="auto", cmap="inferno",
                          vmin=0, vmax=100, interpolation="nearest")
        ax1.set_yticks(range(len(tol_labels))); ax1.set_yticklabels(tol_labels)
        ax1.set_xticks(range(n_epochs))
        ax1.set_xticklabels([str(e) for e in epochs])
        ax1.set_xlabel("Epoch"); ax1.set_title("Train: % within tolerance", fontsize=11, fontweight="bold")
        for i in range(len(tol_labels)):
            for j in range(n_epochs):
                v = train_mat[i][j]
                c = "white" if v < 60 else "black"
                ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=c)

        im2 = ax2.imshow(val_mat, aspect="auto", cmap="inferno",
                          vmin=0, vmax=100, interpolation="nearest")
        ax2.set_yticks(range(len(tol_labels))); ax2.set_yticklabels(tol_labels)
        ax2.set_xticks(range(n_epochs))
        ax2.set_xticklabels([str(e) for e in epochs])
        ax2.set_xlabel("Epoch"); ax2.set_title("Val: % within tolerance", fontsize=11, fontweight="bold")
        for i in range(len(tol_labels)):
            for j in range(n_epochs):
                v = val_mat[i][j]
                c = "white" if v < 60 else "black"
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=c)

        fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="% of samples")
        fig.suptitle("Prediction Tolerance Heatmap", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        save(fig, "tolerance_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# 10. GENERALIZATION GAP
# ═══════════════════════════════════════════════════════════════
if has_epoch_loss:
    epochs = col_int(el_rows, "epoch")
    train_loss = col_float(el_rows, "train_loss")
    val_loss = col_float(el_rows, "val_loss")

    valid = [(e, t-v) for e, t, v in zip(epochs, train_loss, val_loss)
             if all(x is not None for x in (e, t, v))]
    if valid:
        ep = [v[0] for v in valid]
        gap = [v[1] for v in valid]
        colors_bar = [GREEN if g > 0 else RED for g in gap]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(ep, gap, color=colors_bar, alpha=0.85, edgecolor="#2a2f55", linewidth=0.5)
        ax.axhline(y=0, color="#6b70a0", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss − Val Loss")
        ax.set_title("Generalization Gap (positive = val generalizes better)", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        save(fig, "generalization_gap.png")


# ═══════════════════════════════════════════════════════════════
# 11. COMBINED 2×2 DASHBOARD
# ═══════════════════════════════════════════════════════════════
if has_epoch_loss and has_epoch_log:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Loss
    epochs = col_int(el_rows, "epoch")
    tl = col_float(el_rows, "train_loss")
    vl = col_float(el_rows, "val_loss")
    ex, ty = clean_pairs(epochs, tl)
    ex2, vy = clean_pairs(epochs, vl)
    axes[0,0].plot(ex, ty, color=ACCENT, lw=2, marker="o", ms=4, label="Train")
    axes[0,0].plot(ex2, vy, color=GREEN, lw=2, marker="s", ms=4, label="Val")
    axes[0,0].set_title("Loss", fontweight="bold"); axes[0,0].legend(fontsize=7)
    axes[0,0].grid(True, ls="--", alpha=0.3)
    axes[0,0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # (0,1) Accuracy
    epochs2 = col_int(log_rows, "epoch")
    ta = col_float(log_rows, "train_accuracy_pct")
    va = col_float(log_rows, "val_accuracy_pct")
    ex3, ty3 = clean_pairs(epochs2, ta)
    ex4, vy4 = clean_pairs(epochs2, va)
    axes[0,1].plot(ex3, ty3, color=PINK, lw=2, marker="D", ms=4, label="Train")
    axes[0,1].plot(ex4, vy4, color=CYAN, lw=2, marker="^", ms=4, label="Val")
    axes[0,1].set_ylim(-2, 105)
    axes[0,1].set_title("Accuracy %", fontweight="bold"); axes[0,1].legend(fontsize=7)
    axes[0,1].grid(True, ls="--", alpha=0.3)
    axes[0,1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # (1,0) Grad norms
    mg = col_float(log_rows, "mean_grad_norm")
    xg = col_float(log_rows, "max_grad_norm")
    ex5, my5 = clean_pairs(epochs2, mg)
    ex6, xy6 = clean_pairs(epochs2, xg)
    axes[1,0].plot(ex5, my5, color=ORANGE, lw=2, marker="o", ms=4, label="Mean")
    axes[1,0].plot(ex6, xy6, color=RED, lw=1.5, marker="x", ms=4, ls="--", label="Max")
    axes[1,0].set_title("Gradient Norms", fontweight="bold"); axes[1,0].legend(fontsize=7)
    axes[1,0].grid(True, ls="--", alpha=0.3)
    axes[1,0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # (1,1) Parse rate
    pr_t = col_float(log_rows, "train_parse_rate_mean")
    pr_v = col_float(log_rows, "val_parse_rate_mean")
    # Convert to percentage
    pr_t_pct = [v*100 if v is not None else None for v in pr_t]
    pr_v_pct = [v*100 if v is not None else None for v in pr_v]
    ex7, py7 = clean_pairs(epochs2, pr_t_pct)
    ex8, py8 = clean_pairs(epochs2, pr_v_pct)
    axes[1,1].plot(ex7, py7, color=YELLOW, lw=2, marker="o", ms=4, label="Train Parse %")
    axes[1,1].plot(ex8, py8, color=GREEN, lw=2, marker="s", ms=4, label="Val Parse %")
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
