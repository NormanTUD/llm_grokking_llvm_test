#!/bin/bash
#SBATCH --job-name=falsify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=falsify_%j.out
#SBATCH --error=falsify_%j.err

set -euo pipefail

# ── Bootstrap uv if not available ──
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv: $(which uv)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Starting falsification at $(date)"

# ── Run with uv (handles all deps automatically) ──
uv run --quiet falsification.py --compare-all --device cuda

echo "Finished at $(date)"
