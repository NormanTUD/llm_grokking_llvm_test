#!/usr/bin/env bash
set -euo pipefail

# ── Auto-install dependencies in venv if not already in one ─────────
if [ -z "${VIRTUAL_ENV:-}" ]; then
  VENV_DIR=".venv_ci"
  if [ ! -d "$VENV_DIR" ]; then
    echo "▶ Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
  fi
  echo "▶ Activating virtual environment..."
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

if ! python3 -c "import pytest" 2>/dev/null; then
  echo "▶ Installing test dependencies..."
  pip install --quiet --upgrade pip
  pip install --quiet pytest torch
  if [ -f requirements.txt ]; then
    pip install --quiet -r requirements.txt
  fi
fi

RUN_DIR="test_run_ci"
COMMON_ARGS=(
  --epochs 1
  --batches-per-epoch 1
  --val-batches 1
  --batch-size 2
  --target-params 500
  --max-params 2
  --max-ops 1
  --param-min -5
  --param-max 5
  --tokenizer_initial_nr 50
  --bpe-vocab-size 128
  --device cpu
  --dont-plot
  --no-plot
  --no-plot-window
  --seed 42
  --replay-save-rate 0
  --scheduler none
  --optimizer adamw
  --lr 1e-3
)

python3 -c "
from random_infix_gen import generate_random_function
code, res = generate_random_function(num_params=2, params=[5,3], allowed_ops=['add','sub'], num_operations=2, seed=42)
assert isinstance(res, int), 'Result should be int'
print(f'✓ Infix gen: {code} = {res}')
"

cleanup() {
  rm -rf "$RUN_DIR" "${RUN_DIR}_best" llvm_gpt_model llvm_gpt_model_best \
         training_plot.png batch_log.csv epoch_log.csv
}
trap cleanup EXIT

echo "═══════════════════════════════════════════"
echo "  CI Smoke Test: train.py (3 runs)"
echo "═══════════════════════════════════════════"

# ── Run 1: Fresh training ────────────────────────────────────────────
echo ""
echo "▶ Run 1/3: Fresh training (1 epoch, 1 batch)..."
python3 train.py "${COMMON_ARGS[@]}" --run-dir "$RUN_DIR"
rc1=$?
echo "  Exit code: $rc1"
if [ "$rc1" -ne 0 ]; then
  echo "✗ Run 1 FAILED (exit $rc1)"
  exit 1
fi
echo "✓ Run 1 passed"

# Find the run subdirectory (e.g. test_run_ci/0)
LAST_RUN=$(find "${RUN_DIR}" -maxdepth 1 -type d -regex '.*/[0-9]+$' | sort -n | tail -1)
if [ -z "$LAST_RUN" ]; then
  echo "✗ Could not find run directory under ${RUN_DIR}/"
  exit 1
fi
echo "  Run dir: $LAST_RUN"

# ── Run 2: Continue from run 1 ──────────────────────────────────────
echo ""
echo "▶ Run 2/3: Continue training (+1 epoch)..."
python3 train.py "${COMMON_ARGS[@]}" --continue "$LAST_RUN"
rc2=$?
echo "  Exit code: $rc2"
if [ "$rc2" -ne 0 ]; then
  echo "✗ Run 2 FAILED (exit $rc2)"
  exit 1
fi
echo "✓ Run 2 passed"

# ── Run 3: Continue again ───────────────────────────────────────────
echo ""
echo "▶ Run 3/3: Continue training (+1 more epoch)..."
python3 train.py "${COMMON_ARGS[@]}" --continue "$LAST_RUN"
rc3=$?
echo "  Exit code: $rc3"
if [ "$rc3" -ne 0 ]; then
  echo "✗ Run 3 FAILED (exit $rc3)"
  exit 1
fi
echo "✓ Run 3 passed"

# ── Run pytest suite ────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  CI Unit Tests: pytest"
echo "═══════════════════════════════════════════"
echo ""
python3 -m pytest test_sample_logging.py -v
rc_pytest=$?
if [ "$rc_pytest" -ne 0 ]; then
  echo "✗ pytest FAILED (exit $rc_pytest)"
  exit 1
fi
echo "✓ pytest passed"

if ! ./tiny_turnstile --epochs=1; then
	echo "tiny_turnstile failed"
	exit 1
fi


# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  All checks passed"
echo "    Train runs:  $rc1, $rc2, $rc3"
echo "    Pytest:      $rc_pytest"
echo "═══════════════════════════════════════════"
exit 0
