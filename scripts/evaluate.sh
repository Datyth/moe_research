#!/usr/bin/env bash
# Evaluate any trained checkpoint (Dice, IoU, HD95, boundary F1). Reusable
# across models: the checkpoint's own saved metadata drives what gets
# rebuilt, so no per-model/per-experiment logic lives here.
#
# Usage:
#   bash scripts/evaluate.sh runs/isic2018_e1/<run-id>/best.pt
#   SPLIT=val OUTPUT_DIR=results/my_run bash scripts/evaluate.sh <checkpoint>
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${1:-${CHECKPOINT:-}}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: bash scripts/evaluate.sh <checkpoint.pt>  (or set CHECKPOINT=...)" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Error: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/dataset/isic2018_task1}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_VISUALIZATIONS="${NUM_VISUALIZATIONS:-0}"
RUN_NAME="$(basename "$(dirname "$CHECKPOINT")")"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/$RUN_NAME}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "Error: Python was not found. Activate the project environment or set PYTHON_BIN." >&2
  exit 1
fi

if [[ "$DEVICE" == cuda* ]]; then
  if ! "$PYTHON" -c \
    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "Error: DEVICE=$DEVICE but PyTorch cannot access CUDA." >&2
    echo "Set DEVICE=cpu to run without a GPU." >&2
    exit 1
  fi
fi

echo "=== Evaluation ==="
echo "Python     : $PYTHON"
echo "Checkpoint : $CHECKPOINT"
echo "Dataset    : $DATA_ROOT ($SPLIT split)"
echo "Device     : $DEVICE"
echo "Output dir : $OUTPUT_DIR"

exec "$PYTHON" scripts/evaluation/evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-visualizations "$NUM_VISUALIZATIONS"
