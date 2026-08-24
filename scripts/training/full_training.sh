#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "Error: Python was not found. Create .venv or set PYTHON_BIN." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-/home/teama/projects/project_01/dataset/isic2018_task1}"
EPOCHS="${EPOCHS:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
PREDICTION_THRESHOLD="${PREDICTION_THRESHOLD:-0.5}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-unet}"
HISTORY_PATH="${HISTORY_PATH:-$ROOT_DIR/results/unet_training_history.json}"
USE_AMP="${USE_AMP:-1}"

if [[ ! -f "$DATA_ROOT/dataset.json" ]]; then
  echo "Error: dataset manifest not found: $DATA_ROOT/dataset.json" >&2
  echo "Prepare it first with:" >&2
  echo "  $PYTHON scripts/data/prepare_isic2018.py --data-root $DATA_ROOT" >&2
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

TRAIN_ARGS=(
  --data-root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --base-channels "$BASE_CHANNELS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --device "$DEVICE"
  --prediction-threshold "$PREDICTION_THRESHOLD"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --checkpoint-prefix "$CHECKPOINT_PREFIX"
  --history-path "$HISTORY_PATH"
)

if [[ "$USE_AMP" == "0" || "$DEVICE" != cuda* ]]; then
  TRAIN_ARGS+=(--no-amp)
fi

echo "=== Full UNet training ==="
echo "Python          : $PYTHON"
echo "Dataset         : $DATA_ROOT"
echo "Device          : $DEVICE"
echo "Epochs          : $EPOCHS"
echo "Image size      : $IMAGE_SIZE"
echo "Batch size      : $BATCH_SIZE"
echo "Workers         : $NUM_WORKERS"
echo "Checkpoint dir  : $CHECKPOINT_DIR"
echo "Checkpoint prefix: $CHECKPOINT_PREFIX"
echo "History         : $HISTORY_PATH"
echo "AMP             : $([[ "$USE_AMP" == "0" || "$DEVICE" != cuda* ]] && echo disabled || echo enabled)"

exec "$PYTHON" scripts/training/train_unet.py "${TRAIN_ARGS[@]}"
