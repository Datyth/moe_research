#!/usr/bin/env bash
# Train the E-SAM model on ISIC2018. Thin wrapper around
# scripts/run_experiment.py; default config is isic2018_e1.yaml (MoE-SAM),
# pass CONFIG=configs/isic2018_e0.yaml for the no-MoE baseline.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/isic2018_e1.yaml}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/dataset/isic2018_task1}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/checkpoints/sam_vit_b_01ec64.pth}"

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

if [[ ! -f "$DATA_ROOT/dataset.json" ]]; then
  echo "Error: dataset manifest not found: $DATA_ROOT/dataset.json" >&2
  echo "Prepare it first with: bash scripts/02_prepare_isic2018_dataset.sh" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Error: SAM ViT-B checkpoint not found: $CHECKPOINT" >&2
  echo "Download it first with:" >&2
  echo "  mkdir -p checkpoints" >&2
  echo "  curl -L -o $CHECKPOINT https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" >&2
  exit 1
fi

if grep -q "device: cuda" "$CONFIG" && ! "$PYTHON" -c \
  'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "Error: $CONFIG requests device: cuda but PyTorch cannot access CUDA." >&2
  echo "Edit the config's training.device to cpu, or fix the CUDA install." >&2
  exit 1
fi

echo "=== E-SAM training on ISIC2018 ==="
echo "Python     : $PYTHON"
echo "Config     : $CONFIG"
echo "Dataset    : $DATA_ROOT"
echo "Checkpoint : $CHECKPOINT"

exec "$PYTHON" scripts/run_experiment.py --config "$CONFIG"
