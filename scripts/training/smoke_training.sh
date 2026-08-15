#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR/SegMoTE"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
CHECKPOINT="${CHECKPOINT:-checkpoints/segmote.pth}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/moe-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/moe-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python train.py \
  --data_dir dataset \
  --dataset_list isic2018_task1 \
  --work_dir work_dir \
  --task_name smoke_isic2018 \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$EPOCHS" \
  --device "$DEVICE" \
  --gpu_ids 0 \
  --mask_num 1 \
  -num_workers 8 \
  --pretrain_path "$CHECKPOINT"
