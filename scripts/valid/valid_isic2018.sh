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
BATCH_SIZE="${BATCH_SIZE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
CHECKPOINT="${CHECKPOINT:-checkpoints/segmote_best.pth}"
PROMPT_MODE="${PROMPT_MODE:-bboxes}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/moe-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/moe-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python test.py \
  --data_dir dataset \
  --dataset_list isic2018_task1 \
  --work_dir work_dir \
  --task_name valid_isic2018 \
  --output_dir outputs/isic2018 \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --mask_num 1 \
  --device "$DEVICE" \
  --gpu_ids 0 \
  -num_workers 2 \
  --pretrain_path "$CHECKPOINT" \
  --prompt_mode "$PROMPT_MODE"
