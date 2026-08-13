#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/SegMoTE/raw_data/isic2018"
OUT_DIR="$ROOT_DIR/SegMoTE/dataset/isic2018_task1"

download() {
  local url="$1"
  local file="$RAW_DIR/$(basename "$url")"

  if command -v wget >/dev/null 2>&1; then
    wget -c "$url" -O "$file"
  else
    curl -L -C - "$url" -o "$file"
  fi
}

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

if [ -f "$ROOT_DIR/SegMoTE/.venv/bin/activate" ]; then
  source "$ROOT_DIR/SegMoTE/.venv/bin/activate"
elif [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

download "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip"
download "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip"
download "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
download "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"

unzip -n -q ISIC2018_Task1-2_Training_Input.zip
unzip -n -q ISIC2018_Task1_Training_GroundTruth.zip
unzip -n -q ISIC2018_Task1-2_Validation_Input.zip
unzip -n -q ISIC2018_Task1_Validation_GroundTruth.zip

cd "$ROOT_DIR"

python SegMoTE/tools/prepare_isic2018.py \
  --train_images "$RAW_DIR/ISIC2018_Task1-2_Training_Input" \
  --train_masks "$RAW_DIR/ISIC2018_Task1_Training_GroundTruth" \
  --val_images "$RAW_DIR/ISIC2018_Task1-2_Validation_Input" \
  --val_masks "$RAW_DIR/ISIC2018_Task1_Validation_GroundTruth" \
  --output "$OUT_DIR"

find "$OUT_DIR" -maxdepth 2 -type d
