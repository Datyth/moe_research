#!/usr/bin/env bash
set -euo pipefail

# Prepare the ACDC dataset for MoE-SAM-style experiments.
# Expects the raw ACDC archive (e.g. acdc.zip) at $RAW_DIR, either already
# extracted or downloaded beforehand (the challenge requires registration).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ACDC_RAW_DIR:-dataset/raw/acdc}"
OUT_DIR="${ACDC_OUT_DIR:-dataset/acdc}"

mkdir -p "$RAW_DIR"

# Extract the archive if it exists and the patients are not yet extracted.
for archive in "$RAW_DIR"/*.zip; do
  [ -e "$archive" ] || continue
  echo "Extracting $(basename "$archive")"
  unzip -n -q "$archive" -d "$RAW_DIR"
done

if ! ls "$RAW_DIR"/patient*/Info.cfg >/dev/null 2>&1; then
  echo "No ACDC patient folders found under $RAW_DIR."
  echo "Place acdc.zip (from https://www.creatis.insa-lyon.fr/Challenge/acdc/) there and re-run."
  exit 1
fi

cd "$ROOT_DIR"

python scripts/data/prepare_acdc.py \
  --raw-root "$RAW_DIR" \
  --data-root "$OUT_DIR" \
  --manifest-output manifests/acdc_v1.json \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --seed 42

find "$OUT_DIR" -maxdepth 2 -type d
