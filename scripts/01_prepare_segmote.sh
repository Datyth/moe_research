#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/SegMoTE"

uv venv --python 3.10 .venv
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -U huggingface_hub

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
PY

mkdir -p checkpoints

hf download yujielu/SegMoTE \
  --local-dir checkpoints

ls -lh checkpoints
