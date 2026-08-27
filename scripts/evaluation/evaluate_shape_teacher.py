"""Evaluate a Shape Teacher checkpoint on clean or fixed corrupted masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shape_teacher.engine import evaluate_shape_teacher_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Shape Teacher.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--input-mode", choices=("clean", "corrupted"), required=True
    )
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()
    metrics = evaluate_shape_teacher_checkpoint(
        arguments.checkpoint,
        split=arguments.split,
        input_mode=arguments.input_mode,
        device=arguments.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
