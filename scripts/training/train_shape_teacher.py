"""Train clean or denoising Gaussian Shape Teacher from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shape_teacher.engine import (  # noqa: E402
    execute_shape_teacher_experiment,
    load_shape_teacher_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a mask-only Shape Teacher.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()
    config = load_shape_teacher_config(arguments.config, project_root=PROJECT_ROOT)
    if arguments.device:
        config["training"]["device"] = arguments.device
    run_directory = execute_shape_teacher_experiment(config, command=sys.argv)
    print(f"Completed Shape Teacher run: {run_directory}")


if __name__ == "__main__":
    main()
