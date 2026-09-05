#!/usr/bin/env python3
"""Run a fresh Phase-A shape experiment or resume an existing run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import load_experiment_config
from src.experiments import execute_shape_pretraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="YAML config for a fresh run.")
    mode.add_argument("--resume", type=Path, help="Existing run directory to resume.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is not None:
        config = load_experiment_config(
            args.config,
            project_root=PROJECT_ROOT,
        )
        execute_shape_pretraining(config)
        return

    run_dir = args.resume.expanduser().resolve()
    config = load_experiment_config(
        run_dir / "config.yaml",
        project_root=PROJECT_ROOT,
    )
    execute_shape_pretraining(config, run_dir=run_dir, resume=True)


if __name__ == "__main__":
    main()
