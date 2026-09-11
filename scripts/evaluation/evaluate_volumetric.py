#!/usr/bin/env python3
"""Case-level (3D-volume) Dice/HD for a trained CT multiclass checkpoint.

`scripts/run_experiment.py` (via `src/engine/evaluate`) reports test Dice/HD
as a flat mean over individual 2D slices. Synapse/BTCV literature — every
row in MoE-SAM's Table 1 included — instead reconstructs each test case's
3D volume from its slices and reports Dice/HD per case, averaged over organs
then over patients. The two numbers are not comparable; see
`src/metrics/volumetric.py` for why. This script recomputes the second kind
from an already-trained checkpoint, without retraining and without touching
the dataset split (it reuses the run's own tracked manifest test split as-is).

Usage:
    python scripts/evaluation/evaluate_volumetric.py \\
        --run-dir runs/synapse_e3/20260830T062450Z_seed-42 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataset
from src.experiment import build_dataset_config
from src.metrics import compute_case_metrics_3d
from src.models import SegmentationOutput, build_model


SAMPLE_ID_PATTERN = re.compile(r"^(?P<case_id>.+)_slice(?P<slice_index>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run folder containing config.yaml and a checkpoint (e.g. runs/synapse_e3/<run-id>).",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint file inside --run-dir to evaluate (default: best.pt).",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=None,
        help="Defaults to the value stored in the run's config.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/<split>_metrics_volumetric.json.",
    )
    return parser.parse_args()


def _parse_sample_id(sample_id: str) -> tuple[str, int]:
    match = SAMPLE_ID_PATTERN.match(sample_id)
    if match is None:
        raise ValueError(
            f"sample_id {sample_id!r} does not match '<case_id>_slice<index>'; "
            "evaluate_volumetric.py only supports CT slice datasets built by "
            "scripts/data/ct_conversion.py."
        )
    return match.group("case_id"), int(match.group("slice_index"))


def _stack_case_volumes(
    slices_by_case: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct each case's [Z, H, W] prediction/target volumes.

    Slices are placed at their true (index - min_index) position; any slice
    index missing from the case's test records (dropped at conversion time
    for having no foreground label in that case's own volume — see
    scripts/data/ct_conversion.py) is left as background in both volumes.
    """

    volumes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case_id, slices in slices_by_case.items():
        indices = sorted(slices)
        min_index, max_index = indices[0], indices[-1]
        height, width = next(iter(slices.values()))[0].shape
        depth = max_index - min_index + 1

        prediction_volume = np.zeros((depth, height, width), dtype=np.int64)
        target_volume = np.zeros((depth, height, width), dtype=np.int64)
        for slice_index, (prediction, target) in slices.items():
            z = slice_index - min_index
            prediction_volume[z] = prediction
            target_volume[z] = target
        volumes[case_id] = (prediction_volume, target_volume)
    return volumes


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / args.checkpoint_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    if config["dataset"]["task"] != "multiclass":
        raise ValueError(
            "evaluate_volumetric.py only supports dataset.task='multiclass' "
            f"(CT slice datasets), got {config['dataset']['task']!r}."
        )

    dataset_config = build_dataset_config(config)
    dataset = build_dataset(dataset_config, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    model_config = {
        **config["model"],
        "in_channels": dataset_config.in_channels,
        "num_classes": dataset_config.num_classes,
        "task": dataset_config.task,
    }
    model = build_model(model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    model.to(device)
    model.eval()

    boundary_tolerance = (
        args.boundary_tolerance
        if args.boundary_tolerance is not None
        else float(config["training"].get("boundary_tolerance", 2.0))
    )

    slices_by_case: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            output = model(images)
            if not isinstance(output, SegmentationOutput):
                raise TypeError(
                    "Model forward must return SegmentationOutput, got "
                    f"{type(output).__name__}."
                )
            predictions = output.logits.argmax(dim=1).detach().cpu().numpy()
            targets = batch["mask"].numpy()

            for sample_id, prediction, target in zip(batch["sample_id"], predictions, targets):
                case_id, slice_index = _parse_sample_id(str(sample_id))
                slices_by_case[case_id][slice_index] = (
                    prediction.astype(np.int64),
                    target.astype(np.int64),
                )

    if not slices_by_case:
        raise ValueError(f"No samples found for split={args.split!r}.")

    volumes = _stack_case_volumes(slices_by_case)

    per_case: dict[str, dict[str, Any]] = {}
    for case_id, (prediction_volume, target_volume) in sorted(volumes.items()):
        per_case[case_id] = compute_case_metrics_3d(
            prediction_volume,
            target_volume,
            dataset_config.num_classes,
            boundary_tolerance=boundary_tolerance,
        )

    metric_keys = ("dice", "iou", "hd", "hd95", "assd", "boundary_f1")
    mean = {
        key: float(np.mean([case[key] for case in per_case.values()]))
        for key in metric_keys
    }

    payload = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "aggregation": "3d-volume-per-case",
        "num_cases": len(per_case),
        "boundary_tolerance": boundary_tolerance,
        "mean": mean,
        "per_case": per_case,
    }

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / f"{args.split}_metrics_volumetric.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    split_label = args.split.capitalize()
    print(f"Cases            : {len(per_case)}")
    print(f"{split_label} Dice (3D)   : {mean['dice']:.6f}")
    print(f"{split_label} IoU (3D)    : {mean['iou']:.6f}")
    print(f"{split_label} HD (3D)     : {mean['hd']:.6f}")
    print(f"{split_label} HD95 (3D)   : {mean['hd95']:.6f}")
    print(f"{split_label} ASSD (3D)   : {mean['assd']:.6f}")
    print(f"{split_label} Boundary F1 (3D): {mean['boundary_f1']:.6f}")
    print(f"Metrics          : {output_path}")


if __name__ == "__main__":
    main()
