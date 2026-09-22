#!/usr/bin/env python3
"""Evaluate deployable-prior and teacher-oracle paths of a Phase-C checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.dataset import DatasetConfig
from src.data import build_dataset
from src.engine import evaluate
from src.losses import build_loss
from src.models import build_model
from src.tasks import PhaseCDistillTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _checkpoint_configuration(
    checkpoint: dict[str, Any],
    *,
    data_root: Path | None,
) -> tuple[dict[str, Any], DatasetConfig, dict[str, Any], dict[str, Any]]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Phase-C checkpoint metadata is required.")
    model_config = metadata.get("model_config")
    data_config = metadata.get("data_config")
    loss_config = metadata.get("loss_config")
    task_config = checkpoint.get("task_config")
    if not all(
        isinstance(value, dict)
        for value in (model_config, data_config, loss_config, task_config)
    ):
        raise ValueError("Phase-C model/data/loss/task metadata is incomplete.")
    if model_config.get("name") != "phase_c_b6_distill":
        raise ValueError(
            "Evaluator requires model.name='phase_c_b6_distill'."
        )
    if task_config.get("name") != "phase_c_distill":
        raise ValueError("Evaluator requires task.name='phase_c_distill'.")

    root = Path(data_root or data_config["root"]).expanduser().resolve()
    manifest = Path(data_config["manifest"]).expanduser()
    if not manifest.is_absolute():
        manifest = (PROJECT_ROOT / manifest).resolve()
    dataset_config = DatasetConfig(
        name=str(data_config["name"]),
        root=root,
        manifest=manifest,
        version=str(data_config["version"]),
        task=str(data_config["task"]),
        num_classes=int(data_config["num_classes"]),
        in_channels=int(data_config["in_channels"]),
        image_size=tuple(int(value) for value in data_config["image_size"]),
        image_mean=tuple(float(value) for value in data_config["image_mean"]),
        image_std=tuple(float(value) for value in data_config["image_std"]),
        mask_threshold=float(data_config.get("mask_threshold", 0.5)),
    )
    return (
        dict(model_config),
        dataset_config,
        dict(loss_config),
        dict(task_config),
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative.")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError(f"Invalid Phase-C checkpoint: {checkpoint_path}")
    if checkpoint.get("model_class") != "PhaseCB6PriorDistill":
        raise ValueError(
            "Evaluator requires model_class='PhaseCB6PriorDistill'."
        )

    model_config, dataset_config, loss_config, task_config = (
        _checkpoint_configuration(checkpoint, data_root=args.data_root)
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    dataset = build_dataset(dataset_config, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = build_loss(loss_config)
    task = PhaseCDistillTask(
        criterion=criterion,
        threshold=float(task_config.get("threshold", 0.5)),
        boundary_tolerance=float(task_config.get("boundary_tolerance", 2.0)),
        task=dataset_config.task,
        lambda_latent=float(task_config["lambda_latent"]),
        lambda_route=float(task_config["lambda_route"]),
        lambda_deploy=float(task_config["lambda_deploy"]),
    )
    metrics = evaluate(
        model=model,
        loader=loader,
        task=task,
        device=device,
    )
    phase_c_metadata = getattr(model, "phase_c_checkpoint_metadata")()
    payload = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "evaluation_mode": task.evaluation_mode,
        "teacher_evaluation_mode": task.teacher_evaluation_mode,
        "metrics": metrics,
        "phase_c": phase_c_metadata,
    }
    output_path = args.output_dir.expanduser().resolve() / "metrics.json"
    _save_json(output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
