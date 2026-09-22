#!/usr/bin/env python3
"""Evaluate and benchmark the isolated Phase-B build-up ladder B0-B6."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.dataset import DatasetConfig
from src.data import build_dataset
from src.engine import evaluate
from src.losses import build_loss
from src.models import build_model
from src.models.phase_b.experts import ExpertBank
from src.models.phase_b.studies.build_up import DenseTokenFFN
from src.models.phase_b.studies.build_up.common import (
    IMAGE_ONLY,
    POSTERIOR_ORACLE,
)
from src.tasks import PhaseBBuildUpTask, SegmentationTask


BUILD_UP_MODEL_NAMES = {
    "phase_b_b1_multilevel",
    "phase_b_b2_dense",
    "phase_b_b3_image_moe",
    "phase_b_b4_shape_direct",
    "phase_b_b5_gaussian",
    "phase_b_b6_hierarchical",
}
SUPPORTED_MODEL_NAMES = {"esam", *BUILD_UP_MODEL_NAMES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-warmups", type=int, default=10)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_configuration(
    checkpoint: dict[str, Any],
    *,
    data_root: Path | None,
) -> tuple[dict[str, Any], DatasetConfig, dict[str, Any], dict[str, Any], int]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is required.")
    model_config = metadata.get("model_config")
    data_config = metadata.get("data_config")
    loss_config = metadata.get("loss_config")
    if not all(isinstance(value, dict) for value in (model_config, data_config, loss_config)):
        raise ValueError("Checkpoint model/data/loss metadata is incomplete.")

    model_name = str(model_config["name"])
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unsupported build-up evaluator model: {model_name!r}.")
    root = Path(data_root or data_config["root"]).expanduser().resolve()
    manifest = Path(data_config["manifest"]).expanduser()
    if not manifest.is_absolute():
        manifest = (PROJECT_ROOT / manifest).resolve()
    dataset_config = DatasetConfig(
        name=str(data_config["name"]),
        root=root,
        manifest=manifest,
        version=str(data_config.get("version", "legacy-unversioned")),
        task=str(data_config["task"]),
        num_classes=int(data_config["num_classes"]),
        in_channels=int(data_config["in_channels"]),
        image_size=tuple(int(value) for value in data_config["image_size"]),
        image_mean=tuple(float(value) for value in data_config["image_mean"]),
        image_std=tuple(float(value) for value in data_config["image_std"]),
        mask_threshold=float(data_config.get("mask_threshold", 0.5)),
    )
    task_config = checkpoint.get("task_config")
    if not isinstance(task_config, dict):
        task_config = {}
    return (
        dict(model_config),
        dataset_config,
        dict(loss_config),
        dict(task_config),
        int(metadata.get("seed", 42)),
    )


def evaluation_mode_for(
    model_name: str,
    task_config: dict[str, Any],
) -> str:
    if model_name == "esam":
        return IMAGE_ONLY
    mode = task_config.get("evaluation_mode")
    if mode not in {IMAGE_ONLY, POSTERIOR_ORACLE}:
        raise ValueError("Build-up checkpoint lacks a valid evaluation_mode.")
    return str(mode)


def build_evaluation_task(
    *,
    model_name: str,
    dataset_config: DatasetConfig,
    criterion: torch.nn.Module,
    task_config: dict[str, Any],
):
    common = {
        "criterion": criterion,
        "threshold": float(task_config.get("threshold", 0.5)),
        "boundary_tolerance": float(task_config.get("boundary_tolerance", 2.0)),
        "task": dataset_config.task,
    }
    if model_name == "esam":
        return SegmentationTask(**common)
    return PhaseBBuildUpTask(
        **common,
        evaluation_mode=evaluation_mode_for(model_name, task_config),
        lambda_balance=float(task_config.get("lambda_balance", 0.0)),
    )


def efficiency_metadata(model: torch.nn.Module) -> dict[str, int]:
    """Exact parameter and sparse transformation counts."""

    expert_banks = [module for module in model.modules() if isinstance(module, ExpertBank)]
    dense_modules = [module for module in model.modules() if isinstance(module, DenseTokenFFN)]
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "expert_bank_parameters": sum(
            parameter.numel()
            for bank in expert_banks
            for parameter in bank.parameters()
        ),
        "dense_ffn_parameters": sum(
            parameter.numel()
            for dense in dense_modules
            for parameter in dense.parameters()
        ),
        "active_experts_per_sample": int(getattr(model, "active_experts", 0)),
        "expert_transformations_per_sample": int(
            getattr(model, "expert_transformations_per_sample", 0)
        ),
    }


def _forward(
    model: torch.nn.Module,
    images: Tensor,
    targets: Tensor,
    evaluation_mode: str,
):
    if evaluation_mode == POSTERIOR_ORACLE:
        return model(images, masks=targets)
    return model(images)


def benchmark_efficiency(
    *,
    model: torch.nn.Module,
    images: Tensor,
    targets: Tensor,
    evaluation_mode: str,
    warmups: int,
    iterations: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Measure forward-only latency, throughput and peak allocated memory."""

    if warmups < 0 or iterations <= 0:
        raise ValueError("Benchmark warmups must be non-negative and iterations positive.")
    device = images.device
    model.eval()
    set_seed(seed)
    with torch.inference_mode():
        for _ in range(warmups):
            _forward(model, images, targets, evaluation_mode)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iterations):
                _forward(model, images, targets, evaluation_mode)
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
            peak_memory: int | None = int(torch.cuda.max_memory_allocated(device))
        else:
            start = time.perf_counter()
            for _ in range(iterations):
                _forward(model, images, targets, evaluation_mode)
            elapsed_seconds = time.perf_counter() - start
            peak_memory = None

    batch_size = int(images.shape[0])
    return {
        "warmup_iterations": warmups,
        "measured_iterations": iterations,
        "batch_size": batch_size,
        "latency_ms_per_image": (
            elapsed_seconds * 1000.0 / (iterations * batch_size)
        ),
        "throughput_images_per_second": (
            iterations * batch_size / elapsed_seconds
        ),
        "peak_cuda_memory_bytes": peak_memory,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

    model_config, dataset_config, loss_config, task_config, seed = (
        checkpoint_configuration(checkpoint, data_root=args.data_root)
    )
    model_name = str(model_config["name"])
    evaluation_mode = evaluation_mode_for(model_name, task_config)
    set_seed(seed)

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
    model.to(device)
    criterion = build_loss(loss_config).to(device)
    task = build_evaluation_task(
        model_name=model_name,
        dataset_config=dataset_config,
        criterion=criterion,
        task_config=task_config,
    )
    metrics = evaluate(model=model, loader=loader, task=task, device=device)

    first_batch = next(iter(loader))
    images = first_batch["image"].to(device, dtype=torch.float32)
    targets = first_batch["mask"].to(device, dtype=torch.float32)
    benchmark = None
    if not args.skip_benchmark:
        benchmark = benchmark_efficiency(
            model=model,
            images=images,
            targets=targets,
            evaluation_mode=evaluation_mode,
            warmups=args.benchmark_warmups,
            iterations=args.benchmark_iterations,
            seed=seed,
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "evaluation_mode": evaluation_mode,
        "metrics": metrics,
        "efficiency": efficiency_metadata(model),
        "benchmark": benchmark,
    }
    output_dir = args.output_dir.expanduser().resolve()
    save_json(output_dir / "metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
