#!/usr/bin/env python3
"""Evaluate and benchmark the controlled Phase B A0-A4 ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.dataset import DatasetConfig
from src.data import build_dataset
from src.losses import build_loss
from src.metrics import compute_binary_dice_iou, compute_binary_surface_metrics
from src.models import SegmentationOutput, build_model
from src.tasks.phase_b_diagnostics import (
    actual_layer_weights,
    enhancement_metrics,
    layer_fusion_metrics,
    routing_metrics,
)


TRANSFER_MODEL_NAMES = {
    "phase_b_moe",
    "phase_b_a4_shape_conditioned",
}
DIRECT_FUSE_MODEL_NAME = "phase_b_a2_direct_fuse"
IMAGE_ONLY_MODEL_NAMES = {
    "phase_b_a1_no_expert",
    "phase_b_no_moe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Phase B controlled ablation checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--paths",
        choices=("auto", "posterior", "prior", "both"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--boundary-tolerance", type=float, default=None)
    parser.add_argument("--benchmark-warmups", type=int, default=10)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def resolve_evaluation_paths(
    model_name: str,
    requested: str,
) -> tuple[str, ...]:
    """Resolve CLI path selection against each ablation's capabilities."""

    if requested not in {"auto", "posterior", "prior", "both"}:
        raise ValueError(f"Unknown path selection: {requested!r}.")

    if model_name in IMAGE_ONLY_MODEL_NAMES:
        if requested != "auto":
            raise ValueError(
                f"{model_name} is image-only; use --paths auto."
            )
        return ("image",)
    if model_name == DIRECT_FUSE_MODEL_NAME:
        if requested in {"prior", "both"}:
            raise ValueError(
                "A2 direct-fuse requires the ground-truth mask and has no "
                "image-only prior path."
            )
        return ("posterior",)
    if model_name not in TRANSFER_MODEL_NAMES:
        raise ValueError(
            "Unsupported Phase B ablation model for this evaluator: "
            f"{model_name!r}."
        )
    if requested == "auto" or requested == "both":
        return ("posterior", "prior")
    return (requested,)


def _checkpoint_configuration(
    checkpoint: dict[str, Any],
    *,
    data_root: Path | None,
    threshold: float | None,
    boundary_tolerance: float | None,
) -> tuple[dict[str, Any], DatasetConfig, dict[str, Any], float, float]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is required for ablation evaluation.")

    model_config = metadata.get("model_config")
    data_config = metadata.get("data_config")
    loss_config = metadata.get("loss_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint metadata.model_config is missing.")
    if not isinstance(data_config, dict):
        raise ValueError("Checkpoint metadata.data_config is missing.")
    if not isinstance(loss_config, dict):
        raise ValueError("Checkpoint metadata.loss_config is missing.")

    root_value = data_root if data_root is not None else data_config.get("root")
    if root_value is None:
        raise ValueError("Dataset root is absent; pass --data-root.")
    resolved_root = Path(root_value).expanduser().resolve()
    manifest_value = data_config.get("manifest", resolved_root / "dataset.json")
    manifest = Path(manifest_value).expanduser()
    if not manifest.is_absolute():
        manifest = (PROJECT_ROOT / manifest).resolve()

    dataset_config = DatasetConfig(
        name=str(data_config["name"]),
        root=resolved_root,
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
    resolved_threshold = float(
        threshold if threshold is not None else task_config.get("threshold", 0.5)
    )
    resolved_tolerance = float(
        boundary_tolerance
        if boundary_tolerance is not None
        else task_config.get("boundary_tolerance", 2.0)
    )
    if not 0.0 <= resolved_threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if not math.isfinite(resolved_tolerance) or resolved_tolerance < 0:
        raise ValueError("boundary_tolerance must be non-negative.")

    return (
        dict(model_config),
        dataset_config,
        dict(loss_config),
        resolved_threshold,
        resolved_tolerance,
    )


def _pixel_perimeter(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    vertical = np.logical_xor(padded[1:, :], padded[:-1, :]).sum()
    horizontal = np.logical_xor(padded[:, 1:], padded[:, :-1]).sum()
    return float(vertical + horizontal)


def mask_geometry(mask: np.ndarray | Tensor) -> dict[str, float]:
    """Compute finite ground-truth geometry from foreground pixel cells."""

    if torch.is_tensor(mask):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"mask must resolve to 2D, got {array.shape}.")
    foreground = array >= 0.5
    height, width = foreground.shape
    area = float(foreground.sum())
    if area == 0.0:
        return {
            "lesion_area": 0.0,
            "lesion_area_fraction": 0.0,
            "lesion_perimeter": 0.0,
            "circularity": 0.0,
            "boundary_complexity": 0.0,
            "solidity": 0.0,
            "eccentricity": 0.0,
        }

    perimeter = _pixel_perimeter(foreground)
    circularity = (
        4.0 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0.0
        else 0.0
    )
    boundary_complexity = (
        perimeter * perimeter / (4.0 * math.pi * area)
        if area > 0.0
        else 0.0
    )

    rows, columns = np.nonzero(foreground)
    corners = np.concatenate(
        [
            np.column_stack((columns, rows)),
            np.column_stack((columns + 1, rows)),
            np.column_stack((columns, rows + 1)),
            np.column_stack((columns + 1, rows + 1)),
        ],
        axis=0,
    ).astype(np.float64)
    corners = np.unique(corners, axis=0)
    try:
        hull_area = float(ConvexHull(corners).volume)
    except (QhullError, ValueError):
        hull_area = 0.0
    solidity = min(area / hull_area, 1.0) if hull_area > 0.0 else 0.0

    coordinates = np.column_stack((rows, columns)).astype(np.float64)
    eccentricity = 0.0
    if coordinates.shape[0] >= 2:
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
        major = float(np.max(eigenvalues))
        minor = max(float(np.min(eigenvalues)), 0.0)
        if major > 0.0:
            eccentricity = math.sqrt(max(0.0, 1.0 - minor / major))

    values = {
        "lesion_area": area,
        "lesion_area_fraction": area / float(height * width),
        "lesion_perimeter": perimeter,
        "circularity": circularity,
        "boundary_complexity": boundary_complexity,
        "solidity": solidity,
        "eccentricity": eccentricity,
    }
    return {
        key: float(value) if math.isfinite(float(value)) else 0.0
        for key, value in values.items()
    }


def routing_path_comparison(
    posterior_diagnostics: dict[str, Any],
    prior_diagnostics: dict[str, Any],
) -> dict[str, Tensor]:
    """Return per-sample q-to-p KL and Top-K agreement."""

    posterior_stage = posterior_diagnostics.get("phase_b_router")
    prior_stage = prior_diagnostics.get("phase_b_router")
    if posterior_stage is None or prior_stage is None:
        raise ValueError("Both diagnostics must contain phase_b_router.")

    q = posterior_stage.routing.dense_probs.detach()
    p = prior_stage.routing.dense_probs.detach()
    epsilon = torch.finfo(q.dtype).tiny
    kl = (
        q.clamp_min(epsilon)
        * (q.clamp_min(epsilon).log() - p.clamp_min(epsilon).log())
    ).sum(dim=1)

    q_indices = posterior_stage.routing.expert_indices.detach()
    p_indices = prior_stage.routing.expert_indices.detach()
    exact = torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
    jaccard = torch.zeros_like(exact)
    for sample_index in range(q.shape[0]):
        q_set = set(int(value) for value in q_indices[sample_index].tolist())
        p_set = set(int(value) for value in p_indices[sample_index].tolist())
        exact[sample_index] = float(q_set == p_set)
        union = q_set | p_set
        jaccard[sample_index] = (
            float(len(q_set & p_set) / len(union)) if union else 1.0
        )
    return {
        "categorical_routing_kl": kl,
        "topk_exact_match": exact,
        "topk_jaccard": jaccard,
    }


def _forward_path(
    model: torch.nn.Module,
    images: Tensor,
    targets: Tensor,
    path: str,
) -> SegmentationOutput:
    if path == "image" or path == "prior":
        output = model(images)
    elif path == "posterior":
        output = model(images, masks=targets)
    else:
        raise ValueError(f"Unknown forward path: {path!r}.")
    if not isinstance(output, SegmentationOutput):
        raise TypeError(
            "Phase B model must return SegmentationOutput, got "
            f"{type(output).__name__}."
        )
    return output


def _sample_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float,
    boundary_tolerance: float,
) -> dict[str, Tensor]:
    dice, iou = compute_binary_dice_iou(
        logits,
        targets,
        threshold=threshold,
    )
    hd, hd95, assd, boundary_f1 = compute_binary_surface_metrics(
        logits,
        targets,
        threshold=threshold,
        boundary_tolerance=boundary_tolerance,
    )
    return {
        "dice": dice,
        "iou": iou,
        "hd": hd,
        "hd95": hd95,
        "assd": assd,
        "boundary_f1": boundary_f1,
    }


def _diagnostic_sample_values(
    diagnostics: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = [dict() for _ in range(batch_size)]
    weights, level_ids = actual_layer_weights(diagnostics)
    if weights is not None:
        values = weights.detach().cpu()
        for sample_index in range(batch_size):
            rows[sample_index]["layer_weights"] = json.dumps(
                [float(value) for value in values[sample_index]]
            )
            if level_ids is not None:
                for index, level in enumerate(level_ids):
                    rows[sample_index][f"layer_weight_{level}"] = float(
                        values[sample_index, index]
                    )

    router_stage = diagnostics.get("phase_b_router")
    if router_stage is not None:
        routing = router_stage.routing
        indices = routing.expert_indices.detach().cpu()
        dense = routing.dense_probs.detach().cpu()
        sparse = routing.routing_probs.detach().cpu()
        for sample_index in range(batch_size):
            rows[sample_index]["expert_indices"] = json.dumps(
                [int(value) for value in indices[sample_index]]
            )
            rows[sample_index]["routing_dense_probs"] = json.dumps(
                [float(value) for value in dense[sample_index]]
            )
            rows[sample_index]["routing_probs"] = json.dumps(
                [float(value) for value in sparse[sample_index]]
            )

    stage = diagnostics.get("phase_b_moe")
    if stage is None:
        stage = diagnostics.get("phase_b_ablation")
    for name, attribute in (
        ("enhancement_aux_ratio", "aux_norm_ratio"),
        ("fused_token_norm", "fused_token_norm"),
        ("enhanced_token_norm", "enhanced_token_norm"),
    ):
        value = getattr(stage, attribute, None)
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(batch_size, -1).mean(dim=1)
            for sample_index in range(batch_size):
                rows[sample_index][name] = float(value[sample_index])
    return rows


def _batch_diagnostic_metrics(
    diagnostics: dict[str, Any],
) -> dict[str, Tensor]:
    metrics = layer_fusion_metrics(diagnostics)
    router_stage = diagnostics.get("phase_b_router")
    if router_stage is not None:
        metrics.update(routing_metrics(router_stage))
    enhancement_stage = diagnostics.get("phase_b_moe")
    if enhancement_stage is None:
        enhancement_stage = diagnostics.get("phase_b_ablation")
    metrics.update(enhancement_metrics(enhancement_stage))
    return metrics


def evaluate_paths(
    *,
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    paths: tuple[str, ...],
    criterion: torch.nn.Module,
    threshold: float,
    boundary_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Tensor]]:
    """Evaluate requested paths and return aggregate, CSV rows and benchmark batch."""

    path_values: dict[str, dict[str, list[float]]] = {
        path: defaultdict(list) for path in paths
    }
    diagnostic_sums: dict[str, dict[str, float]] = {
        path: defaultdict(float) for path in paths
    }
    diagnostic_counts = {path: 0 for path in paths}
    comparison_values: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    benchmark_batch: dict[str, Tensor] | None = None

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            targets = batch["mask"].to(device, dtype=torch.float32)
            batch_size = images.shape[0]
            if benchmark_batch is None:
                benchmark_batch = {
                    "images": images.clone(),
                    "targets": targets.clone(),
                }

            batch_rows = []
            for sample_index in range(batch_size):
                row = {
                    "sample_id": str(batch["sample_id"][sample_index]),
                }
                row.update(mask_geometry(targets[sample_index]))
                batch_rows.append(row)

            outputs: dict[str, SegmentationOutput] = {}
            for path in paths:
                output = _forward_path(model, images, targets, path)
                outputs[path] = output
                sample_metrics = _sample_metrics(
                    output.logits,
                    targets,
                    threshold=threshold,
                    boundary_tolerance=boundary_tolerance,
                )
                loss = float(criterion(output.logits, targets).detach().cpu())
                path_values[path]["loss"].extend([loss] * batch_size)
                for metric_name, tensor in sample_metrics.items():
                    values = tensor.detach().cpu().tolist()
                    path_values[path][metric_name].extend(
                        float(value) for value in values
                    )
                    for sample_index, value in enumerate(values):
                        batch_rows[sample_index][f"{path}_{metric_name}"] = float(
                            value
                        )

                sample_diagnostics = _diagnostic_sample_values(
                    output.diagnostics,
                    batch_size,
                )
                for sample_index, values in enumerate(sample_diagnostics):
                    for name, value in values.items():
                        batch_rows[sample_index][f"{path}_{name}"] = value

                batch_diagnostics = _batch_diagnostic_metrics(output.diagnostics)
                for name, value in batch_diagnostics.items():
                    diagnostic_sums[path][name] += (
                        float(value.detach().cpu()) * batch_size
                    )
                diagnostic_counts[path] += batch_size

            if "posterior" in outputs and "prior" in outputs:
                comparison = routing_path_comparison(
                    outputs["posterior"].diagnostics,
                    outputs["prior"].diagnostics,
                )
                for name, tensor in comparison.items():
                    values = tensor.detach().cpu().tolist()
                    comparison_values[name].extend(float(value) for value in values)
                    for sample_index, value in enumerate(values):
                        batch_rows[sample_index][name] = float(value)

            rows.extend(batch_rows)

    if benchmark_batch is None:
        raise ValueError("Evaluation loader is empty.")

    aggregates: dict[str, Any] = {}
    for path in paths:
        path_aggregate = {
            name: float(np.mean(values))
            for name, values in path_values[path].items()
        }
        count = diagnostic_counts[path]
        if count:
            path_aggregate.update(
                {
                    name: total / count
                    for name, total in diagnostic_sums[path].items()
                }
            )
        aggregates[path] = path_aggregate

    if comparison_values:
        comparison = {
            name: float(np.mean(values))
            for name, values in comparison_values.items()
        }
        for metric_name in ("dice", "hd95", "boundary_f1"):
            gap = (
                aggregates["posterior"][metric_name]
                - aggregates["prior"][metric_name]
            )
            comparison[f"transfer_gap_{metric_name}"] = gap
        comparison["posterior_prior_dice_gap"] = comparison[
            "transfer_gap_dice"
        ]
        aggregates["posterior_prior"] = comparison

    return aggregates, rows, benchmark_batch


def model_efficiency_metadata(
    model: torch.nn.Module,
    model_name: str,
) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    router = getattr(model, "router", None)
    active_experts = int(getattr(router, "active_experts", 0))
    levels = tuple(getattr(getattr(model, "image_descriptor", None), "levels", ()))
    if model_name in IMAGE_ONLY_MODEL_NAMES:
        calls = 0
    elif model_name == "phase_b_a4_shape_conditioned":
        calls = active_experts
    elif getattr(model, "enhancement_mode", None) == "shape_conditioned":
        calls = active_experts
    else:
        calls = active_experts * len(levels)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "active_experts": active_experts,
        "expert_calls_per_sample": calls,
    }


def benchmark_forward_paths(
    *,
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    paths: tuple[str, ...],
    warmups: int,
    iterations: int,
) -> dict[str, dict[str, float | int | None]]:
    """Benchmark model forward only; tensors are already resident on device."""

    if warmups < 0 or iterations <= 0:
        raise ValueError("warmups must be non-negative and iterations positive.")
    images = batch["images"]
    targets = batch["targets"]
    device = images.device
    model.eval()
    results: dict[str, dict[str, float | int | None]] = {}

    with torch.inference_mode():
        for path in paths:
            for _ in range(warmups):
                _forward_path(model, images, targets, path)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(iterations):
                    _forward_path(model, images, targets, path)
                end_event.record()
                torch.cuda.synchronize(device)
                elapsed_seconds = (
                    float(start_event.elapsed_time(end_event)) / 1000.0
                )
                peak_memory = int(torch.cuda.max_memory_allocated(device))
            else:
                start = time.monotonic()
                for _ in range(iterations):
                    _forward_path(model, images, targets, path)
                elapsed_seconds = time.monotonic() - start
                peak_memory = None

            latency_ms = elapsed_seconds * 1000.0 / iterations
            throughput = images.shape[0] * iterations / elapsed_seconds
            results[path] = {
                "warmup_iterations": warmups,
                "measured_iterations": iterations,
                "batch_size": int(images.shape[0]),
                "latency_ms": latency_ms,
                "throughput_samples_per_second": throughput,
                "peak_allocated_cuda_bytes": peak_memory,
            }
    return results


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty per-sample CSV.")
    preferred = [
        "sample_id",
        "lesion_area",
        "lesion_area_fraction",
        "lesion_perimeter",
        "circularity",
        "boundary_complexity",
        "solidity",
        "eccentricity",
    ]
    remaining = sorted(set().union(*(row.keys() for row in rows)) - set(preferred))
    fieldnames = preferred + remaining
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

    (
        model_config,
        dataset_config,
        loss_config,
        threshold,
        boundary_tolerance,
    ) = _checkpoint_configuration(
        checkpoint,
        data_root=args.data_root,
        threshold=args.threshold,
        boundary_tolerance=args.boundary_tolerance,
    )
    paths = resolve_evaluation_paths(str(model_config["name"]), args.paths)
    dataset = build_dataset(dataset_config, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    criterion = build_loss(loss_config).to(device)

    aggregates, rows, benchmark_batch = evaluate_paths(
        model=model,
        loader=loader,
        device=device,
        paths=paths,
        criterion=criterion,
        threshold=threshold,
        boundary_tolerance=boundary_tolerance,
    )
    efficiency = model_efficiency_metadata(model, str(model_config["name"]))
    benchmark = None
    if not args.skip_benchmark:
        benchmark = benchmark_forward_paths(
            model=model,
            batch=benchmark_batch,
            paths=paths,
            warmups=args.benchmark_warmups,
            iterations=args.benchmark_iterations,
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "paths": list(paths),
        "threshold": threshold,
        "boundary_tolerance": boundary_tolerance,
        "metrics": aggregates,
        "efficiency": efficiency,
        "benchmark": benchmark,
    }
    output_dir = args.output_dir.expanduser().resolve()
    save_json(output_dir / "metrics.json", payload)
    write_per_sample_csv(output_dir / "per_sample.csv", rows)

    print(json.dumps(payload, indent=2))
    print(f"Per-sample CSV: {output_dir / 'per_sample.csv'}")


if __name__ == "__main__":
    main()
