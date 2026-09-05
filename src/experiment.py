"""Build and execute reproducible segmentation experiments."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.configs import DatasetConfig
from src.data import build_dataset
from src.engine import Trainer, TrainerConfig, evaluate
from src.losses import build_loss
from src.models import build_model
from src.tasks import SegmentationTask


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(path)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    temporary_path.replace(path)


def config_fingerprint(config: dict[str, Any]) -> str:
    normalized = deepcopy(config)
    normalized.pop("seed", None)
    experiment = normalized.get("experiment")
    if isinstance(experiment, dict):
        experiment.pop("output_root", None)
    serialized = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    parent = Path(experiment["output_root"]) / experiment["name"]
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{timestamp}_seed-{config['seed']}"
    candidate = parent / base_name
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{base_name}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate.resolve()


def git_metadata() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return commit, bool(dirty_result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def device_metadata(requested: str) -> dict[str, str | None]:
    device = torch.device(requested)
    device_name: str | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(index)
    elif device.type == "cpu":
        device_name = "CPU"
    return {
        "requested": requested,
        "resolved": str(device),
        "name": device_name,
    }


def build_dataset_config(config: dict[str, Any]) -> DatasetConfig:
    dataset = config["dataset"]
    return DatasetConfig(
        name=str(dataset["name"]),
        root=Path(dataset["root"]),
        manifest=Path(dataset["manifest"]),
        version=str(dataset["version"]),
        task=str(dataset["task"]),
        num_classes=int(dataset["num_classes"]),
        in_channels=int(dataset["in_channels"]),
        image_size=tuple(int(value) for value in dataset["image_size"]),
        image_mean=tuple(float(value) for value in dataset["image_mean"]),
        image_std=tuple(float(value) for value in dataset["image_std"]),
        mask_threshold=float(dataset["mask_threshold"]),
    )


def build_optimizer(
    config: dict[str, Any],
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    if optimizer_config["name"] != "adamw":
        raise ValueError("Only optimizer.name='adamw' is supported.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def build_scheduler(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config["name"]
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(scheduler_config["eta_min"]),
        )
    if scheduler_name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=float(scheduler_config["min_lr"]),
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def build_loaders(
    config: dict[str, Any],
    dataset_config: DatasetConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    training = config["training"]
    seed = int(config["seed"])
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": str(training["device"]).startswith("cuda"),
        "persistent_workers": int(training["num_workers"]) > 0,
        "drop_last": False,
    }
    train_dataset = build_dataset(dataset_config, split="train")
    val_dataset = build_dataset(dataset_config, split="val")
    test_dataset = build_dataset(dataset_config, split="test")
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def build_checkpoint_metadata(
    config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_name": config["experiment"]["name"],
        "seed": config["seed"],
        "model_config": model_config,
        "data_config": deepcopy(config["dataset"]),
        "loss_config": deepcopy(config["loss"]),
        "optimizer_config": deepcopy(config["optimizer"]),
        "scheduler_config": deepcopy(config["scheduler"]),
    }


def execute_experiment(
    config: dict[str, Any],
    *,
    run_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    """Execute one fresh or resumed experiment and return its run directory."""

    if resume and run_dir is None:
        raise ValueError("Resume requires an existing run directory.")
    if not resume and run_dir is not None:
        raise ValueError("Fresh experiments create their own run directory.")

    seed = int(config["seed"])
    set_seed(seed)
    resolved_run_dir = (
        Path(run_dir).expanduser().resolve()
        if run_dir is not None
        else create_run_directory(config)
    )
    if resume:
        if not resolved_run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {resolved_run_dir}")
    else:
        save_yaml(resolved_run_dir / "config.yaml", config)

    manifest_path = Path(config["dataset"]["manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Tracked dataset manifest not found: {manifest_path}")

    metadata_path = resolved_run_dir / "metadata.json"
    git_commit, git_dirty = git_metadata()
    if resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Resume metadata not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
        metadata.setdefault("resumed_at", []).append(utc_now())
    else:
        metadata = {
            "experiment_name": config["experiment"]["name"],
            "run_id": resolved_run_dir.name,
            "seed": seed,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "dataset": {
                "name": config["dataset"]["name"],
                "version": config["dataset"]["version"],
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
            },
            "device": device_metadata(config["training"]["device"]),
            "started_at": utc_now(),
            "resume_count": 0,
            "config_fingerprint": config_fingerprint(config),
        }
    metadata["status"] = "running"
    metadata.pop("error", None)
    metadata.pop("ended_at", None)
    save_json(metadata_path, metadata)

    try:
        dataset_config = build_dataset_config(config)
        train_loader, val_loader, test_loader = build_loaders(
            config,
            dataset_config,
        )

        shared_model_config = {
            "in_channels": dataset_config.in_channels,
            "num_classes": dataset_config.num_classes,
            "task": dataset_config.task,
        }
        model_config = {**config["model"], **shared_model_config}
        model = build_model(model_config)
        criterion = build_loss(config["loss"])
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer)
        training = config["training"]
        threshold = float(training["prediction_threshold"])
        boundary_tolerance = float(training["boundary_tolerance"])
        task = SegmentationTask(
            criterion=criterion,
            threshold=threshold,
            boundary_tolerance=boundary_tolerance,
        )
        task_config = {
            "name": "segmentation",
            "threshold": threshold,
            "boundary_tolerance": boundary_tolerance,
        }

        trainer = Trainer(
            model=model,
            task=task,
            task_config=task_config,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TrainerConfig(
                epochs=int(training["epochs"]),
                device=str(training["device"]),
                last_checkpoint_path=resolved_run_dir / "last.pt",
                best_checkpoint_path=resolved_run_dir / "best.pt",
                history_path=resolved_run_dir / "history.json",
                use_amp=bool(training["amp"]),
                amp_dtype=str(training["amp_dtype"]),
                log_interval=int(training["log_interval"]),
                gradient_clip_norm=training["gradient_clip_norm"],
                monitor=str(training["monitor"]),
                monitor_mode=str(training["monitor_mode"]),
            ),
            checkpoint_metadata=build_checkpoint_metadata(config, model_config),
        )
        if resume:
            trainer.resume(resolved_run_dir / "last.pt")

        print(f"Run directory : {resolved_run_dir}")
        print(f"Experiment    : {config['experiment']['name']}")
        print(f"Seed          : {seed}")
        print(f"Device        : {training['device']}")
        trainer.train()

        best_checkpoint_path = resolved_run_dir / "best.pt"
        if not best_checkpoint_path.is_file():
            raise FileNotFoundError(
                "Training produced no best checkpoint; validation is required."
            )
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            task=task,
            device=training["device"],
        )
        test_payload = {
            "checkpoint": "best.pt",
            "split": "test",
            "loss": test_metrics["loss"],
            "dice": test_metrics["dice"],
            "iou": test_metrics["iou"],
            "hd95": test_metrics["hd95"],
            "assd": test_metrics["assd"],
            "boundary_f1": test_metrics["boundary_f1"],
        }
        save_json(resolved_run_dir / "test_metrics.json", test_payload)

        metadata["status"] = "completed"
        metadata["ended_at"] = utc_now()
        metadata["best_epoch"] = int(best_checkpoint["epoch"])
        metadata["monitor_name"] = best_checkpoint["monitor_name"]
        metadata["monitor_mode"] = best_checkpoint["monitor_mode"]
        metadata["best_monitor_value"] = best_checkpoint["best_monitor_value"]
        if (
            best_checkpoint["monitor_name"] == "dice"
            and best_checkpoint["monitor_mode"] == "max"
        ):
            metadata["best_val_dice"] = best_checkpoint["best_monitor_value"]
        else:
            metadata.pop("best_val_dice", None)
        save_json(metadata_path, metadata)
        print(f"Test Dice        : {test_metrics['dice']:.6f}")
        print(f"Test IoU         : {test_metrics['iou']:.6f}")
        print(f"Test HD95        : {test_metrics['hd95']:.6f}")
        print(f"Test ASSD        : {test_metrics['assd']:.6f}")
        print(f"Test Boundary F1 : {test_metrics['boundary_f1']:.6f}")
        return resolved_run_dir
    except Exception as error:
        metadata["status"] = "failed"
        metadata["ended_at"] = utc_now()
        metadata["error"] = f"{type(error).__name__}: {error}"
        save_json(metadata_path, metadata)
        raise
