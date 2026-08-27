"""Training and evaluation lifecycle for mask-only Shape Teachers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.configs.experiment import load_experiment_config
from src.engine import (
    MetricAccumulator,
    compute_binary_dice_iou,
    compute_binary_surface_metrics,
)
from src.experiment import (
    config_fingerprint,
    create_run_directory,
    file_sha256,
    save_json,
    save_yaml,
    set_seed,
    utc_now,
)

from .corruptions import MaskCorruptor
from .data import MaskOnlyDataset, audit_mask_splits, build_mask_datasets
from .losses import ShapeTeacherLoss, ShapeTeacherLosses
from .model import ShapeTeacher
from .qualitative import (
    collect_qualitative_records,
    select_representatives,
)
from .visualization import save_corruption_grid, save_reconstruction_grid


class ScaleAccumulator:
    """Collect exact epoch-level posterior scale statistics."""

    def __init__(self) -> None:
        self.values: list[Tensor] = []

    def update(self, sigma: Tensor) -> None:
        self.values.append(sigma.detach().float().cpu().flatten())

    def summary(self) -> dict[str, float]:
        if not self.values:
            raise ValueError("No posterior scales were accumulated.")
        values = torch.cat(self.values)
        return {
            "sigma_mean": float(values.mean()),
            "sigma_median": float(values.median()),
            "sigma_below_1e-3_pct": float(values.lt(1.0e-3).float().mean() * 100.0),
        }


def validate_shape_teacher_config(config: dict[str, Any]) -> None:
    if config["model"].get("name") != "shape_teacher":
        raise ValueError("model.name must be 'shape_teacher'.")
    if config["loss"].get("name") != "shape_teacher":
        raise ValueError("loss.name must be 'shape_teacher'.")
    mode = config["training"].get("input_mode")
    if mode not in {"clean", "denoise"}:
        raise ValueError("training.input_mode must be clean or denoise.")
    corruption = config.get("corruption")
    if not isinstance(corruption, dict):
        raise ValueError("corruption must be a mapping.")
    enabled = corruption.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("corruption.enabled must be a boolean.")
    if enabled != (mode == "denoise"):
        raise ValueError(
            "corruption.enabled must be false for clean and true for denoise."
        )


def load_shape_teacher_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    config = load_experiment_config(path, project_root=project_root)
    validate_shape_teacher_config(config)
    return config


def build_shape_teacher(config: dict[str, Any]) -> ShapeTeacher:
    values = dict(config["model"])
    values.pop("name")
    values.setdefault("image_size", config["dataset"]["image_size"])
    return ShapeTeacher(**values)


def build_shape_teacher_loss(config: dict[str, Any]) -> ShapeTeacherLoss:
    values = dict(config["loss"])
    values.pop("name")
    return ShapeTeacherLoss(**values)


def build_mask_loaders(
    config: dict[str, Any],
) -> tuple[dict[str, MaskOnlyDataset], dict[str, DataLoader]]:
    datasets = build_mask_datasets(config)
    training = config["training"]
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": str(training["device"]).startswith("cuda"),
        "persistent_workers": int(training["num_workers"]) > 0,
        "drop_last": False,
    }
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loaders = {
        "train": DataLoader(
            datasets["train"], shuffle=True, generator=generator, **common
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return datasets, loaders


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be in [0, epochs).")

    def multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def run_epoch(
    *,
    model: ShapeTeacher,
    loader: DataLoader,
    criterion: ShapeTeacherLoss,
    device: torch.device,
    input_mode: str,
    corruptor: MaskCorruptor | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    gradient_clip_norm: float | None = None,
    threshold: float = 0.5,
    boundary_tolerance: float = 2.0,
    include_surface_metrics: bool = False,
    epoch: int = 1,
    epochs: int = 1,
    log_interval: int = 20,
) -> dict[str, float]:
    if input_mode not in {"clean", "corrupted"}:
        raise ValueError("input_mode must be clean or corrupted.")
    if input_mode == "corrupted" and corruptor is None:
        raise ValueError("corrupted input_mode requires a MaskCorruptor.")
    training = optimizer is not None
    model.train(training)
    metrics = MetricAccumulator()
    scales = ScaleAccumulator()
    if corruptor is not None:
        corruptor.reset_summary()

    for step, batch in enumerate(loader, start=1):
        clean_cpu = batch["mask"].float()
        paths = [str(path) for path in batch["mask_path"]]
        if input_mode == "clean":
            input_cpu = clean_cpu
        else:
            input_cpu = corruptor(clean_cpu, keys=paths, split=str(batch["split"][0]))
        clean = clean_cpu.to(device, non_blocking=True)
        teacher_input = input_cpu.to(device, non_blocking=True)
        batch_size = clean.shape[0]

        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(teacher_input, sample=training)
                losses: ShapeTeacherLosses = criterion(output.logits, clean)
            if not torch.isfinite(losses.total):
                raise FloatingPointError(
                    f"Non-finite Shape Teacher loss at epoch {epoch}, step {step}."
                )
            if training:
                if scaler is None:
                    raise ValueError("Training requires a GradScaler.")
                scaler.scale(losses.total).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(gradient_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()

        scales.update(output.sigma)
        logits = output.logits.detach().float()
        dice, iou = compute_binary_dice_iou(logits, clean, threshold=threshold)
        values = losses.as_dict()
        values.update(hard_dice=float(dice.mean()), iou=float(iou.mean()))
        if include_surface_metrics:
            hd95, assd, boundary_f1 = compute_binary_surface_metrics(
                logits,
                clean,
                threshold=threshold,
                boundary_tolerance=boundary_tolerance,
            )
            values.update(
                hd95=float(hd95.mean()),
                assd=float(assd.mean()),
                boundary_f1=float(boundary_f1.mean()),
            )
        metrics.update(values, batch_size)

        if training and (
            step == 1 or step % log_interval == 0 or step == len(loader)
        ):
            running = metrics.summary()
            print(
                f"Epoch {epoch}/{epochs} - batch {step}/{len(loader)} "
                f"- loss: {running['loss']:.4f} "
                f"- soft dice: {running['soft_dice']:.4f} "
                f"- hard dice: {running['hard_dice']:.4f}"
            )

    summary = {**metrics.summary(), **scales.summary()}
    if corruptor is not None:
        summary.update(corruptor.summary.as_dict())
    return summary


def _optimizer(config: dict[str, Any], model: ShapeTeacher) -> torch.optim.AdamW:
    values = config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(values["lr"]),
        weight_decay=float(values["weight_decay"]),
    )


def _save_checkpoint(
    path: Path,
    *,
    model: ShapeTeacher,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_dice: float,
    best_loss: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": model.model_config(),
            "epoch": epoch,
            "best_validation_clean_dice": best_dice,
            "best_validation_clean_loss": best_loss,
            "experiment_config": deepcopy(config),
        },
        path,
    )


def _environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _data_source_metadata(
    source: Any,
    datasets: dict[str, MaskOnlyDataset],
) -> dict[str, Any]:
    """Record reproducible identities for one manifest or three split sources."""

    if not isinstance(source, dict):
        path = Path(source)
        return {
            "manifest": str(path),
            "manifest_sha256": file_sha256(path),
        }

    sources: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        specification = source[split]
        if isinstance(specification, dict):
            kind, raw_path = next(iter(specification.items()))
        else:
            raw_path = specification
            kind = "directory" if Path(raw_path).is_dir() else "manifest"
        path = Path(raw_path)
        entry: dict[str, Any] = {"kind": kind, "path": str(path)}
        if path.is_file():
            entry["sha256"] = file_sha256(path)
        else:
            digest = hashlib.sha256()
            for mask_path in datasets[split].mask_paths:
                try:
                    identity = str(mask_path.relative_to(path))
                except ValueError:
                    identity = str(mask_path)
                digest.update(identity.encode("utf-8"))
                digest.update(b"\0")
                digest.update(file_sha256(mask_path).encode("ascii"))
                digest.update(b"\n")
            entry.update(
                file_count=len(datasets[split]),
                content_sha256=digest.hexdigest(),
            )
        sources[split] = entry
    return {"data_sources": sources}


def execute_shape_teacher_experiment(
    config: dict[str, Any],
    *,
    command: list[str] | None = None,
) -> Path:
    """Train, reload and evaluate one clean or denoising Shape Teacher."""

    validate_shape_teacher_config(config)
    set_seed(int(config["seed"]))
    training = config["training"]
    requested_device = str(training["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested '{requested_device}' but CUDA is unavailable.")
    device = torch.device(requested_device)
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    run_directory = create_run_directory(config)
    save_yaml(run_directory / "config.yaml", config)

    datasets, loaders = build_mask_loaders(config)
    audit = audit_mask_splits(datasets, scan_masks=True)
    save_json(run_directory / "data_audit.json", audit)
    model = build_shape_teacher(config).to(device)
    criterion = build_shape_teacher_loss(config).to(device)
    optimizer = _optimizer(config, model)
    epochs = int(training["epochs"])
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=int(training.get("warmup_epochs", 0)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    corruption_config = config["corruption"]
    train_corruptor = MaskCorruptor(
        corruption_config,
        seed=int(config["seed"]),
        evaluation=False,
    )
    eval_corruptor = MaskCorruptor(
        corruption_config,
        seed=int(config["seed"]),
        evaluation=True,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "status": "running",
        "phase": "shape-teacher",
        "experiment_name": config["experiment"]["name"],
        "run_id": run_directory.name,
        "seed": int(config["seed"]),
        "started_at": utc_now(),
        "command": command or sys.argv,
        "config_fingerprint": config_fingerprint(config),
        "device": {
            "requested": requested_device,
            "resolved": str(device),
            "name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
        },
        "amp_enabled": amp_enabled,
        "parameters": parameter_count,
        "environment": _environment_metadata(),
    }
    metadata.update(
        _data_source_metadata(config["dataset"]["manifest"], datasets)
    )
    save_json(run_directory / "metadata.json", metadata)

    preview_batch = next(iter(loaders["train"]))
    if training["input_mode"] == "denoise":
        preview_corruptor = MaskCorruptor(
            corruption_config, seed=int(config["seed"]), evaluation=False
        )
        preview_inputs = preview_corruptor(
            preview_batch["mask"], keys=list(preview_batch["mask_path"]), split="train"
        )
        save_corruption_grid(
            run_directory / "corruption_preview.png",
            targets=preview_batch["mask"],
            corrupted=preview_inputs,
        )

    history: list[dict[str, Any]] = []
    best_dice = float("-inf")
    best_loss = float("inf")
    epochs_without_improvement = 0
    patience = int(training.get("early_stopping_patience", 0))
    started = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            train_mode = (
                "clean" if training["input_mode"] == "clean" else "corrupted"
            )
            train_metrics = run_epoch(
                model=model,
                loader=loaders["train"],
                criterion=criterion,
                device=device,
                input_mode=train_mode,
                corruptor=train_corruptor if train_mode == "corrupted" else None,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                gradient_clip_norm=training.get("gradient_clip_norm"),
                threshold=float(training["prediction_threshold"]),
                epoch=epoch,
                epochs=epochs,
                log_interval=int(training["log_interval"]),
            )
            validation_clean = run_epoch(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
                input_mode="clean",
                threshold=float(training["prediction_threshold"]),
            )
            validation_corrupted = run_epoch(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
                input_mode="corrupted",
                corruptor=eval_corruptor,
                threshold=float(training["prediction_threshold"]),
            )
            scheduler.step()
            entry = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_clean_{key}": value
                    for key, value in validation_clean.items()
                },
                **{
                    f"validation_corrupted_{key}": value
                    for key, value in validation_corrupted.items()
                },
            }
            history.append(entry)
            save_json(run_directory / "history.json", history)

            current_dice = validation_clean["soft_dice"]
            current_loss = validation_clean["loss"]
            improved = current_dice > best_dice or (
                math.isclose(current_dice, best_dice, rel_tol=0.0, abs_tol=1.0e-12)
                and current_loss < best_loss
            )
            _save_checkpoint(
                run_directory / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_dice=max(best_dice, current_dice),
                best_loss=min(best_loss, current_loss),
                config=config,
            )
            if improved:
                best_dice, best_loss = current_dice, current_loss
                epochs_without_improvement = 0
                _save_checkpoint(
                    run_directory / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_dice=best_dice,
                    best_loss=best_loss,
                    config=config,
                )
            else:
                epochs_without_improvement += 1
            print(
                f"Epoch {epoch}/{epochs} - val clean soft dice: "
                f"{current_dice:.4f} - val corrupted soft dice: "
                f"{validation_corrupted['soft_dice']:.4f} - sigma mean: "
                f"{train_metrics['sigma_mean']:.6f}"
            )
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

        best = torch.load(run_directory / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"], strict=True)
        final_validation = {
            "clean": run_epoch(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
                input_mode="clean",
                threshold=float(training["prediction_threshold"]),
            ),
            "corrupted": run_epoch(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
                input_mode="corrupted",
                corruptor=eval_corruptor,
                threshold=float(training["prediction_threshold"]),
            ),
        }
        test_metrics = {
            mode: run_epoch(
                model=model,
                loader=loaders["test"],
                criterion=criterion,
                device=device,
                input_mode=mode,
                corruptor=eval_corruptor if mode == "corrupted" else None,
                threshold=float(training["prediction_threshold"]),
                boundary_tolerance=float(training["boundary_tolerance"]),
                include_surface_metrics=True,
            )
            for mode in ("clean", "corrupted")
        }
        save_json(
            run_directory / "validation_metrics.json",
            {"checkpoint": "best.pt", **final_validation},
        )
        save_json(
            run_directory / "test_metrics.json",
            {"checkpoint": "best.pt", **test_metrics},
        )

        qualitative_records = collect_qualitative_records(
            model,
            loaders["test"],
            device=device,
        )
        qualitative_selection = select_representatives(qualitative_records)
        save_json(
            run_directory / "qualitative_selection.json",
            {
                "checkpoint": "best.pt",
                "selection_input": "clean_target",
                "reconstruction_sampling": False,
                "tie_break": ["mask_path", "dataset_index"],
                "representatives": qualitative_selection,
            },
        )
        visual_samples = [
            datasets["test"][int(record["dataset_index"])]
            for record in qualitative_selection
        ]
        visual_targets = torch.stack(
            [sample["mask"].float() for sample in visual_samples]
        )
        visual_paths = [str(sample["mask_path"]) for sample in visual_samples]
        visual_corrupted = eval_corruptor(
            visual_targets,
            keys=visual_paths,
            split="test",
        )
        with torch.no_grad():
            clean_output = model(visual_targets.to(device), sample=False)
            corrupted_output = model(visual_corrupted.to(device), sample=False)
        row_labels = [
            f"{record['category']}: {record['sample_id']}"
            for record in qualitative_selection
        ]
        save_reconstruction_grid(
            run_directory / "qualitative_clean.png",
            targets=visual_targets,
            inputs=visual_targets,
            logits=clean_output.logits,
            threshold=float(training["prediction_threshold"]),
            row_labels=row_labels,
        )
        save_reconstruction_grid(
            run_directory / "qualitative_corrupted.png",
            targets=visual_targets,
            inputs=visual_corrupted,
            logits=corrupted_output.logits,
            threshold=float(training["prediction_threshold"]),
            row_labels=row_labels,
        )
        main_inputs, main_logits = (
            (visual_targets, clean_output.logits)
            if training["input_mode"] == "clean"
            else (visual_corrupted, corrupted_output.logits)
        )
        save_reconstruction_grid(
            run_directory / "reconstruction_grid.png",
            targets=visual_targets,
            inputs=main_inputs,
            logits=main_logits,
            threshold=float(training["prediction_threshold"]),
            row_labels=row_labels,
        )

        metadata.update(
            status="completed",
            ended_at=utc_now(),
            elapsed_seconds=time.perf_counter() - started,
            completed_epochs=len(history),
            best_epoch=int(best["epoch"]),
            best_validation_clean_dice=float(
                best["best_validation_clean_dice"]
            ),
            posterior_scale_collapsed=bool(
                final_validation["clean"]["sigma_below_1e-3_pct"] >= 95.0
            ),
        )
        save_json(run_directory / "metadata.json", metadata)
        return run_directory
    except Exception as error:
        metadata.update(
            status="failed",
            ended_at=utc_now(),
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )
        save_json(run_directory / "metadata.json", metadata)
        raise


def evaluate_shape_teacher_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str,
    input_mode: str,
    device: str | None = None,
    include_surface_metrics: bool = True,
) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["experiment_config"]
    requested = device or config["training"]["device"]
    resolved = torch.device(requested)
    model = ShapeTeacher(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved).eval()
    criterion = build_shape_teacher_loss(config).to(resolved)
    _, loaders = build_mask_loaders(config)
    corruptor = MaskCorruptor(
        config["corruption"], seed=int(config["seed"]), evaluation=True
    )
    return run_epoch(
        model=model,
        loader=loaders[split],
        criterion=criterion,
        device=resolved,
        input_mode=input_mode,
        corruptor=corruptor if input_mode == "corrupted" else None,
        threshold=float(config["training"]["prediction_threshold"]),
        boundary_tolerance=float(config["training"]["boundary_tolerance"]),
        include_surface_metrics=include_surface_metrics,
    )
