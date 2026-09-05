"""Build and execute Phase-A mask reconstruction experiments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from src.engine import Trainer, TrainerConfig, evaluate
from src.experiment import (
    build_checkpoint_metadata,
    build_dataset_config,
    build_loaders,
    build_optimizer,
    build_scheduler,
    config_fingerprint,
    create_run_directory,
    device_metadata,
    file_sha256,
    git_metadata,
    save_json,
    save_yaml,
    set_seed,
    utc_now,
)
from src.losses import build_loss
from src.models.shape import (
    ReconstructionDecoder,
    ShapeAutoencoder,
    SmallCNN,
    SpatialProjector,
)
from src.tasks import MaskReconstructionTask


def _require_fixed_value(
    mapping: dict[str, Any],
    key: str,
    expected: Any,
    *,
    location: str,
) -> None:
    value = mapping.get(key)
    if value != expected:
        raise ValueError(
            f"{location}.{key} must be {expected!r} for the Phase-A baseline, "
            f"got {value!r}."
        )


def build_shape_autoencoder(config: dict[str, Any]) -> ShapeAutoencoder:
    """Validate and directly construct the fixed Small-CNN baseline."""

    if list(config["dataset"]["image_size"]) != [256, 256]:
        raise ValueError(
            "Phase-A Small-CNN requires dataset.image_size=[256, 256]."
        )
    model_config = config["model"]
    _require_fixed_value(
        model_config,
        "name",
        "shape_autoencoder",
        location="model",
    )
    encoder_config = model_config.get("encoder")
    projector_config = model_config.get("projector")
    decoder_config = model_config.get("decoder")
    if not isinstance(encoder_config, dict):
        raise ValueError("model.encoder must be a mapping.")
    if not isinstance(projector_config, dict):
        raise ValueError("model.projector must be a mapping.")
    if not isinstance(decoder_config, dict):
        raise ValueError("model.decoder must be a mapping.")
    _require_fixed_value(
        encoder_config,
        "name",
        "small_cnn",
        location="model.encoder",
    )
    for key, expected in (
        ("channels", 64),
        ("spatial_size", 4),
        ("bottleneck_dim", 256),
    ):
        _require_fixed_value(
            projector_config,
            key,
            expected,
            location="model.projector",
        )
    for key, expected in (("start_channels", 128), ("start_size", 8)):
        _require_fixed_value(
            decoder_config,
            key,
            expected,
            location="model.decoder",
        )

    return ShapeAutoencoder(
        encoder=SmallCNN(),
        projector=SpatialProjector(),
        decoder=ReconstructionDecoder(),
    )


def execute_shape_pretraining(
    config: dict[str, Any],
    *,
    run_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    """Execute one fresh or resumed Phase-A experiment."""

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
        model = build_shape_autoencoder(config)
        criterion = build_loss(config["loss"])
        training = config["training"]
        threshold = float(training["prediction_threshold"])
        task = MaskReconstructionTask(
            criterion=criterion,
            threshold=threshold,
        )
        task_config = {
            "name": "mask_reconstruction",
            "threshold": threshold,
        }
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer)
        model_config = deepcopy(config["model"])
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
        save_json(
            resolved_run_dir / "test_metrics.json",
            {
                "checkpoint": "best.pt",
                "split": "test",
                "loss": test_metrics["loss"],
                "dice": test_metrics["dice"],
            },
        )

        metadata["status"] = "completed"
        metadata["ended_at"] = utc_now()
        metadata["best_epoch"] = int(best_checkpoint["epoch"])
        metadata["monitor_name"] = best_checkpoint["monitor_name"]
        metadata["monitor_mode"] = best_checkpoint["monitor_mode"]
        metadata["best_monitor_value"] = best_checkpoint["best_monitor_value"]
        metadata.pop("best_val_dice", None)
        save_json(metadata_path, metadata)
        print(f"Test Loss : {test_metrics['loss']:.6f}")
        print(f"Test Dice : {test_metrics['dice']:.6f}")
        return resolved_run_dir
    except Exception as error:
        metadata["status"] = "failed"
        metadata["ended_at"] = utc_now()
        metadata["error"] = f"{type(error).__name__}: {error}"
        save_json(metadata_path, metadata)
        raise
