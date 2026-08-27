"""Load and validate lightweight YAML experiment configurations."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = (
    "experiment",
    "dataset",
    "model",
    "loss",
    "optimizer",
    "scheduler",
    "training",
)
EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


def _require_keys(section: dict[str, Any], name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(
            f"Configuration section '{name}' is missing: {', '.join(missing)}."
        )


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (
        result == 0.0 and not allow_zero
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _resolve_path(value: Any, project_root: Path, name: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_experiment_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Read, validate and resolve an experiment YAML file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)
    if not isinstance(raw_config, dict):
        raise ValueError("Experiment config must contain a YAML mapping.")
    return resolve_experiment_config(raw_config, project_root=project_root)


def resolve_experiment_config(
    raw_config: dict[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Return a validated, serializable config with resolved paths."""

    if not isinstance(raw_config, dict):
        raise ValueError("Experiment config must be a mapping.")
    config = copy.deepcopy(raw_config)
    root = Path(project_root).expanduser().resolve()

    for section_name in REQUIRED_SECTIONS:
        _require_mapping(config, section_name)
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise ValueError("Configuration field 'seed' must be an integer.")

    experiment = config["experiment"]
    _require_keys(experiment, "experiment", ("name", "output_root"))
    name = experiment["name"]
    if not isinstance(name, str) or not EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "experiment.name may contain only letters, numbers, '.', '_' and '-'."
        )
    experiment["output_root"] = _resolve_path(
        experiment["output_root"], root, "experiment.output_root"
    )

    dataset = config["dataset"]
    _require_keys(
        dataset,
        "dataset",
        (
            "name",
            "root",
            "manifest",
            "version",
            "task",
            "num_classes",
            "in_channels",
            "image_size",
        ),
    )
    if dataset["task"] != "binary":
        raise ValueError("Experiment runner currently supports task='binary' only.")
    if dataset["num_classes"] != 1:
        raise ValueError("Binary segmentation requires dataset.num_classes=1.")
    _positive_int(dataset["in_channels"], "dataset.in_channels")
    image_size = dataset["image_size"]
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ValueError("dataset.image_size must contain two positive integers.")
    dataset["image_size"] = [int(value) for value in image_size]
    dataset["root"] = _resolve_path(dataset["root"], root, "dataset.root")
    manifest = dataset["manifest"]
    if isinstance(manifest, dict):
        split_keys = ("train", "val", "test")
        missing = [split for split in split_keys if split not in manifest]
        extra = sorted(set(manifest) - set(split_keys))
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"unknown {', '.join(extra)}")
            raise ValueError(
                "dataset.manifest split mapping is invalid: " + "; ".join(details)
            )
        resolved_manifest: dict[str, object] = {}
        for split in split_keys:
            source = manifest[split]
            field = f"dataset.manifest.{split}"
            if isinstance(source, (str, Path)):
                resolved_manifest[split] = _resolve_path(source, root, field)
                continue
            if not isinstance(source, dict):
                raise ValueError(f"{field} must be a path or mapping.")
            source_keys = [
                key for key in ("manifest", "directory") if key in source
            ]
            if len(source_keys) != 1 or len(source) != 1:
                raise ValueError(
                    f"{field} needs exactly one of 'manifest' or 'directory'."
                )
            source_key = source_keys[0]
            resolved_manifest[split] = {
                source_key: _resolve_path(
                    source[source_key], root, f"{field}.{source_key}"
                )
            }
        dataset["manifest"] = resolved_manifest
    else:
        dataset["manifest"] = _resolve_path(manifest, root, "dataset.manifest")
    if not isinstance(dataset["version"], str) or not dataset["version"]:
        raise ValueError("dataset.version must be a non-empty string.")
    dataset.setdefault("image_mean", [0.485, 0.456, 0.406])
    dataset.setdefault("image_std", [0.229, 0.224, 0.225])
    dataset.setdefault("mask_threshold", 0.5)
    if len(dataset["image_mean"]) != dataset["in_channels"]:
        raise ValueError("dataset.image_mean must match dataset.in_channels.")
    if len(dataset["image_std"]) != dataset["in_channels"]:
        raise ValueError("dataset.image_std must match dataset.in_channels.")

    model = config["model"]
    _require_keys(model, "model", ("name",))
    shared_model_fields = {"task", "in_channels", "num_classes"}
    duplicated_fields = sorted(shared_model_fields.intersection(model))
    if duplicated_fields:
        raise ValueError(
            "Shared dataset fields must not be repeated in model config: "
            f"{', '.join(duplicated_fields)}."
        )
    loss = config["loss"]
    _require_keys(loss, "loss", ("name",))

    optimizer = config["optimizer"]
    _require_keys(optimizer, "optimizer", ("name", "lr", "weight_decay"))
    if optimizer["name"] != "adamw":
        raise ValueError("Only optimizer.name='adamw' is supported.")
    optimizer["lr"] = _positive_float(optimizer["lr"], "optimizer.lr")
    optimizer["weight_decay"] = _positive_float(
        optimizer["weight_decay"], "optimizer.weight_decay", allow_zero=True
    )

    scheduler = config["scheduler"]
    _require_keys(scheduler, "scheduler", ("name",))
    scheduler_name = scheduler["name"]
    if scheduler_name not in {"none", "cosine", "reduce_on_plateau"}:
        raise ValueError(
            "scheduler.name must be one of: none, cosine, reduce_on_plateau."
        )
    if scheduler_name == "cosine":
        scheduler.setdefault("eta_min", 0.0)
        scheduler["eta_min"] = _positive_float(
            scheduler["eta_min"], "scheduler.eta_min", allow_zero=True
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler.setdefault("factor", 0.1)
        scheduler.setdefault("patience", 5)
        scheduler.setdefault("min_lr", 0.0)
        factor = _positive_float(scheduler["factor"], "scheduler.factor")
        if factor >= 1.0:
            raise ValueError("scheduler.factor must be less than 1.")
        scheduler["factor"] = factor
        scheduler["patience"] = _positive_int(
            scheduler["patience"], "scheduler.patience", allow_zero=True
        )
        scheduler["min_lr"] = _positive_float(
            scheduler["min_lr"], "scheduler.min_lr", allow_zero=True
        )

    training = config["training"]
    _require_keys(training, "training", ("epochs", "batch_size", "num_workers", "device", "amp"))
    training["epochs"] = _positive_int(training["epochs"], "training.epochs")
    training["batch_size"] = _positive_int(
        training["batch_size"], "training.batch_size"
    )
    training["num_workers"] = _positive_int(
        training["num_workers"], "training.num_workers", allow_zero=True
    )
    if not isinstance(training["device"], str) or not training["device"]:
        raise ValueError("training.device must be a non-empty string.")
    if not isinstance(training["amp"], bool):
        raise ValueError("training.amp must be a boolean.")
    training.setdefault("prediction_threshold", 0.5)
    training.setdefault("boundary_tolerance", 2)
    training.setdefault("log_interval", 20)
    training.setdefault("gradient_clip_norm", None)
    threshold = float(training["prediction_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("training.prediction_threshold must be in [0, 1].")
    training["prediction_threshold"] = threshold
    training["boundary_tolerance"] = _positive_float(
        training["boundary_tolerance"],
        "training.boundary_tolerance",
        allow_zero=True,
    )
    training["log_interval"] = _positive_int(
        training["log_interval"], "training.log_interval"
    )
    gradient_clip_norm = training["gradient_clip_norm"]
    if gradient_clip_norm is not None:
        training["gradient_clip_norm"] = _positive_float(
            gradient_clip_norm, "training.gradient_clip_norm"
        )

    return config
