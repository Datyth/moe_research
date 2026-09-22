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
SUPPORTED_TASKS = (
    "segmentation",
    "phase_b_a1_no_expert",
    "phase_b_build_up",
    "phase_b_fuse",
    "phase_b_router",
    "phase_b_moe",
    "phase_c_distill",
)
# Model fields naming a file on disk; resolved against the project root so a
# config stays runnable from any working directory.
MODEL_PATH_FIELDS = (
    "checkpoint",
    "shape_teacher_checkpoint",
    "teacher_checkpoint",
)


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


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must contain a YAML mapping: {path}")
    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` onto `base`, recursing into shared mapping keys."""

    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def _load_raw_config_with_extends(
    path: Path,
    *,
    seen: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Resolve a config's optional `extends: <relative-path>` chain."""

    resolved_path = path.resolve()
    if resolved_path in seen:
        raise ValueError(
            "Circular 'extends' chain detected at "
            f"{resolved_path}."
        )
    raw = _read_yaml_mapping(resolved_path)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    if not isinstance(extends, str) or not extends:
        raise ValueError(f"'extends' must be a non-empty path string: {resolved_path}")
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = resolved_path.parent / base_path
    base = _load_raw_config_with_extends(base_path, seen=seen | {resolved_path})
    return _deep_merge(base, raw)


def load_experiment_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Read, validate and resolve an experiment YAML file.

    Supports an optional `extends: <relative-path>` deep-merge before
    validation (chains, but not cycles).
    """

    config_path = Path(path).expanduser().resolve()
    raw_config = _load_raw_config_with_extends(config_path)
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
    if dataset["task"] not in {"binary", "multiclass"}:
        raise ValueError("dataset.task must be 'binary' or 'multiclass'.")
    if dataset["task"] == "binary" and dataset["num_classes"] != 1:
        raise ValueError("Binary segmentation requires dataset.num_classes=1.")
    if dataset["task"] == "multiclass" and dataset["num_classes"] < 2:
        raise ValueError("Multiclass segmentation requires dataset.num_classes >= 2.")
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
    dataset["manifest"] = _resolve_path(
        dataset["manifest"], root, "dataset.manifest"
    )
    if not isinstance(dataset["version"], str) or not dataset["version"]:
        raise ValueError("dataset.version must be a non-empty string.")
    dataset.setdefault("image_mean", [0.485, 0.456, 0.406])
    dataset.setdefault("image_std", [0.229, 0.224, 0.225])
    dataset.setdefault("mask_threshold", 0.5)
    if len(dataset["image_mean"]) != dataset["in_channels"]:
        raise ValueError("dataset.image_mean must match dataset.in_channels.")
    if len(dataset["image_std"]) != dataset["in_channels"]:
        raise ValueError("dataset.image_std must match dataset.in_channels.")

    # `task` is optional: configs written before the Task abstraction, and
    # every ordinary segmentation config, omit it entirely.
    task_config = config.setdefault("task", {"name": "segmentation"})
    if not isinstance(task_config, dict):
        raise ValueError("Configuration section 'task' must be a mapping.")
    task_config.setdefault("name", "segmentation")
    if task_config["name"] not in SUPPORTED_TASKS:
        raise ValueError(
            f"task.name must be one of: {', '.join(sorted(SUPPORTED_TASKS))}."
        )
    # Router/MoE-stage loss weights. Defaulted here (not in the task class)
    # so the resolved config — and therefore every saved run folder — always
    # records the weights a training run actually used.
    if task_config["name"] in ("phase_b_router", "phase_b_moe"):
        task_config.setdefault("lambda_latent", 0.1)
        task_config.setdefault("lambda_balance", 0.01)
        task_config["lambda_latent"] = _positive_float(
            task_config["lambda_latent"], "task.lambda_latent", allow_zero=True
        )
        task_config["lambda_balance"] = _positive_float(
            task_config["lambda_balance"], "task.lambda_balance", allow_zero=True
        )
    if task_config["name"] == "phase_b_build_up":
        evaluation_mode = task_config.get("evaluation_mode")
        if evaluation_mode not in {"image_only", "posterior_oracle"}:
            raise ValueError(
                "task.evaluation_mode must be image_only or posterior_oracle."
            )
        task_config.setdefault("lambda_balance", 0.0)
        task_config["lambda_balance"] = _positive_float(
            task_config["lambda_balance"],
            "task.lambda_balance",
            allow_zero=True,
        )
    if task_config["name"] == "phase_c_distill":
        task_config.setdefault("lambda_latent", 1.0)
        task_config.setdefault("lambda_route", 1.0)
        task_config.setdefault("lambda_deploy", 0.0)
        for weight_name in (
            "lambda_latent",
            "lambda_route",
            "lambda_deploy",
        ):
            task_config[weight_name] = _positive_float(
                task_config[weight_name],
                f"task.{weight_name}",
                allow_zero=True,
            )

    model = config["model"]
    _require_keys(model, "model", ("name",))
    shared_model_fields = {"task", "in_channels", "num_classes"}
    duplicated_fields = sorted(shared_model_fields.intersection(model))
    if duplicated_fields:
        raise ValueError(
            "Shared dataset fields must not be repeated in model config: "
            f"{', '.join(duplicated_fields)}."
        )
    for field_name in MODEL_PATH_FIELDS:
        if model.get(field_name) is not None:
            model[field_name] = _resolve_path(
                model[field_name], root, f"model.{field_name}"
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
    if scheduler_name not in {"none", "cosine", "reduce_on_plateau", "warmup_poly"}:
        raise ValueError(
            "scheduler.name must be one of: none, cosine, reduce_on_plateau, "
            "warmup_poly."
        )
    if scheduler_name == "cosine":
        scheduler.setdefault("eta_min", 0.0)
        scheduler["eta_min"] = _positive_float(
            scheduler["eta_min"], "scheduler.eta_min", allow_zero=True
        )
    elif scheduler_name == "warmup_poly":
        scheduler.setdefault("warmup_steps", 250)
        scheduler.setdefault("power", 0.9)
        scheduler["warmup_steps"] = _positive_int(
            scheduler["warmup_steps"], "scheduler.warmup_steps", allow_zero=True
        )
        power = _positive_float(scheduler["power"], "scheduler.power")
        if power > 1.0:
            raise ValueError("scheduler.power must be in (0, 1].")
        scheduler["power"] = power
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
    training.setdefault("amp_dtype", "float16")
    if training["amp_dtype"] not in {"float16", "bfloat16"}:
        raise ValueError("training.amp_dtype must be 'float16' or 'bfloat16'.")
    training.setdefault("prediction_threshold", 0.5)
    training.setdefault("boundary_tolerance", 2)
    training.setdefault("log_interval", 20)
    training.setdefault("gradient_clip_norm", None)
    training.setdefault("early_stopping_patience", None)
    training.setdefault("monitor", "dice")
    training.setdefault("monitor_mode", "max")
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
    if not isinstance(training["monitor"], str) or not training["monitor"]:
        raise ValueError("training.monitor must be a non-empty string.")
    if (
        not isinstance(training["monitor_mode"], str)
        or training["monitor_mode"] not in {"min", "max"}
    ):
        raise ValueError("training.monitor_mode must be 'min' or 'max'.")
    gradient_clip_norm = training["gradient_clip_norm"]
    if gradient_clip_norm is not None:
        training["gradient_clip_norm"] = _positive_float(
            gradient_clip_norm, "training.gradient_clip_norm"
        )
    early_stopping_patience = training["early_stopping_patience"]
    if early_stopping_patience is not None:
        training["early_stopping_patience"] = _positive_int(
            early_stopping_patience, "training.early_stopping_patience"
        )

    return config
