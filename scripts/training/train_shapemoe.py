"""Phase 1: train the ShapeMoE segmenter under a frozen Phase 0 shape teacher.

The teacher reads the ground-truth mask and supplies the target posterior
q_T(z|M); the student must reproduce it from the image alone. Only the student,
router and experts receive gradients.

    python scripts/training/train_shapemoe.py \
        --config configs/shapemoe_isic2018.yaml \
        --teacher runs/teacher_vae_isic2018/<run-id>/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import DatasetConfig  # noqa: E402
from src.configs.experiment import load_experiment_config  # noqa: E402
from src.engine import (  # noqa: E402
    MetricAccumulator,
    compute_binary_dice_iou,
    compute_binary_surface_metrics,
)
from src.experiment import (  # noqa: E402
    build_dataset_config,
    build_optimizer,
    build_scheduler,
    config_fingerprint,
    create_run_directory,
    save_json,
    save_yaml,
    set_seed,
    utc_now,
)
from src.losses.shapemoe import ShapeMoELoss, ShapeMoELosses  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.shapemoe import MaskVAETeacher, ShapeMoESegmenter  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Phase 1 ShapeMoE segmenter.",
    )
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument(
        "--teacher",
        default=None,
        help=(
            "Phase 0 teacher checkpoint. Overrides training.teacher_checkpoint. "
            "Omit both to train without distillation."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device from the config.",
    )
    return parser.parse_args()


def load_teacher(
    checkpoint_path: str | Path,
    *,
    dataset_config: DatasetConfig,
    latent_dim: int,
    device: torch.device,
) -> MaskVAETeacher:
    """Rebuild the Phase 0 teacher from its checkpoint and freeze it."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not a Phase 0 teacher checkpoint: no "
            "'model_config' entry."
        )

    teacher = MaskVAETeacher(**checkpoint["model_config"])
    teacher.load_state_dict(checkpoint["model_state_dict"])

    if teacher.latent_dim != latent_dim:
        raise ValueError(
            f"Teacher latent_dim={teacher.latent_dim} does not match the "
            f"student's latent_dim={latent_dim}. The distillation KL needs both "
            "posteriors in the same space."
        )
    if tuple(teacher.image_size) != tuple(dataset_config.image_size):
        raise ValueError(
            f"Teacher was trained on {tuple(teacher.image_size)} masks but the "
            f"dataset yields {tuple(dataset_config.image_size)}."
        )

    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def build_student(
    config: dict[str, Any],
    dataset_config: DatasetConfig,
) -> ShapeMoESegmenter:
    model_config = dict(config["model"])
    if model_config.get("name") != "shapemoe_unet":
        raise ValueError(
            "This script only trains model.name='shapemoe_unet', got "
            f"'{model_config.get('name')}'."
        )
    model_config.update(
        task=dataset_config.task,
        in_channels=dataset_config.in_channels,
        num_classes=dataset_config.num_classes,
    )
    return build_model(model_config)


def build_criterion(config: dict[str, Any]) -> ShapeMoELoss:
    loss_config = dict(config["loss"])
    name = loss_config.pop("name")
    if name != "shapemoe":
        raise ValueError(
            f"This script only supports loss.name='shapemoe', got '{name}'."
        )
    return ShapeMoELoss(**loss_config)


def build_loaders(
    config: dict[str, Any],
    dataset_config: DatasetConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    from src.data import build_dataset

    training = config["training"]
    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": str(training["device"]).startswith("cuda"),
        "persistent_workers": int(training["num_workers"]) > 0,
        "drop_last": False,
    }
    return (
        DataLoader(
            build_dataset(dataset_config, split="train"),
            shuffle=True,
            generator=generator,
            **common,
        ),
        DataLoader(build_dataset(dataset_config, split="val"), shuffle=False, **common),
        DataLoader(build_dataset(dataset_config, split="test"), shuffle=False, **common),
    )


def teacher_posterior(
    teacher: MaskVAETeacher | None,
    masks: Tensor,
) -> tuple[Tensor, Tensor] | None:
    """Posterior parameters from the ground-truth mask, without gradients."""

    if teacher is None:
        return None
    with torch.no_grad():
        return teacher.encode(masks)


def run_epoch(
    *,
    model: ShapeMoESegmenter,
    loader: DataLoader,
    criterion: ShapeMoELoss,
    device: torch.device,
    teacher: MaskVAETeacher | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    gradient_clip_norm: float | None = None,
    threshold: float = 0.5,
    boundary_tolerance: float = 2,
    include_surface_metrics: bool = False,
    epoch: int = 1,
    epochs: int = 1,
    log_interval: int = 20,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator()
    number_of_batches = len(loader)
    usage = torch.zeros(model.num_experts, dtype=torch.long)

    for step, batch in enumerate(loader, start=1):
        if "image" not in batch or "mask" not in batch:
            raise KeyError("Each batch must contain 'image' and 'mask'.")
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).float()
        batch_size = images.shape[0]

        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(images)
                losses: ShapeMoELosses = criterion(
                    output,
                    masks,
                    teacher_posterior=teacher_posterior(teacher, masks),
                )

            if not torch.isfinite(losses.total):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, step {step}."
                )

            if training:
                scaler.scale(losses.total).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()

        metrics = losses.as_dict()
        dice, iou = compute_binary_dice_iou(
            output.logits.detach().float(),
            masks,
            threshold=threshold,
        )
        metrics["dice"] = float(dice.mean())
        metrics["iou"] = float(iou.mean())
        if include_surface_metrics:
            hd95, assd, boundary_f1 = compute_binary_surface_metrics(
                output.logits.detach().float(),
                masks,
                threshold=threshold,
                boundary_tolerance=boundary_tolerance,
            )
            metrics["hd95"] = float(hd95.mean())
            metrics["assd"] = float(assd.mean())
            metrics["boundary_f1"] = float(boundary_f1.mean())
        accumulator.update(metrics, batch_size)
        usage += model.experts.expert_usage(output.diagnostics["pi"]).cpu()

        if training and (
            step == 1
            or step % log_interval == 0
            or step == number_of_batches
        ):
            running = accumulator.summary()
            print(
                f"Epoch {epoch}/{epochs} - batch {step}/{number_of_batches} "
                f"- loss: {running['loss']:.4f} "
                f"- seg: {running['segmentation']:.4f} "
                f"- cv2: {running['balance']:.4f} "
                f"- kl: {running['distillation']:.4f} "
                f"- dice: {running['dice']:.4f} "
                f"- iou: {running['iou']:.4f}"
            )

    summary = accumulator.summary()
    total_routed = int(usage.sum())
    for expert in range(model.num_experts):
        share = int(usage[expert]) / total_routed if total_routed else 0.0
        summary[f"expert_{expert}_share"] = share
    return summary


def save_checkpoint(
    path: Path,
    *,
    model: ShapeMoESegmenter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    config: dict[str, Any],
    teacher_checkpoint: str | None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.model_config(),
            "epoch": epoch,
            "best_validation_dice": best_dice,
            "teacher_checkpoint": teacher_checkpoint,
            "experiment_config": config,
        },
        path,
    )


def main() -> None:
    arguments = parse_arguments()
    config = load_experiment_config(arguments.config, project_root=PROJECT_ROOT)
    if arguments.device:
        config["training"]["device"] = arguments.device

    set_seed(int(config["seed"]))
    dataset_config = build_dataset_config(config)
    training = config["training"]

    requested_device = str(training["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested_device}' but CUDA is unavailable."
        )
    device = torch.device(requested_device)
    amp_enabled = bool(training["amp"]) and device.type == "cuda"

    model = build_student(config, dataset_config).to(device)
    criterion = build_criterion(config).to(device)

    teacher_path = arguments.teacher or training.get("teacher_checkpoint")
    teacher = (
        None
        if not teacher_path
        else load_teacher(
            teacher_path,
            dataset_config=dataset_config,
            latent_dim=model.latent_dim,
            device=device,
        )
    )
    if teacher is None:
        print(
            "WARNING: no teacher checkpoint given. Training without "
            "distillation; the shape encoder then receives gradients from no "
            "term at all and its posterior stays at initialization."
        )

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    train_loader, validation_loader, test_loader = build_loaders(
        config,
        dataset_config,
    )

    run_directory = create_run_directory(config)
    save_yaml(run_directory / "config.yaml", config)
    save_json(
        run_directory / "metadata.json",
        {
            "phase": "1-shapemoe",
            "created_at": utc_now(),
            "config_fingerprint": config_fingerprint(config),
            "device": {"requested": requested_device, "resolved": str(device)},
            "amp_enabled": amp_enabled,
            "torch_version": torch.__version__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "latent_dim": model.latent_dim,
            "num_experts": model.num_experts,
            "top_k": model.top_k,
            "teacher_checkpoint": str(teacher_path) if teacher_path else None,
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(validation_loader.dataset),
            "test_samples": len(test_loader.dataset),
        },
    )
    print(f"Run directory: {run_directory}")

    epochs = int(training["epochs"])
    threshold = float(training["prediction_threshold"])
    boundary_tolerance = float(training["boundary_tolerance"])
    history: list[dict[str, Any]] = []
    best_dice = float("-inf")

    for epoch in range(1, epochs + 1):
        shared = {
            "model": model,
            "criterion": criterion,
            "device": device,
            "teacher": teacher,
            "threshold": threshold,
            "boundary_tolerance": boundary_tolerance,
            "epoch": epoch,
            "epochs": epochs,
        }
        train_metrics = run_epoch(
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_clip_norm=training["gradient_clip_norm"],
            log_interval=int(training["log_interval"]),
            **shared,
        )
        validation_metrics = run_epoch(
            loader=validation_loader,
            amp_enabled=amp_enabled,
            include_surface_metrics=True,
            **shared,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        save_json(run_directory / "history.json", history)

        shares = " ".join(
            f"{validation_metrics[f'expert_{index}_share']:.2f}"
            for index in range(model.num_experts)
        )
        print(
            f"Epoch {epoch}/{epochs} "
            f"- train loss: {train_metrics['loss']:.4f} "
            f"- val dice: {validation_metrics['dice']:.4f} "
            f"- val iou: {validation_metrics['iou']:.4f} "
            f"- val hd95: {validation_metrics['hd95']:.4f} "
            f"- val assd: {validation_metrics['assd']:.4f} "
            f"- val boundary f1: {validation_metrics['boundary_f1']:.4f} "
            f"- val seg: {validation_metrics['segmentation']:.4f} "
            f"- val cv2: {validation_metrics['balance']:.4f} "
            f"- val kl: {validation_metrics['distillation']:.4f} "
            f"- expert shares: [{shares}]"
        )

        save_checkpoint(
            run_directory / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_dice=best_dice,
            config=config,
            teacher_checkpoint=str(teacher_path) if teacher_path else None,
        )
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            save_checkpoint(
                run_directory / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dice,
                config=config,
                teacher_checkpoint=str(teacher_path) if teacher_path else None,
            )
            print(f"New best validation Dice: {best_dice:.4f}")

    checkpoint = torch.load(
        run_directory / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        teacher=teacher,
        amp_enabled=amp_enabled,
        threshold=threshold,
        boundary_tolerance=boundary_tolerance,
        include_surface_metrics=True,
    )
    save_json(
        run_directory / "test_metrics.json",
        {
            "best_epoch": checkpoint["epoch"],
            "best_validation_dice": best_dice,
            **test_metrics,
        },
    )
    print(f"Test Loss        : {test_metrics['loss']:.6f}")
    print(f"Test Dice        : {test_metrics['dice']:.6f}")
    print(f"Test IoU         : {test_metrics['iou']:.6f}")
    print(f"Test HD95        : {test_metrics['hd95']:.6f}")
    print(f"Test ASSD        : {test_metrics['assd']:.6f}")
    print(f"Test Boundary F1 : {test_metrics['boundary_f1']:.6f}")


if __name__ == "__main__":
    main()
