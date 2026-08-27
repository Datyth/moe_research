"""Phase 0: pretrain the mask-VAE shape teacher on ground-truth masks.

The teacher consumes only ``batch["mask"]``; images are ignored on purpose, so
the learned latent describes shape alone. Run folders follow the same layout as
``scripts/run_experiment.py`` (config, metadata, history, checkpoints, test
metrics) so downstream phases can load a teacher the same way they load any
other run.

    python scripts/training/pretrain_teacher_vae.py \
        --config configs/teacher_vae_isic2018.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import DatasetConfig  # noqa: E402
from src.configs.experiment import load_experiment_config  # noqa: E402
from src.data import build_dataset  # noqa: E402
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
from src.losses.vae import MaskVAELoss, MaskVAELosses  # noqa: E402
from src.models.shapemoe import MaskVAETeacher  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the Phase 0 mask-VAE shape teacher.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the teacher experiment YAML.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device from the config.",
    )
    return parser.parse_args()


def build_teacher(
    config: dict[str, Any],
    dataset_config: DatasetConfig,
) -> MaskVAETeacher:
    """Instantiate the teacher, taking the mask resolution from the dataset."""

    model_config = dict(config["model"])
    name = model_config.pop("name")
    if name != "mask_vae_teacher":
        raise ValueError(
            f"This script only trains 'mask_vae_teacher', got '{name}'."
        )
    model_config.setdefault("image_size", dataset_config.image_size)
    for key in ("encoder_channels", "decoder_channels"):
        if key in model_config:
            model_config[key] = tuple(model_config[key])
    return MaskVAETeacher(**model_config)


def build_criterion(config: dict[str, Any]) -> MaskVAELoss:
    loss_config = dict(config["loss"])
    name = loss_config.pop("name")
    if name != "mask_vae":
        raise ValueError(
            f"This script only supports loss.name='mask_vae', got '{name}'."
        )
    return MaskVAELoss(**loss_config)


def build_loaders(
    config: dict[str, Any],
    dataset_config: DatasetConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    training = config["training"]
    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": str(training["device"]).startswith("cuda"),
        "persistent_workers": int(training["num_workers"]) > 0,
        "drop_last": False,
    }
    train_loader = DataLoader(
        build_dataset(dataset_config, split="train"),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        build_dataset(dataset_config, split="val"),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        build_dataset(dataset_config, split="test"),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, test_loader


def reconstruction_dice(
    recon_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Mean per-sample Dice between the thresholded reconstruction and M.

    This is a readability metric for Phase 0, not part of the objective: it
    says how much shape survives the latent bottleneck.
    """

    predicted = (recon_logits.sigmoid() >= threshold).float().flatten(1)
    reference = targets.float().flatten(1)
    intersection = (predicted * reference).sum(dim=1)
    denominator = predicted.sum(dim=1) + reference.sum(dim=1)
    return ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def run_epoch(
    *,
    model: MaskVAETeacher,
    loader: DataLoader,
    criterion: MaskVAELoss,
    device: torch.device,
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

    for step, batch in enumerate(loader, start=1):
        if "mask" not in batch:
            raise KeyError("Each batch must contain 'mask'.")
        masks = batch["mask"].to(device, non_blocking=True).float()
        batch_size = masks.shape[0]

        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(masks)
                losses: MaskVAELosses = criterion(output, masks)

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
        reconstruction_logits = output.recon_logits.detach().float()
        dice, iou = compute_binary_dice_iou(
            reconstruction_logits,
            masks,
            threshold=threshold,
        )
        metrics["dice"] = float(dice.mean())
        metrics["iou"] = float(iou.mean())
        if include_surface_metrics:
            hd95, assd, boundary_f1 = compute_binary_surface_metrics(
                reconstruction_logits,
                masks,
                threshold=threshold,
                boundary_tolerance=boundary_tolerance,
            )
            metrics["hd95"] = float(hd95.mean())
            metrics["assd"] = float(assd.mean())
            metrics["boundary_f1"] = float(boundary_f1.mean())
        metrics["active_units"] = float(
            (output.mu.detach().float().var(dim=0) > 1e-2).sum()
        )
        accumulator.update(metrics, batch_size)

        if training and (
            step == 1
            or step % log_interval == 0
            or step == number_of_batches
        ):
            running = accumulator.summary()
            print(
                f"Epoch {epoch}/{epochs} - batch {step}/{number_of_batches} "
                f"- loss: {running['loss']:.4f} "
                f"- rec: {running['reconstruction']:.4f} "
                f"- kl: {running['kl']:.4f} "
                f"- dice: {running['dice']:.4f} "
                f"- iou: {running['iou']:.4f}"
            )

    return accumulator.summary()


def save_checkpoint(
    path: Path,
    *,
    model: MaskVAETeacher,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.model_config(),
            # The deliverable of Phase 0: E_M on its own, so later phases can
            # load and freeze it without carrying the decoder and the heads.
            "mask_embedding_state_dict": model.mask_embedding_state_dict(),
            "embedding_geometry": model.embedding_geometry(),
            "latent_dim": model.latent_dim,
            "epoch": epoch,
            "best_validation_loss": best_loss,
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

    model = build_teacher(config, dataset_config).to(device)
    criterion = build_criterion(config).to(device)

    if amp_enabled and criterion.recon_reduction == "sum":
        pixels = dataset_config.image_size[0] * dataset_config.image_size[1]
        print(
            "WARNING: AMP is on while the reconstruction term sums over "
            f"{pixels} pixels, so the loss is around "
            f"{0.7 * pixels:.0f} and its scaled gradients tend to overflow "
            "float16. Expect GradScaler to skip most early steps. Set "
            "training.amp=false, or switch loss.recon_reduction to 'mean' and "
            "reduce beta accordingly."
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
            "phase": "0-teacher-vae",
            "created_at": utc_now(),
            "config_fingerprint": config_fingerprint(config),
            "device": {"requested": requested_device, "resolved": str(device)},
            "amp_enabled": amp_enabled,
            "torch_version": torch.__version__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "latent_dim": model.latent_dim,
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
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        shared = {
            "model": model,
            "criterion": criterion,
            "device": device,
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
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()

        entry = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        history.append(entry)
        save_json(run_directory / "history.json", history)
        print(
            f"Epoch {epoch}/{epochs} "
            f"- train loss: {train_metrics['loss']:.4f} "
            f"- val loss: {validation_metrics['loss']:.4f} "
            f"- val rec: {validation_metrics['reconstruction']:.4f} "
            f"- val kl: {validation_metrics['kl']:.4f} "
            f"- val dice: {validation_metrics['dice']:.4f} "
            f"- val iou: {validation_metrics['iou']:.4f} "
            f"- val hd95: {validation_metrics['hd95']:.4f} "
            f"- val assd: {validation_metrics['assd']:.4f} "
            f"- val boundary f1: {validation_metrics['boundary_f1']:.4f} "
            f"- active units: {validation_metrics['active_units']:.1f}"
            f"/{model.latent_dim}"
        )

        save_checkpoint(
            run_directory / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_loss=best_loss,
            config=config,
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            save_checkpoint(
                run_directory / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                config=config,
            )
            print(f"New best validation loss: {best_loss:.4f}")

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
        amp_enabled=amp_enabled,
        threshold=threshold,
        boundary_tolerance=boundary_tolerance,
        include_surface_metrics=True,
    )
    save_json(
        run_directory / "test_metrics.json",
        {
            "best_epoch": checkpoint["epoch"],
            "best_validation_loss": best_loss,
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
