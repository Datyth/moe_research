"""Phase C distillation from the privileged B6 posterior to an image prior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.models.base import SegmentationOutput
from src.models.phase_b.image_descriptor import (
    DEFAULT_LEVELS,
    MultiLevelImageDescriptorOutput,
)
from src.models.phase_b.posterior import DiagonalGaussian, GaussianParameterHead
from src.models.phase_b.router import RoutingOutput
from src.models.phase_b.studies.build_up.b6_hierarchical import (
    PhaseBB6Hierarchical,
)
from src.models.phase_b.studies.build_up.common import (
    IMAGE_ONLY,
    decode_sam,
    inject_enhancement,
    run_sam_encoder,
)
from src.models.registry import register_model


EXPECTED_TEACHER_MODEL = "phase_b_b6_hierarchical"
EXPECTED_TEACHER_CLASS = "PhaseBB6Hierarchical"


@dataclass
class _EncodedImageState:
    image_embeddings: Tensor
    descriptor: MultiLevelImageDescriptorOutput


@dataclass
class PhaseCDistillationState:
    """Posterior/prior state produced from one shared SAM encoder pass."""

    posterior: DiagonalGaussian
    prior: DiagonalGaussian
    posterior_routing: RoutingOutput
    prior_routing: RoutingOutput
    prior_logits: Tensor | None = None
    posterior_logits: Tensor | None = None
    prior_iou_predictions: Tensor | None = None
    posterior_iou_predictions: Tensor | None = None


@dataclass
class PhaseCSegmentationOutput(SegmentationOutput):
    """SegmentationOutput whose public logits are always prior-routed."""

    prior_logits: Tensor | None = None
    posterior_logits: Tensor | None = None
    distillation_state: PhaseCDistillationState | None = None


def _normalized_config_value(name: str, value: Any) -> Any:
    if name == "levels":
        return tuple(int(level) for level in value)
    if name == "std_floor":
        return float(value)
    return value


def load_b6_teacher_checkpoint(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and strict-load the fixed B6 teacher checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Phase-C B6 teacher checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Phase-C teacher checkpoint must be a dictionary.")
    if checkpoint.get("format_version") != 2:
        raise ValueError("Phase-C teacher must be a format-version 2 checkpoint.")
    if checkpoint.get("model_class") != EXPECTED_TEACHER_CLASS:
        raise ValueError(
            "Phase-C teacher model_class must be "
            f"{EXPECTED_TEACHER_CLASS!r}, got {checkpoint.get('model_class')!r}."
        )

    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Phase-C teacher checkpoint metadata is required.")
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Phase-C teacher checkpoint lacks model_config metadata.")
    if model_config.get("name") != EXPECTED_TEACHER_MODEL:
        raise ValueError(
            "Phase-C teacher model.name must be "
            f"{EXPECTED_TEACHER_MODEL!r}, got {model_config.get('name')!r}."
        )
    task_config = checkpoint.get("task_config")
    if (
        not isinstance(task_config, dict)
        or task_config.get("name") != "phase_b_build_up"
        or task_config.get("evaluation_mode") != "posterior_oracle"
    ):
        raise ValueError(
            "Phase-C teacher must record task.name='phase_b_build_up' and "
            "task.evaluation_mode='posterior_oracle'."
        )

    for name, expected in (expected_model_config or {}).items():
        if name not in model_config:
            raise ValueError(f"Phase-C teacher model_config lacks {name!r}.")
        actual = _normalized_config_value(name, model_config[name])
        normalized_expected = _normalized_config_value(name, expected)
        if actual != normalized_expected:
            raise ValueError(
                f"Phase-C/B6 architecture mismatch for {name}: "
                f"teacher={actual!r}, requested={normalized_expected!r}."
            )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Phase-C teacher checkpoint lacks model_state_dict.")
    module.load_state_dict(state_dict, strict=True)

    return {
        "path": str(path),
        "format_version": checkpoint["format_version"],
        "model_class": checkpoint["model_class"],
        "originating_experiment": metadata.get("experiment_name"),
        "seed": metadata.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "monitor_name": checkpoint.get("monitor_name"),
        "monitor_mode": checkpoint.get("monitor_mode"),
        "best_monitor_value": checkpoint.get("best_monitor_value"),
        "model_config": dict(model_config),
    }


@register_model("phase_c_b6_distill")
class PhaseCB6PriorDistill(PhaseBB6Hierarchical):
    """Frozen B6 teacher plus one trainable image-conditioned Gaussian prior."""

    evaluation_mode = IMAGE_ONLY

    def __init__(
        self,
        *,
        teacher_checkpoint: str | Path,
        in_channels: int = 3,
        num_classes: int = 1,
        task: str = "binary",
        image_size: int = 256,
        checkpoint: str | None = None,
        use_moe: bool = False,
        use_lpeg: bool = True,
        freeze_backbone: bool = True,
        descriptor_dim: int = 256,
        levels: tuple[int, ...] = DEFAULT_LEVELS,
        scoring_hidden_dim: int = 64,
        router_mode: str = "learned",
        latent_dim: int = 64,
        num_experts: int = 4,
        active_experts: int = 2,
        expert_hidden_ratio: int = 4,
        std_floor: float = 1e-4,
        stochastic: bool = True,
        shape_teacher_checkpoint: str | Path | None = None,
        freeze_shape_teacher: bool = True,
    ) -> None:
        resolved_levels = tuple(int(level) for level in levels)
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
            image_size=image_size,
            checkpoint=checkpoint,
            use_moe=use_moe,
            use_lpeg=use_lpeg,
            freeze_backbone=freeze_backbone,
            descriptor_dim=descriptor_dim,
            levels=resolved_levels,
            scoring_hidden_dim=scoring_hidden_dim,
            router_mode=router_mode,
            latent_dim=latent_dim,
            num_experts=num_experts,
            active_experts=active_experts,
            expert_hidden_ratio=expert_hidden_ratio,
            std_floor=std_floor,
            stochastic=stochastic,
            shape_teacher_checkpoint=shape_teacher_checkpoint,
            freeze_shape_teacher=freeze_shape_teacher,
        )
        expected_teacher_config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "task": task,
            "image_size": image_size,
            "use_moe": use_moe,
            "use_lpeg": use_lpeg,
            "freeze_backbone": freeze_backbone,
            "descriptor_dim": descriptor_dim,
            "levels": resolved_levels,
            "scoring_hidden_dim": scoring_hidden_dim,
            "router_mode": router_mode,
            "latent_dim": latent_dim,
            "num_experts": num_experts,
            "active_experts": active_experts,
            "expert_hidden_ratio": expert_hidden_ratio,
            "std_floor": std_floor,
            "stochastic": stochastic,
            "freeze_shape_teacher": freeze_shape_teacher,
        }
        self.teacher_checkpoint_info = load_b6_teacher_checkpoint(
            self,
            teacher_checkpoint,
            expected_model_config=expected_teacher_config,
        )
        self.teacher_checkpoint = self.teacher_checkpoint_info["path"]

        self.prior_head = GaussianParameterHead(
            in_dim=descriptor_dim,
            latent_dim=latent_dim,
            std_floor=std_floor,
        )
        self._freeze_except_prior()
        self.train(False)

    def _freeze_except_prior(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.prior_head.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "PhaseCB6PriorDistill":
        """Keep every inherited B6 module in eval; only the prior follows mode."""

        nn.Module.train(self, mode)
        for child in self.children():
            if child is not self.prior_head:
                child.eval()
        self.prior_head.train(mode)
        return self

    def phase_c_checkpoint_metadata(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "teacher": dict(self.teacher_checkpoint_info),
            "prior_config": {
                "in_dim": self.prior_head.in_dim,
                "latent_dim": self.prior_head.latent_dim,
                "std_floor": self.prior_head.std_floor,
            },
            "parameter_counts": {
                "total": total,
                "frozen": total - trainable,
                "trainable": trainable,
            },
        }

    def _encode_images(self, images: Tensor) -> _EncodedImageState:
        with torch.no_grad():
            image_embeddings, block_outputs = run_sam_encoder(
                self.backbone,
                images,
            )
            descriptor = self.image_descriptor(block_outputs)
        return _EncodedImageState(
            image_embeddings=image_embeddings,
            descriptor=descriptor,
        )

    def _posterior_from_encoded(
        self,
        encoded: _EncodedImageState,
        masks: Tensor,
    ) -> tuple[DiagonalGaussian, RoutingOutput]:
        with torch.no_grad():
            shape_latent = self.shape_teacher(masks)
            fused = self.fusion(
                encoded.descriptor.descriptor,
                shape_latent,
            ).fused
            posterior = self.conditioner.posterior_head(fused)
            routing = self.conditioner.router(posterior.mean)
        return posterior, routing

    def _prior_from_encoded(
        self,
        encoded: _EncodedImageState,
    ) -> tuple[DiagonalGaussian, RoutingOutput]:
        prior = self.prior_head(encoded.descriptor.descriptor.detach())
        routing = self.conditioner.router(prior.mean)
        return prior, routing

    def _decode_routing(
        self,
        encoded: _EncodedImageState,
        *,
        latent: Tensor,
        routing: RoutingOutput,
    ) -> tuple[Tensor, Tensor]:
        descriptor = encoded.descriptor
        level_pools = torch.stack(
            [tokens.mean(dim=1) for tokens in descriptor.level_tokens],
            dim=1,
        )
        enhancement = self.enhancement(
            descriptor.level_tokens,
            level_pools,
            latent,
            routing.routing_probs,
            routing.expert_indices,
        )
        _, enhanced_embeddings = inject_enhancement(
            encoded.image_embeddings,
            enhancement.fused_tokens,
            self.enhancement_neck,
            embed_dim=self.embed_dim,
        )
        return decode_sam(
            self.backbone,
            enhanced_embeddings,
            image_size=self.image_size,
        )

    def distillation_forward(
        self,
        images: Tensor,
        masks: Tensor,
        *,
        decode_prior: bool = False,
        decode_posterior: bool = False,
    ) -> PhaseCDistillationState:
        """Build fixed posterior targets and the trainable prior from one encode."""

        if masks is None:
            raise ValueError("Phase-C distillation requires ground-truth masks.")
        encoded = self._encode_images(images)
        posterior, posterior_routing = self._posterior_from_encoded(
            encoded,
            masks,
        )
        prior, prior_routing = self._prior_from_encoded(encoded)

        prior_logits = None
        prior_iou = None
        if decode_prior:
            prior_logits, prior_iou = self._decode_routing(
                encoded,
                latent=prior.mean,
                routing=prior_routing,
            )

        posterior_logits = None
        posterior_iou = None
        if decode_posterior:
            with torch.no_grad():
                posterior_logits, posterior_iou = self._decode_routing(
                    encoded,
                    latent=posterior.mean,
                    routing=posterior_routing,
                )

        return PhaseCDistillationState(
            posterior=posterior,
            prior=prior,
            posterior_routing=posterior_routing,
            prior_routing=prior_routing,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            prior_iou_predictions=prior_iou,
            posterior_iou_predictions=posterior_iou,
        )

    def forward(
        self,
        images: Tensor,
        *,
        masks: Tensor | None = None,
        include_posterior: bool = False,
        **kwargs: Any,
    ) -> PhaseCSegmentationOutput:
        """Return deployable prior segmentation; privileged work is opt-in."""

        if include_posterior:
            if masks is None:
                raise ValueError(
                    "include_posterior=True requires ground-truth masks."
                )
            state = self.distillation_forward(
                images,
                masks,
                decode_prior=True,
                decode_posterior=True,
            )
            if state.prior_logits is None or state.posterior_logits is None:
                raise RuntimeError("Phase-C diagnostic decoding produced no logits.")
            return PhaseCSegmentationOutput(
                logits=state.prior_logits,
                prior_logits=state.prior_logits,
                posterior_logits=state.posterior_logits,
                distillation_state=state,
                diagnostics={
                    "phase_c_distill": state,
                    "prior_iou_predictions": state.prior_iou_predictions,
                    "posterior_iou_predictions": state.posterior_iou_predictions,
                },
            )

        # Deliberately ignore an unused mask: without the explicit diagnostic
        # flag the public path must never touch privileged modules.
        encoded = self._encode_images(images)
        prior, prior_routing = self._prior_from_encoded(encoded)
        prior_logits, prior_iou = self._decode_routing(
            encoded,
            latent=prior.mean,
            routing=prior_routing,
        )
        return PhaseCSegmentationOutput(
            logits=prior_logits,
            prior_logits=prior_logits,
            diagnostics={
                "phase_c_prior": {
                    "prior": prior,
                    "routing": prior_routing,
                },
                "prior_iou_predictions": prior_iou,
            },
        )


__all__ = [
    "PhaseCB6PriorDistill",
    "PhaseCDistillationState",
    "PhaseCSegmentationOutput",
    "load_b6_teacher_checkpoint",
]
