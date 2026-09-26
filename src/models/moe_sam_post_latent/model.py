"""Full MoE-SAM E3 with deterministic post-MoE mean conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from ..base import SegmentationOutput
from ..esam import EsamModel
from ..esam._vendor.sam_moe import MoeEncoding
from ..phase_b.image_descriptor import (
    DEFAULT_LEVELS,
    MultiLevelImageDescriptor,
    MultiLevelImageDescriptorOutput,
)
from ..phase_b.shape_teacher import ShapeTeacher, load_shape_teacher
from ..registry import register_model
from .modules import PostLatentFiLMConditioning


@dataclass
class _EncodedImageState:
    moe: MoeEncoding
    descriptor: MultiLevelImageDescriptorOutput
    prior_mean: Tensor


@dataclass
class _DecodedState:
    logits: Tensor
    iou_predictions: Tensor
    delta_norm_ratio: Tensor


@dataclass
class MoeSamPostLatentState:
    """Joint mean-only state consumed by the experiment task."""

    posterior_mean: Tensor
    prior_mean: Tensor
    moe_expert_indices: Tensor | None
    descriptor_level_weights: Tensor
    posterior_logits: Tensor | None = None
    prior_logits: Tensor | None = None
    posterior_iou_predictions: Tensor | None = None
    prior_iou_predictions: Tensor | None = None
    posterior_delta_norm_ratio: Tensor | None = None
    prior_delta_norm_ratio: Tensor | None = None


def _parameter_counts(module: nn.Module | None) -> dict[str, int]:
    parameters = () if module is None else tuple(module.parameters())
    return _counts_from_parameters(parameters)


def _counts_from_parameters(
    parameters: Iterable[nn.Parameter],
) -> dict[str, int]:
    resolved = tuple(parameters)
    total = sum(parameter.numel() for parameter in resolved)
    trainable = sum(
        parameter.numel()
        for parameter in resolved
        if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


@register_model("moe_sam_post_latent_mean")
class MoeSamPostLatentMeanModel(EsamModel):
    """Preserve Full MoE-SAM E3 and inject a deterministic mean after MoE-FEB."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        task: str = "binary",
        image_size: int = 256,
        checkpoint: str | None = None,
        use_moe: bool = True,
        use_lpeg: bool = True,
        moe_num_experts: int = 4,
        moe_top_k_ratio: float = 0.5,
        freeze_backbone: bool = True,
        descriptor_dim: int = 256,
        levels: tuple[int, ...] | list[int] = DEFAULT_LEVELS,
        scoring_hidden_dim: int = 64,
        shape_teacher_checkpoint: str | Path | None = None,
        freeze_shape_teacher: bool = True,
        latent_dim: int = 8,
        latent_projection_dim: int = 64,
    ) -> None:
        if task != "binary" or num_classes != 1:
            raise ValueError(
                "MoE-SAM post-latent mean conditioning supports binary masks only."
            )
        if not use_moe:
            raise ValueError("The experiment requires the original MoE-SAM MoE-FEB.")
        if not use_lpeg:
            raise ValueError("The experiment requires E3 LPEG.")
        if not freeze_backbone:
            raise ValueError(
                "freeze_backbone must be true; E3 still trains its Adapters."
            )
        if not freeze_shape_teacher:
            raise ValueError("The Phase-A Shape Teacher must remain frozen.")
        if shape_teacher_checkpoint is None:
            raise ValueError("shape_teacher_checkpoint is required.")
        if descriptor_dim != 256:
            raise ValueError("descriptor_dim must be 256.")
        if latent_dim != 8:
            raise ValueError("latent_dim must be 8.")
        if latent_projection_dim != 64:
            raise ValueError("latent_projection_dim must be 64.")

        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
            image_size=image_size,
            checkpoint=checkpoint,
            use_moe=True,
            use_lpeg=True,
            moe_num_experts=moe_num_experts,
            moe_top_k_ratio=moe_top_k_ratio,
            freeze_backbone=True,
        )
        self.image_size = image_size
        self.descriptor_dim = descriptor_dim
        self.latent_dim = latent_dim
        self.image_descriptor = MultiLevelImageDescriptor(
            embed_dim=self.network.image_encoder.embed_dim,
            descriptor_dim=descriptor_dim,
            levels=tuple(levels),
            scoring_hidden_dim=scoring_hidden_dim,
        )
        self.shape_teacher: ShapeTeacher = load_shape_teacher(
            shape_teacher_checkpoint,
            freeze=True,
        )
        shape_latent_dim = (
            self.shape_teacher.projector.latent_projection.out_features
        )
        if shape_latent_dim != descriptor_dim:
            raise ValueError(
                "Shape Teacher output must be 256-dimensional, got "
                f"{shape_latent_dim}."
            )

        self.posterior_mean_head = nn.Linear(
            descriptor_dim + shape_latent_dim,
            latent_dim,
        )
        self.prior_mean_head = nn.Linear(descriptor_dim, latent_dim)
        self.post_latent_conditioning = (
            PostLatentFiLMConditioning(
                feature_dim=descriptor_dim,
                latent_dim=latent_dim,
                latent_projection_dim=latent_projection_dim,
            )
        )

        for parameter in self.shape_teacher.parameters():
            parameter.requires_grad_(False)
        self.shape_teacher.eval()
        self._assert_parameter_policy()

    def train(self, mode: bool = True) -> "MoeSamPostLatentMeanModel":
        super().train(mode)
        self.shape_teacher.eval()
        return self

    @staticmethod
    def _require_finite(name: str, tensor: Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN or Inf.")

    def _encode_images(self, images: Tensor) -> _EncodedImageState:
        moe = self.network.encode_with_moe(images)
        # Alignment may train the descriptor, but it must not introduce an
        # auxiliary gradient path into the E3 ViT/Adapters. The segmentation
        # path through moe.image_embeddings remains attached.
        detached_block_outputs = [
            features.detach() for features in moe.block_outputs
        ]
        descriptor = self.image_descriptor(detached_block_outputs)
        prior_mean = self.prior_mean_head(descriptor.descriptor)
        self._require_finite("moe_image_embeddings", moe.image_embeddings)
        self._require_finite("image_descriptor", descriptor.descriptor)
        self._require_finite("prior_mean", prior_mean)
        return _EncodedImageState(
            moe=moe,
            descriptor=descriptor,
            prior_mean=prior_mean,
        )

    def _posterior_mean(
        self,
        descriptor: Tensor,
        masks: Tensor,
    ) -> Tensor:
        shape_descriptor = self.shape_teacher(masks)
        if shape_descriptor.shape != descriptor.shape:
            raise ValueError(
                "Shape and image descriptors must both be [B, 256], got "
                f"{tuple(shape_descriptor.shape)} and {tuple(descriptor.shape)}."
            )
        posterior_mean = self.posterior_mean_head(
            torch.cat(
                [descriptor, shape_descriptor.to(descriptor.dtype)],
                dim=1,
            )
        )
        self._require_finite("posterior_mean", posterior_mean)
        return posterior_mean

    def _decode(self, image_embeddings: Tensor, latent: Tensor) -> _DecodedState:
        conditioned = self.post_latent_conditioning.condition(
            image_embeddings,
            latent,
        )
        logits, iou_predictions, _ = self.network.decode_embeddings(
            conditioned.features,
            multimask_output=True,
            image_size=self.image_size,
            batch_size=image_embeddings.shape[0],
        )
        self._require_finite("segmentation_logits", logits)
        self._require_finite("iou_predictions", iou_predictions)
        self._require_finite(
            "post_conditioning_delta_norm_ratio",
            conditioned.delta_norm_ratio,
        )
        return _DecodedState(
            logits=logits,
            iou_predictions=iou_predictions,
            delta_norm_ratio=conditioned.delta_norm_ratio,
        )

    def joint_forward(
        self,
        images: Tensor,
        masks: Tensor,
        *,
        decode_posterior: bool = True,
        decode_prior: bool = False,
    ) -> MoeSamPostLatentState:
        if masks is None:
            raise ValueError("joint_forward requires ground-truth masks.")
        encoded = self._encode_images(images)
        posterior_mean = self._posterior_mean(
            encoded.descriptor.descriptor,
            masks,
        )

        posterior_decoded = (
            self._decode(encoded.moe.image_embeddings, posterior_mean)
            if decode_posterior
            else None
        )
        prior_decoded = (
            self._decode(encoded.moe.image_embeddings, encoded.prior_mean)
            if decode_prior
            else None
        )
        return MoeSamPostLatentState(
            posterior_mean=posterior_mean,
            prior_mean=encoded.prior_mean,
            moe_expert_indices=encoded.moe.expert_indices,
            descriptor_level_weights=encoded.descriptor.level_weights,
            posterior_logits=(
                None if posterior_decoded is None else posterior_decoded.logits
            ),
            prior_logits=None if prior_decoded is None else prior_decoded.logits,
            posterior_iou_predictions=(
                None
                if posterior_decoded is None
                else posterior_decoded.iou_predictions
            ),
            prior_iou_predictions=(
                None if prior_decoded is None else prior_decoded.iou_predictions
            ),
            posterior_delta_norm_ratio=(
                None
                if posterior_decoded is None
                else posterior_decoded.delta_norm_ratio
            ),
            prior_delta_norm_ratio=(
                None
                if prior_decoded is None
                else prior_decoded.delta_norm_ratio
            ),
        )

    def forward(
        self,
        images: Tensor,
        *,
        masks: Tensor | None = None,
        **kwargs: Any,
    ) -> SegmentationOutput:
        # Ordinary deployment is image-only by construction. Incidental masks
        # and unrelated keyword arguments cannot activate the privileged branch.
        del masks, kwargs
        encoded = self._encode_images(images)
        decoded = self._decode(
            encoded.moe.image_embeddings,
            encoded.prior_mean,
        )
        return SegmentationOutput(
            logits=decoded.logits,
            diagnostics={
                "iou_predictions": decoded.iou_predictions,
                "moe_expert_indices": encoded.moe.expert_indices,
                "image_descriptor": encoded.descriptor.descriptor,
                "descriptor_level_weights": (
                    encoded.descriptor.level_weights
                ),
                "prior_mean": encoded.prior_mean,
                "post_conditioning_delta_norm_ratio": (
                    decoded.delta_norm_ratio.mean()
                ),
            },
        )

    @staticmethod
    def _assert_all_trainable(name: str, module: nn.Module | None) -> None:
        if module is None:
            raise RuntimeError(f"{name} is required.")
        parameters = tuple(module.parameters())
        if not parameters or any(
            not parameter.requires_grad for parameter in parameters
        ):
            raise RuntimeError(f"All {name} parameters must be trainable.")

    def _assert_parameter_policy(self) -> None:
        image_parameters = tuple(
            self.network.image_encoder.named_parameters()
        )
        adapter_parameters = tuple(
            parameter
            for name, parameter in image_parameters
            if "Adapter" in name
        )
        non_adapter_parameters = tuple(
            parameter
            for name, parameter in image_parameters
            if "Adapter" not in name
        )
        if not adapter_parameters or any(
            not parameter.requires_grad for parameter in adapter_parameters
        ):
            raise RuntimeError("All SAM Adapter parameters must be trainable.")
        if any(
            parameter.requires_grad for parameter in non_adapter_parameters
        ):
            raise RuntimeError(
                "All non-Adapter SAM image-encoder parameters must be frozen."
            )
        prompt_non_lpeg_parameters = tuple(
            parameter
            for name, parameter in self.network.prompt_encoder.named_parameters()
            if "lpeg" not in name
        )
        if any(
            parameter.requires_grad
            for parameter in prompt_non_lpeg_parameters
        ):
            raise RuntimeError(
                "Pretrained SAM prompt-encoder parameters must be frozen."
            )
        if any(
            parameter.requires_grad
            for parameter in self.shape_teacher.parameters()
        ):
            raise RuntimeError("The Shape Teacher must be frozen.")

        required_trainable = {
            "original MoE-FEB router": self.network.ExpertChoiceTokenMoE.router,
            "original MoE-FEB experts": self.network.ExpertChoiceTokenMoE.experts,
            "MoE-FEB attention": self.network.attn,
            "MoE-FEB neck5": self.network.neck5,
            "LPEG": self.network.prompt_encoder.lpeg,
            "SAM mask decoder": self.network.mask_decoder,
            "image descriptor": self.image_descriptor,
            "posterior mean head": self.posterior_mean_head,
            "prior mean head": self.prior_mean_head,
            "post-latent conditioning": self.post_latent_conditioning,
        }
        for name, module in required_trainable.items():
            self._assert_all_trainable(name, module)

    def moe_sam_post_latent_checkpoint_metadata(self) -> dict[str, Any]:
        """Return architecture invariants and a module-level parameter audit."""

        self._assert_parameter_policy()
        image_parameters = tuple(
            self.network.image_encoder.named_parameters()
        )
        adapter_parameters = tuple(
            parameter
            for name, parameter in image_parameters
            if "Adapter" in name
        )
        non_adapter_parameters = tuple(
            parameter
            for name, parameter in image_parameters
            if "Adapter" not in name
        )
        prompt_non_lpeg_parameters = tuple(
            parameter
            for name, parameter in self.network.prompt_encoder.named_parameters()
            if "lpeg" not in name
        )
        all_parameters = tuple(self.parameters())
        total = sum(parameter.numel() for parameter in all_parameters)
        trainable = sum(
            parameter.numel()
            for parameter in all_parameters
            if parameter.requires_grad
        )
        groups = {
            "sam_image_encoder_non_adapter": _counts_from_parameters(
                non_adapter_parameters
            ),
            "sam_adapters": _counts_from_parameters(adapter_parameters),
            "sam_prompt_encoder_non_lpeg": _counts_from_parameters(
                prompt_non_lpeg_parameters
            ),
            "original_moe_feb_router": _parameter_counts(
                self.network.ExpertChoiceTokenMoE.router
            ),
            "original_moe_feb_experts": _parameter_counts(
                self.network.ExpertChoiceTokenMoE.experts
            ),
            "moe_feb_attention": _parameter_counts(self.network.attn),
            "moe_feb_neck5": _parameter_counts(self.network.neck5),
            "lpeg": _parameter_counts(self.network.prompt_encoder.lpeg),
            "sam_mask_decoder": _parameter_counts(self.network.mask_decoder),
            "image_descriptor": _parameter_counts(self.image_descriptor),
            "shape_teacher": _parameter_counts(self.shape_teacher),
            "posterior_mean_head": _parameter_counts(
                self.posterior_mean_head
            ),
            "prior_mean_head": _parameter_counts(self.prior_mean_head),
            "post_latent_film_conditioning": _parameter_counts(
                self.post_latent_conditioning
            ),
        }
        return {
            "variant": "moe_sam_post_latent_mean",
            "latent_dim": self.latent_dim,
            "mean_only": True,
            "post_moe": True,
            "uses_original_moe_feb": True,
            "uses_external_hierarchical_moe": False,
            "conditioning": "latent_only_film",
            "parameter_counts": {
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
                "groups": groups,
            },
        }


__all__ = [
    "MoeSamPostLatentMeanModel",
    "MoeSamPostLatentState",
]
