"""C1/C2/C3 joint prior-posterior latent-conditioning model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.models.base import BaseSegmentationModel, SegmentationOutput
from src.models.esam import EsamModel
from src.models.phase_b.fusion import PrivilegedFusion
from src.models.phase_b.image_descriptor import (
    DEFAULT_LEVELS,
    MultiLevelImageDescriptor,
    MultiLevelImageDescriptorOutput,
)
from src.models.phase_b.moe_enhancement import HierarchicalMoEEnhancement
from src.models.phase_b.posterior import DiagonalGaussian, GaussianParameterHead
from src.models.phase_b.router import RoutingOutput, TopKRouter, load_balance_loss
from src.models.phase_b.shape_teacher import ShapeTeacher, load_shape_teacher
from src.models.phase_b.studies.build_up.common import (
    build_enhancement_neck,
    decode_sam,
    inject_enhancement,
    run_sam_encoder,
)
from src.models.registry import register_model

from .modules import PostLatentConditioning, PreMoEContextAdapter


@dataclass
class _EncodedImageState:
    image_embeddings: Tensor
    descriptor: MultiLevelImageDescriptorOutput


@dataclass
class _DecodedState:
    logits: Tensor
    iou_predictions: Tensor


@dataclass
class LatentConditioningState:
    """Joint latent and optional decoded state produced from one encoder pass."""

    posterior: DiagonalGaussian
    prior: DiagonalGaussian
    use_moe: bool
    use_pre_moe_latent: bool
    posterior_context: Tensor | None = None
    prior_context: Tensor | None = None
    posterior_routing: RoutingOutput | None = None
    prior_routing: RoutingOutput | None = None
    posterior_balance: Tensor | None = None
    posterior_logits: Tensor | None = None
    prior_logits: Tensor | None = None
    posterior_iou_predictions: Tensor | None = None
    prior_iou_predictions: Tensor | None = None


@dataclass
class LatentConditioningOutput(SegmentationOutput):
    """Public segmentation result whose main logits always use the prior."""

    prior_logits: Tensor | None = None
    posterior_logits: Tensor | None = None
    joint_state: LatentConditioningState | None = None


def _parameter_counts(module: nn.Module | None) -> dict[str, int]:
    if module is None:
        return {"total": 0, "trainable": 0, "frozen": 0}
    parameters = tuple(module.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


@register_model("latent_conditioning_model")
class LatentConditioningModel(BaseSegmentationModel):
    """One controlled architecture spanning C1, C2, and C3."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        task: str = "binary",
        image_size: int = 256,
        checkpoint: str | Path | None = None,
        use_lpeg: bool = True,
        freeze_backbone: bool = True,
        descriptor_dim: int = 256,
        levels: tuple[int, ...] | list[int] = DEFAULT_LEVELS,
        scoring_hidden_dim: int = 64,
        shape_teacher_checkpoint: str | Path | None = None,
        freeze_shape_teacher: bool = True,
        latent_dim: int = 8,
        std_floor: float = 1e-4,
        stochastic: bool = False,
        num_experts: int = 4,
        active_experts: int = 2,
        expert_hidden_ratio: int = 4,
        moe_context_dim: int = 256,
        latent_projection_dim: int = 64,
        use_moe: bool = False,
        use_pre_moe_latent: bool = False,
        use_post_moe_latent: bool = True,
    ) -> None:
        if task != "binary" or num_classes != 1:
            raise ValueError("Latent conditioning supports binary masks only.")
        if not freeze_backbone:
            raise ValueError("The complete SAM image encoder must remain frozen.")
        if not freeze_shape_teacher:
            raise ValueError("The Phase-A Shape Teacher must remain frozen.")
        if not use_lpeg:
            raise ValueError("Latent conditioning requires LPEG.")
        if stochastic:
            raise ValueError("Latent conditioning requires deterministic mean latents.")
        if latent_dim != 8:
            raise ValueError("C1/C2/C3 require model.latent_dim=8.")
        if descriptor_dim != 256 or moe_context_dim != 256:
            raise ValueError("C1/C2/C3 require 256-D descriptors and MoE contexts.")
        if latent_projection_dim != 64:
            raise ValueError("C1/C2/C3 require latent_projection_dim=64.")
        if not use_post_moe_latent:
            raise ValueError("C1/C2/C3 require post-MoE latent conditioning.")
        if use_pre_moe_latent and not use_moe:
            raise ValueError("Pre-MoE latent conditioning requires model.use_moe=true.")
        if shape_teacher_checkpoint is None:
            raise ValueError("shape_teacher_checkpoint is required.")

        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
        )
        self.image_size = image_size
        self.descriptor_dim = descriptor_dim
        self.latent_dim = latent_dim
        self.moe_context_dim = moe_context_dim
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.use_moe = bool(use_moe)
        self.use_pre_moe_latent = bool(use_pre_moe_latent)
        self.use_post_moe_latent = True

        # The public ``use_moe`` flag controls the external hierarchical MoE.
        # The legacy token MoE inside E-SAM is deliberately always disabled.
        self.backbone = EsamModel(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
            image_size=image_size,
            checkpoint=None if checkpoint is None else str(checkpoint),
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        self.embed_dim = self.backbone.network.image_encoder.embed_dim
        self.image_descriptor = MultiLevelImageDescriptor(
            embed_dim=self.embed_dim,
            descriptor_dim=descriptor_dim,
            levels=tuple(levels),
            scoring_hidden_dim=scoring_hidden_dim,
        )
        self.shape_teacher: ShapeTeacher = load_shape_teacher(
            shape_teacher_checkpoint,
            freeze=True,
        )
        shape_latent_dim = self.shape_teacher.projector.latent_projection.out_features
        if shape_latent_dim != descriptor_dim:
            raise ValueError(
                "Shape Teacher output must match descriptor_dim=256, got "
                f"{shape_latent_dim}."
            )
        self.fusion = PrivilegedFusion(
            descriptor_dim=descriptor_dim,
            shape_latent_dim=shape_latent_dim,
        )
        self.posterior_head = GaussianParameterHead(
            in_dim=self.fusion.output_dim,
            latent_dim=latent_dim,
            std_floor=std_floor,
        )
        self.prior_head = GaussianParameterHead(
            in_dim=descriptor_dim,
            latent_dim=latent_dim,
            std_floor=std_floor,
        )
        self.post_latent_conditioning = PostLatentConditioning(
            feature_dim=descriptor_dim,
            latent_dim=latent_dim,
            latent_projection_dim=latent_projection_dim,
        )

        if self.use_moe:
            self.pre_moe_context_adapter = PreMoEContextAdapter(
                descriptor_dim=descriptor_dim,
                latent_dim=latent_dim,
                context_dim=moe_context_dim,
            )
            self.router = TopKRouter(
                latent_dim=moe_context_dim,
                num_experts=num_experts,
                active_experts=active_experts,
            )
            self.enhancement = HierarchicalMoEEnhancement(
                embed_dim=self.embed_dim,
                num_experts=num_experts,
                num_levels=len(tuple(levels)),
                latent_dim=moe_context_dim,
                expert_hidden_ratio=expert_hidden_ratio,
            )
            self.enhancement_neck = build_enhancement_neck(
                self.embed_dim,
                descriptor_dim,
            )

        self._freeze_privileged_sources()
        # Run the assertions once at construction; the same audit is exported
        # into run/checkpoint metadata by the experiment runner.
        self._assert_parameter_policy()

    @property
    def variant(self) -> str:
        if not self.use_moe:
            return "c2_no_moe_post"
        if self.use_pre_moe_latent:
            return "c3_moe_pre_post"
        return "c1_moe_post"

    def _freeze_privileged_sources(self) -> None:
        for parameter in self.backbone.network.image_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.shape_teacher.parameters():
            parameter.requires_grad_(False)
        self.backbone.network.image_encoder.eval()
        self.shape_teacher.eval()

    def train(self, mode: bool = True) -> "LatentConditioningModel":
        super().train(mode)
        self.backbone.network.image_encoder.eval()
        self.shape_teacher.eval()
        return self

    @staticmethod
    def _require_finite(name: str, tensor: Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN or Inf.")

    @classmethod
    def _validate_gaussian(
        cls,
        name: str,
        distribution: DiagonalGaussian,
    ) -> None:
        cls._require_finite(f"{name}.mean", distribution.mean)
        cls._require_finite(f"{name}.std", distribution.std)
        if bool((distribution.std <= 0).any()):
            raise FloatingPointError(f"{name}.std must be strictly positive.")

    @classmethod
    def _validate_routing(cls, name: str, routing: RoutingOutput) -> None:
        cls._require_finite(f"{name}.logits", routing.logits)
        cls._require_finite(f"{name}.dense_probs", routing.dense_probs)
        cls._require_finite(f"{name}.routing_probs", routing.routing_probs)

    def _encode_images(self, images: Tensor) -> _EncodedImageState:
        with torch.no_grad():
            image_embeddings, block_outputs = run_sam_encoder(
                self.backbone,
                images,
            )
        descriptor = self.image_descriptor(block_outputs)
        self._require_finite("image_descriptor", descriptor.descriptor)
        self._require_finite("image_embeddings", image_embeddings)
        return _EncodedImageState(
            image_embeddings=image_embeddings,
            descriptor=descriptor,
        )

    def _posterior(
        self,
        encoded: _EncodedImageState,
        masks: Tensor,
    ) -> DiagonalGaussian:
        with torch.no_grad():
            shape_latent = self.shape_teacher(masks)
        fused = self.fusion(
            encoded.descriptor.descriptor,
            shape_latent,
        ).fused
        posterior = self.posterior_head(fused)
        self._validate_gaussian("posterior", posterior)
        return posterior

    def _prior(self, encoded: _EncodedImageState) -> DiagonalGaussian:
        prior = self.prior_head(encoded.descriptor.descriptor)
        self._validate_gaussian("prior", prior)
        return prior

    def _context(self, image_descriptor: Tensor, latent: Tensor) -> Tensor:
        if not self.use_moe:
            raise RuntimeError("C2 has no pre-MoE context.")
        latent_slot = latent if self.use_pre_moe_latent else torch.zeros_like(latent)
        context = self.pre_moe_context_adapter(image_descriptor, latent_slot)
        self._require_finite("moe_context", context)
        return context

    def _route(self, context: Tensor) -> RoutingOutput:
        if not self.use_moe:
            raise RuntimeError("C2 has no router.")
        routing = self.router(context)
        self._validate_routing("routing", routing)
        return routing

    def _moe_base_features(
        self,
        encoded: _EncodedImageState,
        context: Tensor,
        routing: RoutingOutput,
    ) -> Tensor:
        if not self.use_moe:
            raise RuntimeError("C2 has no MoE feature path.")
        level_pools = torch.stack(
            [tokens.mean(dim=1) for tokens in encoded.descriptor.level_tokens],
            dim=1,
        )
        enhancement = self.enhancement(
            encoded.descriptor.level_tokens,
            level_pools,
            context,
            routing.routing_probs,
            routing.expert_indices,
        )
        _, base_features = inject_enhancement(
            encoded.image_embeddings,
            enhancement.fused_tokens,
            self.enhancement_neck,
            embed_dim=self.embed_dim,
        )
        self._require_finite("moe_base_features", base_features)
        return base_features

    def _decode(self, base_features: Tensor, latent: Tensor) -> _DecodedState:
        conditioned = self.post_latent_conditioning(base_features, latent)
        self._require_finite("post_latent_features", conditioned)
        logits, iou_predictions = decode_sam(
            self.backbone,
            conditioned,
            image_size=self.image_size,
        )
        self._require_finite("segmentation_logits", logits)
        self._require_finite("iou_predictions", iou_predictions)
        return _DecodedState(logits=logits, iou_predictions=iou_predictions)

    def joint_forward(
        self,
        images: Tensor,
        masks: Tensor,
        *,
        decode_posterior: bool = True,
        decode_prior: bool = False,
    ) -> LatentConditioningState:
        if masks is None:
            raise ValueError("joint_forward requires ground-truth masks.")
        encoded = self._encode_images(images)
        posterior = self._posterior(encoded, masks)
        prior = self._prior(encoded)

        posterior_context = None
        prior_context = None
        posterior_routing = None
        prior_routing = None
        posterior_balance = None
        posterior_decoded = None
        prior_decoded = None

        if not self.use_moe:
            if decode_posterior:
                posterior_decoded = self._decode(
                    encoded.image_embeddings,
                    posterior.mean,
                )
            if decode_prior:
                prior_decoded = self._decode(encoded.image_embeddings, prior.mean)
        elif not self.use_pre_moe_latent:
            # C1 has one strictly image-only route and one MoE feature map.
            posterior_context = self._context(
                encoded.descriptor.descriptor,
                posterior.mean,
            )
            prior_context = posterior_context
            posterior_routing = self._route(posterior_context)
            prior_routing = posterior_routing
            posterior_balance = load_balance_loss(
                posterior_routing.dense_probs,
                posterior_routing.expert_indices,
                num_experts=self.num_experts,
            )
            shared_base = None
            if decode_posterior or decode_prior:
                shared_base = self._moe_base_features(
                    encoded,
                    posterior_context,
                    posterior_routing,
                )
            if decode_posterior:
                posterior_decoded = self._decode(shared_base, posterior.mean)
            if decode_prior:
                prior_decoded = self._decode(shared_base, prior.mean)
        else:
            # C3 uses branch-specific latents before and after the MoE.
            posterior_context = self._context(
                encoded.descriptor.descriptor,
                posterior.mean,
            )
            posterior_routing = self._route(posterior_context)
            posterior_balance = load_balance_loss(
                posterior_routing.dense_probs,
                posterior_routing.expert_indices,
                num_experts=self.num_experts,
            )
            if decode_posterior:
                posterior_base = self._moe_base_features(
                    encoded,
                    posterior_context,
                    posterior_routing,
                )
                posterior_decoded = self._decode(posterior_base, posterior.mean)
            if decode_prior:
                prior_context = self._context(
                    encoded.descriptor.descriptor,
                    prior.mean,
                )
                prior_routing = self._route(prior_context)
                prior_base = self._moe_base_features(
                    encoded,
                    prior_context,
                    prior_routing,
                )
                prior_decoded = self._decode(prior_base, prior.mean)

        if posterior_balance is not None:
            self._require_finite("posterior_balance", posterior_balance)
        return LatentConditioningState(
            posterior=posterior,
            prior=prior,
            use_moe=self.use_moe,
            use_pre_moe_latent=self.use_pre_moe_latent,
            posterior_context=posterior_context,
            prior_context=prior_context,
            posterior_routing=posterior_routing,
            prior_routing=prior_routing,
            posterior_balance=posterior_balance,
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
        )

    def forward(
        self,
        images: Tensor,
        *,
        masks: Tensor | None = None,
        include_posterior: bool = False,
        **kwargs: Any,
    ) -> LatentConditioningOutput:
        if include_posterior:
            if masks is None:
                raise ValueError("include_posterior=True requires masks.")
            state = self.joint_forward(
                images,
                masks,
                decode_posterior=True,
                decode_prior=True,
            )
            if state.prior_logits is None or state.posterior_logits is None:
                raise RuntimeError("Joint diagnostic decoding produced no logits.")
            return LatentConditioningOutput(
                logits=state.prior_logits,
                prior_logits=state.prior_logits,
                posterior_logits=state.posterior_logits,
                joint_state=state,
                diagnostics={
                    "latent_conditioning": state,
                    "prior_iou_predictions": state.prior_iou_predictions,
                    "posterior_iou_predictions": state.posterior_iou_predictions,
                },
            )

        # An incidental mask is deliberately ignored on the deployment path.
        encoded = self._encode_images(images)
        prior = self._prior(encoded)
        context = None
        routing = None
        if self.use_moe:
            context = self._context(encoded.descriptor.descriptor, prior.mean)
            routing = self._route(context)
            base_features = self._moe_base_features(encoded, context, routing)
        else:
            base_features = encoded.image_embeddings
        decoded = self._decode(base_features, prior.mean)
        return LatentConditioningOutput(
            logits=decoded.logits,
            prior_logits=decoded.logits,
            diagnostics={
                "latent_conditioning_prior": {
                    "prior": prior,
                    "context": context,
                    "routing": routing,
                },
                "prior_iou_predictions": decoded.iou_predictions,
            },
        )

    def _assert_parameter_policy(self) -> None:
        if any(
            parameter.requires_grad
            for parameter in self.backbone.network.image_encoder.parameters()
        ):
            raise RuntimeError("SAM image encoder and Adapters must be frozen.")
        if any(
            parameter.requires_grad
            for parameter in self.shape_teacher.parameters()
        ):
            raise RuntimeError("Shape Teacher must be frozen.")
        required_trainable = {
            "image_descriptor": self.image_descriptor,
            "posterior_head": self.posterior_head,
            "prior_head": self.prior_head,
            "post_latent_conditioning": self.post_latent_conditioning,
            "lpeg": self.backbone.network.prompt_encoder.lpeg,
            "sam_mask_decoder": self.backbone.network.mask_decoder,
        }
        if self.use_moe:
            required_trainable.update(
                {
                    "pre_moe_context_adapter": self.pre_moe_context_adapter,
                    "router": self.router,
                    "experts": self.enhancement.experts,
                    "layer_scorer": self.enhancement.layer_scorer,
                    "enhancement_neck": self.enhancement_neck,
                }
            )
        for name, module in required_trainable.items():
            if _parameter_counts(module)["trainable"] <= 0:
                raise RuntimeError(f"{name} must contain trainable parameters.")

    def latent_conditioning_checkpoint_metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata and a module-level parameter audit."""

        self._assert_parameter_policy()
        all_parameters = tuple(self.parameters())
        total = sum(parameter.numel() for parameter in all_parameters)
        trainable = sum(
            parameter.numel()
            for parameter in all_parameters
            if parameter.requires_grad
        )
        adapter_parameters = tuple(
            parameter
            for name, parameter in (
                self.backbone.network.image_encoder.named_parameters()
            )
            if "Adapter" in name
        )
        adapter_total = sum(parameter.numel() for parameter in adapter_parameters)
        adapter_trainable = sum(
            parameter.numel()
            for parameter in adapter_parameters
            if parameter.requires_grad
        )
        groups: dict[str, dict[str, int]] = {
            "sam_image_encoder": _parameter_counts(
                self.backbone.network.image_encoder
            ),
            "sam_adapters": {
                "total": adapter_total,
                "trainable": adapter_trainable,
                "frozen": adapter_total - adapter_trainable,
            },
            "image_descriptor": _parameter_counts(self.image_descriptor),
            "shape_teacher": _parameter_counts(self.shape_teacher),
            "posterior_head": _parameter_counts(self.posterior_head),
            "prior_head": _parameter_counts(self.prior_head),
            "pre_moe_context_adapter": _parameter_counts(
                getattr(self, "pre_moe_context_adapter", None)
            ),
            "router": _parameter_counts(getattr(self, "router", None)),
            "experts": _parameter_counts(
                getattr(getattr(self, "enhancement", None), "experts", None)
            ),
            "layer_scorer": _parameter_counts(
                getattr(getattr(self, "enhancement", None), "layer_scorer", None)
            ),
            "enhancement_neck": _parameter_counts(
                getattr(self, "enhancement_neck", None)
            ),
            "post_latent_conditioning": _parameter_counts(
                self.post_latent_conditioning
            ),
            "lpeg": _parameter_counts(self.backbone.network.prompt_encoder.lpeg),
            "sam_mask_decoder": _parameter_counts(
                self.backbone.network.mask_decoder
            ),
        }
        return {
            "variant": self.variant,
            "latent_dim": self.latent_dim,
            "use_moe": self.use_moe,
            "use_pre_moe_latent": self.use_pre_moe_latent,
            "use_post_moe_latent": self.use_post_moe_latent,
            "routing_shared_by_construction": (
                self.use_moe and not self.use_pre_moe_latent
            ),
            "parameter_counts": {
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
                "groups": groups,
            },
        }


__all__ = [
    "LatentConditioningModel",
    "LatentConditioningOutput",
    "LatentConditioningState",
]
