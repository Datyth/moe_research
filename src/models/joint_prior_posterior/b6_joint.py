"""Joint training of the B6 privileged posterior and image-only prior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from src.models.base import SegmentationOutput
from src.models.phase_b.image_descriptor import MultiLevelImageDescriptorOutput
from src.models.phase_b.posterior import DiagonalGaussian, GaussianParameterHead
from src.models.phase_b.router import RoutingOutput, load_balance_loss
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


@dataclass
class _EncodedImageState:
    image_embeddings: Tensor
    descriptor: MultiLevelImageDescriptorOutput


@dataclass
class _DecodedRoutingState:
    logits: Tensor
    iou_predictions: Tensor


@dataclass
class JointPriorPosteriorState:
    """Posterior/prior state computed from one frozen SAM encoder pass."""

    posterior: DiagonalGaussian
    prior: DiagonalGaussian
    posterior_routing: RoutingOutput
    prior_routing: RoutingOutput
    posterior_balance: Tensor
    posterior_logits: Tensor | None = None
    prior_logits: Tensor | None = None
    posterior_iou_predictions: Tensor | None = None
    prior_iou_predictions: Tensor | None = None


@dataclass
class JointPriorPosteriorOutput(SegmentationOutput):
    """Segmentation output whose public logits always use the image prior."""

    prior_logits: Tensor | None = None
    posterior_logits: Tensor | None = None
    joint_state: JointPriorPosteriorState | None = None


@register_model("joint_prior_posterior_b6")
class JointPriorPosteriorB6(PhaseBB6Hierarchical):
    """Train B6 and an image prior jointly while freezing privileged sources."""

    evaluation_mode = IMAGE_ONLY

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        latent_dim: int = 64,
        std_floor: float = 1e-4,
        stochastic: bool = False,
        **kwargs: Any,
    ) -> None:
        if stochastic:
            raise ValueError(
                "Joint prior-posterior routing is deterministic; "
                "model.stochastic must be false."
            )
        super().__init__(
            descriptor_dim=descriptor_dim,
            latent_dim=latent_dim,
            std_floor=std_floor,
            stochastic=False,
            **kwargs,
        )
        self.prior_head = GaussianParameterHead(
            in_dim=descriptor_dim,
            latent_dim=latent_dim,
            std_floor=std_floor,
        )
        self._freeze_privileged_sources()

    def _freeze_privileged_sources(self) -> None:
        """Freeze every SAM image-encoder and Phase-A teacher parameter."""

        for parameter in self.backbone.network.image_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.shape_teacher.parameters():
            parameter.requires_grad_(False)
        self.backbone.network.image_encoder.eval()
        self.shape_teacher.eval()

    def train(self, mode: bool = True) -> "JointPriorPosteriorB6":
        """Train joint modules but keep both frozen feature sources in eval."""

        super().train(mode)
        self.backbone.network.image_encoder.eval()
        self.shape_teacher.eval()
        return self

    def _encode_images(self, images: Tensor) -> _EncodedImageState:
        # The SAM encoder is a fixed feature extractor. The descriptor stays
        # outside this context so its projections and level scorer receive
        # gradients from both the posterior segmentation path and latent KL.
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
    ) -> tuple[DiagonalGaussian, RoutingOutput, Tensor]:
        with torch.no_grad():
            shape_latent = self.shape_teacher(masks)
        fused = self.fusion(
            encoded.descriptor.descriptor,
            shape_latent,
        ).fused
        posterior = self.conditioner.posterior_head(fused)
        posterior_routing = self.conditioner.router(posterior.mean)
        posterior_balance = load_balance_loss(
            posterior_routing.dense_probs,
            posterior_routing.expert_indices,
            num_experts=self.conditioner.router.num_experts,
        )
        return posterior, posterior_routing, posterior_balance

    def _prior_from_encoded(
        self,
        encoded: _EncodedImageState,
    ) -> tuple[DiagonalGaussian, RoutingOutput]:
        prior = self.prior_head(encoded.descriptor.descriptor)
        prior_routing = self.conditioner.router(prior.mean)
        return prior, prior_routing

    def _decode_routing(
        self,
        encoded: _EncodedImageState,
        *,
        latent: Tensor,
        routing: RoutingOutput,
    ) -> _DecodedRoutingState:
        level_pools = torch.stack(
            [tokens.mean(dim=1) for tokens in encoded.descriptor.level_tokens],
            dim=1,
        )
        enhancement = self.enhancement(
            encoded.descriptor.level_tokens,
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
        logits, iou_predictions = decode_sam(
            self.backbone,
            enhanced_embeddings,
            image_size=self.image_size,
        )
        return _DecodedRoutingState(
            logits=logits,
            iou_predictions=iou_predictions,
        )

    def joint_forward(
        self,
        images: Tensor,
        masks: Tensor,
        *,
        decode_posterior: bool = True,
        decode_prior: bool = False,
    ) -> JointPriorPosteriorState:
        """Compute joint latent state and optionally decode either route."""

        if masks is None:
            raise ValueError("Joint posterior diagnostics require ground-truth masks.")
        encoded = self._encode_images(images)
        posterior, posterior_routing, posterior_balance = (
            self._posterior_from_encoded(encoded, masks)
        )
        prior, prior_routing = self._prior_from_encoded(encoded)

        posterior_decoded = (
            self._decode_routing(
                encoded,
                latent=posterior.mean,
                routing=posterior_routing,
            )
            if decode_posterior
            else None
        )
        prior_decoded = (
            self._decode_routing(
                encoded,
                latent=prior.mean,
                routing=prior_routing,
            )
            if decode_prior
            else None
        )
        return JointPriorPosteriorState(
            posterior=posterior,
            prior=prior,
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
    ) -> JointPriorPosteriorOutput:
        """Return deployable prior logits; privileged diagnostics are opt-in."""

        if include_posterior:
            if masks is None:
                raise ValueError(
                    "include_posterior=True requires ground-truth masks."
                )
            state = self.joint_forward(
                images,
                masks,
                decode_posterior=True,
                decode_prior=True,
            )
            if state.prior_logits is None or state.posterior_logits is None:
                raise RuntimeError("Joint diagnostic decoding produced no logits.")
            return JointPriorPosteriorOutput(
                logits=state.prior_logits,
                prior_logits=state.prior_logits,
                posterior_logits=state.posterior_logits,
                joint_state=state,
                diagnostics={
                    "joint_prior_posterior": state,
                    "prior_iou_predictions": state.prior_iou_predictions,
                    "posterior_iou_predictions": state.posterior_iou_predictions,
                },
            )

        # Deliberately ignore an incidental mask. The default deployment path
        # must never execute privileged Shape-Teacher or fusion computation.
        encoded = self._encode_images(images)
        prior, prior_routing = self._prior_from_encoded(encoded)
        decoded = self._decode_routing(
            encoded,
            latent=prior.mean,
            routing=prior_routing,
        )
        return JointPriorPosteriorOutput(
            logits=decoded.logits,
            prior_logits=decoded.logits,
            diagnostics={
                "joint_prior": {
                    "prior": prior,
                    "routing": prior_routing,
                },
                "prior_iou_predictions": decoded.iou_predictions,
            },
        )


__all__ = [
    "JointPriorPosteriorB6",
    "JointPriorPosteriorOutput",
    "JointPriorPosteriorState",
]
