"""Shared encoder-to-decoder pipeline for Phase-B build-up B1-B6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ....base import BaseSegmentationModel, SegmentationOutput
from ....esam import EsamModel
from ....esam._vendor.common import LayerNorm2d
from ...fusion import PrivilegedFusion
from ...image_descriptor import (
    DEFAULT_LEVELS,
    MultiLevelImageDescriptor,
    MultiLevelImageDescriptorOutput,
)
from ...moe_enhancement import HierarchicalMoEEnhancement
from ...shape_teacher import ShapeTeacher, load_shape_teacher
from .modules import (
    ConditioningOutput,
    DirectConditioner,
    GaussianConditioner,
    GlobalEnhancementOutput,
    RandomTopKRouter,
    SparseMoEEnhancer,
)
from ...router import TopKRouter


IMAGE_ONLY = "image_only"
POSTERIOR_ORACLE = "posterior_oracle"
SUPPORTED_EVALUATION_MODES = (IMAGE_ONLY, POSTERIOR_ORACLE)


@dataclass
class BuildUpVariantOutput:
    """Variant output consumed by the one shared neck/decoder path."""

    fused_tokens: Tensor
    fusion_weights: Tensor
    routing: ConditioningOutput | None = None
    expert_layer_weights: Tensor | None = None


@dataclass
class BuildUpStageOutput:
    """Stable diagnostics emitted by every B1-B6 model."""

    evaluation_mode: str
    descriptor_level_weights: Tensor
    fusion_weights: Tensor
    aux_norm_ratio: Tensor
    fused_token_norm: Tensor
    routing: ConditioningOutput | None = None
    expert_layer_weights: Tensor | None = None


def build_enhancement_neck(embed_dim: int, output_dim: int) -> nn.Sequential:
    """Map raw ViT tokens onto SAM's decoder embedding width."""

    return nn.Sequential(
        nn.Conv2d(embed_dim, output_dim, kernel_size=1, bias=False),
        LayerNorm2d(output_dim),
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(output_dim),
    )


def run_sam_encoder(
    backbone: EsamModel,
    images: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    """Run the frozen SAM image encoder exactly once."""

    network = backbone.network
    input_images = network.preprocess(images)
    return network.image_encoder(input_images)


def tokens_to_spatial(
    tokens: Tensor,
    image_embeddings: Tensor,
    *,
    embed_dim: int,
) -> Tensor:
    """Reshape [B,P,C] using the actual SAM embedding grid."""

    if tokens.ndim != 3 or tokens.shape[2] != embed_dim:
        raise ValueError(
            f"tokens must be [B, P, {embed_dim}], got {tuple(tokens.shape)}."
        )
    if image_embeddings.ndim != 4:
        raise ValueError("image_embeddings must be [B, C, H_p, W_p].")
    batch_size, _, height, width = image_embeddings.shape
    if tokens.shape[:2] != (batch_size, height * width):
        raise ValueError(
            "Token grid must match SAM image embeddings: "
            f"tokens={tuple(tokens.shape)}, embeddings={tuple(image_embeddings.shape)}."
        )
    return tokens.transpose(1, 2).reshape(batch_size, embed_dim, height, width)


def inject_enhancement(
    image_embeddings: Tensor,
    fused_tokens: Tensor,
    neck: nn.Module,
    *,
    embed_dim: int,
) -> tuple[Tensor, Tensor]:
    """Apply the enhancement neck and residual add."""

    spatial = tokens_to_spatial(
        fused_tokens,
        image_embeddings,
        embed_dim=embed_dim,
    )
    auxiliary = neck(spatial)
    if auxiliary.shape != image_embeddings.shape:
        raise ValueError(
            "Enhancement neck output must match SAM embeddings, got "
            f"{tuple(auxiliary.shape)} and {tuple(image_embeddings.shape)}."
        )
    return auxiliary, image_embeddings + auxiliary


def decode_sam(
    backbone: EsamModel,
    enhanced_embeddings: Tensor,
    *,
    image_size: int,
) -> tuple[Tensor, Tensor]:
    """Run the unchanged LPEG prompt and SAM mask decoder path."""

    network = backbone.network
    batch_size = enhanced_embeddings.shape[0]
    sparse_embeddings, dense_embeddings = network.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
        image_embedding=enhanced_embeddings if network.use_lpeg else None,
        batch_size=batch_size,
    )
    low_res_masks, iou_predictions = network.mask_decoder(
        image_embeddings=enhanced_embeddings,
        image_pe=network.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    masks = network.postprocess_masks(
        low_res_masks,
        input_size=(image_size, image_size),
        original_size=(image_size, image_size),
    )
    return masks, iou_predictions


def token_norm(tokens: Tensor) -> Tensor:
    return tokens.flatten(1).norm(dim=1).detach()


class BuildUpBase(BaseSegmentationModel):
    """One shared SAM pipeline; subclasses only implement token production."""

    evaluation_mode = IMAGE_ONLY
    expert_transformations_per_sample = 0

    def __init__(
        self,
        *,
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
    ) -> None:
        if use_moe:
            raise ValueError("Phase-B build-up requires legacy use_moe=False.")
        if not use_lpeg:
            raise ValueError("Phase-B build-up requires use_lpeg=True.")
        if not freeze_backbone:
            raise ValueError("Phase-B build-up requires freeze_backbone=True.")
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
        )
        self.backbone = EsamModel(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
            image_size=image_size,
            checkpoint=checkpoint,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        self.image_size = image_size
        embed_dim = self.backbone.network.image_encoder.embed_dim
        self.image_descriptor = MultiLevelImageDescriptor(
            embed_dim=embed_dim,
            descriptor_dim=descriptor_dim,
            levels=tuple(levels),
            scoring_hidden_dim=scoring_hidden_dim,
        )
        self.enhancement_neck = build_enhancement_neck(embed_dim, descriptor_dim)
        self.embed_dim = embed_dim
        self.descriptor_dim = descriptor_dim
        self.num_levels = len(levels)

    def _variant_forward(
        self,
        descriptor: MultiLevelImageDescriptorOutput,
        masks: Tensor | None,
    ) -> BuildUpVariantOutput:
        raise NotImplementedError

    def forward(
        self,
        images: Tensor,
        *,
        masks: Tensor | None = None,
        **kwargs: Any,
    ) -> SegmentationOutput:
        if self.evaluation_mode == POSTERIOR_ORACLE and masks is None:
            raise ValueError(
                f"{type(self).__name__} is a privileged posterior/oracle model "
                "and requires masks; image-only fallback is not implemented."
            )

        image_embeddings, block_outputs = run_sam_encoder(self.backbone, images)
        descriptor = self.image_descriptor(block_outputs)
        variant = self._variant_forward(descriptor, masks)
        auxiliary, enhanced_embeddings = inject_enhancement(
            image_embeddings,
            variant.fused_tokens,
            self.enhancement_neck,
            embed_dim=self.embed_dim,
        )
        logits, iou_predictions = decode_sam(
            self.backbone,
            enhanced_embeddings,
            image_size=self.image_size,
        )
        denominator = enhanced_embeddings.flatten(1).norm(dim=1).clamp_min(1e-12)
        stage = BuildUpStageOutput(
            evaluation_mode=self.evaluation_mode,
            descriptor_level_weights=descriptor.level_weights,
            fusion_weights=variant.fusion_weights,
            aux_norm_ratio=(
                auxiliary.flatten(1).norm(dim=1) / denominator
            ).detach(),
            fused_token_norm=token_norm(variant.fused_tokens),
            routing=variant.routing,
            expert_layer_weights=variant.expert_layer_weights,
        )
        return SegmentationOutput(
            logits=logits,
            diagnostics={
                "iou_predictions": iou_predictions,
                "moe_expert_indices": None,
                "image_descriptor": descriptor.descriptor,
                "level_weights": descriptor.level_weights,
                "level_ids": self.image_descriptor.levels,
                "phase_b_build_up": stage,
            },
        )


class RoutedBuildUpBase(BuildUpBase):
    """Shared direct/Gaussian routing and global/hierarchical enhancement."""

    def __init__(
        self,
        *,
        conditioning_dim: int,
        conditioner_kind: str,
        routing_source: str,
        router_mode: str = "learned",
        latent_dim: int = 64,
        num_experts: int = 4,
        active_experts: int = 2,
        expert_hidden_ratio: int = 4,
        std_floor: float = 1e-4,
        stochastic: bool = True,
        hierarchical: bool = False,
        **kwargs: Any,
    ) -> None:
        if router_mode not in ("learned", "random"):
            raise ValueError("router_mode must be 'learned' or 'random'.")
        if conditioner_kind not in ("direct", "gaussian"):
            raise ValueError("conditioner_kind must be 'direct' or 'gaussian'.")
        if conditioner_kind == "gaussian" and router_mode != "learned":
            raise ValueError("Gaussian build-up routing requires router_mode='learned'.")
        super().__init__(**kwargs)
        if conditioner_kind == "direct":
            router: TopKRouter | RandomTopKRouter
            if router_mode == "random":
                router = RandomTopKRouter(
                    latent_dim=conditioning_dim,
                    num_experts=num_experts,
                    active_experts=active_experts,
                )
            else:
                router = TopKRouter(
                    latent_dim=conditioning_dim,
                    num_experts=num_experts,
                    active_experts=active_experts,
                )
            self.conditioner: DirectConditioner | GaussianConditioner = (
                DirectConditioner(router=router, source=routing_source)
            )
        else:
            self.conditioner = GaussianConditioner(
                in_dim=conditioning_dim,
                latent_dim=latent_dim,
                num_experts=num_experts,
                active_experts=active_experts,
                std_floor=std_floor,
                stochastic=stochastic,
            )

        self.hierarchical = bool(hierarchical)
        effective_latent_dim = latent_dim if hierarchical else 0
        if self.hierarchical:
            self.enhancement: SparseMoEEnhancer | HierarchicalMoEEnhancement = (
                HierarchicalMoEEnhancement(
                    embed_dim=self.embed_dim,
                    num_experts=num_experts,
                    num_levels=self.num_levels,
                    latent_dim=effective_latent_dim,
                    expert_hidden_ratio=expert_hidden_ratio,
                )
            )
        else:
            self.enhancement = SparseMoEEnhancer(
                embed_dim=self.embed_dim,
                num_levels=self.num_levels,
                num_experts=num_experts,
                expert_hidden_ratio=expert_hidden_ratio,
            )
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.router_mode = router_mode
        self.expert_transformations_per_sample = self.num_levels * active_experts

    @property
    def router(self) -> TopKRouter | RandomTopKRouter:
        return self.conditioner.router

    def _conditioning_descriptor(
        self,
        descriptor: MultiLevelImageDescriptorOutput,
        masks: Tensor | None,
    ) -> Tensor:
        return descriptor.descriptor

    def _variant_forward(
        self,
        descriptor: MultiLevelImageDescriptorOutput,
        masks: Tensor | None,
    ) -> BuildUpVariantOutput:
        conditioning_descriptor = self._conditioning_descriptor(descriptor, masks)
        routing = self.conditioner(conditioning_descriptor)
        if self.hierarchical:
            level_pools = torch.stack(
                [tokens.mean(dim=1) for tokens in descriptor.level_tokens],
                dim=1,
            )
            enhancement = self.enhancement(
                descriptor.level_tokens,
                level_pools,
                routing.latent,
                routing.routing.routing_probs,
                routing.routing.expert_indices,
            )
            return BuildUpVariantOutput(
                fused_tokens=enhancement.fused_tokens,
                fusion_weights=enhancement.layer_weights,
                routing=routing,
                expert_layer_weights=enhancement.expert_layer_weights,
            )

        enhancement = self.enhancement(
            descriptor.level_tokens,
            descriptor.level_weights,
            routing.routing.routing_probs,
            routing.routing.expert_indices,
        )
        return BuildUpVariantOutput(
            fused_tokens=enhancement.fused_tokens,
            fusion_weights=descriptor.level_weights,
            routing=routing,
        )


class PrivilegedRoutedBuildUpBase(RoutedBuildUpBase):
    """Routed base whose conditioning descriptor is privileged h_q."""

    evaluation_mode = POSTERIOR_ORACLE

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        shape_teacher_checkpoint: str | Path | None = None,
        freeze_shape_teacher: bool = True,
        conditioner_kind: str,
        **kwargs: Any,
    ) -> None:
        if shape_teacher_checkpoint is None:
            raise ValueError(
                "Privileged build-up models require shape_teacher_checkpoint."
            )
        if not freeze_shape_teacher:
            raise ValueError("Build-up Shape Teacher must remain frozen.")
        teacher = load_shape_teacher(
            shape_teacher_checkpoint,
            freeze=True,
        )
        shape_latent_dim = teacher.projector.latent_projection.out_features
        super().__init__(
            descriptor_dim=descriptor_dim,
            conditioning_dim=descriptor_dim + shape_latent_dim,
            conditioner_kind=conditioner_kind,
            **kwargs,
        )
        self.shape_teacher: ShapeTeacher = teacher
        self.fusion = PrivilegedFusion(
            descriptor_dim=descriptor_dim,
            shape_latent_dim=shape_latent_dim,
        )

    def _conditioning_descriptor(
        self,
        descriptor: MultiLevelImageDescriptorOutput,
        masks: Tensor | None,
    ) -> Tensor:
        if masks is None:
            raise ValueError("Privileged build-up routing requires masks.")
        shape_latent = self.shape_teacher(masks)
        return self.fusion(descriptor.descriptor, shape_latent).fused
