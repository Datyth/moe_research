"""A2: privileged fused descriptor routed directly, without Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ...base import SegmentationOutput
from ...registry import register_model
from ..fuse_stage import PhaseBFuseStage
from ..image_descriptor import DEFAULT_LEVELS
from ..moe_enhancement import HierarchicalMoEEnhancement
from ..phase_b_moe import EnhancementStageOutput
from ..router import RoutingOutput, TopKRouter, load_balance_loss
from .common import (
    build_enhancement_neck,
    enhanced_level_norm,
    inject_enhancement,
    make_segmentation_output,
    run_sam_encoder,
    token_norm,
)


@dataclass
class DirectFuseRouterStageOutput:
    """Router diagnostic compatible with PhaseBMoETask without Gaussian state."""

    source: str
    latent: Tensor
    routing: RoutingOutput
    balance: Tensor
    prior: None = None
    posterior: None = None
    latent_kl: None = None


@register_model("phase_b_a2_direct_fuse")
class PhaseBA2DirectFuseStage(PhaseBFuseStage):
    """Route and condition hierarchical experts directly with h_q [B, 512]."""

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
        shape_teacher_checkpoint: str | Path | None = None,
        shape_teacher: dict[str, Any] | None = None,
        freeze_shape_teacher: bool = True,
        num_experts: int = 4,
        active_experts: int = 2,
        expert_hidden_ratio: int = 4,
    ) -> None:
        if use_moe:
            raise ValueError("phase_b_a2_direct_fuse requires use_moe=False.")
        if not use_lpeg:
            raise ValueError("phase_b_a2_direct_fuse requires use_lpeg=True.")
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
            image_size=image_size,
            checkpoint=checkpoint,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=freeze_backbone,
            descriptor_dim=descriptor_dim,
            levels=levels,
            scoring_hidden_dim=scoring_hidden_dim,
            shape_teacher_checkpoint=shape_teacher_checkpoint,
            shape_teacher=shape_teacher,
            freeze_shape_teacher=freeze_shape_teacher,
        )
        direct_dim = self.fused_dim
        embed_dim = self.backbone.network.image_encoder.embed_dim
        self.router = TopKRouter(
            latent_dim=direct_dim,
            num_experts=num_experts,
            active_experts=active_experts,
        )
        self.enhancement = HierarchicalMoEEnhancement(
            embed_dim=embed_dim,
            num_experts=num_experts,
            num_levels=len(levels),
            latent_dim=direct_dim,
            expert_hidden_ratio=expert_hidden_ratio,
        )
        self.enhancement_neck = build_enhancement_neck(embed_dim, descriptor_dim)

    def forward(
        self,
        images: Tensor,
        *,
        masks: Tensor | None = None,
        **kwargs,
    ) -> SegmentationOutput:
        if masks is None:
            raise ValueError(
                "phase_b_a2_direct_fuse is a privileged/oracle ablation and "
                "requires masks to construct h_q; no image-only fallback exists."
            )

        image_embeddings, block_outputs = run_sam_encoder(self.backbone, images)
        descriptor = self.image_descriptor(block_outputs)
        shape_latent = self.shape_teacher(masks)
        fusion = self.fusion(descriptor.descriptor, shape_latent)
        h_q = fusion.fused
        routing = self.router(h_q)
        balance = load_balance_loss(
            routing.dense_probs,
            routing.expert_indices,
            num_experts=self.router.num_experts,
        )
        router_stage = DirectFuseRouterStageOutput(
            source="direct_fuse",
            latent=h_q,
            routing=routing,
            balance=balance,
        )

        level_tokens = descriptor.level_tokens
        level_pools = torch.stack(
            [tokens.mean(dim=1) for tokens in level_tokens], dim=1
        )
        enhancement = self.enhancement(
            level_tokens,
            level_pools,
            h_q,
            routing.routing_probs,
            routing.expert_indices,
        )
        e_aux, e_enh = inject_enhancement(
            image_embeddings,
            enhancement.fused_tokens,
            self.enhancement_neck,
            embed_dim=self.enhancement.embed_dim,
        )
        aux_ratio = (
            e_aux.flatten(1).norm(dim=1) / e_enh.flatten(1).norm(dim=1)
        ).detach()
        enhancement_stage = EnhancementStageOutput(
            layer_weights=enhancement.layer_weights,
            expert_layer_weights=enhancement.expert_layer_weights,
            shape_fusion_layer_weights=None,
            aux_norm_ratio=aux_ratio,
            fused_token_norm=token_norm(enhancement.fused_tokens),
            enhanced_token_norm=enhanced_level_norm(enhancement.enhanced_tokens),
        )
        diagnostics = {
            "image_descriptor": descriptor.descriptor,
            "level_weights": descriptor.level_weights,
            "level_ids": self.image_descriptor.levels,
            "fuse_stage": fusion,
            "phase_b_router": router_stage,
            "phase_b_moe": enhancement_stage,
            "moe_expert_indices": None,
        }
        return make_segmentation_output(
            backbone=self.backbone,
            enhanced_embeddings=e_enh,
            image_size=self.image_size,
            diagnostics=diagnostics,
        )
