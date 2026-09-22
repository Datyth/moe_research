"""Phase B full: routed expert enhancement injected into the SAM decoder.

Extends the router stage with the proposal's Section 1.3 wiring (Eqs. 45-56):

    X_hat^(l) = X^(l) + sum_{k in K_b} pi_{b,k} beta_{b,k,l} E_k(X^(l))
    gamma_{b,l} = sum_{k in K_b} pi_{b,k} beta_{b,k,l}
    Z_fused = sum_l gamma_{b,l} X_hat^(l)}                       (47-50)
    E_aux = Neck(Z_fused)                       (53)
    E_enh = E_SAM + E_aux                       (54)
    (prompt, mask) = D_SAM(E_enh)               (55-56)

The neck mirrors the vendored MoE-FEB ``neck5`` (Conv 1x1 768->256, LayerNorm,
Conv 3x3, LayerNorm): ``Z_fused`` arrives at 768 channels on the 16x16 patch
grid and must land at SAM's 256-channel image embedding before the residual
add. The residual-add injection point itself is the same one MoE-FEB already
uses (``image_embeddings + neck5(...)`` in ``sam_moe.py``), so this stage is a
second, shape-conditioned consumer of the pattern.

Because E_aux now perturbs the decoder input, this stage **breaks bit-identity
with E3 by design** — that was the point of the fuse/router stages, which
verified the plumbing without disturbing the baseline. Two things remain
untouched: the frozen SAM backbone (experts read X^(l) but nothing writes back
into the encoder), and the deployable inference path (no mask -> prior mean
mu_p routes; beta and the experts need only image-side inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..base import SegmentationOutput
from ..registry import register_model
from .phase_b_router import PhaseBRouterStage
from .moe_enhancement import HierarchicalMoEEnhancement
from .shape_conditioned_enhancement import ShapeConditionedFusionMoEEnhancement
from .ablations.common import (
    build_enhancement_neck,
    enhanced_level_norm,
    token_norm,
)


@dataclass
class EnhancementStageOutput:
    """Diagnostics for the enhancement stage."""

    layer_weights: Tensor
    """gamma for hierarchical or alpha for shape-conditioned, [B, L_levels]."""

    expert_layer_weights: Tensor | None
    """beta_{b,k,l} for hierarchical mode; otherwise None."""

    shape_fusion_layer_weights: Tensor | None
    """Shape-conditioned alpha [B, L_levels]; otherwise None."""

    aux_norm_ratio: Tensor
    """||E_aux|| / ||E_enh|| per sample, [B]; how much the enhancement
    contributes to the decoder input (0 would mean the experts did nothing)."""

    fused_token_norm: Tensor | None = None
    """Per-sample L2 norm of the representation entering fusion/neck."""

    enhanced_token_norm: Tensor | None = None
    """Per-sample enhanced-token norm when the mode exposes one."""


@register_model("phase_b_moe")
class PhaseBMoEStage(PhaseBRouterStage):
    """Router stage + routed expert enhancement + decoder injection."""

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
        levels: tuple[int, ...] = (3, 6, 9, 12),
        scoring_hidden_dim: int = 64,
        shape_teacher_checkpoint: str | Path | None = None,
        shape_teacher: dict[str, Any] | None = None,
        freeze_shape_teacher: bool = True,
        latent_dim: int = 64,
        num_experts: int = 4,
        active_experts: int = 2,
        std_floor: float = 1e-4,
        stochastic: bool = True,
        expert_hidden_ratio: int = 4,
        enhancement_mode: str = "hierarchical",
    ) -> None:
        if enhancement_mode not in ("hierarchical", "shape_conditioned"):
            raise ValueError(
                "enhancement_mode must be 'hierarchical' or "
                f"'shape_conditioned', got {enhancement_mode!r}."
            )
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
            image_size=image_size,
            checkpoint=checkpoint,
            use_moe=use_moe,
            use_lpeg=use_lpeg,
            moe_num_experts=moe_num_experts,
            moe_top_k_ratio=moe_top_k_ratio,
            freeze_backbone=freeze_backbone,
            descriptor_dim=descriptor_dim,
            levels=levels,
            scoring_hidden_dim=scoring_hidden_dim,
            shape_teacher_checkpoint=shape_teacher_checkpoint,
            shape_teacher=shape_teacher,
            freeze_shape_teacher=freeze_shape_teacher,
            latent_dim=latent_dim,
            num_experts=num_experts,
            active_experts=active_experts,
            std_floor=std_floor,
            stochastic=stochastic,
        )
        embed_dim = self.backbone.network.image_encoder.embed_dim
        num_levels = len(levels)
        self.enhancement_mode = enhancement_mode
        enhancement_class = (
            HierarchicalMoEEnhancement
            if enhancement_mode == "hierarchical"
            else ShapeConditionedFusionMoEEnhancement
        )
        self.enhancement = enhancement_class(
            embed_dim=embed_dim,
            num_experts=num_experts,
            num_levels=num_levels,
            latent_dim=latent_dim,
            expert_hidden_ratio=expert_hidden_ratio,
        )
        # Neck (Eq. 53): Z_fused [B, P, 768] -> spatial [B, 256, H_p, W_p]
        # to match SAM's image embedding. Mirrors neck5 in sam_moe.py.
        self.enhancement_neck = build_enhancement_neck(embed_dim, descriptor_dim)

    def forward(
        self,
        images: Tensor,
        *,
        masks: Tensor | None = None,
        **kwargs,
    ) -> SegmentationOutput:
        # 1. Backbone forward, but we need the image embeddings *before* the
        # decoder, so we call the vendored network in stages rather than
        # re-running the whole forward.
        network = self.backbone.network
        input_images = network.preprocess(images)
        image_embeddings, low_image_embeddings = network.image_encoder(input_images)
        block_outputs = low_image_embeddings

        # 2. MoE-FEB path (unchanged from the vendored forward).
        indices = None
        if network.use_moe:
            embedding_moe = torch.cat(
                [embed.unsqueeze(1) for embed in low_image_embeddings], dim=1
            ).permute(0, 1, 4, 2, 3).contiguous()
            embedding_moe = embedding_moe.permute(0, 1, 3, 4, 2).contiguous()
            bs, num_features, h, w, dim = embedding_moe.shape
            embedding_moe, indices = network.ExpertChoiceTokenMoE(
                embedding_moe.reshape(bs * num_features, h * w, dim)
            )
            embedding_moe = embedding_moe.reshape(bs, -1, dim)
            embedding_moe = network.attn(embedding_moe, embedding_moe, embedding_moe).reshape(
                bs, num_features, h, w, dim
            ).permute(0, 1, 4, 2, 3).contiguous()
            embedding_moe = embedding_moe.mean(1)
            image_embeddings = image_embeddings + network.neck5(embedding_moe)

        # 3. Descriptor / fuse / router — same as PhaseBRouterStage, but on
        # block_outputs captured above (the encoder already ran).
        descriptor = self.image_descriptor(block_outputs)
        diagnostics: dict[str, Any] = {
            "iou_predictions": None,
            "moe_expert_indices": indices,
            "image_descriptor": descriptor.descriptor,
            "level_weights": descriptor.level_weights,
            "level_ids": self.image_descriptor.levels,
        }
        fuse_stage = None
        if masks is not None:
            shape_latent = self.shape_teacher(masks)
            fuse_stage = self.fusion(descriptor.descriptor, shape_latent)
            diagnostics["fuse_stage"] = fuse_stage

        # Router decision
        fused = fuse_stage.fused if (masks is not None and fuse_stage is not None) else None
        router_stage = self._head(
            descriptor.descriptor,
            fused,
            sample=self.training and self.stochastic,
        )
        diagnostics["phase_b_router"] = router_stage

        # 4. Routed enhancement. The default hierarchical path is unchanged;
        # the controlled ablation fuses levels before sparse expert dispatch.
        level_tokens = descriptor.level_tokens  # raw X^(l), [B, P, 768]
        if self.enhancement_mode == "hierarchical":
            # v^(l) = GAP(X^(l)): raw pooling, independent of the P_l-projected
            # u^(l) the descriptor uses — g_layer sees unpooled geometry.
            level_pools = torch.stack(
                [tokens.mean(dim=1) for tokens in level_tokens], dim=1
            )  # [B, L_levels, 768]
            enhancement_out = self.enhancement(
                level_tokens,
                level_pools,
                router_stage.latent,
                router_stage.routing.routing_probs,
                router_stage.routing.expert_indices,
            )
            tokens_for_neck = enhancement_out.fused_tokens
            expert_layer_weights = enhancement_out.expert_layer_weights
            shape_fusion_layer_weights = None
            enhanced_tokens_norm = enhanced_level_norm(
                enhancement_out.enhanced_tokens
            )
        else:
            enhancement_out = self.enhancement(
                level_tokens,
                router_stage.latent,
                router_stage.routing.routing_probs,
                router_stage.routing.expert_indices,
            )
            tokens_for_neck = enhancement_out.enhanced_fused_tokens
            expert_layer_weights = None
            shape_fusion_layer_weights = enhancement_out.layer_weights
            enhanced_tokens_norm = token_norm(
                enhancement_out.enhanced_fused_tokens
            )

        # 5. Neck + residual injection (Eqs. 53-54).
        batch_size = images.shape[0]
        patch_grid = int(tokens_for_neck.shape[1] ** 0.5)
        spatial = tokens_for_neck.transpose(1, 2).reshape(
            batch_size, self.enhancement.embed_dim, patch_grid, patch_grid
        )
        e_aux = self.enhancement_neck(spatial)  # [B, 256, H_p, W_p]
        e_enh = image_embeddings + e_aux  # Eq. 54

        # 6. Decoder with the enhanced embedding (Eqs. 55-56): LPEG prompt
        # from E_enh, then the mask decoder.
        sparse_embeddings, dense_embeddings = network.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            image_embedding=e_enh if network.use_lpeg else None,
            batch_size=batch_size,
        )
        low_res_masks, iou_predictions = network.mask_decoder(
            image_embeddings=e_enh,
            image_pe=network.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        masks_out = network.postprocess_masks(
            low_res_masks,
            input_size=(self.image_size, self.image_size),
            original_size=(self.image_size, self.image_size),
        )

        diagnostics["iou_predictions"] = iou_predictions
        diagnostics["phase_b_moe"] = EnhancementStageOutput(
            layer_weights=enhancement_out.layer_weights,
            expert_layer_weights=expert_layer_weights,
            shape_fusion_layer_weights=shape_fusion_layer_weights,
            aux_norm_ratio=(
                e_aux.flatten(1).norm(dim=1) / e_enh.flatten(1).norm(dim=1)
            ).detach(),
            fused_token_norm=token_norm(enhancement_out.fused_tokens),
            enhanced_token_norm=enhanced_tokens_norm,
        )

        return SegmentationOutput(logits=masks_out, diagnostics=diagnostics)
