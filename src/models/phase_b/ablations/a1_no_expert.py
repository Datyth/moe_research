"""A1: multi-level residual enhancement without any expert or router."""

from __future__ import annotations

from torch import Tensor

from ...base import BaseSegmentationModel, SegmentationOutput
from ...esam import EsamModel
from ...registry import register_model
from ..image_descriptor import DEFAULT_LEVELS, MultiLevelImageDescriptor
from .common import (
    AblationEnhancementOutput,
    build_enhancement_neck,
    fuse_level_tokens,
    inject_enhancement,
    make_segmentation_output,
    run_sam_encoder,
    token_norm,
)


@register_model("phase_b_no_moe")
@register_model("phase_b_a1_no_expert")
class PhaseBNoMoEStage(BaseSegmentationModel):
    """A1 model; the historical class/name remains a compatibility alias."""

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
            raise ValueError("phase_b_a1_no_expert requires use_moe=False.")
        if not use_lpeg:
            raise ValueError("phase_b_a1_no_expert requires use_lpeg=True.")
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
            freeze_backbone=freeze_backbone,
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

    # Preserve the helper used by the existing public tests/callers.
    fuse_level_tokens = staticmethod(fuse_level_tokens)

    def forward(self, images: Tensor, **kwargs) -> SegmentationOutput:
        image_embeddings, block_outputs = run_sam_encoder(self.backbone, images)
        descriptor = self.image_descriptor(block_outputs)
        fused_tokens = fuse_level_tokens(
            descriptor.level_tokens,
            descriptor.level_weights,
        )
        e_aux, e_enh = inject_enhancement(
            image_embeddings,
            fused_tokens,
            self.enhancement_neck,
            embed_dim=self.image_descriptor.embed_dim,
        )
        aux_ratio = (
            e_aux.flatten(1).norm(dim=1) / e_enh.flatten(1).norm(dim=1)
        ).detach()
        stage = AblationEnhancementOutput(
            layer_weights=descriptor.level_weights,
            aux_norm_ratio=aux_ratio,
            fused_token_norm=token_norm(fused_tokens),
        )
        diagnostics = {
            "image_descriptor": descriptor.descriptor,
            "level_weights": descriptor.level_weights,
            "level_ids": self.image_descriptor.levels,
            "enhancement_aux_ratio": aux_ratio,
            "fused_token_norm": stage.fused_token_norm,
            "level_weight_entropy": (
                -(
                    descriptor.level_weights.clamp_min(1e-9).log()
                    * descriptor.level_weights
                ).sum(dim=1)
            ).detach(),
            "phase_b_ablation": stage,
        }
        return make_segmentation_output(
            backbone=self.backbone,
            enhanced_embeddings=e_enh,
            image_size=self.image_size,
            diagnostics=diagnostics,
        )


PhaseBA1NoExpertStage = PhaseBNoMoEStage
