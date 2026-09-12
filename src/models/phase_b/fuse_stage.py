"""Phase B up to the fuse stage.

Wires the two branches the proposal's Phase B diagram shows converging:

    I --> SAM ViT-B (MoE-SAM) --> F^(l), l in {3,6,9,12} --> h_I  \\
                                                                   >--> h_q
    M --> Shape Teacher G_M (frozen, from Phase A)          --> h_M /

The segmentation head of the MoE-SAM branch is kept and still produces mask
logits, so this module is a complete trainable segmentation model that
additionally exposes h_q. The posterior q(z | I, M), Top-K routing and
hierarchical enhancement live in the later stages of Phase B
(``phase_b_router`` and ``phase_b_moe``) and are not built here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor

from ..base import BaseSegmentationModel, SegmentationOutput
from ..esam import EsamModel
from ..registry import register_model
from .fusion import PrivilegedFusion
from .image_descriptor import DEFAULT_LEVELS, MultiLevelImageDescriptor
from .shape_teacher import ShapeTeacher, build_shape_teacher, load_shape_teacher


@dataclass
class FuseStageOutput:
    """Everything the fuse stage produces, beyond the mask logits."""

    fused: Tensor
    image_descriptor: Tensor
    shape_latent: Tensor
    level_weights: Tensor
    level_tokens: tuple[Tensor, ...]


@register_model("phase_b_fuse")
class PhaseBFuseStage(BaseSegmentationModel):
    """MoE-SAM segmentation plus the privileged fuse stage h_q = [h_I ; h_M].

    The ground-truth mask reaches `forward` through the `masks` keyword. It is
    privileged information, so `forward` must tolerate its absence: without it
    the model still segments and simply reports no h_q, which is what the
    Phase-C prior path and inference will do.
    """

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
        levels: tuple[int, ...] = DEFAULT_LEVELS,
        scoring_hidden_dim: int = 64,
        shape_teacher_checkpoint: str | Path | None = None,
        shape_teacher: dict[str, Any] | None = None,
        freeze_shape_teacher: bool = True,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, task=task)

        self.backbone = EsamModel(
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
        )
        self.image_size = image_size

        self.image_descriptor = MultiLevelImageDescriptor(
            embed_dim=self.backbone.network.image_encoder.embed_dim,
            descriptor_dim=descriptor_dim,
            levels=tuple(levels),
            scoring_hidden_dim=scoring_hidden_dim,
        )

        # A randomly initialized teacher would make h_M pure noise while every
        # shape check downstream still "passes", so an explicit checkpoint is
        # the normal path and the untrained fallback is opt-in.
        if shape_teacher_checkpoint is not None:
            self.shape_teacher: ShapeTeacher = load_shape_teacher(
                shape_teacher_checkpoint,
                freeze=freeze_shape_teacher,
            )
        else:
            self.shape_teacher = build_shape_teacher(shape_teacher or {})
            self.shape_teacher.frozen = bool(freeze_shape_teacher)
            if freeze_shape_teacher:
                for parameter in self.shape_teacher.parameters():
                    parameter.requires_grad = False

        shape_latent_dim = self.shape_teacher.projector.latent_projection.out_features
        self.fusion = PrivilegedFusion(
            descriptor_dim=descriptor_dim,
            shape_latent_dim=shape_latent_dim,
        )

    @property
    def fused_dim(self) -> int:
        """Width of h_q, for sizing the posterior heads built on top."""

        return self.fusion.output_dim

    def forward(
        self,
        images: Tensor,
        *,
        masks: Tensor | None = None,
        **kwargs,
    ) -> SegmentationOutput:
        outputs = self.backbone.network(
            images,
            multimask_output=True,
            image_size=self.image_size,
            **kwargs,
        )
        block_outputs = outputs["block_outputs"]
        descriptor = self.image_descriptor(block_outputs)

        diagnostics: dict[str, Any] = {
            "iou_predictions": outputs["iou_predictions"],
            "moe_expert_indices": outputs["indices"],
            "image_descriptor": descriptor.descriptor,
            "level_weights": descriptor.level_weights,
        }

        if masks is not None:
            shape_latent = self.shape_teacher(masks)
            fusion = self.fusion(descriptor.descriptor, shape_latent)
            diagnostics["fuse_stage"] = FuseStageOutput(
                fused=fusion.fused,
                image_descriptor=fusion.image_descriptor,
                shape_latent=fusion.shape_latent,
                level_weights=descriptor.level_weights,
                level_tokens=descriptor.level_tokens,
            )

        return SegmentationOutput(logits=outputs["masks"], diagnostics=diagnostics)
