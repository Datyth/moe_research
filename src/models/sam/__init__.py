"""Promptless, adapter-free SAM ViT-B segmentation baseline.

This is a supervised semantic-segmentation baseline rather than SAM's original
prompted zero-shot interface: the image encoder is vanilla SAM, prompts are
empty, and the task-specific mask decoder is fine-tuned to emit exactly the
number of channels required by the dataset.
"""

from __future__ import annotations

from torch import Tensor

from ..base import BaseSegmentationModel, SegmentationOutput
from ..registry import register_model
from .build import build_promptless_sam_vit_b


@register_model("sam")
class PromptlessSamModel(BaseSegmentationModel):
    """Plain SAM ViT-B with no Adapter, MoE, LPEG, point, box, or mask prompt."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        task: str = "binary",
        image_size: int = 256,
        checkpoint: str | None = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
        )
        if in_channels != 3:
            raise ValueError(
                "PromptlessSamModel only supports in_channels=3 "
                "(RGB SAM backbone)."
            )
        if image_size <= 0 or image_size % 16 != 0:
            raise ValueError("PromptlessSamModel image_size must be divisible by 16.")

        self.image_size = image_size
        self.network = build_promptless_sam_vit_b(
            image_size=image_size,
            num_classes=num_classes,
            checkpoint=checkpoint,
        )

        if freeze_image_encoder:
            self.network.image_encoder.requires_grad_(False)
        if freeze_prompt_encoder:
            self.network.prompt_encoder.requires_grad_(False)

    def forward(self, images: Tensor, **kwargs) -> SegmentationOutput:
        outputs = self.network(
            images,
            multimask_output=True,
            image_size=self.image_size,
        )
        return SegmentationOutput(
            logits=outputs["masks"],
            diagnostics={"iou_predictions": outputs["iou_predictions"]},
        )


__all__ = ["PromptlessSamModel"]
