"""E-SAM: SAM ViT backbone + Adapter fine-tuning + Sparse MoE routing.

Ported from https://github.com/Asphyxiate-Rye/E-SAM (SAM-MoE, MICCAI 2025);
see `_vendor/` for the vendored architecture and its deviations from upstream.
"""

from __future__ import annotations

from torch import Tensor

from ..base import BaseSegmentationModel, SegmentationOutput
from ..registry import register_model
from ._vendor.build import build_sam_vit_b


class _UpstreamArgs:
    """Stand-in for upstream's argparse.Namespace; `batch_size` is dead config,
    never actually read back by the vendored code."""

    batch_size = 1


@register_model("esam")
class EsamModel(BaseSegmentationModel):
    """SAM ViT-B + Adapter + Sparse MoE, adapted to this project's contract.

    Only Adapter parameters in the backbone are trained; the prompt encoder
    stays frozen (matches upstream's train.py). Set `use_moe=False` for a
    no-MoE ablation baseline with everything else identical.
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
        moe_num_experts: int = 4,
        moe_top_k_ratio: float = 0.5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, task=task)

        if in_channels != 3:
            raise ValueError("EsamModel only supports in_channels=3 (RGB SAM backbone).")

        self.image_size = image_size
        self.network = build_sam_vit_b(
            args=_UpstreamArgs(),
            image_size=image_size,
            num_classes=num_classes,
            checkpoint=checkpoint,
            moe_top_k_ratio=moe_top_k_ratio,
            moe_num_experts=moe_num_experts,
            use_moe=use_moe,
        )

        if freeze_backbone:
            self._freeze_pretrained_backbone()

    def _freeze_pretrained_backbone(self) -> None:
        for name, param in self.network.image_encoder.named_parameters():
            param.requires_grad = "Adapter" in name
        for param in self.network.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, images: Tensor, **kwargs) -> SegmentationOutput:
        outputs = self.network(
            images,
            multimask_output=True,
            image_size=self.image_size,
        )
        return SegmentationOutput(
            logits=outputs["masks"],
            diagnostics={
                "iou_predictions": outputs["iou_predictions"],
                "moe_expert_indices": outputs["indices"],
            },
        )
