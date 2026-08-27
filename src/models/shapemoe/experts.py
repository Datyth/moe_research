"""Shape-specialized segmentation experts (ShapeMoE Sec. 3.5).

The paper's analysis is that duplicating a whole mask decoder per expert is
wasteful: the expensive two-way Transformer only refines the image feature F and
does not produce masks, while the lightweight hyper-network that emits the mask
weights w is what actually decides the shape. So only the hyper-network is
replicated across K branches.

This project has no SAM decoder. The structural analogue in a UNet is exact: the
decoder produces a full-resolution feature map F, and a single 1x1 convolution
turns it into mask logits. That final projection is the mask-weight producer, so
it is the part replicated per expert; the encoder and decoder stay shared.

Dispatch is genuinely sparse. Each expert runs on the sub-batch routed to it, so
with top_k=1 every sample is processed by exactly one head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ExpertMaskHeads(nn.Module):
    """K lightweight mask heads combined with the router's sparse weights."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_experts: int,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.heads = nn.ModuleList(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
            for _ in range(num_experts)
        )

    def forward(self, features: Tensor, pi: Tensor) -> Tensor:
        """Combine the selected experts' logits weighted by ``pi``."""

        if features.ndim != 4:
            raise ValueError(
                f"features must have shape [B, C, H, W], got "
                f"{tuple(features.shape)}."
            )
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"features must have {self.in_channels} channels, got "
                f"{features.shape[1]}."
            )
        if pi.ndim != 2 or pi.shape[1] != self.num_experts:
            raise ValueError(
                f"pi must have shape [B, {self.num_experts}], got "
                f"{tuple(pi.shape)}."
            )
        if pi.shape[0] != features.shape[0]:
            raise ValueError(
                f"pi and features disagree on batch size: {pi.shape[0]} vs "
                f"{features.shape[0]}."
            )

        batch_size, _, height, width = features.shape
        logits = torch.zeros(
            batch_size,
            self.num_classes,
            height,
            width,
            dtype=features.dtype,
            device=features.device,
        )

        for expert, head in enumerate(self.heads):
            weights = pi[:, expert]
            selected = torch.nonzero(weights, as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            contribution = head(features.index_select(0, selected))
            contribution = contribution * weights.index_select(
                0,
                selected,
            ).view(-1, 1, 1, 1)
            logits = logits.index_add(0, selected, contribution.to(logits.dtype))

        return logits

    def expert_usage(self, pi: Tensor) -> Tensor:
        """Per-expert sample counts in this batch, for logging."""

        return (pi > 0).sum(dim=0)
