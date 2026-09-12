"""Multi-level SAM image representation and the routing descriptor h_I.

Implements the proposal's "Overall Framework and Multi-Level Image
Representation": intermediate ViT-B features are taken from layers
L = {3, 6, 9, 12}, each projected to a common dimension C_s by a learnable
1x1 projection P_l, globally pooled into u^(l), and combined by a softmax
level attention into h_I. The untouched token representations X^(l) are
returned alongside, because the later MoE enhancement stage consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


DEFAULT_LEVELS = (3, 6, 9, 12)


@dataclass
class MultiLevelImageDescriptorOutput:
    """The routing descriptor plus everything downstream stages still need."""

    descriptor: Tensor
    """h_I, shape [B, C_s]."""

    level_weights: Tensor
    """alpha, shape [B, len(levels)]; sums to 1 along dim 1."""

    level_descriptors: Tensor
    """Stacked u^(l), shape [B, len(levels), C_s]."""

    level_tokens: tuple[Tensor, ...]
    """X^(l), each [B, P, C] with P = H_p * W_p; retained for MoE enhancement."""


class MultiLevelImageDescriptor(nn.Module):
    """Turn selected ViT block outputs into the image-only routing descriptor.

    `levels` is 1-indexed over transformer blocks to match the proposal's
    L = {3, 6, 9, 12}; block outputs arrive 0-indexed, so layer l is
    `block_outputs[l - 1]`.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        descriptor_dim: int = 256,
        levels: tuple[int, ...] = DEFAULT_LEVELS,
        scoring_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive.")
        if scoring_hidden_dim <= 0:
            raise ValueError("scoring_hidden_dim must be positive.")
        if len(levels) == 0:
            raise ValueError("levels must not be empty.")
        if len(set(levels)) != len(levels):
            raise ValueError(f"levels must be unique, got {levels}.")
        if any(level < 1 for level in levels):
            raise ValueError(f"levels are 1-indexed and must be >= 1, got {levels}.")

        self.embed_dim = embed_dim
        self.descriptor_dim = descriptor_dim
        self.levels = tuple(levels)

        # P_l is indexed by l in the proposal, so each level gets its own
        # projection rather than one shared across levels.
        self.level_projections = nn.ModuleList(
            nn.Conv2d(embed_dim, descriptor_dim, kernel_size=1, bias=True)
            for _ in self.levels
        )
        # g_level scores one pooled level at a time and is shared across
        # levels; a per-level scorer could not be compared on a common scale.
        self.level_scoring = nn.Sequential(
            nn.Linear(descriptor_dim, scoring_hidden_dim),
            nn.GELU(),
            nn.Linear(scoring_hidden_dim, 1),
        )

    @staticmethod
    def _to_channels_first(features: Tensor) -> Tensor:
        """Accept SAM's [B, H_p, W_p, C] block output or [B, C, H_p, W_p]."""

        if features.ndim != 4:
            raise ValueError(
                "Each level feature map must be 4-dimensional, got "
                f"{tuple(features.shape)}."
            )
        return features.permute(0, 3, 1, 2).contiguous()

    def _select_levels(self, block_outputs: list[Tensor]) -> list[Tensor]:
        required = max(self.levels)
        if len(block_outputs) < required:
            raise ValueError(
                f"levels={self.levels} needs at least {required} block outputs, "
                f"got {len(block_outputs)}."
            )
        return [block_outputs[level - 1] for level in self.levels]

    def forward(self, block_outputs: list[Tensor]) -> MultiLevelImageDescriptorOutput:
        selected = self._select_levels(list(block_outputs))

        level_descriptors = []
        level_tokens = []
        for features, projection in zip(selected, self.level_projections):
            channels_first = self._to_channels_first(features)
            if channels_first.shape[1] != self.embed_dim:
                raise ValueError(
                    "Level features must be [B, H_p, W_p, C] with "
                    f"C={self.embed_dim}, got {tuple(features.shape)}."
                )
            projected = projection(channels_first)
            level_descriptors.append(projected.mean(dim=(2, 3)))
            level_tokens.append(channels_first.flatten(2).transpose(1, 2))

        stacked = torch.stack(level_descriptors, dim=1)
        scores = self.level_scoring(stacked).squeeze(-1)
        level_weights = torch.softmax(scores, dim=1)
        descriptor = (level_weights.unsqueeze(-1) * stacked).sum(dim=1)

        return MultiLevelImageDescriptorOutput(
            descriptor=descriptor,
            level_weights=level_weights,
            level_descriptors=stacked,
            level_tokens=tuple(level_tokens),
        )
