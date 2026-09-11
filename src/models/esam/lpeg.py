"""Lightweight Prompt Embedding Generator (LPEG) from MoE-SAM (MICCAI 2025).

Extracted from the LPEG block previously inlined in
`_vendor/prompt_encoder.py` so it can be toggled and trained independently of
the original SAM prompt-encoder parameters.

The paper describes the bottleneck as Linear-GELU-Linear; the released
implementation used LayerNorm-GELU-pool followed by Linear-ReLU-Linear. This
module follows the released front end (LayerNorm -> GELU -> AdaptiveAvgPool)
but replaces the inner ReLU with GELU per the paper's description.

Contract: consumes an image embedding `[B, C, H, W]` and produces a single
sparse prompt token `[B, 1, C]`. It never produces a dense feature map; the
dense prompt remains SAM's learned `no_mask_embed`.
"""

from __future__ import annotations

import torch
from torch import nn

from ._vendor.common import LayerNorm2d


class LPEG(nn.Module):
    """Image embedding -> one sparse prompt token `[B, 1, embed_dim]`."""

    def __init__(self, embed_dim: int = 256, bottleneck_ratio: int = 16) -> None:
        super().__init__()
        bottleneck = max(1, embed_dim // bottleneck_ratio)
        self.norm = LayerNorm2d(embed_dim)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(embed_dim, bottleneck)
        self.act2 = nn.GELU()
        self.fc2 = nn.Linear(bottleneck, embed_dim)

    def forward(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """Return `[B, 1, embed_dim]` from an image embedding `[B, C, H, W]`."""
        feature = self.norm(image_embedding)
        feature = self.act(feature)
        feature = self.pool(feature).flatten(1)
        feature = self.act2(self.fc1(feature))
        return self.fc2(feature).unsqueeze(1)
