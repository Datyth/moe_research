"""B3: image-only learned or random sparse expert routing."""

from __future__ import annotations

from typing import Any

from ....registry import register_model
from .common import RoutedBuildUpBase


@register_model("phase_b_b3_image_moe")
class PhaseBB3ImageMoE(RoutedBuildUpBase):
    """Route from h_I and fuse enhanced levels with global alpha."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        router_mode: str = "learned",
        num_experts: int = 4,
        active_experts: int = 2,
        expert_hidden_ratio: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            descriptor_dim=descriptor_dim,
            conditioning_dim=descriptor_dim,
            conditioner_kind="direct",
            routing_source="image",
            router_mode=router_mode,
            num_experts=num_experts,
            active_experts=active_experts,
            expert_hidden_ratio=expert_hidden_ratio,
            hierarchical=False,
            **kwargs,
        )
