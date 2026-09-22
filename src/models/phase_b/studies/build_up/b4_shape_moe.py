"""B4: privileged direct shape routing without a Gaussian latent."""

from __future__ import annotations

from typing import Any

from ....registry import register_model
from .common import PrivilegedRoutedBuildUpBase


@register_model("phase_b_b4_shape_direct")
class PhaseBB4ShapeDirect(PrivilegedRoutedBuildUpBase):
    """Route directly from h_q=[h_I;h_M], then use B3 enhancement."""

    def __init__(
        self,
        *,
        router_mode: str = "learned",
        num_experts: int = 4,
        active_experts: int = 2,
        expert_hidden_ratio: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            conditioner_kind="direct",
            routing_source="shape_direct",
            router_mode=router_mode,
            num_experts=num_experts,
            active_experts=active_experts,
            expert_hidden_ratio=expert_hidden_ratio,
            hierarchical=False,
            **kwargs,
        )
