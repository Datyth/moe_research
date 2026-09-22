"""B6: posterior routing plus expert-conditioned layer preferences."""

from __future__ import annotations

from typing import Any

from ....registry import register_model
from .common import PrivilegedRoutedBuildUpBase


@register_model("phase_b_b6_hierarchical")
class PhaseBB6Hierarchical(PrivilegedRoutedBuildUpBase):
    """Reuse HierarchicalMoEEnhancement for beta/gamma routed fusion."""

    def __init__(
        self,
        *,
        router_mode: str = "learned",
        latent_dim: int = 64,
        num_experts: int = 4,
        active_experts: int = 2,
        expert_hidden_ratio: int = 4,
        std_floor: float = 1e-4,
        stochastic: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            conditioner_kind="gaussian",
            routing_source="posterior",
            router_mode=router_mode,
            latent_dim=latent_dim,
            num_experts=num_experts,
            active_experts=active_experts,
            expert_hidden_ratio=expert_hidden_ratio,
            std_floor=std_floor,
            stochastic=stochastic,
            hierarchical=True,
            **kwargs,
        )
