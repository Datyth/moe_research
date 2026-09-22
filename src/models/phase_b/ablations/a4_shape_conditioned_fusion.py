"""A4: Gaussian routing with shape-conditioned fusion before sparse experts."""

from __future__ import annotations

from typing import Any

from torch import nn

from ...registry import register_model
from ..phase_b_moe import PhaseBMoEStage
from ..shape_conditioned_enhancement import (
    ShapeConditionedFusionMoEEnhancement,
    ShapeConditionedFusionOutput,
)


@register_model("phase_b_a4_shape_conditioned")
class PhaseBA4ShapeConditionedStage(PhaseBMoEStage):
    """Dedicated A4 registry with stable uniform level fusion at init."""

    def __init__(self, **kwargs: Any) -> None:
        requested_mode = kwargs.pop("enhancement_mode", "shape_conditioned")
        if requested_mode != "shape_conditioned":
            raise ValueError(
                "phase_b_a4_shape_conditioned fixes enhancement_mode to "
                "'shape_conditioned'."
            )
        super().__init__(enhancement_mode="shape_conditioned", **kwargs)
        final_linear = self.enhancement.g_fuse[-1]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("A4 g_fuse must end in nn.Linear.")
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)


__all__ = [
    "PhaseBA4ShapeConditionedStage",
    "ShapeConditionedFusionMoEEnhancement",
    "ShapeConditionedFusionOutput",
]
