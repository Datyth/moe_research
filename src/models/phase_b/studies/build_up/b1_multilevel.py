"""B1: multi-level residual enhancement without dense or expert modules."""

from __future__ import annotations

from typing import Any

from ....registry import register_model
from ...image_descriptor import MultiLevelImageDescriptorOutput
from .common import BuildUpBase, BuildUpVariantOutput
from .modules import IdentityEnhancer


@register_model("phase_b_b1_multilevel")
class PhaseBB1MultiLevel(BuildUpBase):
    """Fuse raw multi-level tokens with alpha before decoder injection."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.enhancement = IdentityEnhancer(
            embed_dim=self.embed_dim,
            num_levels=self.num_levels,
        )

    def _variant_forward(
        self,
        descriptor: MultiLevelImageDescriptorOutput,
        masks,
    ) -> BuildUpVariantOutput:
        enhancement = self.enhancement(
            descriptor.level_tokens,
            descriptor.level_weights,
        )
        return BuildUpVariantOutput(
            fused_tokens=enhancement.fused_tokens,
            fusion_weights=descriptor.level_weights,
        )
