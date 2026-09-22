"""B2: parameter-matched shared dense enhancement."""

from __future__ import annotations

from typing import Any

from ....registry import register_model
from ...image_descriptor import MultiLevelImageDescriptorOutput
from .common import BuildUpBase, BuildUpVariantOutput
from .modules import DenseEnhancer


DENSE_HIDDEN_DIM = 12_292


@register_model("phase_b_b2_dense")
class PhaseBB2Dense(BuildUpBase):
    """Apply one shared dense residual FFN independently at every level."""

    def __init__(
        self,
        *,
        dense_hidden_dim: int = DENSE_HIDDEN_DIM,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.enhancement = DenseEnhancer(
            embed_dim=self.embed_dim,
            num_levels=self.num_levels,
            hidden_dim=dense_hidden_dim,
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
