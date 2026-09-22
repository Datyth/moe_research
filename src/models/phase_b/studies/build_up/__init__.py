"""Phase-B build-up ablation ladder B1-B6."""

from .b1_multilevel import PhaseBB1MultiLevel
from .b2_dense import DENSE_HIDDEN_DIM, PhaseBB2Dense
from .b3_image_moe import PhaseBB3ImageMoE
from .b4_shape_moe import PhaseBB4ShapeDirect
from .b5_gaussian_moe import PhaseBB5Gaussian
from .b6_hierarchical import PhaseBB6Hierarchical
from .common import (
    IMAGE_ONLY,
    POSTERIOR_ORACLE,
    BuildUpBase,
    BuildUpStageOutput,
    BuildUpVariantOutput,
)
from .modules import (
    ConditioningOutput,
    DenseEnhancer,
    DenseTokenFFN,
    DirectConditioner,
    GaussianConditioner,
    IdentityEnhancer,
    RandomTopKRouter,
    SparseMoEEnhancer,
    nearest_dense_hidden_dim,
)

__all__ = [
    "IMAGE_ONLY",
    "POSTERIOR_ORACLE",
    "DENSE_HIDDEN_DIM",
    "BuildUpBase",
    "BuildUpStageOutput",
    "BuildUpVariantOutput",
    "ConditioningOutput",
    "DenseEnhancer",
    "DenseTokenFFN",
    "DirectConditioner",
    "GaussianConditioner",
    "IdentityEnhancer",
    "PhaseBB1MultiLevel",
    "PhaseBB2Dense",
    "PhaseBB3ImageMoE",
    "PhaseBB4ShapeDirect",
    "PhaseBB5Gaussian",
    "PhaseBB6Hierarchical",
    "RandomTopKRouter",
    "SparseMoEEnhancer",
    "nearest_dense_hidden_dim",
]
