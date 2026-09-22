"""Phase B: privileged shape-aware routing and hierarchical enhancement."""

from .experts import ExpertBank, FeedForwardExpert
from .fuse_stage import FuseStageOutput, PhaseBFuseStage
from .fusion import PrivilegedFusion, PrivilegedFusionOutput
from .image_descriptor import (
    DEFAULT_LEVELS,
    MultiLevelImageDescriptor,
    MultiLevelImageDescriptorOutput,
)
from .layer_attention import LayerPreferenceScorer
from .moe_enhancement import EnhancementOutput, HierarchicalMoEEnhancement
from .phase_b_moe import EnhancementStageOutput, PhaseBMoEStage
from .phase_b_no_moe import PhaseBA1NoExpertStage, PhaseBNoMoEStage
from .ablations.a2_direct_fuse import (
    DirectFuseRouterStageOutput,
    PhaseBA2DirectFuseStage,
)
from .ablations.a4_shape_conditioned_fusion import (
    PhaseBA4ShapeConditionedStage,
)
from .phase_b_router import PhaseBRouterHead, PhaseBRouterStage, RouterStageOutput
from .posterior import DiagonalGaussian, GaussianParameterHead, gaussian_kl
from .router import RoutingOutput, TopKRouter, load_balance_loss
from .shape_teacher import ShapeTeacher, build_shape_teacher, load_shape_teacher
from .shape_conditioned_enhancement import (
    ShapeConditionedFusionMoEEnhancement,
    ShapeConditionedFusionOutput,
)
from .studies.build_up import (
    PhaseBB1MultiLevel,
    PhaseBB2Dense,
    PhaseBB3ImageMoE,
    PhaseBB4ShapeDirect,
    PhaseBB5Gaussian,
    PhaseBB6Hierarchical,
)

__all__ = [
    "DEFAULT_LEVELS",
    "DiagonalGaussian",
    "EnhancementOutput",
    "EnhancementStageOutput",
    "ExpertBank",
    "FeedForwardExpert",
    "FuseStageOutput",
    "GaussianParameterHead",
    "HierarchicalMoEEnhancement",
    "LayerPreferenceScorer",
    "MultiLevelImageDescriptor",
    "MultiLevelImageDescriptorOutput",
    "DirectFuseRouterStageOutput",
    "PhaseBA1NoExpertStage",
    "PhaseBA2DirectFuseStage",
    "PhaseBA4ShapeConditionedStage",
    "PhaseBB1MultiLevel",
    "PhaseBB2Dense",
    "PhaseBB3ImageMoE",
    "PhaseBB4ShapeDirect",
    "PhaseBB5Gaussian",
    "PhaseBB6Hierarchical",
    "PhaseBFuseStage",
    "PhaseBMoEStage",
    "PhaseBNoMoEStage",
    "PhaseBRouterHead",
    "PhaseBRouterStage",
    "PrivilegedFusion",
    "PrivilegedFusionOutput",
    "RouterStageOutput",
    "RoutingOutput",
    "ShapeTeacher",
    "ShapeConditionedFusionMoEEnhancement",
    "ShapeConditionedFusionOutput",
    "TopKRouter",
    "build_shape_teacher",
    "gaussian_kl",
    "load_balance_loss",
    "load_shape_teacher",
]
