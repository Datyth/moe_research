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
from .phase_b_router import PhaseBRouterHead, PhaseBRouterStage, RouterStageOutput
from .posterior import DiagonalGaussian, GaussianParameterHead, gaussian_kl
from .router import RoutingOutput, TopKRouter, load_balance_loss
from .shape_teacher import ShapeTeacher, build_shape_teacher, load_shape_teacher

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
    "PhaseBFuseStage",
    "PhaseBMoEStage",
    "PhaseBRouterHead",
    "PhaseBRouterStage",
    "PrivilegedFusion",
    "PrivilegedFusionOutput",
    "RouterStageOutput",
    "RoutingOutput",
    "ShapeTeacher",
    "TopKRouter",
    "build_shape_teacher",
    "gaussian_kl",
    "load_balance_loss",
    "load_shape_teacher",
]
