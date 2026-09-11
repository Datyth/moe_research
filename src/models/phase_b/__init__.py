"""Phase B: privileged shape-aware routing, up to the fuse stage h_q."""

from .fuse_stage import FuseStageOutput, PhaseBFuseStage
from .fusion import PrivilegedFusion, PrivilegedFusionOutput
from .image_descriptor import (
    DEFAULT_LEVELS,
    MultiLevelImageDescriptor,
    MultiLevelImageDescriptorOutput,
)
from .shape_teacher import ShapeTeacher, build_shape_teacher, load_shape_teacher

__all__ = [
    "DEFAULT_LEVELS",
    "FuseStageOutput",
    "MultiLevelImageDescriptor",
    "MultiLevelImageDescriptorOutput",
    "PhaseBFuseStage",
    "PrivilegedFusion",
    "PrivilegedFusionOutput",
    "ShapeTeacher",
    "build_shape_teacher",
    "load_shape_teacher",
]
