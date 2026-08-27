"""ShapeMoE components.

Phase 0 provides the mask-VAE shape teacher; Phase 1 adds the student shape
encoder, the shape-aware sparse router, the expert mask heads, and the segmenter
that assembles them.
"""

from .experts import ExpertMaskHeads
from .router import RouterOutput, ShapeAwareSparseRouter
from .shapemoe import ShapeMoESegmenter
from .student import ShapeDistributionEncoder
from .teacher import (
    FrozenModule,
    MaskVAEOutput,
    MaskVAETeacher,
    load_mask_embedding_encoder,
)

__all__ = [
    "ExpertMaskHeads",
    "FrozenModule",
    "MaskVAEOutput",
    "MaskVAETeacher",
    "load_mask_embedding_encoder",
    "RouterOutput",
    "ShapeAwareSparseRouter",
    "ShapeDistributionEncoder",
    "ShapeMoESegmenter",
]
