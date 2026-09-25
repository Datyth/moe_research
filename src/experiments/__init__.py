"""Research experiment entry points."""

from .shape_pretraining import build_shape_autoencoder, execute_shape_pretraining
from .phase_c_comprehensive import PHASE_C_COMPREHENSIVE_SPECS

__all__ = [
    "build_shape_autoencoder",
    "execute_shape_pretraining",
    "PHASE_C_COMPREHENSIVE_SPECS",
]
