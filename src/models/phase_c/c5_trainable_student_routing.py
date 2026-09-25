"""Phase-C C5: segmentation-only adaptation of student routing policy."""

from __future__ import annotations

from typing import Any

from src.models.registry import register_model

from .adaptive_student import PhaseCAdaptiveStudent


@register_model("phase_c_c5_trainable_student_routing")
class PhaseCC5TrainableStudentRouting(PhaseCAdaptiveStudent):
    """Compatibility wrapper for the completed C5/D5 routing ablation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            train_student_router=True,
            train_student_layer_router=True,
            train_student_experts=False,
            train_student_image_descriptor=False,
            **kwargs,
        )

    def _initialize_student_routing(self) -> None:
        """Retain the local C5 tiny-test initialization hook."""

        self.train_student_router = True
        self.train_student_layer_router = True
        self.train_student_experts = False
        self.train_student_image_descriptor = False
        self._initialize_adaptive_student()

    def phase_c_checkpoint_metadata(self) -> dict[str, Any]:
        metadata = super().phase_c_checkpoint_metadata()
        metadata.pop("adaptive_student", None)
        metadata["parameter_counts"]["trainable_groups"] = {
            name: count
            for name, count in metadata["parameter_counts"][
                "trainable_groups"
            ].items()
            if count > 0
        }
        metadata["student_routing"] = {
            "router_initialized_from_teacher": True,
            "layer_scorer_initialized_from_teacher": True,
            "experts_shared_frozen": True,
        }
        return metadata


__all__ = ["PhaseCC5TrainableStudentRouting"]
