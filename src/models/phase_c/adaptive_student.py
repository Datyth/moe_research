"""Configurable Phase-C student adaptation with an immutable B6 teacher."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from src.models.phase_b.experts import ExpertBank
from src.models.phase_b.image_descriptor import MultiLevelImageDescriptor
from src.models.phase_b.layer_attention import LayerPreferenceScorer
from src.models.phase_b.posterior import DiagonalGaussian
from src.models.phase_b.router import RoutingOutput, TopKRouter
from src.models.phase_b.studies.build_up.common import run_sam_encoder
from src.models.registry import register_model

from .b6_prior_distill import PhaseCB6PriorDistill, _EncodedImageState


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _assert_independent_equal_copy(
    name: str,
    teacher: nn.Module,
    student: nn.Module,
) -> None:
    """Validate numerical initialization without allowing storage aliasing."""

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    if teacher_state.keys() != student_state.keys():
        raise RuntimeError(f"{name} copy has a different state-dict schema.")
    for key, teacher_tensor in teacher_state.items():
        student_tensor = student_state[key]
        if not torch.equal(teacher_tensor, student_tensor):
            raise RuntimeError(f"{name} copy differs at initialization: {key}.")
        if teacher_tensor.data_ptr() == student_tensor.data_ptr():
            raise RuntimeError(f"{name} copy aliases teacher storage: {key}.")


@register_model("phase_c_adaptive_student")
class PhaseCAdaptiveStudent(PhaseCB6PriorDistill):
    """B6 Phase-C student with independently configurable adaptation scope.

    The teacher remains the inherited B6 object graph. Student router and
    layer scorer objects are always independent copies. Expert and image
    descriptor copies are created only when those groups are trainable.

    Student: frozen SAM -> student/shared descriptor -> prior mean -> student
    router/scorer -> student/shared experts -> frozen neck and SAM decoder.
    Teacher diagnostics: frozen teacher descriptor + Shape Teacher -> frozen
    posterior/router/scorer/experts -> the same frozen decoder.
    """

    def __init__(
        self,
        *,
        train_student_router: bool,
        train_student_layer_router: bool,
        train_student_experts: bool,
        train_student_image_descriptor: bool,
        **kwargs: Any,
    ) -> None:
        flags = {
            "train_student_router": _require_bool(
                "train_student_router", train_student_router
            ),
            "train_student_layer_router": _require_bool(
                "train_student_layer_router", train_student_layer_router
            ),
            "train_student_experts": _require_bool(
                "train_student_experts", train_student_experts
            ),
            "train_student_image_descriptor": _require_bool(
                "train_student_image_descriptor",
                train_student_image_descriptor,
            ),
        }
        super().__init__(**kwargs)
        for name, value in flags.items():
            setattr(self, name, value)
        self._initialize_adaptive_student()

    @property
    def teacher_image_descriptor(self) -> MultiLevelImageDescriptor:
        """Frozen B6 descriptor, exposed without registering another alias."""

        return self.image_descriptor

    def _initialize_adaptive_student(self) -> None:
        """Create student-owned modules from the already-loaded B6 teacher."""

        if hasattr(self, "student_router"):
            raise RuntimeError("Adaptive student modules are already initialized.")

        self.student_router = deepcopy(self.teacher_router)
        self.student_layer_scorer = deepcopy(self.teacher_layer_scorer)
        if self.train_student_experts:
            self.student_experts = deepcopy(self.teacher_experts)
        if self.train_student_image_descriptor:
            self.student_image_descriptor = deepcopy(
                self.teacher_image_descriptor
            )

        _assert_independent_equal_copy(
            "student_router", self.teacher_router, self.student_router
        )
        _assert_independent_equal_copy(
            "student_layer_scorer",
            self.teacher_layer_scorer,
            self.student_layer_scorer,
        )
        if self.train_student_experts:
            _assert_independent_equal_copy(
                "student_experts", self.teacher_experts, self.student_experts
            )
        if self.train_student_image_descriptor:
            _assert_independent_equal_copy(
                "student_image_descriptor",
                self.teacher_image_descriptor,
                self.student_image_descriptor,
            )

        self._apply_trainability()
        self._assert_trainable_contract()
        self.train(False)

    def _apply_trainability(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.prior_head.parameters():
            parameter.requires_grad_(True)

        enabled_modules = (
            (self.train_student_router, self.student_router),
            (self.train_student_layer_router, self.student_layer_scorer),
            (
                self.train_student_experts,
                getattr(self, "student_experts", None),
            ),
            (
                self.train_student_image_descriptor,
                getattr(self, "student_image_descriptor", None),
            ),
        )
        for enabled, module in enabled_modules:
            if enabled:
                if module is None:
                    raise RuntimeError("Enabled student module was not initialized.")
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "PhaseCAdaptiveStudent":
        """Train only enabled student groups; keep the B6 teacher in eval."""

        PhaseCB6PriorDistill.train(self, mode)
        modules = (
            (
                getattr(self, "train_student_router", False),
                getattr(self, "student_router", None),
            ),
            (
                getattr(self, "train_student_layer_router", False),
                getattr(self, "student_layer_scorer", None),
            ),
            (
                getattr(self, "train_student_experts", False),
                getattr(self, "student_experts", None),
            ),
            (
                getattr(self, "train_student_image_descriptor", False),
                getattr(self, "student_image_descriptor", None),
            ),
        )
        for enabled, module in modules:
            if module is not None:
                module.train(mode if enabled else False)
        return self

    def _prior_router_module(self) -> TopKRouter:
        return self.student_router

    def _prior_layer_scorer_module(self) -> LayerPreferenceScorer:
        return self.student_layer_scorer

    def _prior_expert_bank_module(self) -> ExpertBank:
        if self.train_student_experts:
            return self.student_experts
        return self.teacher_experts

    def _encode_images(
        self,
        images: Tensor,
        *,
        include_teacher_descriptor: bool = False,
    ) -> _EncodedImageState:
        if not getattr(self, "train_student_image_descriptor", False):
            return super()._encode_images(
                images,
                include_teacher_descriptor=include_teacher_descriptor,
            )

        with torch.no_grad():
            image_embeddings, block_outputs = run_sam_encoder(
                self.backbone,
                images,
            )
            detached_outputs = tuple(
                output.detach() for output in block_outputs
            )

        # Frozen SAM outputs are constants, but this descriptor call must be
        # tracked so gradients reach level_projections and level_scoring.
        student_descriptor = self.student_image_descriptor(
            list(detached_outputs)
        )
        teacher_descriptor = None
        if include_teacher_descriptor:
            with torch.no_grad():
                teacher_descriptor = self.teacher_image_descriptor(
                    list(detached_outputs)
                )
        return _EncodedImageState(
            image_embeddings=image_embeddings,
            descriptor=student_descriptor,
            teacher_descriptor=teacher_descriptor,
            block_outputs=detached_outputs,
        )

    def _prior_from_encoded(
        self,
        encoded: _EncodedImageState,
    ) -> tuple[DiagonalGaussian, RoutingOutput]:
        descriptor = encoded.descriptor.descriptor
        if not self.train_student_image_descriptor:
            descriptor = descriptor.detach()
        prior = self.prior_head(descriptor)
        routing = self.student_router(prior.mean)
        return prior, routing

    def trainable_parameter_groups(
        self,
    ) -> dict[str, tuple[nn.Parameter, ...]]:
        groups: dict[str, tuple[nn.Parameter, ...]] = {
            "prior_head": tuple(self.prior_head.parameters()),
        }
        if self.train_student_router:
            groups["student_router"] = tuple(self.student_router.parameters())
        if self.train_student_layer_router:
            groups["student_layer_scorer"] = tuple(
                self.student_layer_scorer.parameters()
            )
        if self.train_student_experts:
            groups["student_experts"] = tuple(
                self.student_experts.parameters()
            )
        if self.train_student_image_descriptor:
            groups["student_image_descriptor"] = tuple(
                self.student_image_descriptor.parameters()
            )
        return groups

    def optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for parameters in self.trainable_parameter_groups().values()
            for parameter in parameters
        )

    def _assert_trainable_contract(self) -> None:
        groups = self.trainable_parameter_groups()
        allowed = [
            parameter
            for parameters in groups.values()
            for parameter in parameters
        ]
        allowed_ids = [id(parameter) for parameter in allowed]
        if len(allowed_ids) != len(set(allowed_ids)):
            raise RuntimeError("Adaptive student parameter groups overlap.")
        if not all(parameter.requires_grad for parameter in allowed):
            raise RuntimeError("Every optimizer parameter must require gradients.")

        actual_ids = {
            id(parameter)
            for parameter in self.parameters()
            if parameter.requires_grad
        }
        if actual_ids != set(allowed_ids):
            raise RuntimeError(
                "Trainable parameters do not match the configured student groups."
            )

        teacher_modules = (
            self.conditioner,
            self.enhancement,
            self.backbone,
            self.teacher_image_descriptor,
            self.shape_teacher,
            self.fusion,
            self.enhancement_neck,
        )
        teacher_ids = {
            id(parameter)
            for module in teacher_modules
            for parameter in module.parameters()
        }
        if teacher_ids.intersection(allowed_ids):
            raise RuntimeError(
                "Teacher, SAM, neck, or decoder parameters entered the optimizer."
            )

    def validate_optimizer_parameters(
        self,
        parameters: Iterable[nn.Parameter],
    ) -> None:
        actual = tuple(parameters)
        actual_ids = [id(parameter) for parameter in actual]
        expected_ids = [id(parameter) for parameter in self.optimizer_parameters()]
        if len(actual_ids) != len(set(actual_ids)):
            raise RuntimeError("Adaptive student optimizer has duplicates.")
        if set(actual_ids) != set(expected_ids):
            raise RuntimeError(
                "Optimizer membership does not match configured student groups."
            )
        if not all(parameter.requires_grad for parameter in actual):
            raise RuntimeError("Optimizer contains a frozen parameter.")
        self._assert_trainable_contract()

    def phase_c_checkpoint_metadata(self) -> dict[str, Any]:
        metadata = super().phase_c_checkpoint_metadata()
        group_counts = {
            "prior_head": 0,
            "student_router": 0,
            "student_layer_scorer": 0,
            "student_experts": 0,
            "student_image_descriptor": 0,
        }
        group_counts.update(
            {
                name: sum(parameter.numel() for parameter in parameters)
                for name, parameters in self.trainable_parameter_groups().items()
            }
        )
        if sum(group_counts.values()) != metadata["parameter_counts"][
            "trainable"
        ]:
            raise RuntimeError("Adaptive student parameter counts are inconsistent.")
        metadata["parameter_counts"]["trainable_groups"] = group_counts
        metadata["adaptive_student"] = {
            "train_student_router": self.train_student_router,
            "train_student_layer_router": self.train_student_layer_router,
            "train_student_experts": self.train_student_experts,
            "train_student_image_descriptor": (
                self.train_student_image_descriptor
            ),
            "router_initialized_from_teacher": True,
            "layer_scorer_initialized_from_teacher": True,
            "experts_initialized_from_teacher": self.train_student_experts,
            "image_descriptor_initialized_from_teacher": (
                self.train_student_image_descriptor
            ),
        }
        return metadata


__all__ = ["PhaseCAdaptiveStudent"]
