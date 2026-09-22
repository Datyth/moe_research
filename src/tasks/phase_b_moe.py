"""Phase B full-stage task: segmentation + KL + load-balance.

Loss-wise this stage adds nothing over the router task — the routed expert
enhancement is in the model's forward path, so the segmentation loss itself
now backpropagates through the experts, ``beta``, and the level attention
(the first stage where ``h_I``'s parameters receive gradient). The extra
terms remain the two routing losses:

    L = L_seg + lambda_latent * KL(sg[q] || p) + lambda_bal * L_bal

What *is* new is strict verification that the enhancement actually ran: a run
that stopped emitting the ``phase_b_moe`` diagnostic fails loudly rather than
silently degrading to the router stage. Evaluation additionally reports
``enhancement_aux_ratio`` — the mean ||E_aux||/||E_enh|| — and the fused
gamma level weights, so a dead enhancement (ratio 0) or a collapsed level
fusion shows up in the metrics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import TaskStepOutput
from .phase_b_diagnostics import enhancement_metrics
from .phase_b_router import PhaseBRouterTask


class PhaseBMoETask(PhaseBRouterTask):
    """Router task plus strict enhancement-stage verification and metrics."""

    def __init__(
        self,
        *,
        criterion: nn.Module,
        threshold: float = 0.5,
        boundary_tolerance: float = 2,
        task: str = "binary",
        strict_fuse: bool = True,
        strict_router: bool = True,
        strict_enhancement: bool = True,
        lambda_latent: float = 0.1,
        lambda_balance: float = 0.01,
    ) -> None:
        super().__init__(
            criterion=criterion,
            threshold=threshold,
            boundary_tolerance=boundary_tolerance,
            task=task,
            strict_fuse=strict_fuse,
            strict_router=strict_router,
            lambda_latent=lambda_latent,
            lambda_balance=lambda_balance,
        )
        self.strict_enhancement = bool(strict_enhancement)

    def _enhancement_stage(self, diagnostics: dict[str, Any]):
        stage = diagnostics.get("phase_b_moe")
        if stage is None and self.strict_enhancement:
            raise ValueError(
                "Phase B MoE task expected a 'phase_b_moe' diagnostic; got "
                f"keys {sorted(diagnostics)}. Is this a phase_b_moe model?"
            )
        return stage

    @staticmethod
    def _enhancement_metrics(stage: Any) -> dict[str, torch.Tensor]:
        return enhancement_metrics(stage)

    def training_step(self, model: nn.Module, batch: Any, device: Any) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, diagnostics = self._forward_with_privileged_mask(model, images, targets)
        self._router_stage(diagnostics)
        self._enhancement_stage(diagnostics)

        loss = self.criterion(logits, targets)
        stage = diagnostics.get("phase_b_router")
        if stage is not None and stage.latent_kl is not None:
            loss = loss + self.lambda_latent * stage.latent_kl
        if stage is not None and stage.balance is not None:
            loss = loss + self.lambda_balance * stage.balance

        return TaskStepOutput(loss=loss, metrics={}, batch_size=images.shape[0])

    def evaluation_step(self, model: nn.Module, batch: Any, device: Any) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, diagnostics = self._forward_with_privileged_mask(model, images, targets)
        stage = self._router_stage(diagnostics)
        enhancement = self._enhancement_stage(diagnostics)

        metrics = self._segmentation_metrics(logits, targets)
        metrics.update(self._level_weight_metrics(diagnostics))
        if stage is not None:
            metrics.update(self._routing_metrics(stage))
        metrics.update(self._enhancement_metrics(enhancement))

        loss = self.criterion(logits, targets)
        if stage is not None and stage.latent_kl is not None:
            loss = loss + self.lambda_latent * stage.latent_kl
        if stage is not None and stage.balance is not None:
            loss = loss + self.lambda_balance * stage.balance

        return TaskStepOutput(loss=loss, metrics=metrics, batch_size=images.shape[0])
