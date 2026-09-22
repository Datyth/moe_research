"""Phase B router-stage task: segmentation loss + KL + load-balance.

Adds the two privileged-routing terms to the fuse-stage task. The segmentation
loss still comes from the backbone logits (the router does not yet drive the
mask); the extra terms train the posterior/prior/router:

    L = L_seg + lambda_latent * L_latent + lambda_bal * L_bal

``L_latent = KL(sg[q] || p)`` pulls the deployable prior toward the privileged
posterior, and ``L_bal`` keeps expert load from collapsing. ``strict_router``
defaults to True so a run that stopped emitting the ``phase_b_router`` diagnostic
fails loudly instead of quietly training on the segmentation loss alone.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import TaskStepOutput
from .phase_b_diagnostics import routing_metrics
from .phase_b_fuse import PhaseBFuseTask


class PhaseBRouterTask(PhaseBFuseTask):
    """Fuse-stage task plus the KL and load-balance routing losses."""

    def __init__(
        self,
        *,
        criterion: nn.Module,
        threshold: float = 0.5,
        boundary_tolerance: float = 2,
        task: str = "binary",
        strict_fuse: bool = True,
        strict_router: bool = True,
        lambda_latent: float = 0.1,
        lambda_balance: float = 0.01,
    ) -> None:
        super().__init__(
            criterion=criterion,
            threshold=threshold,
            boundary_tolerance=boundary_tolerance,
            task=task,
            strict_fuse=strict_fuse,
        )
        if lambda_latent < 0 or lambda_balance < 0:
            raise ValueError("Loss weights must be non-negative.")
        self.strict_router = bool(strict_router)
        self.lambda_latent = float(lambda_latent)
        self.lambda_balance = float(lambda_balance)

    def _router_stage(self, diagnostics: dict[str, Any]):
        stage = diagnostics.get("phase_b_router")
        if stage is None and self.strict_router:
            raise ValueError(
                "Phase B router task expected a 'phase_b_router' diagnostic; got "
                f"keys {sorted(diagnostics)}. Is this a phase_b_router model?"
            )
        return stage

    @staticmethod
    def _routing_metrics(stage: Any) -> dict[str, torch.Tensor]:
        return routing_metrics(stage)

    def training_step(self, model: nn.Module, batch: Any, device: Any) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, diagnostics = self._forward_with_privileged_mask(model, images, targets)
        stage = self._router_stage(diagnostics)

        loss = self.criterion(logits, targets)
        if stage is not None and stage.latent_kl is not None:
            loss = loss + self.lambda_latent * stage.latent_kl
        if stage is not None and stage.balance is not None:
            loss = loss + self.lambda_balance * stage.balance

        return TaskStepOutput(loss=loss, metrics={}, batch_size=images.shape[0])

    def evaluation_step(self, model: nn.Module, batch: Any, device: Any) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, diagnostics = self._forward_with_privileged_mask(model, images, targets)
        stage = self._router_stage(diagnostics)

        metrics = self._segmentation_metrics(logits, targets)
        metrics.update(self._level_weight_metrics(diagnostics))
        if stage is not None:
            metrics.update(self._routing_metrics(stage))

        loss = self.criterion(logits, targets)
        if stage is not None and stage.latent_kl is not None:
            loss = loss + self.lambda_latent * stage.latent_kl
        if stage is not None and stage.balance is not None:
            loss = loss + self.lambda_balance * stage.balance

        return TaskStepOutput(loss=loss, metrics=metrics, batch_size=images.shape[0])
