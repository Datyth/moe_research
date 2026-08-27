"""Phase 1 objective: segmentation, expert balancing, and shape distillation.

    L = L_seg(M_hat, M)
      + lambda_balance     * L_CV2(pi)
      + lambda_distillation * KL( q_S(z|I) || q_T(z|M) )

The first two terms are ShapeMoE Eq. (5), with cross-entropy generalised to the
repository's configurable segmentation losses. The third is this project's own
addition and is not in the paper: it is what transfers the Phase 0 teacher's
privileged shape posterior into the student encoder.

Like ``MaskVAELoss``, this is not registered in ``LOSS_REGISTRY`` because it
does not obey the ``criterion(logits, targets)`` contract that ``Trainer`` uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.models.base import SegmentationOutput

from .registry import build_loss


@dataclass
class ShapeMoELosses:
    """Total objective plus each term, kept for logging."""

    total: Tensor
    segmentation: Tensor
    balance: Tensor
    distillation: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "segmentation": float(self.segmentation.detach()),
            "balance": float(self.balance.detach()),
            "distillation": float(self.distillation.detach()),
        }


def expert_balance_cv2_loss(probabilities: Tensor, *, epsilon: float = 1e-8) -> Tensor:
    """Squared coefficient of variation of expert importance (Shazeer et al. 2017).

    Importance is the per-expert sum of routing probability over the batch. A
    perfectly balanced router gives every expert the same importance, so the
    coefficient of variation, and therefore this loss, is zero.

    The dense softmax is the right input here rather than the sparse ``pi`` of
    Eq. (4): with top_k=1 the sparse weights are identically 1.0 for the selected
    expert, which is a constant with no gradient to the routing matrix W.
    """

    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must have shape [B, K], got "
            f"{tuple(probabilities.shape)}."
        )
    if probabilities.shape[1] < 2:
        return probabilities.sum() * 0.0

    importance = probabilities.sum(dim=0)
    mean = importance.mean()
    variance = importance.var(unbiased=False)
    return variance / (mean.pow(2) + epsilon)


def gaussian_kl_divergence(
    mu_q: Tensor,
    logvar_q: Tensor,
    mu_p: Tensor,
    logvar_p: Tensor,
) -> Tensor:
    """Per-sample KL( N(mu_q, diag) || N(mu_p, diag) ), summed over dimensions.

    Closed form: 0.5 * sum( logvar_p - logvar_q
                            + (var_q + (mu_q - mu_p)^2) / var_p - 1 ).
    """

    shapes = {tuple(tensor.shape) for tensor in (mu_q, logvar_q, mu_p, logvar_p)}
    if len(shapes) != 1:
        raise ValueError(f"All posterior tensors must share a shape, got {shapes}.")
    if mu_q.ndim != 2:
        raise ValueError(
            f"Posterior tensors must have shape [B, D], got {tuple(mu_q.shape)}."
        )

    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    return 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0,
        dim=1,
    )


class ShapeMoELoss(nn.Module):
    """Segmentation loss plus expert balancing plus posterior distillation."""

    def __init__(
        self,
        *,
        segmentation: dict[str, Any] | None = None,
        balance_weight: float = 1.0,
        distillation_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if balance_weight < 0.0:
            raise ValueError("balance_weight must be non-negative.")
        if distillation_weight < 0.0:
            raise ValueError("distillation_weight must be non-negative.")

        self.segmentation = build_loss(
            dict(segmentation) if segmentation else {"name": "bce_dice"}
        )
        self.balance_weight = float(balance_weight)
        self.distillation_weight = float(distillation_weight)

    def forward(
        self,
        output: SegmentationOutput,
        targets: Tensor,
        *,
        teacher_posterior: tuple[Tensor, Tensor] | None = None,
    ) -> ShapeMoELosses:
        diagnostics = output.diagnostics
        for key in ("router_probabilities", "mu", "logvar"):
            if key not in diagnostics:
                raise KeyError(
                    f"SegmentationOutput.diagnostics is missing '{key}'; this "
                    "loss expects a ShapeMoE model."
                )

        segmentation = self.segmentation(output.logits, targets)
        balance = expert_balance_cv2_loss(diagnostics["router_probabilities"])

        if teacher_posterior is None:
            distillation = segmentation.new_zeros(())
        else:
            teacher_mu, teacher_logvar = teacher_posterior
            distillation = gaussian_kl_divergence(
                diagnostics["mu"],
                diagnostics["logvar"],
                teacher_mu,
                teacher_logvar,
            ).mean()

        total = (
            segmentation
            + self.balance_weight * balance
            + self.distillation_weight * distillation
        )
        return ShapeMoELosses(
            total=total,
            segmentation=segmentation,
            balance=balance,
            distillation=distillation,
        )
