"""Phase 0 objective for the mask-VAE shape teacher.

    L_rec   = BCE(M, M_hat_T)
    L_prior = KL( q_T(z|M) || N(0, I) )
    L_VAE   = L_rec + beta * L_prior

This loss is deliberately absent from ``LOSS_REGISTRY``. Registered losses obey
the ``criterion(logits, targets) -> scalar`` contract that ``Trainer`` and
``evaluate`` rely on, whereas this one consumes the full posterior and returns
its components separately. Registering it would let it be selected by a
segmentation config that cannot call it correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.shapemoe.teacher import MaskVAEOutput


@dataclass
class MaskVAELosses:
    """Total objective plus the two terms, kept for logging."""

    total: Tensor
    reconstruction: Tensor
    kl: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "reconstruction": float(self.reconstruction.detach()),
            "kl": float(self.kl.detach()),
        }


def gaussian_kl_to_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """Per-sample KL( N(mu, diag(exp(logvar))) || N(0, I) ), summed over dims.

    Closed form: 0.5 * sum(mu^2 + sigma^2 - 1 - log sigma^2).
    """

    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu and logvar must have the same shape, got {tuple(mu.shape)} "
            f"and {tuple(logvar.shape)}."
        )
    if mu.ndim != 2:
        raise ValueError(
            f"mu and logvar must have shape [B, D], got {tuple(mu.shape)}."
        )

    return 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - 1.0 - logvar,
        dim=1,
    )


class MaskVAELoss(nn.Module):
    """Reconstruction plus KL, weighted by beta.

    ``recon_reduction='sum'`` sums BCE over the pixels of each mask and averages
    over the batch. This is the usual VAE convention and it keeps the two terms
    on a comparable scale, so beta=1 corresponds to the plain ELBO. Switching to
    ``'mean'`` divides the reconstruction term by the pixel count, which makes
    the KL term dominate unless beta is reduced by roughly the same factor.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        recon_reduction: str = "sum",
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()

        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if recon_reduction not in {"sum", "mean"}:
            raise ValueError("recon_reduction must be 'sum' or 'mean'.")

        weight_tensor = (
            None
            if pos_weight is None
            else torch.as_tensor(pos_weight, dtype=torch.float32)
        )
        self.register_buffer("pos_weight", weight_tensor)
        self.beta = float(beta)
        self.recon_reduction = recon_reduction

    def forward(
        self,
        output: MaskVAEOutput,
        targets: Tensor,
        *,
        beta: float | None = None,
    ) -> MaskVAELosses:
        recon_logits = output.recon_logits
        if recon_logits.shape != targets.shape:
            raise ValueError(
                f"reconstruction and targets must have the same shape, got "
                f"{tuple(recon_logits.shape)} and {tuple(targets.shape)}."
            )

        per_pixel = F.binary_cross_entropy_with_logits(
            recon_logits,
            targets.to(dtype=recon_logits.dtype),
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if self.recon_reduction == "sum":
            reconstruction = per_pixel.flatten(1).sum(dim=1).mean()
        else:
            reconstruction = per_pixel.flatten(1).mean(dim=1).mean()

        kl = gaussian_kl_to_standard_normal(output.mu, output.logvar).mean()
        weight = self.beta if beta is None else float(beta)

        return MaskVAELosses(
            total=reconstruction + weight * kl,
            reconstruction=reconstruction,
            kl=kl,
        )
