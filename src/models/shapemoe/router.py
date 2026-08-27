"""Shape-Aware Sparse Router (ShapeMoE Sec. 3.4, Eq. 2-4).

    l_o = mu + sigma * eta,   eta ~ N(0, I)          (Eq. 2)
    s   = W * l_o                                    (Eq. 3)
    pi  = Softmax(TopK(s, k))                        (Eq. 4)

Eq. (2) in the paper reads ``l_o = mu + Softplus(sigma) * eta``. Here sigma comes
from a log-variance head, so the equivalent step is ``exp(0.5 * logvar)``; see
docs/shapemoe_assumptions.md for why the whole pipeline uses log-variance.

The router also exposes the dense softmax over all experts. Eq. (4) masks every
non-selected score to -inf, which makes ``pi`` a constant 1.0 whenever k = 1 and
therefore carries no gradient to W. The balancing loss of Sec. 3.6 needs a
gradient to do anything at all, so it consumes the dense probabilities while the
expert combination follows Eq. (4) literally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RouterOutput:
    """Routing decision for one batch."""

    pi: Tensor
    indices: Tensor
    probabilities: Tensor
    scores: Tensor
    latent: Tensor


class ShapeAwareSparseRouter(nn.Module):
    """Sample a latent shape code and route it to the top-k experts."""

    def __init__(
        self,
        *,
        latent_dim: int,
        num_experts: int,
        top_k: int = 1,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts={num_experts}], got {top_k}."
            )

        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Eq. (3): a plain trainable matrix, no bias.
        self.gate = nn.Linear(latent_dim, num_experts, bias=False)

    def sample_latent(
        self,
        mu: Tensor,
        logvar: Tensor,
        *,
        sample: bool = True,
    ) -> Tensor:
        """Eq. (2). Falls back to the mean when sampling is disabled."""

        if mu.shape != logvar.shape:
            raise ValueError(
                f"mu and logvar must have the same shape, got {tuple(mu.shape)} "
                f"and {tuple(logvar.shape)}."
            )
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(
        self,
        mu: Tensor,
        logvar: Tensor,
        *,
        sample: bool = True,
    ) -> RouterOutput:
        latent = self.sample_latent(mu, logvar, sample=sample)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape [B, {self.latent_dim}], got "
                f"{tuple(latent.shape)}."
            )

        scores = self.gate(latent)
        probabilities = scores.softmax(dim=1)

        top_values, indices = scores.topk(self.top_k, dim=1)
        masked = torch.full_like(scores, float("-inf"))
        masked = masked.scatter(1, indices, top_values)
        pi = masked.softmax(dim=1)

        return RouterOutput(
            pi=pi,
            indices=indices,
            probabilities=probabilities,
            scores=scores,
            latent=latent,
        )
