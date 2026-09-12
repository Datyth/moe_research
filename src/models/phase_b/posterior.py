"""Diagonal-Gaussian latents for privileged routing (proposal Phase B).

Two heads with identical shape but different roles:

- the **posterior** ``q_phi(z | I, M)`` reads the fused descriptor ``h_q = [h_I ; h_M]``,
  so it sees the ground-truth mask and is *privileged*;
- the **prior** ``p_theta(z | I)`` reads only ``h_I``, so it is *deployable* and is
  the branch that survives at inference.

Both predict a diagonal Gaussian ``N(mu, diag(sigma^2))`` over the routing latent
``z in R^{d_z}``. The posterior is the teacher; the prior is pulled toward it by the
latent KL (see :func:`gaussian_kl`), which is why the two share this one head class
rather than diverging in parametrization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class DiagonalGaussian:
    """A batch of diagonal Gaussians in the routing-latent space."""

    mean: Tensor
    """mu, shape [B, d_z]."""

    std: Tensor
    """sigma > 0 (standard deviation, not variance), shape [B, d_z]."""

    def rsample(self, generator: torch.Generator | None = None) -> Tensor:
        """Reparameterized draw ``z = mu + sigma * eps``, ``eps ~ N(0, I)``.

        Differentiable in both ``mu`` and ``sigma`` (the point of the
        reparameterization trick): the randomness sits in ``eps``, which carries
        no gradient, so router gradients flow back into the distribution heads.
        """

        eps = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + self.std * eps

    def detach(self) -> "DiagonalGaussian":
        """A gradient-stopped copy, for the ``sg[q]`` in ``KL(sg[q] || p)``."""

        return DiagonalGaussian(mean=self.mean.detach(), std=self.std.detach())


class GaussianParameterHead(nn.Module):
    """Map a descriptor to ``(mu, sigma)`` of a diagonal Gaussian.

    Follows the proposal: two independent affine maps produce the mean and the
    pre-activation scale ``rho``, and ``sigma = Softplus(rho) + eps``. Softplus
    keeps ``sigma`` strictly positive while staying differentiable, and the small
    floor ``eps`` stops the KL's ``1/sigma_p^2`` and ``log sigma`` terms from
    blowing up when a scale is driven toward zero.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        latent_dim: int = 64,
        std_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if std_floor < 0:
            raise ValueError("std_floor must be non-negative.")

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.std_floor = std_floor
        self.mean_head = nn.Linear(in_dim, latent_dim)
        self.scale_head = nn.Linear(in_dim, latent_dim)

    def forward(self, descriptor: Tensor) -> DiagonalGaussian:
        if descriptor.ndim != 2:
            raise ValueError(
                f"descriptor must be [B, in_dim], got {tuple(descriptor.shape)}."
            )
        if descriptor.shape[1] != self.in_dim:
            raise ValueError(
                f"descriptor must have in_dim={self.in_dim}, got {descriptor.shape[1]}."
            )
        mean = self.mean_head(descriptor)
        std = torch.nn.functional.softplus(self.scale_head(descriptor)) + self.std_floor
        return DiagonalGaussian(mean=mean, std=std)


def gaussian_kl(q: DiagonalGaussian, p: DiagonalGaussian) -> Tensor:
    """Closed-form ``KL(q || p)`` for diagonal Gaussians, per sample.

    Returns shape ``[B]`` (summed over the latent dimension); the caller reduces
    over the batch. To realize the proposal's ``KL(sg[q] || p)`` — the posterior
    is a fixed target and only the prior is trained — pass ``q.detach()``.

        KL = 1/2 * sum_j [ log(sigma_p_j^2 / sigma_q_j^2)
                           + (sigma_q_j^2 + (mu_q_j - mu_p_j)^2) / sigma_p_j^2 - 1 ]
    """

    if q.mean.shape != p.mean.shape:
        raise ValueError(
            f"q and p must share shape, got {tuple(q.mean.shape)} and "
            f"{tuple(p.mean.shape)}."
        )
    var_q = q.std.pow(2)
    var_p = p.std.pow(2)
    # log(var_p/var_q) via 2*(log sigma_p - log sigma_q): numerically steadier
    # than logging a ratio when either scale is small.
    log_ratio = 2.0 * (p.std.log() - q.std.log())
    term = log_ratio + (var_q + (q.mean - p.mean).pow(2)) / var_p - 1.0
    return 0.5 * term.sum(dim=1)
