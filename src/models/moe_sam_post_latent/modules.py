"""Post-MoE conditioning used only by the controlled mean-latent experiment."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class PostLatentFiLMConditioningOutput:
    """Conditioned features and diagnostics for latent-only FiLM."""

    features: Tensor
    delta: Tensor
    delta_norm_ratio: Tensor
    gamma: Tensor
    beta: Tensor


class PostLatentFiLMConditioning(nn.Module):
    """Scale and shift an embedding using parameters generated only from z."""

    def __init__(
        self,
        *,
        feature_dim: int = 256,
        latent_dim: int = 8,
        latent_projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if latent_projection_dim <= 0:
            raise ValueError("latent_projection_dim must be positive.")

        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.latent_projection_dim = int(latent_projection_dim)
        self.latent_projection = nn.Linear(
            self.latent_dim,
            self.latent_projection_dim,
            bias=False,
        )
        self.activation = nn.GELU()
        self.film_projection = nn.Linear(
            self.latent_projection_dim,
            2 * self.feature_dim,
            bias=False,
        )
        nn.init.zeros_(self.film_projection.weight)

    def condition(
        self,
        features: Tensor,
        latent: Tensor,
    ) -> PostLatentFiLMConditioningOutput:
        if features.ndim != 4 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be [B, {self.feature_dim}, H, W], got "
                f"{tuple(features.shape)}."
            )
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must be [B, {self.latent_dim}], got "
                f"{tuple(latent.shape)}."
            )
        if features.shape[0] != latent.shape[0]:
            raise ValueError("features and latent must have the same batch size.")

        hidden = self.activation(self.latent_projection(latent))
        gamma, beta = self.film_projection(hidden).chunk(2, dim=1)
        gamma = gamma.to(features.dtype).unsqueeze(-1).unsqueeze(-1)
        beta = beta.to(features.dtype).unsqueeze(-1).unsqueeze(-1)
        delta = features * gamma + beta
        conditioned = features + delta

        delta_norm = delta.float().flatten(1).norm(dim=1)
        base_norm = features.float().flatten(1).norm(dim=1)
        ratio = delta_norm / (base_norm + 1e-12)
        return PostLatentFiLMConditioningOutput(
            features=conditioned,
            delta=delta,
            delta_norm_ratio=ratio,
            gamma=gamma,
            beta=beta,
        )

    def forward(self, features: Tensor, latent: Tensor) -> Tensor:
        return self.condition(features, latent).features


__all__ = [
    "PostLatentFiLMConditioning",
    "PostLatentFiLMConditioningOutput",
]
