"""Shape Distribution Encoder E_S (ShapeMoE Sec. 3.3, Eq. 1).

The paper feeds E_S a mask embedding ``e_m`` that SAM's prompt encoder produces
from the visible mask. This project has no prompt branch, and the student is not
allowed to see any ground-truth mask, so E_S reads an image feature map instead:
the segmentation trunk's bottleneck, pooled to a vector.

The structure the paper does prescribe is preserved exactly: one shared input
feeding two separate projections, an expectation encoder and a variance encoder.
As in the Phase 0 teacher, the variance head predicts log-variance rather than
the raw sigma of Eq. (2), so that every KL in the pipeline stays closed-form.
"""

from __future__ import annotations

from torch import Tensor, nn


class ShapeDistributionEncoder(nn.Module):
    """Map a feature map to the Gaussian shape posterior q_S(z|I).

        h_S      = GAP(features)
        mu_S     = E_mu(h_S)
        logvar_S = E_sigma(h_S)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max.")

        self.latent_dim = latent_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.pool = nn.AdaptiveAvgPool2d(1)
        if hidden_dim is None:
            self.trunk: nn.Module = nn.Identity()
            head_dim = in_channels
        else:
            self.trunk = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(inplace=True),
            )
            head_dim = hidden_dim

        # Eq. (1): two separate encoders sharing one input.
        self.fc_mu = nn.Linear(head_dim, latent_dim)
        self.fc_logvar = nn.Linear(head_dim, latent_dim)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 4:
            raise ValueError(
                f"features must have shape [B, C, H, W], got "
                f"{tuple(features.shape)}."
            )

        hidden = self.trunk(self.pool(features).flatten(1))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar
