"""The fuse stage: h_q = [h_I ; h_M].

This is where the image-only routing descriptor meets the privileged shape
latent. Everything downstream of this point (the posterior q(z | I, M), the
Top-K routing it induces, and the hierarchical enhancement) consumes h_q; at
inference the h_M half disappears and the prior p(z | I) reads h_I alone.
See `moe_enhancement` for the stage that consumes the routing downstream of h_q.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PrivilegedFusionOutput:
    """The fused routing representation and the two halves that formed it."""

    fused: Tensor
    """h_q, shape [B, C_s + d_M]."""

    image_descriptor: Tensor
    """h_I, shape [B, C_s]."""

    shape_latent: Tensor
    """h_M, shape [B, d_M]."""


class PrivilegedFusion(nn.Module):
    """Concatenate h_I and h_M into the privileged routing representation.

    The proposal defines the fusion as plain concatenation, so no parameters
    are introduced here; the posterior's linear heads are what learn from h_q.
    The module exists to pin the contract — dimensions, batch agreement, and
    the recorded output width other stages build against.
    """

    def __init__(self, *, descriptor_dim: int = 256, shape_latent_dim: int = 256) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive.")
        if shape_latent_dim <= 0:
            raise ValueError("shape_latent_dim must be positive.")
        self.descriptor_dim = descriptor_dim
        self.shape_latent_dim = shape_latent_dim

    @property
    def output_dim(self) -> int:
        """Width of h_q, which the posterior heads are sized against."""

        return self.descriptor_dim + self.shape_latent_dim

    def forward(
        self,
        image_descriptor: Tensor,
        shape_latent: Tensor,
    ) -> PrivilegedFusionOutput:
        if image_descriptor.ndim != 2:
            raise ValueError(
                "h_I must have shape [B, C_s], got "
                f"{tuple(image_descriptor.shape)}."
            )
        if shape_latent.ndim != 2:
            raise ValueError(
                f"h_M must have shape [B, d_M], got {tuple(shape_latent.shape)}."
            )
        if image_descriptor.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"h_I must have C_s={self.descriptor_dim}, got "
                f"{image_descriptor.shape[1]}."
            )
        if shape_latent.shape[1] != self.shape_latent_dim:
            raise ValueError(
                f"h_M must have d_M={self.shape_latent_dim}, got "
                f"{shape_latent.shape[1]}."
            )
        if image_descriptor.shape[0] != shape_latent.shape[0]:
            raise ValueError(
                "h_I and h_M must share a batch size, got "
                f"{image_descriptor.shape[0]} and {shape_latent.shape[0]}."
            )

        return PrivilegedFusionOutput(
            fused=torch.cat(
                [image_descriptor, shape_latent.to(image_descriptor.dtype)],
                dim=1,
            ),
            image_descriptor=image_descriptor,
            shape_latent=shape_latent,
        )
