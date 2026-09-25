"""Shared conditioning modules for the C1/C2/C3 latent experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PreMoEContextAdapter(nn.Module):
    """Map ``[h_I; latent_slot]`` onto the shared MoE context width."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        latent_dim: int = 8,
        context_dim: int = 256,
    ) -> None:
        super().__init__()
        for name, value in (
            ("descriptor_dim", descriptor_dim),
            ("latent_dim", latent_dim),
            ("context_dim", context_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.descriptor_dim = descriptor_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.adapter = nn.Sequential(
            nn.Linear(descriptor_dim + latent_dim, context_dim),
            nn.GELU(),
        )

    def forward(self, image_descriptor: Tensor, latent_slot: Tensor) -> Tensor:
        if image_descriptor.ndim != 2:
            raise ValueError(
                "image_descriptor must be [B, descriptor_dim], got "
                f"{tuple(image_descriptor.shape)}."
            )
        if latent_slot.ndim != 2:
            raise ValueError(
                f"latent_slot must be [B, latent_dim], got {tuple(latent_slot.shape)}."
            )
        if image_descriptor.shape != (
            latent_slot.shape[0],
            self.descriptor_dim,
        ):
            raise ValueError(
                "image_descriptor and latent_slot must share B and use "
                f"descriptor_dim={self.descriptor_dim}; got "
                f"{tuple(image_descriptor.shape)} and {tuple(latent_slot.shape)}."
            )
        if latent_slot.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent_slot must have latent_dim={self.latent_dim}, got "
                f"{latent_slot.shape[1]}."
            )
        return self.adapter(
            torch.cat(
                [image_descriptor, latent_slot.to(image_descriptor.dtype)],
                dim=1,
            )
        )


class PostLatentConditioning(nn.Module):
    """Inject an eight-dimensional latent into a SAM decoder feature map."""

    def __init__(
        self,
        *,
        feature_dim: int = 256,
        latent_dim: int = 8,
        latent_projection_dim: int = 64,
    ) -> None:
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("latent_dim", latent_dim),
            ("latent_projection_dim", latent_projection_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.latent_projection_dim = latent_projection_dim
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_projection_dim),
            nn.GELU(),
        )
        self.fusion = nn.Conv2d(
            feature_dim + latent_projection_dim,
            feature_dim,
            kernel_size=1,
        )

    def forward(self, features: Tensor, latent: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be [B, {self.feature_dim}, H, W], got "
                f"{tuple(features.shape)}."
            )
        if latent.ndim != 2 or latent.shape != (
            features.shape[0],
            self.latent_dim,
        ):
            raise ValueError(
                f"latent must be [B, {self.latent_dim}] aligned with features, "
                f"got {tuple(latent.shape)}."
            )
        projected = self.latent_projection(latent)
        projected = projected.to(features.dtype).view(
            features.shape[0],
            self.latent_projection_dim,
            1,
            1,
        )
        projected = projected.expand(
            -1,
            -1,
            features.shape[2],
            features.shape[3],
        )
        delta = self.fusion(torch.cat([features, projected], dim=1))
        return features + delta


__all__ = ["PostLatentConditioning", "PreMoEContextAdapter"]
