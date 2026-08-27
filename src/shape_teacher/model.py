"""Gaussian posterior Shape Teacher from ``docs/teacher_plan.md``."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ShapeTeacherOutput(NamedTuple):
    """Public output contract: logits, sampled latent, mean and positive scale."""

    logits: Tensor
    z: Tensor
    mu: Tensor
    sigma: Tensor


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


class ShapeTeacher(nn.Module):
    """Encode a binary mask into a Gaussian shape posterior and reconstruct it.

    The posterior deliberately has no KL regularizer. Training therefore observes
    whether the learned scale collapses under reconstruction-only supervision.
    """

    def __init__(
        self,
        *,
        image_size: tuple[int, int] | list[int] = (256, 256),
        mask_channels: int = 1,
        encoder_channels: tuple[int, ...] | list[int] = (32, 64, 128, 256),
        feature_dim: int = 256,
        latent_dim: int = 128,
        decoder_channels: tuple[int, ...] | list[int] = (128, 64, 32, 16),
        sigma_floor: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.image_size = tuple(int(value) for value in image_size)
        self.mask_channels = int(mask_channels)
        self.encoder_channels = tuple(int(value) for value in encoder_channels)
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.decoder_channels = tuple(int(value) for value in decoder_channels)
        self.sigma_floor = float(sigma_floor)
        self._validate_config()

        encoder: list[nn.Module] = []
        previous = self.mask_channels
        for channels in self.encoder_channels:
            encoder.append(_encoder_block(previous, channels))
            previous = channels
        self.encoder = nn.Sequential(*encoder)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_projection = (
            nn.Identity()
            if previous == self.feature_dim
            else nn.Linear(previous, self.feature_dim)
        )
        self.mean_head = nn.Linear(self.feature_dim, self.latent_dim)
        self.scale_head = nn.Linear(self.feature_dim, self.latent_dim)

        downsample_factor = 2 ** len(self.encoder_channels)
        self.seed_size = tuple(value // downsample_factor for value in self.image_size)
        seed_elements = self.feature_dim * self.seed_size[0] * self.seed_size[1]
        self.decoder_input = nn.Linear(self.latent_dim, seed_elements)

        decoder: list[nn.Module] = []
        previous = self.feature_dim
        for channels in self.decoder_channels:
            decoder.append(_decoder_block(previous, channels))
            previous = channels
        self.decoder = nn.Sequential(*decoder)
        self.decoder_output = nn.Conv2d(
            previous,
            self.mask_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _validate_config(self) -> None:
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("image_size must contain two positive integers.")
        if self.mask_channels <= 0:
            raise ValueError("mask_channels must be positive.")
        if not self.encoder_channels or any(value <= 0 for value in self.encoder_channels):
            raise ValueError("encoder_channels must contain positive integers.")
        if not self.decoder_channels or any(value <= 0 for value in self.decoder_channels):
            raise ValueError("decoder_channels must contain positive integers.")
        if len(self.decoder_channels) != len(self.encoder_channels):
            raise ValueError(
                "decoder_channels and encoder_channels must have the same length."
            )
        if self.feature_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("feature_dim and latent_dim must be positive.")
        if self.sigma_floor <= 0.0:
            raise ValueError("sigma_floor must be positive.")
        factor = 2 ** len(self.encoder_channels)
        if any(value % factor for value in self.image_size):
            raise ValueError(
                f"image_size {self.image_size} must be divisible by {factor}."
            )

    def _validate_masks(self, masks: Tensor) -> None:
        expected = (self.mask_channels, *self.image_size)
        if masks.ndim != 4 or tuple(masks.shape[1:]) != expected:
            raise ValueError(
                f"masks must have shape [B, {expected[0]}, {expected[1]}, "
                f"{expected[2]}], got {tuple(masks.shape)}."
            )
        if not masks.is_floating_point():
            raise TypeError("masks must be floating point.")

    def encode(self, masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``h_T``, ``mu_T`` and strictly positive ``sigma_T``."""

        self._validate_masks(masks)
        h = self.pool(self.encoder(masks)).flatten(1)
        h = self.feature_projection(h)
        mu = self.mean_head(h)
        sigma = F.softplus(self.scale_head(h)) + self.sigma_floor
        return h, mu, sigma

    @staticmethod
    def reparameterize(mu: Tensor, sigma: Tensor) -> Tensor:
        if mu.shape != sigma.shape:
            raise ValueError("mu and sigma must have the same shape.")
        return mu + sigma * torch.randn_like(sigma)

    def decode(self, z: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must have shape [B, {self.latent_dim}], got {tuple(z.shape)}."
            )
        seed = self.decoder_input(z).view(
            z.shape[0],
            self.feature_dim,
            self.seed_size[0],
            self.seed_size[1],
        )
        logits = self.decoder_output(self.decoder(seed))
        if tuple(logits.shape[2:]) != self.image_size:
            raise RuntimeError(
                f"decoder produced {tuple(logits.shape[2:])}, expected {self.image_size}."
            )
        return logits

    def forward(self, masks: Tensor, sample: bool = True) -> ShapeTeacherOutput:
        _, mu, sigma = self.encode(masks)
        z = self.reparameterize(mu, sigma) if sample else mu
        return ShapeTeacherOutput(self.decode(z), z, mu, sigma)

    def model_config(self) -> dict[str, object]:
        return {
            "image_size": list(self.image_size),
            "mask_channels": self.mask_channels,
            "encoder_channels": list(self.encoder_channels),
            "feature_dim": self.feature_dim,
            "latent_dim": self.latent_dim,
            "decoder_channels": list(self.decoder_channels),
            "sigma_floor": self.sigma_floor,
        }
