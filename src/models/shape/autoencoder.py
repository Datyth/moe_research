"""Latent-only shape autoencoder composition."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class ShapeAutoencoderOutput:
    """Raw reconstruction logits and compact shape latent."""

    reconstruction_logits: Tensor
    latent: Tensor


class ShapeAutoencoder(nn.Module):
    """Compose a mask encoder, spatial projector, and latent-only decoder."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder

    def forward(self, masks: Tensor) -> ShapeAutoencoderOutput:
        if masks.ndim != 4:
            raise ValueError(
                "ShapeAutoencoder input must have shape [B, 1, 256, 256], "
                f"got {tuple(masks.shape)}."
            )
        if masks.shape[1] != 1:
            raise ValueError(
                "ShapeAutoencoder input must have exactly one channel, "
                f"got {masks.shape[1]}."
            )
        if masks.shape[-2:] != (256, 256):
            raise ValueError(
                "ShapeAutoencoder input spatial size must be (256, 256), "
                f"got {tuple(masks.shape[-2:])}."
            )

        features = self.encoder(masks)
        latent = self.projector(features)
        reconstruction_logits = self.decoder(latent)
        return ShapeAutoencoderOutput(
            reconstruction_logits=reconstruction_logits,
            latent=latent,
        )
