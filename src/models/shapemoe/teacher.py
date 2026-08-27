"""Mask-VAE shape teacher (Phase 0) for the ShapeMoE-style pipeline.

The teacher observes the ground-truth mask directly, which makes it a
privileged-information model: it never sees the input image. Its purpose is to
learn a latent shape distribution that later phases can distil into a student
encoder.

The Gaussian head follows Eq. (1) of ShapeMoE (arXiv 2508.01664), where a shared
trunk feeds two separate projections for the mean and the spread. The
parametrisation differs deliberately: ShapeMoE Eq. (2) applies Softplus to a raw
sigma, while this module predicts log-variance so that the KL term against
N(0, I) stays closed-form and numerically stable. See
docs/shapemoe_assumptions.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass
class MaskVAEOutput:
    """Everything Phase 0 needs from one teacher forward pass."""

    recon_logits: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Stride-2 convolution that halves the spatial resolution."""

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Transposed convolution that doubles the spatial resolution."""

    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FrozenModule(nn.Module):
    """Wrap a module so it stays frozen even when a parent calls ``.train()``.

    ``requires_grad_(False)`` alone is not enough to freeze a block containing
    BatchNorm: in training mode those layers keep updating their running
    statistics, so the block's output drifts even though no gradient reaches it.
    ``.eval()`` fixes that but does not stick, because ``nn.Module.train()``
    recurses into every child. Overriding ``train`` here is what makes it stick.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenModule":
        super().train(False)
        self.module.eval()
        return self

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MaskVAETeacher(nn.Module):
    """Encode a binary mask into a Gaussian shape posterior and rebuild it.

    Forward pass, matching the Phase 0 specification:

        M -> E_T(M) -> h_T
        mu_T     = W_mu    h_T + b_mu
        logvar_T = W_sigma h_T + b_sigma
        z_T      = mu_T + sigma_T * eps,  eps ~ N(0, I)
        M_hat_T  = D_T(z_T)

    The decoder returns raw logits, so the reconstruction term must be a
    logit-space BCE. That is numerically equivalent to
    ``BCE(M, sigmoid(logits))``.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        image_size: tuple[int, int] | list[int] = (256, 256),
        mask_channels: int = 1,
        encoder_channels: tuple[int, ...] = (32, 64, 128, 256),
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16, 16, 16),
        decoder_seed_size: int = 4,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__()

        image_size = tuple(int(size) for size in image_size)
        encoder_channels = tuple(int(size) for size in encoder_channels)
        decoder_channels = tuple(int(size) for size in decoder_channels)

        self._validate(
            latent_dim=latent_dim,
            image_size=image_size,
            mask_channels=mask_channels,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            decoder_seed_size=decoder_seed_size,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.mask_channels = mask_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.decoder_seed_size = decoder_seed_size
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self._mask_embedding_frozen = False

        # E_M: mask -> mask embedding e_m, still a feature map. This is the part
        # later phases load and freeze, mirroring the frozen Mask Embedding
        # Encoder of the paper's Fig. 2.
        encoder_layers: list[nn.Module] = []
        previous = mask_channels
        for channels in encoder_channels:
            encoder_layers.append(_conv_block(previous, channels))
            previous = channels
        self.mask_embedding_encoder = nn.Sequential(*encoder_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.hidden_dim = previous
        self.embedding_channels = previous

        # Two separate projection heads, as in ShapeMoE Eq. (1).
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim)

        # D_T: project the latent to a small feature map, then upsample.
        seed_channels = decoder_channels[0]
        self.decoder_input = nn.Linear(
            latent_dim,
            seed_channels * decoder_seed_size * decoder_seed_size,
        )
        decoder_layers: list[nn.Module] = []
        previous = seed_channels
        for channels in decoder_channels[1:]:
            decoder_layers.append(_deconv_block(previous, channels))
            previous = channels
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder_output = nn.Conv2d(
            previous,
            mask_channels,
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def _validate(
        *,
        latent_dim: int,
        image_size: tuple[int, ...],
        mask_channels: int,
        encoder_channels: tuple[int, ...],
        decoder_channels: tuple[int, ...],
        decoder_seed_size: int,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if mask_channels <= 0:
            raise ValueError("mask_channels must be positive.")
        if decoder_seed_size <= 0:
            raise ValueError("decoder_seed_size must be positive.")
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max.")
        if len(image_size) != 2 or any(size <= 0 for size in image_size):
            raise ValueError("image_size must contain two positive integers.")
        if not encoder_channels:
            raise ValueError("encoder_channels must not be empty.")
        if len(decoder_channels) < 2:
            raise ValueError("decoder_channels needs at least two entries.")

        downsample = 2 ** len(encoder_channels)
        if any(size % downsample for size in image_size):
            raise ValueError(
                f"image_size {image_size} must be divisible by {downsample} "
                f"for {len(encoder_channels)} stride-2 encoder blocks."
            )

        upsample = 2 ** (len(decoder_channels) - 1)
        decoded = tuple(decoder_seed_size * upsample for _ in image_size)
        if decoded != image_size:
            raise ValueError(
                f"Decoder reconstructs {decoded} but image_size is "
                f"{image_size}. Adjust decoder_seed_size or decoder_channels."
            )

    def embed(self, masks: Tensor) -> Tensor:
        """Run E_M only: mask -> mask embedding e_m, shape [B, C, H/16, W/16].

        This is the boundary the paper draws between the frozen Mask Embedding
        Encoder and the trainable Gaussian branches, and therefore the part
        Phase 0 exists to pretrain.
        """

        self._validate_masks(masks)
        return self.mask_embedding_encoder(masks)

    def freeze_mask_embedding_encoder(self) -> None:
        """Freeze E_M, as the paper's Fig. 2 marks it.

        The freeze survives later ``.train()`` calls on this module, so the
        BatchNorm statistics inside E_M stop moving for good.
        """

        self.mask_embedding_encoder.eval()
        for parameter in self.mask_embedding_encoder.parameters():
            parameter.requires_grad_(False)
        self._mask_embedding_frozen = True

    def train(self, mode: bool = True) -> "MaskVAETeacher":
        super().train(mode)
        if getattr(self, "_mask_embedding_frozen", False):
            self.mask_embedding_encoder.eval()
        return self

    def mask_embedding_state_dict(self) -> dict[str, Tensor]:
        """Weights of E_M alone, for loading into a later phase."""

        return {
            key: value.detach().clone()
            for key, value in self.mask_embedding_encoder.state_dict().items()
        }

    def encode(self, masks: Tensor) -> tuple[Tensor, Tensor]:
        """Return the posterior parameters (mu_T, logvar_T) for a mask batch."""

        hidden = self.pool(self.embed(masks)).flatten(1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Draw z_T = mu_T + sigma_T * eps with eps ~ N(0, I)."""

        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: Tensor) -> Tensor:
        """Map a latent shape code back to mask logits."""

        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must have shape [B, {self.latent_dim}], got "
                f"{tuple(z.shape)}."
            )
        seed = self.decoder_input(z).view(
            z.shape[0],
            -1,
            self.decoder_seed_size,
            self.decoder_seed_size,
        )
        return self.decoder_output(self.decoder(seed))

    def forward(self, masks: Tensor, *, sample: bool = True) -> MaskVAEOutput:
        mu, logvar = self.encode(masks)
        z = self.reparameterize(mu, logvar) if sample else mu
        return MaskVAEOutput(
            recon_logits=self.decode(z),
            mu=mu,
            logvar=logvar,
            z=z,
        )

    def _validate_masks(self, masks: Tensor) -> None:
        if masks.ndim != 4:
            raise ValueError(
                f"masks must have shape [B, C, H, W], got {tuple(masks.shape)}."
            )
        if masks.shape[1] != self.mask_channels:
            raise ValueError(
                f"masks must have {self.mask_channels} channel(s), got "
                f"{masks.shape[1]}."
            )
        if tuple(masks.shape[2:]) != self.image_size:
            raise ValueError(
                f"masks must be {self.image_size}, got {tuple(masks.shape[2:])}."
            )
        if not masks.is_floating_point():
            raise TypeError("masks must be a floating-point tensor.")

    def embedding_geometry(self) -> dict[str, object]:
        """Shape facts a later phase needs to consume e_m."""

        stride = 2 ** len(self.encoder_channels)
        return {
            "embedding_channels": self.embedding_channels,
            "embedding_stride": stride,
            "embedding_size": [size // stride for size in self.image_size],
            "mask_channels": self.mask_channels,
            "image_size": list(self.image_size),
        }

    def model_config(self) -> dict[str, object]:
        """Serializable constructor arguments, stored inside checkpoints."""

        return {
            "latent_dim": self.latent_dim,
            "image_size": list(self.image_size),
            "mask_channels": self.mask_channels,
            "encoder_channels": tuple(self.encoder_channels),
            "decoder_channels": tuple(self.decoder_channels),
            "decoder_seed_size": self.decoder_seed_size,
            "logvar_min": self.logvar_min,
            "logvar_max": self.logvar_max,
        }


def load_mask_embedding_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    freeze: bool = True,
) -> tuple[nn.Module, dict[str, object]]:
    """Load E_M alone from a Phase 0 checkpoint, frozen by default.

    Phase 0 exists to produce this: an unsupervised, pretrained mask embedding
    encoder that later phases load and keep fixed, the way ShapeMoE keeps SAM2's
    Mask Embedding Encoder frozen. Returns the module and its output geometry.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not a Phase 0 teacher checkpoint: it needs "
            "'model_config' and 'model_state_dict'."
        )

    teacher = MaskVAETeacher(**checkpoint["model_config"])
    teacher.load_state_dict(checkpoint["model_state_dict"])
    encoder = teacher.mask_embedding_encoder
    if freeze:
        encoder = FrozenModule(encoder)
    return encoder, teacher.embedding_geometry()
