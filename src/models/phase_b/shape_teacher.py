"""The retained Phase-A Shape Teacher G_M = P_M o E_M.

Phase A trains a mask autoencoder; the proposal discards the reconstruction
decoder D_M afterwards and keeps only the encoder and its spatial projection,
which together map a ground-truth mask to the compact shape latent h_M. This
module loads such a Phase-A checkpoint, drops the decoder, and exposes h_M.

h_M is privileged information: it is available only during training, and at
inference the whole branch is removed. It is therefore frozen and kept in eval
mode by default, so its BatchNorm/dropout-free statistics and its weights stay
exactly what Phase A produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..shape import ShapeAutoencoder, SmallCNN, SpatialProjector


SUPPORTED_ENCODERS = {"small_cnn": SmallCNN}


class ShapeTeacher(nn.Module):
    """Map a ground-truth mask to the shape latent h_M."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        projector: nn.Module,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.frozen = bool(freeze)
        if self.frozen:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "ShapeTeacher":
        """Keep a frozen teacher in eval mode even inside a training loop."""

        if self.frozen:
            return super().train(False)
        return super().train(mode)

    def forward(self, masks: Tensor) -> Tensor:
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(
                "ShapeTeacher input must have shape [B, 1, H, W], got "
                f"{tuple(masks.shape)}."
            )
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            return self.projector(self.encoder(masks))


def build_shape_teacher(config: dict[str, Any]) -> ShapeTeacher:
    """Instantiate an untrained Shape Teacher from a Phase-A model config."""

    encoder_name = config.get("encoder", {}).get("name", "small_cnn")
    if encoder_name not in SUPPORTED_ENCODERS:
        raise ValueError(
            f"Unsupported shape encoder {encoder_name!r}; "
            f"expected one of {sorted(SUPPORTED_ENCODERS)}."
        )
    return ShapeTeacher(
        encoder=SUPPORTED_ENCODERS[encoder_name](),
        projector=SpatialProjector(),
    )


def load_shape_teacher(
    checkpoint_path: str | Path,
    *,
    freeze: bool = True,
    map_location: str | torch.device = "cpu",
) -> ShapeTeacher:
    """Load a Phase-A checkpoint and keep only G_M, discarding the decoder."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Phase-A checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{path} is not a Phase-A experiment checkpoint "
            "(no 'model_state_dict' entry)."
        )
    model_config = checkpoint.get("metadata", {}).get("model_config", {})
    teacher = build_shape_teacher(model_config)
    teacher.frozen = bool(freeze)

    state_dict = checkpoint["model_state_dict"]
    prefixes = ("encoder.", "projector.")
    teacher_state = {
        name: tensor
        for name, tensor in state_dict.items()
        if name.startswith(prefixes)
    }
    if not teacher_state:
        raise ValueError(
            f"{path} has no 'encoder.'/'projector.' weights; it does not look "
            "like a ShapeAutoencoder checkpoint."
        )
    # strict=True: a silently partial load would leave the teacher randomly
    # initialized, which is exactly the failure this stage cannot detect later.
    teacher.load_state_dict(teacher_state, strict=True)

    if freeze:
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher.eval()
    return teacher


__all__ = [
    "ShapeTeacher",
    "ShapeAutoencoder",
    "build_shape_teacher",
    "load_shape_teacher",
]
