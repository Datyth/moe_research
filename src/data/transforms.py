"""Joint image-mask transforms for segmentation datasets."""

from collections.abc import Sequence

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.configs.dataset import DatasetConfig


class SegmentationTransform:
    """Resize and normalize an image while keeping its mask aligned."""

    def __init__(
        self,
        image_size: Sequence[int],
        image_mean: Sequence[float],
        image_std: Sequence[float],
        *,
        training: bool = False,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.0,
        rotation_degrees: float = 0.0,
        scale_range: tuple[float, float] | None = None,
        intensity_jitter: float = 0.0,
    ) -> None:
        if len(image_size) != 2 or any(size <= 0 for size in image_size):
            raise ValueError("image_size must contain two positive integers.")

        for name, probability in (
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("vertical_flip_probability", vertical_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if rotation_degrees < 0.0:
            raise ValueError("rotation_degrees must be non-negative.")
        if scale_range is not None:
            low, high = scale_range
            if not 0.0 < low <= high:
                raise ValueError("scale_range must satisfy 0 < low <= high.")
        if intensity_jitter < 0.0:
            raise ValueError("intensity_jitter must be non-negative.")

        self.image_size = tuple(int(size) for size in image_size)
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.intensity_jitter = intensity_jitter

    def __call__(self, image: Image.Image, mask: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image.")
        if mask.ndim != 3:
            raise ValueError(
                f"mask must have shape [C, H, W], got {tuple(mask.shape)}."
            )

        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )

        if self.training and torch.rand(()) < self.horizontal_flip_probability:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if self.training and torch.rand(()) < self.vertical_flip_probability:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if self.training and (self.rotation_degrees > 0.0 or self.scale_range is not None):
            angle = (
                float(torch.empty(()).uniform_(-self.rotation_degrees, self.rotation_degrees))
                if self.rotation_degrees > 0.0
                else 0.0
            )
            scale = (
                float(torch.empty(()).uniform_(*self.scale_range))
                if self.scale_range is not None
                else 1.0
            )
            image = TF.affine(
                image,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
            )

        image_tensor = TF.pil_to_tensor(image).float().div(255.0)
        if self.training and self.intensity_jitter > 0.0:
            shift = float(torch.empty(()).uniform_(-self.intensity_jitter, self.intensity_jitter))
            image_tensor = (image_tensor + shift).clamp(0.0, 1.0)
        image_tensor = TF.normalize(
            image_tensor,
            self.image_mean,
            self.image_std,
        )

        return image_tensor.contiguous(), mask.float().contiguous()


# Datasets matching MoE-SAM (MICCAI 2025)'s reported "flipping, rotation,
# scaling, and intensity shifting" augmentation protocol. Scoped to the
# dataset the protocol is being aligned against (Synapse/BTCV) rather than
# applied everywhere, so ISIC2018/AMOS22 training is unaffected.
_ROTATION_SCALE_INTENSITY_DATASETS = {"synapse_btcv"}


def build_segmentation_transform(config: DatasetConfig, split: str) -> SegmentationTransform:
    """Build deterministic evaluation and augmented training transforms."""

    extra_augmentation = (
        {"rotation_degrees": 15.0, "scale_range": (0.9, 1.1), "intensity_jitter": 0.1}
        if config.name in _ROTATION_SCALE_INTENSITY_DATASETS
        else {}
    )
    return SegmentationTransform(
        image_size = config.image_size,
        image_mean = config.image_mean,
        image_std = config.image_std,
        training = split == "train",
        **extra_augmentation,
    )
