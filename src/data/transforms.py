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
    ) -> None:
        if len(image_size) != 2 or any(size <= 0 for size in image_size):
            raise ValueError("image_size must contain two positive integers.")

        for name, probability in (
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("vertical_flip_probability", vertical_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

        self.image_size = tuple(int(size) for size in image_size)
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability

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

        image_tensor = TF.pil_to_tensor(image).float().div(255.0)
        image_tensor = TF.normalize(
            image_tensor,
            self.image_mean,
            self.image_std,
        )

        return image_tensor.contiguous(), mask.float().contiguous()


def build_segmentation_transform(config: DatasetConfig, split: str) -> SegmentationTransform:
    """Build deterministic evaluation and augmented training transforms."""

    return SegmentationTransform(
        image_size = config.image_size,
        image_mean = config.image_mean,
        image_std = config.image_std,
        training = split == "train",
    )
