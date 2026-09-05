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
        rotation_range: float = 0.0,
        scaling_range: tuple[float, float] = (1.0, 1.0),
        intensity_shift_range: float = 0.0,
    ) -> None:
        if len(image_size) != 2 or any(size <= 0 for size in image_size):
            raise ValueError("image_size must contain two positive integers.")

        for name, probability in (
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("vertical_flip_probability", vertical_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if rotation_range < 0.0:
            raise ValueError("rotation_range must be non-negative.")
        if len(scaling_range) != 2 or not 0 < scaling_range[0] <= scaling_range[1]:
            raise ValueError(
                "scaling_range must be (min_scale, max_scale) with "
                "0 < min_scale <= max_scale."
            )
        if intensity_shift_range < 0:
            raise ValueError("intensity_shift_range must be non-negative.")

        self.image_size = tuple(int(size) for size in image_size)
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.rotation_range = float(rotation_range)
        self.scaling_range = tuple(float(value) for value in scaling_range)
        self.intensity_shift_range = float(intensity_shift_range)

    def __call__(self, image: Image.Image, mask: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image.")
        if mask.ndim != 3:
            raise ValueError(
                f"mask must have shape [C, H, W], got {tuple(mask.shape)}."
            )

        # Geometric operations (affine/rotate) require float tensors; class
        # indices are small integers so float32 represents them exactly and
        # nearest-neighbor interpolation preserves them.
        mask = mask.float()

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

        if self.training and self.rotation_range > 0:
            angle = float(torch.empty(()).uniform_(-self.rotation_range, self.rotation_range))
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if self.training and self.scaling_range != (1.0, 1.0):
            scale = float(
                torch.empty(()).uniform_(self.scaling_range[0], self.scaling_range[1])
            )
            image = TF.affine(
                image,
                angle=0.0,
                translate=(0, 0),
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=0.0,
                translate=(0, 0),
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        image_tensor = TF.pil_to_tensor(image).float().div(255.0)

        if self.training and self.intensity_shift_range > 0:
            # Paper augmentation: random intensity shifting on the raw [0, 1]
            # image, applied before dataset normalization.
            shift = float(
                torch.empty(()).uniform_(
                    -self.intensity_shift_range,
                    self.intensity_shift_range,
                )
            )
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
_ROTATION_SCALE_INTENSITY_DATASETS = {"synapse_btcv", "synapse_ct"}


def build_segmentation_transform(config: DatasetConfig, split: str) -> SegmentationTransform:
    """Build deterministic evaluation and augmented training transforms."""

    extra_augmentation = (
        {
            "rotation_range": 15.0,
            "scaling_range": (0.9, 1.1),
            "intensity_shift_range": 0.1,
        }
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
