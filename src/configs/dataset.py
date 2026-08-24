from dataclasses import dataclass
from pathlib import Path
from typing import Literal


TaskType = Literal["binary", "multiclass"]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    manifest: Path
    version: str

    task: TaskType
    num_classes: int
    in_channels: int = 3
    image_size: tuple[int, int] = (256, 256)

    image_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    image_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    mask_threshold: float = 0.5

    def __post_init__(self):
        if self.task == "binary" and self.num_classes != 1:
            raise ValueError(
                "Binary segmentation requires num_classes=1."
            )

        if not self.version:
            raise ValueError("Dataset version must not be empty.")

        if len(self.image_mean) != self.in_channels:
            raise ValueError(
                "image_mean must match in_channels."
            )

        if len(self.image_std) != self.in_channels:
            raise ValueError(
                "image_std must match in_channels."
            )

@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False  # evaluation should remain False


# dataset_config = DatasetConfig(
#     name="isic2018",
#     root=Path("dataset/isic2018_task1"),
#     task="binary",
#     num_classes=1,
#     in_channels=3,
#     image_size=(256, 256),
#     splits={
#         "train": DatasetSplitConfig(
#             images_dir="images/train",
#             masks_dir="labels/train",
#         ),
#         "test": DatasetSplitConfig(
#             images_dir="images/test",
#             masks_dir="labels/test",
#         ),
#     },
# )