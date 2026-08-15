from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from src.configs.dataset import DatasetConfig


class BaseSegmentationDataset(Dataset, ABC):
    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transform=None,
    ):
        if split not in config.splits:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Available: {tuple(config.splits)}"
            )

        self.config = config
        self.split = split
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError