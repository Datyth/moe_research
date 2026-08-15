from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal, Any

import torch
from torch import Tensor, nn

TaskType = Literal["binary", "multiclass"]

@dataclass
class SegmentationOutput:
    logits: Tensor
    aux_logits: tuple[Tensor, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationPrediction:
    masks: Tensor
    probabilities: Tensor
    logits: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict) 

class BaseSegmentationModel(nn.Module, ABC):
    def __init__(self, *, in_channels:int, num_classes: int, task: TaskType):
        super().__init__()

        if task == "binary" and num_classes != 1:
            raise ValueError("Binary segmentation requires num_classes = 1.")
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.task = task

    @abstractmethod
    def forward(self, images: Tensor, **kwargs):
        """Return raw, unnormalized logits."""
        raise NotImplementedError

    
    @torch.inference_mode() #Báo cho PyTorch không cần theo dõi gradient, giúp tiết kiệm bộ nhớ và chạy nhanh hơn.
    def predict(self, images: Tensor, *, threshold: float = 0.5, **kwargs):
        was_training = self.training
        try:
            self.eval()
            output = self(images, **kwargs)

            if self.task == "multiclass":
                probabilities = output.logits.softmax(dim = 1)
                masks = probabilities.argmax(dim = 1)

            else:
                probabilities = output.logits.sigmoid()
                masks = probabilities >= threshold
            
            return SegmentationPrediction(
                masks = masks,
                probabilities = probabilities,
                logits = output.logits,
                diagnostics = output.diagnostics
            )
        except Exception as e:
            print(f"Error {e}")
            raise e
        finally:
            self.train(was_training) # Trả lại trạng thái gốc sau khi infer