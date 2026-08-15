from collections.abc import Callable
from typing import Any

from .base import BaseSegmentationModel

MODEL_REGISTRY: dict[str, Callable[..., BaseSegmentationModel]] = {}

def register_model(name: str):
    def decorator(model_class):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        
        MODEL_REGISTRY[name] = model_class
        return model_class
    
    return decorator

def build_model(config: dict[str, Any]) -> BaseSegmentationModel:
    config = config.copy()
    model_name = config.pop("name")

    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        ) from error

    return model_class(**config)


# How to use?: 

# @register_model("my_unet")
# class UNet(BaseSegmentationModel):
#     def __init__(self, in_channels: int, num_classes: int, task: TaskType, hidden_dim: int = 64):
#         super().__init__(in_channels=in_channels, num_classes=num_classes, task=task)
#         self.hidden_dim = hidden_dim
#         # ... logic tạo layers UNet ...

#     def forward(self, images, **kwargs):
#         pass # Code chạy forward

# Cấu hình này thường được đọc từ file config.yaml
# user_config = {
#     "name": "my_unet",
#     "in_channels": 3,
#     "num_classes": 2,
#     "task": "multiclass",
#     "hidden_dim": 128
# }

# # Tự động tạo mô hình!
# model = build_model(user_config) 

# print(type(model)) # Output: <class '__main__.UNet'>