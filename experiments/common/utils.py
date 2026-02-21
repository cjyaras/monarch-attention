from typing import Dict, List, TypeVar

import torch
from transformers.image_processing_base import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding

T = TypeVar("T")

Tensor = torch.Tensor


def move(obj: T, device) -> T:
    if isinstance(obj, (Tensor, BatchFeature, BatchEncoding)):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, List):
        return [move(v, device) for v in obj]  # type: ignore
    elif isinstance(obj, Dict):
        return {k: move(v, device) for k, v in obj.items()}  # type: ignore
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
