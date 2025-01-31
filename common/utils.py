from typing import Dict, List, TypeVar

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType
from transformers.image_processing_base import BatchFeature

from common.baselines import Softmax, Sparsemax

T = TypeVar("T")

Tensor = torch.Tensor


def move(obj: T, device: DeviceLikeType) -> T:
    if isinstance(obj, (Tensor, BatchFeature)):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, List):
        return [move(v, device) for v in obj]  # type: ignore
    elif isinstance(obj, Dict):
        return {k: move(v, device) for k, v in obj.items()}  # type: ignore
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def get_device() -> DeviceLikeType:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_compile(module: nn.Module, mode: str = "reduce-overhead"):
    if torch.cuda.is_available() and not isinstance(module, (Softmax, Sparsemax)):
        module.compile(mode=mode)
