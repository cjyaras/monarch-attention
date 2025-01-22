from typing import TypeVar

import torch
import torch.nn as nn
from common.baselines import Softmax, Sparsemax
from torch._prims_common import DeviceLikeType

T = TypeVar("T")

Tensor = torch.Tensor


def get_device() -> DeviceLikeType:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_compile(module: nn.Module):
    if torch.cuda.is_available() and not isinstance(module, (Softmax, Sparsemax)):
        module.compile()


def move(obj: T, device: DeviceLikeType) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}  # type: ignore
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")
