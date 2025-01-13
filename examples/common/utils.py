from typing import Any, TypeVar

import torch
from torch._prims_common import DeviceLikeType

T = TypeVar("T")


def get_device() -> DeviceLikeType:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move(obj: T, device: DeviceLikeType) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}  # type: ignore
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")
