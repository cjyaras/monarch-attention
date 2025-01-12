from typing import Any, TypeVar

import torch

T = TypeVar("T")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move(obj: T, device) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, dict):
        return {k: v.to(device) for k, v in obj.items()}  # type: ignore
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")
