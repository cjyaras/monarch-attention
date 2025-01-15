from time import time
from typing import Any, Dict, TypeVar

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

T = TypeVar("T")

Tensor = torch.Tensor


def get_device() -> DeviceLikeType:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_compile(module: nn.Module):
    if torch.cuda.is_available():
        module.compile()


def move(obj: T, device: DeviceLikeType) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}  # type: ignore
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def benchmark_time(model: nn.Module, inputs: Dict[str, Tensor]) -> float:
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # type: ignore
        model(**inputs)
        end_event.record()  # type: ignore
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)
    else:
        start = time()
        model(**inputs)
        total_ms = (time() - start) * 1000

    return total_ms


def benchmark_flops(
    model: nn.Module, inputs: Dict[str, Tensor], module_name: str
) -> float:
    from torchtnt.utils.flops import FlopTensorDispatchMode

    with FlopTensorDispatchMode(model) as ftdm:
        model(**inputs)
        flops_counts = ftdm.flop_counts[module_name]["bmm.default"]

    return flops_counts
