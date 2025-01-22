from typing import TypeVar

import torch
import torch.nn as nn
from common.baselines import Softmax, Sparsemax
from torch._prims_common import DeviceLikeType

T = TypeVar("T")

Tensor = torch.Tensor


def get_device() -> DeviceLikeType:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_compile(module: nn.Module, mode: str = "reduce-overhead"):
    if torch.cuda.is_available() and not isinstance(module, (Softmax, Sparsemax)):
        module.compile(mode=mode)
