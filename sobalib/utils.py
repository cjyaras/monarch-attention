from math import sqrt
from typing import Literal, Tuple

import torch

Tensor = torch.Tensor

PadType = Literal["pre", "post"]


def init(shape: Tuple[int, ...], init_scale: float = 1e-3) -> Tensor:
    """
    Approximately initializes from projection of Dirichlet distribution with large scale parameter onto sphere.
    Uses uniform distribution for fast sampling.
    """
    center = 1 / sqrt(shape[-1])
    noise = 2 * init_scale * torch.rand(shape) - init_scale
    return center + noise


def project(x: Tensor, u: Tensor) -> Tensor:
    return torch.einsum("...i,...j,...j->...i", x, x, u)


def inv_norm(x: Tensor) -> Tensor:
    return 1 / torch.linalg.norm(x, dim=-1, keepdims=True)
