from math import ceil, sqrt
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch._prims_common import DeviceLikeType
from torch.nn.functional import normalize, pad

Tensor = torch.Tensor

PadType = Literal["pre", "post"]


def _init(
    shape: Tuple[int, ...],
    perturb_scale: Optional[float] = None,
    device: Optional[DeviceLikeType] = None,
) -> Tensor:
    center = 1 / sqrt(shape[-1])
    if perturb_scale is not None:
        noise = 2 * perturb_scale * torch.rand(shape, device=device) - perturb_scale
        return center + noise
    else:
        return center * torch.ones(shape, device=device)


def _project(x: Tensor, u: Tensor) -> Tensor:
    return torch.einsum("...i,...j,...j->...i", x, x, u)


def _inv_norm(x: Tensor) -> Tensor:
    return 1 / torch.linalg.norm(x, dim=-1, keepdims=True)


class LowRankAttention(nn.Module):

    def __init__(self, rank: int, num_steps: int, step_size: float):
        super().__init__()
        self.rank = rank
        self.num_steps = num_steps
        self.step_size = step_size

    @staticmethod
    def multiply(left: Tensor, right: Tensor, inputs: Tensor) -> Tensor:
        return left @ (right @ inputs)

    @staticmethod
    def get_matrix_from_factors(left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    @staticmethod
    def grad(
        left_params: Tensor,
        right_params: Tensor,
        query: Tensor,
        key: Tensor,
        seq_len: int,
    ) -> Tuple[Tensor, Tensor]:
        right_sphere = normalize(right_params, dim=-1)
        left_sphere = normalize(left_params, dim=-1)
        left = left_sphere**2
        right = right_sphere**2
        d_left = (
            torch.einsum("...jk,...ki,...li->...jl", left, right, right)
            - torch.einsum("...ja,...ia,...li->...jl", query, key, right)
        ) / seq_len**2
        d_left = d_left * 2 * left_sphere
        d_left = d_left - _project(left_sphere, d_left)
        d_left = d_left * _inv_norm(left_params)
        d_right = (
            torch.einsum("...jk,...jl,...li->...ki", left, left, right)
            - torch.einsum("...jk,...ja,...ia->...ki", left, query, key)
        ) / seq_len**2
        d_right = d_right * 2 * right_sphere
        d_right = d_right - _project(right_sphere, d_right)
        d_right = d_right * _inv_norm(right_params)
        return d_left, d_right

    def get_factors(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = query.shape[:-2]
        seq_len = query.shape[-2]

        # Need small perturbation to break symmetry
        left_params = _init(
            batch_size + (seq_len, self.rank), perturb_scale=1e-3, device=query.device
        )
        right_params = _init(
            batch_size + (self.rank, seq_len), perturb_scale=1e-3, device=query.device
        )

        for _ in range(self.num_steps):
            d_left_params, d_right_params = self.grad(
                left_params, right_params, query, key, seq_len
            )
            left_params = left_params - self.step_size * d_left_params
            right_params = right_params - self.step_size * d_right_params

        left = normalize(left_params, dim=-1) ** 2
        right = normalize(right_params, dim=-1) ** 2

        return left, right

    def get_matrix(self, query: Tensor, key: Tensor) -> Tensor:
        left, right = self.get_factors(query, key)
        return self.get_matrix_from_factors(left, right)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        left, right = self.get_factors(query, key)
        return self.multiply(left, right, value)


class MonarchAttention(nn.Module):

    def __init__(
        self,
        block_size: int,
        num_steps: int,
        step_size: float,
        pad_type: PadType,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.pad_type = pad_type

    def get_num_blocks(self, seq_len: int) -> int:
        num_blocks = ceil(seq_len / self.block_size)
        return num_blocks

    def get_pad_amount(self, seq_len: int) -> int:
        num_blocks = self.get_num_blocks(seq_len)
        seq_len_padded = self.block_size * num_blocks
        pad_amount = seq_len_padded - seq_len
        return pad_amount

    def multiply(self, left: Tensor, right: Tensor, inputs: Tensor) -> Tensor:
        pad_amount = self.get_pad_amount(inputs.shape[-2])
        pad_t = (0, 0) + (pad_amount, 0) if self.pad_type == "pre" else (0, pad_amount)
        x = pad(inputs, pad_t)
        X = rearrange(x, "... (k i) a -> ... k i a", i=self.block_size)
        Y = torch.einsum("...kji,...kia->...kja", right, X)
        Z = torch.einsum("...jlk,...kja->...lja", left, Y)
        z = rearrange(Z, "... l j a -> ... (l j) a")
        return (
            z[..., pad_amount:, :]
            if self.pad_type == "pre"
            else z[..., : -pad_amount or None, :]
        )

    def get_matrix_from_factors(self, left: Tensor, right: Tensor) -> Tensor:
        out = torch.einsum("...jlk,...kji->...ljki", left, right)
        out = rearrange(out, "... l j k i -> ... (l j) (k i)")
        return out

    def mask(self, x, pad_amount: int) -> Tensor:
        if self.pad_type == "pre":
            x[..., 0, :, :pad_amount] = 0.0
        else:
            x[..., -1, :, -pad_amount or None :] = 0
        return x

    def grad(
        self,
        left_params: Tensor,
        right_params: Tensor,
        query: Tensor,
        key: Tensor,
        seq_len: int,
    ) -> Tuple[Tensor, Tensor]:
        pad_amount = self.get_pad_amount(seq_len)
        right_params = self.mask(right_params, pad_amount)
        left_sphere = normalize(left_params, dim=-1)
        right_sphere = normalize(right_params, dim=-1)
        left = left_sphere**2
        right = right_sphere**2
        d_left = (
            torch.einsum("...kj,...jlk->...jlk", torch.sum(right**2, dim=-1), left)
            - torch.einsum("...kji,...lja,...kia->...jlk", right, query, key)
        ) / seq_len**2
        d_left = d_left * 2 * left_sphere
        d_left = d_left - _project(left_sphere, d_left)
        d_left = d_left * _inv_norm(left_params)
        d_right = (
            torch.einsum("...jk,...kji->...kji", torch.sum(left**2, dim=-2), right)
            - torch.einsum("...jlk,...lja,...kia->...kji", left, query, key)
        ) / seq_len**2
        d_right = d_right * 2 * right_sphere
        d_right = d_right - _project(right_sphere, d_right)
        d_right = d_right * _inv_norm(right_params)
        return d_left, d_right

    def get_factors(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = query.shape[:-2]
        seq_len = query.shape[-2]
        pad_amount = self.get_pad_amount(seq_len)
        num_blocks = self.get_num_blocks(seq_len)

        pad_t = (0, 0) + (pad_amount, 0) if self.pad_type == "pre" else (0, pad_amount)
        query = pad(query, pad_t)
        key = pad(key, pad_t)
        query = rearrange(query, "... (l j) a -> ... l j a", j=self.block_size)
        key = rearrange(key, "... (k i) a -> ... k i a", i=self.block_size)

        left_params = _init(
            batch_size + (self.block_size, num_blocks, num_blocks), device=query.device
        )
        right_params = _init(
            batch_size + (num_blocks, self.block_size, self.block_size),
            device=query.device,
        )

        for _ in range(self.num_steps):
            d_left_params, d_right_params = self.grad(
                left_params, right_params, query, key, seq_len
            )
            left_params = left_params - self.step_size * d_left_params
            right_params = right_params - self.step_size * d_right_params

        left = normalize(left_params, dim=-1) ** 2
        right = normalize(self.mask(right_params, pad_amount), dim=-1) ** 2

        return left, right

    def get_matrix(self, query: Tensor, key: Tensor) -> Tensor:
        seq_len = query.shape[-2]
        pad_amount = self.get_pad_amount(seq_len)

        left, right = self.get_factors(query, key)
        matrix = self.get_matrix_from_factors(left, right)
        return (
            matrix[..., pad_amount:, pad_amount:]
            if self.pad_type == "pre"
            else matrix[..., : -pad_amount or None, : -pad_amount or None]
        )

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        left, right = self.get_factors(query, key)
        return self.multiply(left, right, value)
