from enum import StrEnum
from math import ceil, sqrt
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch._prims_common import DeviceLikeType

Tensor = torch.Tensor

torch.set_float32_matmul_precision("high")


class PadType(StrEnum):
    pre = "pre"
    post = "post"


class InitType(StrEnum):
    ones = "ones"
    eye = "eye"


class SobaMonarch(nn.Module):

    def __init__(
        self,
        block_size: int,
        num_steps: int,
        pad_type: PadType,
        init_type: InitType,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_steps = num_steps
        self.pad_type = pad_type
        self.init_type = init_type

    def get_matrix(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert query.shape == key.shape
        batch_size, num_heads, seq_len, head_dim = query.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)

        query = query / sqrt(head_dim)

        pad_amount = self._get_pad_amount(seq_len)
        valid_mask = self._get_valid_mask(attention_mask)
        left, right = self._get_factors(query, key, valid_mask)
        matrix = self._get_matrix_from_factors(left, right, pad_amount)
        return matrix

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert query.shape == key.shape and key.shape == value.shape
        batch_size, num_heads, seq_len, head_dim = query.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)

        query = query / sqrt(head_dim)

        valid_mask = self._get_valid_mask(attention_mask)
        left, right = self._get_factors(query, key, valid_mask)
        return self._multiply(left, right, value)

    def _get_num_blocks(self, seq_len: int) -> int:
        num_blocks = ceil(seq_len / self.block_size)
        return num_blocks

    def _get_pad_amount(self, seq_len: int) -> int:
        num_blocks = self._get_num_blocks(seq_len)
        seq_len_padded = self.block_size * num_blocks
        pad_amount = seq_len_padded - seq_len
        return pad_amount

    def _multiply(self, left: Tensor, right: Tensor, inputs: Tensor) -> Tensor:
        seq_len = inputs.shape[-2]
        pad_amount = self._get_pad_amount(seq_len)

        match self.pad_type:
            case PadType.pre:
                pad_t = (0, 0) + (pad_amount, 0)
            case PadType.post:
                pad_t = (0, 0) + (0, pad_amount)
            case _:
                raise ValueError("Invalid pad_type")

        x = F.pad(inputs, pad_t)
        X = rearrange(x, "... (k i) v -> ... k i v", i=self.block_size)
        Y = torch.einsum("...kji,...kiv->...kjv", right, X)
        Z = torch.einsum("...jlk,...kjv->...ljv", left, Y)
        z = rearrange(Z, "... l j v -> ... (l j) v")

        match self.pad_type:
            case PadType.pre:
                return z[..., pad_amount:, :]
            case PadType.post:
                return z[..., : -pad_amount or None, :]
            case _:
                raise ValueError("Invalid pad_type")

    def _get_matrix_from_factors(
        self, left: Tensor, right: Tensor, pad_amount: int
    ) -> Tensor:
        out = torch.einsum("...jlk,...kji->...ljki", left, right)
        out = rearrange(out, "... l j k i -> ... (l j) (k i)")

        match self.pad_type:
            case PadType.pre:
                return out[..., pad_amount:, pad_amount:]
            case PadType.post:
                return out[..., : -pad_amount or None, : -pad_amount or None]
            case _:
                raise ValueError("Invalid pad_type")

    def _get_valid_mask(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")
        return attention_mask

    def _create_left_mask(
        self,
        left_shape: Tuple[int, int, int, int, int],
        valid_mask: Optional[Tensor],
        pad_amount: int,
        device: DeviceLikeType,
    ) -> Tensor:
        left_mask = torch.ones(left_shape, device=device, dtype=torch.bool)
        left_mask_flat = rearrange(left_mask, "... j l k -> ... (l j) k")

        match self.pad_type:
            case PadType.pre:
                left_mask_flat[..., :pad_amount, :] = False
                if valid_mask is not None:
                    left_mask_flat[..., pad_amount:, :] = valid_mask.transpose(
                        -1, -2
                    ).bool()
            case PadType.post:
                left_mask_flat[..., -pad_amount or left_mask_flat.shape[-2] :, :] = (
                    False
                )
                if valid_mask is not None:
                    left_mask_flat[..., : -pad_amount or None, :] = (
                        valid_mask.transpose(-1, -2).bool()
                    )
            case _:
                raise ValueError("Invalid pad_type")

        left_mask = rearrange(
            left_mask_flat, "... (l j) k -> ... j l k", j=self.block_size
        )
        return left_mask

    def _create_right_mask(
        self,
        right_shape: Tuple[int, int, int, int, int],
        valid_mask: Optional[Tensor],
        pad_amount: int,
        device: DeviceLikeType,
    ) -> Tensor:
        right_mask = torch.ones(right_shape, device=device, dtype=torch.bool)
        right_mask_flat = rearrange(right_mask, "... k j i -> ... j (k i)")

        match self.pad_type:
            case PadType.pre:
                right_mask_flat[..., :pad_amount] = False
                if valid_mask is not None:
                    right_mask_flat[..., pad_amount:] = valid_mask.bool()
            case PadType.post:
                right_mask_flat[..., -pad_amount or right_mask_flat.shape[-1] :] = False
                if valid_mask is not None:
                    right_mask_flat[..., : -pad_amount or None] = valid_mask.bool()
            case _:
                raise ValueError("Invalid pad_type")

        right_mask = rearrange(
            right_mask_flat, "... j (k i) -> ... k j i", i=self.block_size
        )
        return right_mask

    def _get_factors(
        self, query: Tensor, key: Tensor, valid_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch_size, num_heads, seq_len, head_dim = query.shape
        pad_amount = self._get_pad_amount(seq_len)
        num_blocks = self._get_num_blocks(seq_len)
        block_size = self.block_size

        if valid_mask is not None:
            query = query * valid_mask.transpose(-1, -2)
            key = key * valid_mask.transpose(-1, -2)

        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == PadType.pre else (0, pad_amount)
        )
        query = F.pad(query, pad_t)
        key = F.pad(key, pad_t)
        query = rearrange(query, "... (l j) v -> ... l j v", j=block_size)
        key = rearrange(key, "... (k i) v -> ... k i v", i=block_size)

        left_shape = (batch_size, num_heads, block_size, num_blocks, num_blocks)
        right_shape = (batch_size, num_heads, num_blocks, block_size, block_size)

        left_mask = self._create_left_mask(
            left_shape, valid_mask, pad_amount, query.device
        )
        right_mask = self._create_right_mask(
            right_shape, valid_mask, pad_amount, query.device
        )

        match self.init_type:
            case InitType.ones:
                left = torch.ones(left_shape, device=query.device) / num_blocks
            case InitType.eye:
                left = torch.eye(num_blocks, device=query.device).expand(left_shape)
            case _:
                raise ValueError("Invalid init_type")

        left = torch.where(left_mask, left, 0.0)

        for _ in range(self.num_steps):

            # Right
            beta = torch.einsum("...jlk,...ljv,...kiv->...kji", left, query, key)
            tau = repeat(torch.sum(left, dim=-2), "... j k -> ... k j 1")
            right = F.softmax(
                torch.where(
                    torch.logical_and(
                        right_mask,
                        torch.logical_not(torch.isclose(tau, torch.zeros_like(tau))),
                    ),
                    beta / tau,
                    torch.finfo(beta.dtype).min,
                ),
                dim=-1,
            )
            right = torch.where(right_mask, right, 0.0)

            alpha = repeat(
                torch.sum(torch.special.xlogy(right, right), dim=-1),
                "... k j -> ... j 1 k",
            )
            beta = torch.einsum("...kji,...ljv,...kiv->...jlk", right, query, key)
            left = F.softmax(
                torch.where(left_mask, beta - alpha, torch.finfo(beta.dtype).min),
                dim=-1,
            )
            left = torch.where(left_mask, left, 0.0)
            print(torch.any(torch.isnan(left)))

        return left, right
