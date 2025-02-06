from enum import StrEnum
from math import ceil, sqrt
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import pad

from common.activations import sparsemax

Tensor = torch.Tensor

torch.set_float32_matmul_precision("high")


def _project(x: Tensor, u: Tensor) -> Tensor:
    return torch.einsum("...i,...j,...j->...i", x, x, u)


def _safe_normalize(x: Tensor, dim: int) -> Tensor:
    norms = torch.linalg.norm(x, dim=dim, keepdims=True)
    return torch.where(norms > 0, x / norms, x)


def _safe_inv_norm(x: Tensor, dim: int) -> Tensor:
    norms = torch.linalg.norm(x, dim=dim, keepdims=True)
    return torch.where(norms > 0, 1 / norms, 0.0)


class PadType(StrEnum):
    pre = "pre"
    post = "post"


class SobaMonarch(nn.Module):

    def __init__(
        self,
        block_size: int,
        num_steps: int,
        num_heads: int,
        pad_type: PadType,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_steps = num_steps
        self.pad_type = pad_type

        self.attention_scale = nn.Parameter(torch.zeros((num_heads,)))
        self.step_size = nn.Parameter(torch.zeros((num_heads, 2, num_steps)))

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
        query = query * torch.nn.functional.softplus(
            self.attention_scale[..., None, None]
        )

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
        query = query * torch.nn.functional.softplus(
            self.attention_scale[..., None, None]
        )

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
        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == PadType.pre else (0, pad_amount)
        )
        x = pad(inputs, pad_t)
        X = rearrange(x, "... (k i) v -> ... k i v", i=self.block_size)
        Y = torch.einsum("...kji,...kiv->...kjv", right, X)
        Z = torch.einsum("...jlk,...kjv->...ljv", left, Y)
        z = rearrange(Z, "... l j v -> ... (l j) v")

        if self.pad_type == PadType.pre:
            return z[..., pad_amount:, :]
        else:
            return z[..., : -pad_amount or None, :]

    def _get_matrix_from_factors(
        self, left: Tensor, right: Tensor, pad_amount: int
    ) -> Tensor:
        out = torch.einsum("...jlk,...kji->...ljki", left, right)
        out = rearrange(out, "... l j k i -> ... (l j) (k i)")

        if self.pad_type == PadType.pre:
            return out[..., pad_amount:, pad_amount:]
        else:
            return out[..., : -pad_amount or None, : -pad_amount or None]

    def _get_valid_mask(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")
        return attention_mask

    def _mask(
        self,
        left_params: Tensor,
        right_params: Tensor,
        valid_mask: Optional[Tensor],
        pad_amount: int,
    ) -> Tuple[Tensor, Tensor]:
        left_params_flat = rearrange(left_params, "... j l k -> ... (l j) k")
        right_params_flat = rearrange(right_params, "... k j i -> ... j (k i)")

        if self.pad_type == PadType.pre:
            left_params_flat[..., :pad_amount, :] = 0.0
            right_params_flat[..., :pad_amount] = 0.0

            if valid_mask is not None:
                left_params_flat[..., pad_amount:, :] = left_params_flat[
                    ..., pad_amount:, :
                ] * valid_mask.transpose(-1, -2)
                right_params_flat[..., pad_amount:] = (
                    right_params_flat[..., pad_amount:] * valid_mask
                )
        else:
            left_params_flat[..., -pad_amount or left_params_flat.shape[-2] :, :] = 0.0
            right_params_flat[..., -pad_amount or right_params_flat.shape[-1] :] = 0.0

            if valid_mask is not None:
                left_params_flat[..., : -pad_amount or None, :] = left_params_flat[
                    ..., : -pad_amount or None, :
                ] * valid_mask.transpose(-1, -2)
                right_params_flat[..., : -pad_amount or None] = (
                    right_params_flat[..., : -pad_amount or None] * valid_mask
                )

        left_params = rearrange(
            left_params_flat, "... (l j) k -> ... j l k", j=self.block_size
        )
        right_params = rearrange(
            right_params_flat, "... j (k i) -> ... k j i", i=self.block_size
        )

        return left_params, right_params

    def _grad(
        self,
        left_params: Tensor,
        right_params: Tensor,
        query: Tensor,
        key: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        left_sphere = _safe_normalize(left_params, dim=-1)
        right_sphere = _safe_normalize(right_params, dim=-1)
        left = left_sphere**2
        right = right_sphere**2
        d_left = torch.einsum(
            "...kj,...jlk->...jlk", torch.sum(right**2, dim=-1), left
        ) - torch.einsum("...kji,...ljv,...kiv->...jlk", right, query, key)
        d_left = d_left * 2 * left_sphere
        d_left = d_left - _project(left_sphere, d_left)
        d_left = d_left * _safe_inv_norm(left_params, dim=-1)
        d_right = torch.einsum(
            "...jk,...kji->...kji", torch.sum(left**2, dim=-2), right
        ) - torch.einsum("...jlk,...ljv,...kiv->...kji", left, query, key)
        d_right = d_right * 2 * right_sphere
        d_right = d_right - _project(right_sphere, d_right)
        d_right = d_right * _safe_inv_norm(right_params, dim=-1)
        return d_left, d_right

    def _get_factors(
        self, query: Tensor, key: Tensor, valid_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch_size, num_heads, seq_len, head_dim = query.shape
        pad_amount = self._get_pad_amount(seq_len)
        num_blocks = self._get_num_blocks(seq_len)

        if valid_mask is not None:
            query = query * valid_mask.transpose(-1, -2)
            key = key * valid_mask.transpose(-1, -2)

        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == PadType.pre else (0, pad_amount)
        )
        query = pad(query, pad_t)
        key = pad(key, pad_t)
        query = rearrange(query, "... (l j) v -> ... l j v", j=self.block_size)
        key = rearrange(key, "... (k i) v -> ... k i v", i=self.block_size)

        left_params = torch.full(
            (batch_size, num_heads, self.block_size, num_blocks, num_blocks),
            1 / sqrt(num_blocks),
            device=query.device,
        )
        right_params = torch.full(
            (batch_size, num_heads, num_blocks, self.block_size, self.block_size),
            1 / sqrt(self.block_size),
            device=query.device,
        )

        left_params, right_params = self._mask(
            left_params, right_params, valid_mask, pad_amount
        )

        for step in range(self.num_steps):
            d_left_params, d_right_params = self._grad(
                left_params, right_params, query, key
            )

            left_params = (
                left_params
                - torch.nn.functional.softplus(
                    self.step_size[:, 0, step, None, None, None]
                )
                * d_left_params
            )
            right_params = (
                right_params
                - torch.nn.functional.softplus(
                    self.step_size[:, 1, step, None, None, None]
                )
                * d_right_params
            )

            left_params, right_params = self._mask(
                left_params, right_params, valid_mask, pad_amount
            )

        left = _safe_normalize(left_params, dim=-1) ** 2
        right = _safe_normalize(right_params, dim=-1) ** 2
        return left, right


class SobaMonarchV2(nn.Module):

    def __init__(
        self,
        block_size: int,
        num_steps: int,
        num_heads: int,
        pad_type: PadType,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_steps = num_steps
        self.pad_type = pad_type

        self.attention_scale = nn.Parameter(torch.zeros((num_heads,)))

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
        query = query * torch.nn.functional.softplus(
            self.attention_scale[..., None, None]
        )

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
        query = query * torch.nn.functional.softplus(
            self.attention_scale[..., None, None]
        )

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
        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == PadType.pre else (0, pad_amount)
        )
        x = pad(inputs, pad_t)
        X = rearrange(x, "... (k i) v -> ... k i v", i=self.block_size)
        Y = torch.einsum("...kji,...kiv->...kjv", right, X)
        Z = torch.einsum("...jlk,...kjv->...ljv", left, Y)
        z = rearrange(Z, "... l j v -> ... (l j) v")

        if self.pad_type == PadType.pre:
            return z[..., pad_amount:, :]
        else:
            return z[..., : -pad_amount or None, :]

    def _get_matrix_from_factors(
        self, left: Tensor, right: Tensor, pad_amount: int
    ) -> Tensor:
        out = torch.einsum("...jlk,...kji->...ljki", left, right)
        out = rearrange(out, "... l j k i -> ... (l j) (k i)")

        if self.pad_type == PadType.pre:
            return out[..., pad_amount:, pad_amount:]
        else:
            return out[..., : -pad_amount or None, : -pad_amount or None]

    def _get_valid_mask(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")
        return attention_mask

    def _mask(
        self,
        right: Tensor,
        valid_mask: Optional[Tensor],
        pad_amount: int,
    ) -> Tensor:
        right_flat = rearrange(right, "... k j i -> ... j (k i)")
        neg_inf = torch.finfo(right_flat.dtype).min

        if valid_mask is not None:
            neg_inf_valid_mask = (1 - valid_mask) * neg_inf

        if self.pad_type == PadType.pre:
            right_flat[..., :pad_amount] += neg_inf

            if valid_mask is not None:
                assert neg_inf_valid_mask is not None
                right_flat[..., pad_amount:] = (
                    right_flat[..., pad_amount:] + neg_inf_valid_mask
                )
        else:
            right_flat[..., -pad_amount or right_flat.shape[-1] :] += neg_inf

            if valid_mask is not None:
                assert neg_inf_valid_mask is not None
                right_flat[..., : -pad_amount or None] = (
                    right_flat[..., : -pad_amount or None] + neg_inf_valid_mask
                )

        right = rearrange(right_flat, "... j (k i) -> ... k j i", i=self.block_size)

        return right

    def _scaled_left_grad(
        self, left: Tensor, right: Tensor, query: Tensor, key: Tensor
    ) -> Tensor:
        alpha = torch.sum(right**2, dim=-1)
        grad = torch.einsum("...kj,...jlk->...jlk", alpha, left) - torch.einsum(
            "...kji,...ljv,...kiv->...jlk", right, query, key
        )
        step_size = 2 / torch.amax(alpha, dim=(-1, -2))
        return step_size[..., None, None, None] * grad

    def _scaled_right_grad(
        self, left: Tensor, right: Tensor, query: Tensor, key: Tensor
    ) -> Tensor:
        alpha = torch.sum(left**2, dim=-2)
        grad = torch.einsum("...jk,...kji->...kji", alpha, right) - torch.einsum(
            "...jlk,...ljv,...kiv->...kji", left, query, key
        )
        step_size = 2 / torch.amax(alpha, dim=(-1, -2))
        return step_size[..., None, None, None] * grad

    def _get_factors(
        self, query: Tensor, key: Tensor, valid_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch_size, num_heads, seq_len, head_dim = query.shape
        pad_amount = self._get_pad_amount(seq_len)
        num_blocks = self._get_num_blocks(seq_len)

        if valid_mask is not None:
            query = query * valid_mask.transpose(-1, -2)
            key = key * valid_mask.transpose(-1, -2)

        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == PadType.pre else (0, pad_amount)
        )
        query = pad(query, pad_t)
        key = pad(key, pad_t)
        query = rearrange(query, "... (l j) v -> ... l j v", j=self.block_size)
        key = rearrange(key, "... (k i) v -> ... k i v", i=self.block_size)

        left = (
            torch.ones(
                (batch_size, num_heads, self.block_size, num_blocks, num_blocks),
                device=query.device,
            )
            / num_blocks
        )
        right = (
            torch.ones(
                (batch_size, num_heads, num_blocks, self.block_size, self.block_size),
                device=query.device,
            )
            / self.block_size
        )
        right = self._mask(right, valid_mask, pad_amount)
        right = sparsemax(right, dim=-1)

        for _ in range(self.num_steps):
            right = right - self._scaled_right_grad(left, right, query, key)
            right = self._mask(right, valid_mask, pad_amount)
            right = sparsemax(right, dim=-1)

            left = left - self._scaled_left_grad(left, right, query, key)
            left = sparsemax(left, dim=-1)

        return left, right
