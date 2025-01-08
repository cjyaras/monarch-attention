from math import ceil, prod, sqrt
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch._prims_common import DeviceLikeType
from torch.nn.functional import pad

Tensor = torch.Tensor


def _fast_simplex_init(
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


def _safe_normalize(x: Tensor, dim: int) -> Tensor:
    norms = torch.linalg.norm(x, dim=dim, keepdims=True)
    return torch.where(norms > 0, x / norms, x)


def _safe_inv_norm(x: Tensor, dim: int) -> Tensor:
    norms = torch.linalg.norm(x, dim=dim, keepdims=True)
    return torch.where(norms > 0, 1 / norms, 0.0)


class LowRankMHA(nn.Module):

    def __init__(self, rank: int, num_steps: int, step_size: float):
        super().__init__()
        self.rank = rank
        self.num_steps = num_steps
        self.step_size = step_size

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
        valid_mask = self._get_valid_mask(attention_mask)
        left, right = self._get_factors(query, key, valid_mask)
        return self._get_matrix_from_factors(left, right)

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

        query = query / sqrt(query.shape[-1])
        valid_mask = self._get_valid_mask(attention_mask)
        left, right = self._get_factors(query, key, valid_mask)
        return self._multiply(left, right, value)

    def _multiply(self, left: Tensor, right: Tensor, inputs: Tensor) -> Tensor:
        return left @ (right @ inputs)

    def _get_matrix_from_factors(self, left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    def _get_valid_mask(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")
        return attention_mask

    def _mask(self, right_params: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        if valid_mask is None:
            return right_params
        return right_params * valid_mask

    def _grad(
        self,
        left_params: Tensor,
        right_params: Tensor,
        query: Tensor,
        key: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        right_sphere = _safe_normalize(right_params, dim=-1)
        left_sphere = _safe_normalize(left_params, dim=-1)
        left = left_sphere**2
        right = right_sphere**2
        d_left = left @ (right @ right.mT) - query @ (key.mT @ right.mT)
        d_left = d_left * 2 * left_sphere
        d_left = d_left - _project(left_sphere, d_left)
        d_left = d_left * _safe_inv_norm(left_params, dim=-1)
        d_right = (left.mT @ left) @ right - (left.mT @ query) @ key.mT
        d_right = d_right * 2 * right_sphere
        d_right = d_right - _project(right_sphere, d_right)
        d_right = d_right * _safe_inv_norm(right_params, dim=-1)
        return d_left, d_right

    def _get_factors(
        self, query: Tensor, key: Tensor, valid_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch_size, num_heads, seq_len, head_dim = query.shape

        left_params = _fast_simplex_init(
            (batch_size, num_heads, seq_len, self.rank),
            perturb_scale=1e-3,
            device=query.device,
        )
        right_params = _fast_simplex_init(
            (batch_size, num_heads, self.rank, seq_len),
            perturb_scale=1e-3,
            device=query.device,
        )

        for _ in range(self.num_steps):
            right_params = self._mask(right_params, valid_mask)
            d_left_params, d_right_params = self._grad(
                left_params, right_params, query, key
            )
            left_params = left_params - self.step_size * d_left_params
            right_params = right_params - self.step_size * d_right_params

        left = _safe_normalize(left_params, dim=-1) ** 2
        right = _safe_normalize(self._mask(right_params, valid_mask), dim=-1) ** 2

        return left, right


PadType = Literal["pre", "post"]


class MonarchMHA(nn.Module):

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

    def get_matrix(
        self, query: Tensor, key: Tensor, attention_mask: Optional[Tensor] = None
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
        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == "pre" else (0, pad_amount)
        )
        x = pad(inputs, pad_t)
        X = rearrange(x, "... (k i) d -> ... k i d", i=self.block_size)
        Y = torch.einsum("...kji,...kid->...kjd", right, X)
        Z = torch.einsum("...jlk,...kjd->...ljd", left, Y)
        z = rearrange(Z, "... l j d -> ... (l j) d")

        if self.pad_type == "pre":
            return z[..., pad_amount:, :]
        else:
            return z[..., : -pad_amount or None, :]

    def _get_matrix_from_factors(
        self, left: Tensor, right: Tensor, pad_amount: int
    ) -> Tensor:
        out = torch.einsum("...jlk,...kji->...ljki", left, right)
        out = rearrange(out, "... l j k i -> ... (l j) (k i)")

        if self.pad_type == "pre":
            return out[..., pad_amount:, pad_amount:]
        else:
            return out[..., : -pad_amount or None, : -pad_amount or None]

    def _get_valid_mask(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")
        return attention_mask

    def _mask(
        self, right_params: Tensor, valid_mask: Optional[Tensor], pad_amount: int
    ) -> Tensor:
        right_params_flat = rearrange(right_params, "... k j i -> ... j (k i)")

        if self.pad_type == "pre":
            right_params_flat[..., :pad_amount] = 0.0

            if valid_mask is not None:
                right_params_flat[..., pad_amount:] = (
                    right_params_flat[..., pad_amount:] * valid_mask
                )
        else:
            right_params_flat[..., -pad_amount or right_params_flat.shape[-1] :] = 0.0

            if valid_mask is not None:
                right_params_flat[..., : -pad_amount or None] = (
                    right_params_flat[..., : -pad_amount or None] * valid_mask
                )

        right_params = rearrange(
            right_params_flat, "... j (k i) -> ... k j i", i=self.block_size
        )

        return right_params

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
        ) - torch.einsum("...kji,...ljd,...kid->...jlk", right, query, key)
        d_left = d_left * 2 * left_sphere
        d_left = d_left - _project(left_sphere, d_left)
        d_left = d_left * _safe_inv_norm(left_params, dim=-1)
        d_right = torch.einsum(
            "...jk,...kji->...kji", torch.sum(left**2, dim=-2), right
        ) - torch.einsum("...jlk,...ljd,...kid->...kji", left, query, key)
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

        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == "pre" else (0, pad_amount)
        )
        query = pad(query, pad_t)
        key = pad(key, pad_t)
        query = rearrange(query, "... (l j) d -> ... l j d", j=self.block_size)
        key = rearrange(key, "... (k i) d -> ... k i d", i=self.block_size)

        left_params = _fast_simplex_init(
            (batch_size, num_heads, self.block_size, num_blocks, num_blocks),
            device=query.device,
        )
        right_params = _fast_simplex_init(
            (batch_size, num_heads, num_blocks, self.block_size, self.block_size),
            device=query.device,
        )

        for _ in range(self.num_steps):
            right_params = self._mask(right_params, valid_mask, pad_amount)
            d_left_params, d_right_params = self._grad(
                left_params, right_params, query, key
            )
            left_params = left_params - self.step_size * d_left_params
            right_params = right_params - self.step_size * d_right_params

        left = _safe_normalize(left_params, dim=-1) ** 2
        right = (
            _safe_normalize(self._mask(right_params, valid_mask, pad_amount), dim=-1)
            ** 2
        )

        return left, right


class MonarchBlockDiagonalMHA(nn.Module):

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

    def get_matrix(
        self, query: Tensor, key: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        assert query.shape == key.shape
        batch_size, num_heads, seq_len, head_dim = query.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)

        query = query / sqrt(head_dim)
        pad_amount = self._get_pad_amount(seq_len)
        valid_mask = self._get_valid_mask(attention_mask)
        block_diag, left, right = self._get_factors(query, key, valid_mask)
        matrix = self._get_matrix_from_factors(block_diag, left, right, pad_amount)
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
        block_diag, left, right = self._get_factors(query, key, valid_mask)
        return self._multiply(block_diag, left, right, value)

    def _get_num_blocks(self, seq_len: int) -> int:
        num_blocks = ceil(seq_len / self.block_size)
        return num_blocks

    def _get_pad_amount(self, seq_len: int) -> int:
        num_blocks = self._get_num_blocks(seq_len)
        seq_len_padded = self.block_size * num_blocks
        pad_amount = seq_len_padded - seq_len
        return pad_amount

    def _multiply(
        self, block_diag: Tensor, left: Tensor, right: Tensor, inputs: Tensor
    ) -> Tensor:
        seq_len = inputs.shape[-2]
        pad_amount = self._get_pad_amount(seq_len)
        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == "pre" else (0, pad_amount)
        )

        x = pad(inputs, pad_t)
        X = rearrange(x, "... (k i) d -> ... k i d", i=self.block_size)

        Y = torch.einsum("...kji,...kid->...kjd", right, X)
        Z = torch.einsum("...jlk,...kjd->...ljd", left, Y)
        Z = Z + torch.einsum("...ljk,...lkd->...ljd", block_diag, X)
        z = rearrange(Z, "... l j d -> ... (l j) d")

        if self.pad_type == "pre":
            return z[..., pad_amount:, :]
        else:
            return z[..., : -pad_amount or None, :]

    def _get_matrix_from_factors(
        self, block_diag: Tensor, left: Tensor, right: Tensor, pad_amount: int
    ) -> Tensor:
        num_blocks = self._get_num_blocks(prod(right.shape[-3:-1]))
        block_diag_out = torch.einsum(
            "...jlk,...kji->...ljki",
            torch.eye(num_blocks).expand(
                block_diag.shape[:-3] + (self.block_size, num_blocks, num_blocks)
            ),
            block_diag,
        )
        monarch_out = torch.einsum("...jlk,...kji->...ljki", left, right)

        out = rearrange(
            block_diag_out + monarch_out,
            "... l j k i -> ... (l j) (k i)",
        )

        if self.pad_type == "pre":
            return out[..., pad_amount:, pad_amount:]
        else:
            return out[..., : -pad_amount or None, : -pad_amount or None]

    def _get_valid_mask(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")
        return attention_mask

    def _mask(
        self,
        block_diag_flat_and_left_flat_params: Tensor,
        right_params: Tensor,
        valid_mask: Optional[Tensor],
        pad_amount: int,
    ) -> Tuple[Tensor, Tensor]:

        block_diag_flat_params = block_diag_flat_and_left_flat_params[
            ..., : self.block_size
        ]
        left_flat_params = block_diag_flat_and_left_flat_params[..., self.block_size :]

        block_diag_flat_params_transposed = rearrange(
            block_diag_flat_params, "... (k j) i -> ... j (k i)", j=self.block_size
        )
        right_flat_params_transposed = rearrange(
            right_params, "... k j i -> ... j (k i)"
        )

        if self.pad_type == "pre":
            block_diag_flat_params_transposed[..., :pad_amount] = 0.0
            right_flat_params_transposed[..., :pad_amount] = 0.0

            if valid_mask is not None:
                block_diag_flat_params_transposed[..., pad_amount:] = (
                    block_diag_flat_params_transposed[..., pad_amount:] * valid_mask
                )
                right_flat_params_transposed[..., pad_amount:] = (
                    right_flat_params_transposed[..., pad_amount:] * valid_mask
                )
        else:
            block_diag_flat_params_transposed[
                ..., -pad_amount or block_diag_flat_params_transposed.shape[-1] :
            ] = 0.0
            right_flat_params_transposed[
                ..., -pad_amount or right_flat_params_transposed.shape[-1] :
            ] = 0.0

            if valid_mask is not None:
                block_diag_flat_params_transposed[..., : -pad_amount or None] = (
                    block_diag_flat_params_transposed[..., : -pad_amount or None]
                    * valid_mask
                )
                right_flat_params_transposed[..., : -pad_amount or None] = (
                    right_flat_params_transposed[..., : -pad_amount or None]
                    * valid_mask
                )

        block_diag_flat_params = rearrange(
            block_diag_flat_params_transposed,
            "... j (k i) -> ... (k j) i",
            i=self.block_size,
        )

        right_params = rearrange(
            right_flat_params_transposed, "... j (k i) -> ... k j i", i=self.block_size
        )

        block_diag_flat_and_left_flat_params = torch.cat(
            [block_diag_flat_params, left_flat_params], dim=-1
        )
        return block_diag_flat_and_left_flat_params, right_params

    def _grad(
        self,
        block_diag_flat_and_left_flat_params: Tensor,
        right_params: Tensor,
        query: Tensor,
        key: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        num_blocks = right_params.shape[-3]
        block_diag_flat_and_left_flat_sphere = _safe_normalize(
            block_diag_flat_and_left_flat_params, dim=-1
        )
        right_sphere = _safe_normalize(right_params, dim=-1)

        block_diag_flat_and_left_flat = block_diag_flat_and_left_flat_sphere**2
        right = right_sphere**2

        block_diag_flat = block_diag_flat_and_left_flat[..., : self.block_size]
        left_flat = block_diag_flat_and_left_flat[..., self.block_size :]

        block_diag = rearrange(
            block_diag_flat, "... (k j) i -> ... k j i", j=self.block_size
        )

        left = rearrange(left_flat, "... (l j) k -> ... j l k", j=self.block_size)

        d_block_diag = (
            block_diag
            + torch.einsum("...jkk->...kj", left)[..., None] * right
            - query @ key.mT
        )

        d_left = torch.einsum(
            "...kj,...jlk->...jlk", torch.sum(right**2, dim=-1), left
        ) - torch.einsum("...kji,...ljd,...kid->...jlk", right, query, key)

        d_left[..., torch.arange(num_blocks), torch.arange(num_blocks)] += torch.einsum(
            "...ijk,...ijk->...ji", right, block_diag
        )

        d_right = (
            torch.einsum("...jk,...kji->...kji", torch.sum(left**2, dim=-2), right)
            - torch.einsum("...jlk,...ljd,...kid->...kji", left, query, key)
            + torch.einsum("...jii,...ijk->...ijk", left, block_diag)
        )

        d_block_diag_flat = rearrange(d_block_diag, "... k j i -> ... (k j) i")
        d_left_flat = rearrange(d_left, "... j l k -> ... (l j) k")
        d_block_diag_flat_and_left_flat = torch.cat(
            [d_block_diag_flat, d_left_flat], dim=-1
        )
        d_block_diag_flat_and_left_flat = (
            d_block_diag_flat_and_left_flat * 2 * block_diag_flat_and_left_flat_sphere
        )
        d_block_diag_flat_and_left_flat = d_block_diag_flat_and_left_flat - _project(
            block_diag_flat_and_left_flat_sphere, d_block_diag_flat_and_left_flat
        )
        d_block_diag_flat_and_left_flat = (
            d_block_diag_flat_and_left_flat
            * _safe_inv_norm(block_diag_flat_and_left_flat_params, dim=-1)
        )

        d_right = d_right * 2 * right_sphere
        d_right = d_right - _project(right_sphere, d_right)
        d_right = d_right * _safe_inv_norm(right_params, dim=-1)

        return d_block_diag_flat_and_left_flat, d_right

    def _get_factors(
        self, query: Tensor, key: Tensor, valid_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, num_heads, seq_len, head_dim = query.shape
        pad_amount = self._get_pad_amount(seq_len)
        num_blocks = self._get_num_blocks(seq_len)
        padded_seq_len = seq_len + pad_amount

        pad_t = (0, 0) + (
            (pad_amount, 0) if self.pad_type == "pre" else (0, pad_amount)
        )
        query = pad(query, pad_t)
        key = pad(key, pad_t)
        query = rearrange(query, "... (l j) d -> ... l j d", j=self.block_size)
        key = rearrange(key, "... (k i) d -> ... k i d", i=self.block_size)

        block_diag_flat_and_left_flat_params = _fast_simplex_init(
            (batch_size, num_heads, padded_seq_len, self.block_size + num_blocks),
            device=query.device,
        )

        right_params = _fast_simplex_init(
            (batch_size, num_heads, num_blocks, self.block_size, self.block_size),
            device=query.device,
        )

        for _ in range(self.num_steps):
            block_diag_flat_and_left_flat_params, right_params = self._mask(
                block_diag_flat_and_left_flat_params,
                right_params,
                valid_mask,
                pad_amount,
            )
            d_block_diag_flat_and_left_flat_params, d_right_params = self._grad(
                block_diag_flat_and_left_flat_params, right_params, query, key
            )
            block_diag_flat_and_left_flat_params = (
                block_diag_flat_and_left_flat_params
                - self.step_size * d_block_diag_flat_and_left_flat_params
            )
            right_params = right_params - self.step_size * d_right_params

        block_diag_flat_and_left_flat = (
            _safe_normalize(block_diag_flat_and_left_flat_params, dim=-1) ** 2
        )
        block_diag_flat = block_diag_flat_and_left_flat[..., : self.block_size]
        left_flat = block_diag_flat_and_left_flat[..., self.block_size :]
        block_diag = rearrange(
            block_diag_flat, "... (k j) i -> ... k j i", j=self.block_size
        )
        left = rearrange(left_flat, "... (l j) k -> ... j l k", j=self.block_size)
        right = _safe_normalize(right_params, dim=-1) ** 2
        return block_diag, left, right


def main():

    import matplotlib.pyplot as plt
    from entmax import sparsemax

    mha = MonarchMHA(2, 100, 1e1, "pre")
    query = torch.randn(1, 1, 16, 4)
    key = torch.randn(1, 1, 16, 4)

    attention_mask = torch.tensor([13 * [1] + 3 * [0]])

    original_matrix = sparsemax(
        query @ key.transpose(-1, -2) / sqrt(query.shape[-1])
        + (1 - attention_mask[:, None, None, :]) * -1e9
    )[  # type: ignore
        0, 0
    ]
    efficient_matrix = mha.get_matrix(query, key, attention_mask)[0, 0]
    print(efficient_matrix)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original_matrix)
    axes[1].imshow(efficient_matrix)
    plt.show()


if __name__ == "__main__":
    main()
