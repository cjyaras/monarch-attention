# TODO: Gradients should be scaled so that same step size works across sequence lengths
from math import ceil, sqrt
from typing import Literal, Tuple

import torch
from einops import rearrange
from torch.distributions import Dirichlet
from torch.nn.functional import normalize, pad

Tensor = torch.Tensor
PadType = Literal["pre", "post"]
# DIRICHLET_SCALE = 1e3

einsum = torch.einsum


def project(x, u):
    return einsum("...i,...j,...j->...i", x, x, u)


def inv_norm(x):
    return 1 / torch.linalg.norm(x, dim=-1, keepdims=True)


## Efficient low-rank attention


def low_rank_multiply(left: Tensor, right: Tensor, inputs: Tensor):
    return left @ (right @ inputs)


def low_rank_matrix(left: Tensor, right: Tensor):
    return left @ right


def low_rank_grad(
    left_params: Tensor, right_params: Tensor, query: Tensor, key: Tensor, seq_len: int
) -> Tuple[Tensor, Tensor]:
    right_sphere = normalize(right_params, dim=-1)
    left_sphere = normalize(left_params, dim=-1)
    left = left_sphere**2
    right = right_sphere**2
    d_left = (
        einsum("...jk,...ki,...li->...jl", left, right, right)
        - einsum("...ja,...ia,...li->...jl", query, key, right)
    ) / seq_len**2
    d_left = d_left * 2 * left_sphere
    d_left = d_left - project(left_sphere, d_left)
    d_left = d_left * inv_norm(left_params)
    d_right = (
        einsum("...jk,...jl,...li->...ki", left, left, right)
        - einsum("...jk,...ja,...ia->...ki", left, query, key)
    ) / seq_len**2
    d_right = d_right * 2 * right_sphere
    d_right = d_right - project(right_sphere, d_right)
    d_right = d_right * inv_norm(right_params)
    return d_left, d_right


def low_rank_attention_factors(
    query: Tensor, key: Tensor, rank: int, num_steps: int, step_size: float
):
    batch_size = query.shape[:-2]
    seq_len = query.shape[-2]

    # left_params = torch.sqrt(
    #     Dirichlet(DIRICHLET_SCALE * torch.ones(rank)).sample(batch_size + (seq_len,))
    # )

    # right_params = torch.sqrt(
    #     Dirichlet(DIRICHLET_SCALE * torch.ones(seq_len)).sample(batch_size + (rank,))
    # )

    left_params = torch.rand(batch_size + (seq_len, rank))
    right_params = torch.rand(batch_size + (rank, seq_len))

    for _ in range(num_steps):
        d_left_params, d_right_params = low_rank_grad(
            left_params, right_params, query, key, seq_len
        )
        left_params = left_params - step_size * d_left_params
        right_params = right_params - step_size * d_right_params

    left = normalize(left_params, dim=-1) ** 2
    right = normalize(right_params, dim=-1) ** 2

    return left, right


def low_rank_attention_matrix(
    query: Tensor, key: Tensor, rank: int, num_steps: int, step_size: float
):
    left, right = low_rank_attention_factors(query, key, rank, num_steps, step_size)
    return low_rank_matrix(left, right)


def low_rank_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    rank: int,
    num_steps: int,
    step_size: float,
):
    left, right = low_rank_attention_factors(query, key, rank, num_steps, step_size)
    return low_rank_multiply(left, right, value)


## Efficient monarch attention


def monarch_multiply(
    left: Tensor,
    right: Tensor,
    inputs: Tensor,
    block_size: int,
    pad_amount: int,
    pad_type: PadType,
):
    pad_t = (0, 0) + (pad_amount, 0) if pad_type == "pre" else (0, pad_amount)
    x = pad(inputs, pad_t)
    X = rearrange(x, "... (k i) a -> ... k i a", i=block_size)
    Y = einsum("...kji,...kia->...kja", right, X)
    Z = einsum("...jlk,...kja->...lja", left, Y)
    z = rearrange(Z, "... l j a -> ... (l j) a")
    return (
        z[..., pad_amount:, :]
        if pad_type == "pre"
        else z[..., : -pad_amount or None, :]
    )


def monarch_matrix(left: Tensor, right: Tensor, pad_amount: int, pad_type: PadType):
    out = einsum("...jlk,...kji->...ljki", left, right)
    out = rearrange(out, "... l j k i -> ... (l j) (k i)")
    return (
        out[..., pad_amount:, pad_amount:]
        if pad_type == "pre"
        else out[..., : -pad_amount or None, : -pad_amount or None]
    )


def monarch_mask(x, pad_type: PadType, pad_amount: int):
    if pad_type == "pre":
        x[..., 0, :, :pad_amount] = 0.0
    else:
        x[..., -1, :, -pad_amount or None :] = 0
    return x


def monarch_grad(
    left_params: Tensor,
    right_params: Tensor,
    query: Tensor,
    key: Tensor,
    seq_len: int,
    pad_type: PadType,
    pad_amount: int,
) -> Tuple[Tensor, Tensor]:
    right_params = monarch_mask(right_params, pad_type, pad_amount)
    left_sphere = normalize(left_params, dim=-1)
    right_sphere = normalize(right_params, dim=-1)
    left = left_sphere**2
    right = right_sphere**2
    d_left = (
        einsum("...kj,...jlk->...jlk", torch.sum(right**2, dim=-1), left)
        - einsum("...kji,...lja,...kia->...jlk", right, query, key)
    ) / seq_len**2
    d_left = d_left * 2 * left_sphere
    d_left = d_left - project(left_sphere, d_left)
    d_left = d_left * inv_norm(left_params)
    d_right = (
        einsum("...jk,...kji->...kji", torch.sum(left**2, dim=-2), right)
        - einsum("...jlk,...lja,...kia->...kji", left, query, key)
    ) / seq_len**2
    d_right = d_right * 2 * right_sphere
    d_right = d_right - project(right_sphere, d_right)
    d_right = d_right * inv_norm(right_params)
    return d_left, d_right


@torch.compile()
def monarch_attention_factors(
    query: Tensor,
    key: Tensor,
    block_size: int,
    num_steps: int,
    step_size: float,
    pad_type: PadType = "pre",
):
    batch_size = query.shape[:-2]
    seq_len = query.shape[-2]
    num_blocks = ceil(seq_len / block_size)
    seq_len_padded = block_size * num_blocks
    pad_amount = seq_len_padded - seq_len

    # Pad and reshape query/key into 3D tensors
    pad_t = (0, 0) + (pad_amount, 0) if pad_type == "pre" else (0, pad_amount)
    query = pad(query, pad_t)
    key = pad(key, pad_t)
    query = rearrange(query, "... (l j) a -> ... l j a", j=block_size)
    key = rearrange(key, "... (k i) a -> ... k i a", i=block_size)

    # left_params = torch.sqrt(
    #     Dirichlet(1e3 * torch.ones(num_blocks)).sample(
    #         batch_size + (block_size, num_blocks)
    #     )
    # )

    # right_params = torch.sqrt(
    #     Dirichlet(1e3 * torch.ones(block_size)).sample(
    #         batch_size + (num_blocks, block_size)
    #     )
    # )

    # left_params = torch.sqrt(
    #     torch.abs(
    #         torch.randn(batch_size + (block_size, num_blocks, num_blocks))
    #         / sqrt(num_blocks)
    #     )
    # )
    # right_params = torch.sqrt(
    #     torch.abs(
    #         torch.rand(batch_size + (num_blocks, block_size, block_size))
    #         / sqrt(block_size)
    #     )
    # )

    left_params = (
        1 / sqrt(num_blocks)
        + 2e-3 * torch.rand(batch_size + (block_size, num_blocks, num_blocks))
        - 1e-3
    )
    right_params = (
        1 / sqrt(block_size)
        + 2e-3 * torch.rand(batch_size + (num_blocks, block_size, block_size))
        - 1e-3
    )

    # left_params = torch.rand(batch_size + (block_size, num_blocks, num_blocks))
    # right_params = torch.rand(batch_size + (num_blocks, block_size, block_size))

    for _ in range(num_steps):
        d_left_params, d_right_params = monarch_grad(
            left_params,
            right_params,
            query,
            key,
            seq_len,
            pad_type,
            pad_amount,
        )
        left_params = left_params - step_size * d_left_params
        right_params = right_params - step_size * d_right_params

    left = normalize(left_params, dim=-1) ** 2
    right = normalize(monarch_mask(right_params, pad_type, pad_amount), dim=-1) ** 2

    return left, right


def monarch_attention_matrix(
    query: Tensor,
    key: Tensor,
    block_size: int,
    num_steps: int,
    step_size: float,
    pad_type: PadType = "pre",
):
    seq_len = query.shape[-2]
    num_blocks = ceil(seq_len / block_size)
    seq_len_padded = block_size * num_blocks
    pad_amount = seq_len_padded - seq_len

    left, right = monarch_attention_factors(
        query, key, block_size, num_steps, step_size, pad_type
    )
    return monarch_matrix(left, right, pad_amount, pad_type)


def monarch_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_size: int,
    num_steps: int,
    step_size: float,
    pad_type: PadType = "pre",
):
    seq_len = query.shape[-2]
    num_blocks = ceil(seq_len / block_size)
    seq_len_padded = block_size * num_blocks
    pad_amount = seq_len_padded - seq_len

    left, right = monarch_attention_factors(
        query, key, block_size, num_steps, step_size, pad_type
    )
    return monarch_multiply(left, right, value, block_size, pad_amount, pad_type)


@torch.no_grad()
def main():

    from time import time

    import matplotlib.pyplot as plt
    from sparsemax import Sparsemax

    seq_len = int(2**16)
    # seq_len = 196
    model_dims = 64
    # query = jr.normal(jr.key(2), (seq_len, model_dims))
    query = torch.randn(1, seq_len, model_dims)
    key = torch.randn(1, seq_len, model_dims)
    # monarch_attn = monarch_attention_matrix(
    #     query, key, block_size=int(2**8), num_steps=10, step_size=1e1
    # )
    monarch_attention_factors(
        query, key, block_size=int(2**8), num_steps=10, step_size=1e1
    )

    start = time()
    monarch_attention_factors(
        query, key, block_size=int(2**8), num_steps=10, step_size=1e1
    )
    print("Monarch factors", time() - start)
    # lr_attn = low_rank_attention_matrix(query, key, rank=2, num_steps=20, step_size=1e1)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(Sparsemax(dim=-1)(einsum("qd,kd->qk", query, key)))
    # ax[1].imshow(monarch_attn)
    # plt.show()


if __name__ == "__main__":
    main()
