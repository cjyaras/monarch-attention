from math import ceil, sqrt
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.func import grad, grad_and_value  # type: ignore

# from torch.jit import script
from tqdm import tqdm

Tensor = torch.Tensor

# TODO: This is twice as slow as the JAX version, can we compute gradients explicitly?


def monarch_multiply(
    left: Tensor,
    right: Tensor,
    inputs: Tensor,
    block_size: int,
    pad_amount: int,
    pad_type: str,
):
    "Assumes that left and right are 2D, inputs is 1D."
    x = torch.nn.functional.pad(
        inputs,
        (pad_amount, 0) if pad_type == "pre" else (0, pad_amount),
    )
    x = rearrange(x, "(m b) -> m b", b=block_size)
    x = torch.einsum("kji,ki->kj", right, x)
    x = torch.einsum("jlk,kj->lj", left, x)
    x = rearrange(x, "m b -> (m b)")
    return x[pad_amount:] if pad_type == "pre" else x[: -pad_amount or None]


def monarch_gram_trace(left: Tensor, right: Tensor):
    "Assumes that left and right are 2D."
    left = torch.matmul(left.mT, left)
    right = torch.matmul(right, right.mT)
    left = torch.vmap(torch.diag)(left)
    right = torch.vmap(torch.diag)(right)
    return torch.sum(left * right.T)


def params_to_simplex(params: Tensor):
    return F.normalize(params, dim=-1) ** 2


def monarch_get_left(left_params: Tensor):
    return params_to_simplex(left_params)


def monarch_get_right(
    right_params: Tensor, block_size: int, pad_amount: int, pad_type: str
):
    mask = torch.ones_like(right_params)
    if pad_type == "pre":
        mask[0, :, :pad_amount] = 0.0
    else:
        mask[-1, :, block_size - pad_amount :] = 0.0
    right_params_zeroed = right_params * mask
    return params_to_simplex(right_params_zeroed)


def monarch_matrix(
    left: Tensor,
    right: Tensor,
    seq_len: int,
    block_size: int,
    pad_amount: int,
    pad_type: str,
):
    return torch.vmap(
        monarch_multiply, in_dims=(None, None, 1, None, None, None), out_dims=1
    )(left, right, torch.eye(seq_len), block_size, pad_amount, pad_type)


def objective(left_params, right_params, key, query, block_size, pad_amount, pad_type):
    left = monarch_get_left(left_params)
    right = monarch_get_right(right_params, block_size, pad_amount, pad_type)
    modified_key = torch.vmap(
        monarch_multiply, in_dims=(None, None, 1, None, None, None), out_dims=1
    )(left, right, key, block_size, pad_amount, pad_type)
    return 1 / 2 * monarch_gram_trace(left, right) - torch.trace(
        torch.matmul(query.T, modified_key)
    )


def monarch_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_size: int,
    pad_type: str = "pre",
    num_steps: int = 10,
    step_size: float = 1e4,
):
    seq_len = query.shape[0]
    m = ceil(seq_len / block_size)
    seq_len_padded = block_size * m
    pad_amount = seq_len_padded - seq_len

    left_params = F.normalize(torch.normal(0.0, 1.0, (block_size, m, m)), dim=-1)
    right_params = F.normalize(
        torch.normal(0.0, 1.0, (m, block_size, block_size)), dim=-1
    )

    for _ in range(num_steps):
        left_grad, right_grad = grad(objective, argnums=(0, 1))(
            left_params, right_params, key, query, block_size, pad_amount, pad_type
        )
        left_params -= step_size * left_grad
        right_params -= step_size * right_grad

    left = monarch_get_left(left_params)
    right = monarch_get_right(right_params, block_size, pad_amount, pad_type)
    return monarch_matrix(left, right, seq_len, block_size, pad_amount, pad_type)
    # return torch.vmap(
    #     monarch_multiply, in_dims=(None, None, 1, None, None, None), out_dims=1
    # )(left, right, value, block_size, pad_amount, pad_type)


def standard_attention(query: Tensor, key: Tensor, value: Tensor):
    attn_matrix = F.softmax(query @ key.T, dim=-1)
    return attn_matrix @ value


def time_attentions(query, n_iters):
    from math import floor
    from time import time

    # start = time()
    # for _ in tqdm(range(n_iters)):
    #     standard_attention(query, query, query)
    # end = time()
    # standard_time = (end - start) / n_iters

    standard_time = 0.0

    start = time()
    for _ in tqdm(range(n_iters)):
        monarch_attention(query, query, query, floor(sqrt(query.shape[0])), num_steps=5)
    end = time()

    efficient_time = (end - start) / n_iters

    return standard_time, efficient_time


def main():

    import matplotlib.pyplot as plt
    from sparsemax import Sparsemax

    torch.manual_seed(0)

    # seq_len = int(2**16)
    seq_len = 16
    model_dims = 64

    query = torch.normal(0, 1, (seq_len, model_dims))
    block_size = 1
    # print(time_attentions(query, 3))
    key = torch.normal(0, 1, (seq_len, model_dims))

    monarch_attn = monarch_attention(
        query, key, query, block_size, num_steps=1000, step_size=0.01
    )

    # plt.plot(values)
    # plt.show()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(Sparsemax()(torch.einsum("qd,kd->qk", query, key)))
    ax[1].imshow(monarch_attn)
    plt.show()


if __name__ == "__main__":
    main()
