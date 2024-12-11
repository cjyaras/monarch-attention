from math import ceil, sqrt
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.func import grad_and_value  # type: ignore

PadType = Literal["pre", "post"]
Tensor = torch.Tensor


def monarch_multiply(
    left: Tensor,
    right: Tensor,
    inputs: Tensor,
    block_size: int,
    pad_amount: int,
    pad_type: PadType,
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
    right_params: Tensor, block_size: int, pad_amount: int, pad_type: PadType
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
    pad_type: PadType,
):
    return torch.vmap(
        monarch_multiply, in_dims=(None, None, 1, None, None, None), out_dims=1
    )(left, right, torch.eye(seq_len), block_size, pad_amount, pad_type)


def monarch_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_size: int,
    pad_type: PadType = "pre",
    num_steps: int = 100,
    step_size: float = 1e3,
):
    seq_len = query.shape[0]
    m = ceil(seq_len / block_size)
    seq_len_padded = block_size * m
    pad_amount = seq_len_padded - seq_len

    def objective(left_params, right_params):
        left = monarch_get_left(left_params)
        right = monarch_get_right(right_params, block_size, pad_amount, pad_type)
        modified_key = torch.vmap(
            monarch_multiply, in_dims=(None, None, 1, None, None, None), out_dims=1
        )(left, right, key, block_size, pad_amount, pad_type)
        return (
            1 / 2 * monarch_gram_trace(left, right)
            - torch.trace(torch.matmul(query.T, modified_key))
            + 1
            / 2
            * torch.trace(
                torch.matmul(
                    torch.matmul(query.T, query),
                    torch.matmul(key.T, key),
                )
            )
        ) / seq_len**2

    left_params = torch.normal(0, 1, (block_size, m, m), requires_grad=False)
    right_params = torch.normal(0, 1, (m, block_size, block_size), requires_grad=False)

    values = []

    for _ in range(num_steps):
        (left_grad, right_grad), value = grad_and_value(objective, argnums=(0, 1))(
            left_params, right_params
        )
        left_params -= step_size * left_grad
        right_params -= step_size * right_grad
        values.append(value)

    left = monarch_get_left(left_params)
    right = monarch_get_right(right_params, block_size, pad_amount, pad_type)
    return (
        monarch_matrix(left, right, seq_len, block_size, pad_amount, pad_type),
        values,
    )


def main():
    import matplotlib.pyplot as plt
    from sparsemax import Sparsemax

    torch.manual_seed(0)

    seq_len = 17
    model_dims = 12
    block_size = 7

    query = torch.normal(0, 1, (seq_len, model_dims)) / sqrt(model_dims)
    key = torch.normal(0, 1, (seq_len, model_dims)) / sqrt(model_dims)

    monarch_attn, values = monarch_attention(query, key, key, block_size)

    plt.plot(values)
    plt.show()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(Sparsemax()(torch.einsum("qd,kd->qk", query, key)))
    ax[1].imshow(monarch_attn)
    plt.show()


if __name__ == "__main__":
    main()
