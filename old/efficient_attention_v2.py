from math import ceil, sqrt
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from einshape import jax_einshape as einshape
from opt_einsum import contract

Array = jax.Array
PadType = Literal["pre", "post"]


def normalize(x):
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def project(x, u):
    return contract("...i,...j,...j->...i", x, x, u)


def inv_norm(x):
    return 1 / jnp.linalg.norm(x, axis=-1, keepdims=True)


# Low-rank attention


def low_rank_multiply(left: Array, right: Array, inputs: Array):
    return left @ (right @ inputs)


def low_rank_matrix(left: Array, right: Array):
    return left @ right


scale = 1e0


def low_rank_attention(
    query: Array, key: Array, rank: int, num_steps: int, step_size: float
):
    batch_size = query.shape[:-2]
    seq_len = query.shape[-2]

    def grad(left_params: Array, right_params: Array) -> Tuple[Array, Array]:
        right_sphere = normalize(right_params)
        left_sphere = normalize(left_params)
        left = left_sphere**2
        right = right_sphere**2
        d_left = (
            contract("...jk,...ki,...li->...jl", left, right, right)
            - contract("...ja,...ia,...li->...jl", query, key, right)
        ) / seq_len**2
        d_left = d_left * 2 * left_sphere
        d_left = d_left - project(left_sphere, d_left)
        d_left = d_left * inv_norm(left_params)
        d_right = (
            contract("...jk,...jl,...li->...ki", left, left, right)
            - contract("...jk,...ja,...ia->...ki", left, query, key)
        ) / seq_len**2
        d_right = d_right * 2 * right_sphere
        d_right = d_right - project(right_sphere, d_right)
        d_right = d_right * inv_norm(right_params)
        return d_left, d_right

    def f(carrys, _):
        left_params, right_params = carrys
        d_left_params, d_right_params = grad(left_params, right_params)
        left_params = left_params - step_size * d_left_params
        right_params = right_params - step_size * d_right_params
        carrys = left_params, right_params
        return carrys, carrys

    init_left_params = jnp.sqrt(
        jr.dirichlet(jr.key(0), scale * jnp.ones(rank), batch_size + (seq_len,))
    )
    init_right_params = jnp.sqrt(
        jr.dirichlet(jr.key(0), scale * jnp.ones(seq_len), batch_size + (rank,))
    )

    _, (left_params_history, right_params_history) = jax.lax.scan(
        f, (init_left_params, init_right_params), jnp.arange(num_steps)
    )

    left_history = normalize(left_params_history) ** 2
    right_history = normalize(right_params_history) ** 2
    return low_rank_matrix(left_history, right_history)


# Monarch attention


def monarch_multiply(
    left: Array,
    right: Array,
    inputs: Array,
    block_size: int,
    pad_amount: int,
    pad_type: PadType,
):
    pad_width = (
        [(0, 0)] * (inputs.ndim - 2)
        + [(pad_amount, 0) if pad_type == "pre" else (0, pad_amount)]
        + [(0, 0)]
    )
    x = jnp.pad(inputs, pad_width)
    X = einshape("...(ki)a->...kia", x, i=block_size)
    Y = contract("...kji,...kia->...kja", right, X)
    Z = contract("...jlk,...kja->...lja", left, Y)
    z = einshape("...lja->...(lj)a", Z)
    return (
        z[..., pad_amount:, :]
        if pad_type == "pre"
        else z[..., : -pad_amount or None, :]
    )


def monarch_matrix(left: Array, right: Array, pad_amount: int, pad_type: PadType):
    matrix = einshape(
        "...ljki->...(lj)(ki)", contract("...jlk,...kji->...ljki", left, right)
    )
    return (
        matrix[..., pad_amount:, pad_amount:]
        if pad_type == "pre"
        else matrix[..., : -pad_amount or None, : -pad_amount or None]
    )


def monarch_attention(
    query: Array,
    key: Array,
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
    pad_width = (
        [(0, 0)] * (query.ndim - 2)
        + [(pad_amount, 0) if pad_type == "pre" else (0, pad_amount)]
        + [(0, 0)]
    )
    query = jnp.pad(query, pad_width)
    key = jnp.pad(key, pad_width)
    query = einshape("...(lj)a->...lja", query, j=block_size)
    key = einshape("...(ki)a->...kia", key, i=block_size)

    def mask(x):
        if pad_type == "pre":
            x = x.at[..., 0, :, :pad_amount].set(0.0)
        else:
            x = x.at[..., -1, :, block_size - pad_amount :].set(0.0)
        return x

    def grad(left_params: Array, right_params: Array) -> Tuple[Array, Array]:
        right_params = mask(right_params)
        left_sphere = normalize(left_params)
        right_sphere = normalize(right_params)
        left = left_sphere**2
        right = right_sphere**2
        d_left = (
            contract("...kj,...jlk->...jlk", jnp.sum(right**2, axis=-1), left)
            - contract("...kji,...lja,...kia->...jlk", right, query, key)
        ) / seq_len**2
        d_left = d_left * 2 * left_sphere
        d_left = d_left - project(left_sphere, d_left)
        d_left = d_left * inv_norm(left_params)
        d_right = (
            contract("...jk,...kji->...kji", jnp.sum(left**2, axis=-2), right)
            - contract("...jlk,...lja,...kia->...kji", left, query, key)
        ) / seq_len**2
        d_right = d_right * 2 * right_sphere
        d_right = d_right - project(right_sphere, d_right)
        d_right = d_right * inv_norm(right_params)
        return d_left, d_right

    def f(carrys, _):
        left_params, right_params = carrys
        d_left_params, d_right_params = grad(left_params, right_params)
        left_params = left_params - step_size * d_left_params
        right_params = right_params - step_size * d_right_params
        carrys = left_params, right_params
        return carrys, carrys

    init_left_params = jnp.sqrt(
        jr.dirichlet(
            jr.key(0),
            scale * jnp.ones(num_blocks),
            batch_size + (block_size, num_blocks),
        )
    )
    init_right_params = jnp.sqrt(
        jr.dirichlet(
            jr.key(1),
            scale * jnp.ones(block_size),
            batch_size + (num_blocks, block_size),
        )
    )

    _, (left_params_history, right_params_history) = jax.lax.scan(
        f, (init_left_params, init_right_params), jnp.arange(num_steps)
    )

    left_history = normalize(left_params_history) ** 2
    right_history = normalize(mask(right_params_history)) ** 2
    return monarch_matrix(left_history, right_history, pad_amount, pad_type)


def main():

    import matplotlib.pyplot as plt

    from projections import sparsemax

    # seq_len = int(2**15)
    seq_len = 16
    model_dims = 8
    query = jr.normal(jr.key(2), (seq_len, model_dims))
    key = jr.normal(jr.key(3), (seq_len, model_dims))
    monarch_attn = monarch_attention(
        query, key, block_size=1, num_steps=100, step_size=1e5
    )
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(sparsemax(contract("qd,kd->qk", query, key), axis=-1))
    ax[1].imshow(monarch_attn[-1])
    plt.show()


if __name__ == "__main__":
    main()
