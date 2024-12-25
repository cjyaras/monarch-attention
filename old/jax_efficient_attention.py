from math import ceil, sqrt
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from einshape import jax_einshape as einshape

Array = jax.Array

from tqdm import tqdm

PadType = Literal["pre", "post"]


def monarch_multiply(
    left: Array,
    right: Array,
    inputs: Array,
    block_size: int,
    pad_amount: int,
    pad_type: PadType,
):
    "Assumes that left and right are 2D, inputs is 1D."
    x = jnp.pad(
        inputs,
        (pad_amount, 0) if pad_type == "pre" else (0, pad_amount),
    )
    x = einshape("(mb)->mb", x, b=block_size)
    x = jnp.einsum("kji,ki->kj", right, x)
    x = jnp.einsum("jlk,kj->lj", left, x)
    x = einshape("mb->(mb)", x)
    return x[pad_amount:] if pad_type == "pre" else x[: -pad_amount or None]


def monarch_gram_trace(left: Array, right: Array):
    "Assumes that left and right are 2D."
    left = jnp.matmul(left.mT, left)
    right = jnp.matmul(right, right.mT)
    left = jax.vmap(jnp.diag)(left)
    right = jax.vmap(jnp.diag)(right)
    return jnp.sum(left * right.T)


def normalize(x):
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def params_to_simplex(params: Array):
    return normalize(params) ** 2


def monarch_get_left(left_params: Array):
    return params_to_simplex(left_params)


def monarch_get_right(
    right_params: Array, block_size: int, pad_amount: int, pad_type: PadType
):
    if pad_type == "pre":
        right_params = right_params.at[0, :, :pad_amount].set(0.0)
    else:
        right_params = right_params.at[-1, :, block_size - pad_amount :].set(0.0)
    return params_to_simplex(right_params)


def monarch_matrix(
    left: Array,
    right: Array,
    seq_len: int,
    block_size: int,
    pad_amount: int,
    pad_type: PadType,
):
    return jax.vmap(
        monarch_multiply, in_axes=(None, None, 1, None, None, None), out_axes=1
    )(left, right, jnp.eye(seq_len), block_size, pad_amount, pad_type)


def monarch_attention(
    query: Array,
    key: Array,
    value: Array,
    block_size: int,
    pad_type: PadType = "pre",
    num_steps: int = 10,
    step_size: float = 1e4,
):
    seq_len = query.shape[0]
    m = ceil(seq_len / block_size)
    seq_len_padded = block_size * m
    pad_amount = seq_len_padded - seq_len

    def objective(left_params, right_params):
        left = monarch_get_left(left_params)
        right = monarch_get_right(right_params, block_size, pad_amount, pad_type)
        modified_key = jax.vmap(
            monarch_multiply, in_axes=(None, None, 1, None, None, None), out_axes=1
        )(left, right, key, block_size, pad_amount, pad_type)
        return 1 / 2 * monarch_gram_trace(left, right) - jnp.trace(
            jnp.matmul(query.T, modified_key)
        )

    left_params = normalize(jr.normal(jr.key(0), (block_size, m, m)))
    right_params = normalize(jr.normal(jr.key(1), (m, block_size, block_size)))

    def f(carrys, _):
        left_params, right_params = carrys
        left_grad, right_grad = jax.grad(objective, argnums=(0, 1))(
            left_params, right_params
        )
        left_params -= step_size * left_grad
        right_params -= step_size * right_grad
        return (left_params, right_params), None

    (left_params, right_params), _ = jax.lax.scan(
        f, (left_params, right_params), jnp.arange(num_steps)
    )

    left = monarch_get_left(left_params)
    right = monarch_get_right(right_params, block_size, pad_amount, pad_type)
    return jax.vmap(
        monarch_multiply, in_axes=(None, None, 1, None, None, None), out_axes=1
    )(left, right, jnp.eye(seq_len), block_size, pad_amount, pad_type)
    # return jax.vmap(
    #     monarch_multiply, in_axes=(None, None, 1, None, None, None), out_axes=1
    # )(left, right, value, block_size, pad_amount, pad_type)


def standard_attention(query: Array, key: Array, value: Array):
    attn_matrix = jax.nn.softmax(query @ key.T, axis=-1)
    return attn_matrix @ value


def time_attentions(query, n_iters):
    from math import floor
    from time import time

    # standard_attention(query, query, query)
    # start = time()
    # for _ in tqdm(range(n_iters)):
    #     result = standard_attention(query, query, query).block_until_ready()
    # end = time()
    # standard_time = (end - start) / n_iters
    standard_time = 0.0

    monarch_attention(query, query, query, floor(sqrt(query.shape[0])))
    start = time()
    for _ in tqdm(range(n_iters)):
        result = monarch_attention(
            query, query, query, floor(sqrt(query.shape[0])), num_steps=5
        ).block_until_ready()
    end = time()

    efficient_time = (end - start) / n_iters

    return standard_time, efficient_time


def main():

    import matplotlib.pyplot as plt

    from projections import sparsemax

    # seq_len = int(2**15)
    seq_len = 32
    model_dims = 8
    query = jr.normal(jr.key(2), (seq_len, model_dims))
    key = jr.normal(jr.key(3), (seq_len, model_dims))
    monarch_attn = monarch_attention(
        query, key, query, block_size=4, num_steps=1000, step_size=1e-1
    )
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(sparsemax(jnp.einsum("qd,kd->qk", query, key), axis=-1))
    ax[1].imshow(monarch_attn)
    plt.show()


if __name__ == "__main__":
    main()
