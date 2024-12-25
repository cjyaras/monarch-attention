from math import ceil, sqrt
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from einshape import jax_einshape as einshape
from opt_einsum import contract

jax.config.update("jax_enable_x64", True)

Array = jax.Array

PadType = Literal["pre", "post"]


def normalize(x):
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def monarch_attention_v1(
    query: Array,
    key: Array,
    block_size: int,
    pad_type: PadType,
):
    seq_len = query.shape[-2]
    num_blocks = ceil(seq_len / block_size)
    seq_len_padded = block_size * num_blocks
    pad_amount = seq_len_padded - seq_len

    def monarch_multiply(left: Array, right: Array, inputs: Array):
        pad_width = (pad_amount, 0) if pad_type == "pre" else (0, pad_amount)
        x = jnp.pad(inputs, pad_width)
        x = einshape("(ki)->ki", x, i=block_size)
        x = jnp.einsum("kji,ki->kj", right, x)
        x = jnp.einsum("jlk,kj->lj", left, x)
        x = einshape("lj->(lj)", x)
        return x[pad_amount:] if pad_type == "pre" else x[: -pad_amount or None]

    def monarch_gram_trace(left: Array, right: Array):
        "Assumes that left and right are 2D."
        left = jnp.matmul(left.mT, left)
        right = jnp.matmul(right, right.mT)
        left = jax.vmap(jnp.diag)(left)
        right = jax.vmap(jnp.diag)(right)
        return jnp.sum(left * right.T)

    def normalize_and_square(x):
        x = normalize(x) ** 2
        return x

    def mask(x):
        if pad_type == "pre":
            x = x.at[0, :, :pad_amount].set(0.0)
        else:
            x = x.at[-1, :, block_size - pad_amount :].set(0.0)
        return x

    def objective(left_params, right_params):
        left = normalize_and_square(left_params)
        right = normalize_and_square(mask(right_params))
        modified_key = jax.vmap(monarch_multiply, in_axes=(None, None, 1), out_axes=1)(
            left, right, key
        )
        return 1 / 2 * monarch_gram_trace(left, right) - jnp.trace(
            jnp.matmul(query.T, modified_key)
        )

    left_params = jr.normal(jr.key(0), (block_size, num_blocks, num_blocks))
    right_params = jr.normal(jr.key(1), (num_blocks, block_size, block_size))

    left_params_grad, right_params_grad = jax.grad(objective, argnums=(0, 1))(
        left_params, right_params
    )

    return left_params_grad, right_params_grad


def monarch_attention_v2(
    query: Array,
    key: Array,
    block_size: int,
    pad_type: PadType,
):
    seq_len = query.shape[-2]
    num_blocks = ceil(seq_len / block_size)
    seq_len_padded = block_size * num_blocks
    pad_amount = seq_len_padded - seq_len

    pad_width = [(pad_amount, 0) if pad_type == "pre" else (0, pad_amount), (0, 0)]
    query = jnp.pad(query, pad_width)
    key = jnp.pad(key, pad_width)
    query = einshape("...(lj)a->...lja", query, j=block_size)
    key = einshape("...(ki)a->...kia", key, i=block_size)

    def mask(x):
        if pad_type == "pre":
            x = x.at[0, :, :pad_amount].set(0.0)
        else:
            x = x.at[-1, :, block_size - pad_amount :].set(0.0)
        return x

    def project(x, u):
        return contract("...i,...j,...j->...i", x, x, u)

    def inv_norm(x):
        return 1 / jnp.linalg.norm(x, axis=-1, keepdims=True)

    def grad(left_params: Array, right_params: Array) -> Tuple[Array, Array]:
        right_params = mask(right_params)
        right_sphere = normalize(right_params)
        left_sphere = normalize(left_params)
        left = left_sphere**2
        right = right_sphere**2
        d_left = contract(
            "...kj,...jlk->...jlk", jnp.sum(right**2, axis=2), left
        ) - contract("...kji,...lja,...kia->...jlk", right, query, key)
        d_left = d_left * 2 * left_sphere
        d_left = d_left - project(left_sphere, d_left)
        d_left = d_left * inv_norm(left_params)
        d_right = contract(
            "...jk,...kji->...kji", jnp.sum(left**2, axis=1), right
        ) - contract("...jlk,...lja,...kia->...kji", left, query, key)
        d_right = d_right * 2 * right_sphere
        d_right = d_right - project(right_sphere, d_right)
        d_right = d_right * inv_norm(right_params)
        return d_left, d_right

    left_params = jr.normal(jr.key(0), (block_size, num_blocks, num_blocks))
    right_params = jr.normal(jr.key(1), (num_blocks, block_size, block_size))

    left_params_grad, right_params_grad = grad(left_params, right_params)

    return left_params_grad, right_params_grad


def main():
    seq_len = 16
    model_dims = 8
    block_size = 4
    query = jr.normal(jr.key(0), (seq_len, model_dims))
    key = jr.normal(jr.key(1), (seq_len, model_dims))

    left_params_grad_v1, right_params_grad_v1 = monarch_attention_v1(
        query, key, block_size, "post"
    )
    left_params_grad_v2, right_params_grad_v2 = monarch_attention_v2(
        query, key, block_size, "post"
    )

    print(right_params_grad_v1[-1, -1, :])
    print(right_params_grad_v2[-1, -1, :])

    assert jnp.allclose(left_params_grad_v1, left_params_grad_v2)
    assert jnp.allclose(right_params_grad_v1, right_params_grad_v2)


if __name__ == "__main__":
    main()
