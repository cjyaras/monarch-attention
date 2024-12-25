from math import ceil
from typing import Literal, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from projections import sparsemax

Array = jax.Array

from einshape import jax_einshape as einshape


def l2_normalize(x: Array):
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def params_to_simplex(x: Array):
    return l2_normalize(x) ** 2


class Parameterization(eqx.Module):
    n: int = eqx.field(static=True)

    def __init__(self, n: int):
        self.n = n

    def get_e2e(self) -> Array:
        return jax.vmap(self.multiply.__func__, in_axes=(None, 1), out_axes=1)(
            self, jnp.eye(self.n)
        )

    def loss_fn(self, query: Array, key: Array) -> Array:
        raise NotImplementedError

    def multiply(self, x: Array) -> Array:
        raise NotImplementedError

    def gram_trace(self) -> Array:
        raise NotImplementedError


class LRParameterization(Parameterization):
    L_params: Array
    R_params: Array

    def __init__(self, n: int):
        super().__init__(n)

    def init_fn(self, key: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

    def loss_fn(self, query: Array, key: Array) -> Array:
        Kbar = jax.vmap(self.multiply.__func__, in_axes=(None, 1), out_axes=1)(
            self, key
        )
        QTQ = query.T @ query
        KTK = key.T @ key
        return (
            1 / 2 * self.gram_trace()
            - jnp.trace(query.T @ Kbar)
            + 1 / 2 * jnp.trace(QTQ @ KTK)
        ) / self.n**2


class LowRank(LRParameterization):
    rank: int = eqx.field(static=True)

    def __init__(self, n: int, rank: int):
        super().__init__(n)
        self.rank = rank
        self.L_params, self.R_params = self.init_fn(jr.key(0))

    def init_fn(self, key: Array) -> Tuple[Array, Array]:
        key1, key2 = jr.split(key)
        scale = 1e0
        L_params = jnp.sqrt(
            jr.dirichlet(key1, alpha=scale * jnp.ones(self.rank), shape=(self.n,))
        )
        R_params = jnp.sqrt(
            jr.dirichlet(key2, alpha=scale * jnp.ones(self.n), shape=(self.rank,))
        )
        return L_params, R_params

    def get_L(self) -> Array:
        return params_to_simplex(self.L_params)

    def get_R(self) -> Array:
        return params_to_simplex(self.R_params)

    def multiply(self, x: Array) -> Array:
        L, R = self.get_L(), self.get_R()
        result = L @ (R @ x)
        return result

    def gram_trace(self) -> Array:
        L, R = self.get_L(), self.get_R()
        return jnp.trace((L.T @ L) @ (R @ R.T))


PaddingType = Literal["pre", "post"]


class Monarch(LRParameterization):
    b: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    n_padded: int = eqx.field(static=True)
    padding_type: PaddingType = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        block_size: int,
        padding_type: PaddingType,
    ):
        super().__init__(n)
        self.b = block_size
        self.m = ceil(n / block_size)
        self.n_padded = block_size * self.m
        self.padding_type = padding_type
        self.L_params, self.R_params = self.init_fn(jr.key(0))

    def init_fn(self, key: Array) -> Tuple[Array, Array]:
        key1, key2 = jr.split(key)
        scale = 1e0
        L_params = jnp.sqrt(
            jr.dirichlet(key1, alpha=scale * jnp.ones(self.m), shape=(self.b, self.m))
        )
        R_params = jnp.sqrt(
            jr.dirichlet(key2, alpha=scale * jnp.ones(self.b), shape=(self.m, self.b))
        )
        return L_params, R_params

    def get_L(self) -> Array:
        L_params = self.L_params
        return params_to_simplex(L_params)

    def get_R(self) -> Array:
        R_params = self.R_params
        pad_amount = self.n_padded - self.n

        if self.padding_type == "pre":
            R_params_zeroed = R_params.at[0, :, :pad_amount].set(0.0)
        else:
            R_params_zeroed = R_params.at[-1, :, self.b - pad_amount :].set(0.0)

        return params_to_simplex(R_params_zeroed)

    def multiply(self, x: Array) -> Array:
        pad_amount = self.n_padded - self.n
        if self.padding_type == "pre":
            x_padded = jnp.pad(x, (pad_amount, 0))
        else:
            x_padded = jnp.pad(x, (0, pad_amount))

        L, R = self.get_L(), self.get_R()

        X = einshape("(ki)->ki", x_padded, i=self.b)  # X ∈ (m, b)
        Y = jnp.einsum("kji,ki->kj", R, X)  # Y ∈ (m, b)
        Z = jnp.einsum("jlk,kj->lj", L, Y)  # Z ∈ (m, b)
        result = einshape("lj->(lj)", Z)

        if self.padding_type == "pre":
            return result[pad_amount:]
        else:
            return result[: self.n]

    def gram_trace(self) -> Array:
        L, R = self.get_L(), self.get_R()
        Lhat = jnp.matmul(L.mT, L)
        Rhat = jnp.matmul(R, R.mT)
        Lhat_diag = jax.vmap(jnp.diag)(Lhat)
        Rhat_diag = jax.vmap(jnp.diag)(Rhat)
        return jnp.sum(Lhat_diag * Rhat_diag.T)


def solve(
    p_init: LRParameterization,
    Q: Array,
    K: Array,
    num_steps: int,
    tx: optax.GradientTransformation,
    return_history: bool,
) -> LRParameterization:
    n = Q.shape[0]

    def f(carrys, _):
        p, opt_state = carrys
        _, grads = jax.value_and_grad(p.loss_fn.__func__)(p, Q, K)
        updates, opt_state = tx.update(grads, opt_state, params=p)
        new_p = optax.apply_updates(p, updates)
        carrys = new_p, opt_state
        return carrys, p

    p = p_init
    opt_state = tx.init(p)  # type: ignore

    carrys, p_history = jax.lax.scan(f, (p, opt_state), jnp.arange(num_steps))

    if return_history:
        return p_history
    else:
        p, _ = carrys
        assert isinstance(p, LRParameterization)
        return p


def low_rank_solve(
    query: Array,
    key: Array,
    num_steps: int,
    tx: optax.GradientTransformation,
    rank: int,
    return_history: bool,
) -> Parameterization:
    p_init = LowRank(query.shape[0], rank)
    return solve(p_init, query, key, num_steps, tx, return_history)


def monarch_solve(
    query: Array,
    key: Array,
    num_steps: int,
    tx: optax.GradientTransformation,
    block_size: int,
    padding_type: PaddingType,
    return_history: bool,
) -> Parameterization:
    p_init = Monarch(query.shape[0], block_size, padding_type)
    return solve(p_init, query, key, num_steps, tx, return_history)


def main():
    b = 1
    n = 16
    d = 4
    query = jr.normal(jr.key(0), (b, n, d)) / jnp.sqrt(d)
    key = jr.normal(jr.key(1), (b, n, d)) / jnp.sqrt(d)

    in_axes = (0, 0) + 5 * (None,)
    # in_axes = (0, 0) + 4 * (None,)
    # attn_weight = jax.vmap(
    #     low_rank_solve,
    #     in_axes=in_axes,
    # )(query, key, 100, optax.adam(0.1), 4, False)

    # attn_weight = low_rank_solve(query, key, 100, optax.adam(0.1), 4, False)
    attn_weight = jax.vmap(
        monarch_solve,
        in_axes=in_axes,
    )(query, key, 100, optax.adam(0.1), 4, "pre", False)
    attn_weight_e2e = jax.vmap(attn_weight.get_e2e.__func__)(attn_weight)
    # attn_weight_e2e = attn_weight.get_e2e()

    # print(attn_weight.multiply(jnp.eye(n)[:, 0]))
    # print(attn_weight.multiply.__func__(attn_weight, jnp.eye(n)[:, 0]))

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(sparsemax(jnp.einsum("bqd,bkd->bqk", query, key)[0], axis=-1))
    # ax[1].imshow(attn_weight_e2e[0])
    # plt.show()
    print(
        jnp.einsum("...qk,...kd->...qd", attn_weight_e2e, query)
        - jax.vmap(
            jax.vmap(attn_weight.multiply.__func__, in_axes=(None, 1), out_axes=1),
            in_axes=(0, 0),
        )(attn_weight, query)
    )
    # print(
    #     attn_weight_e2e @ query
    #     == jax.vmap(attn_weight.multiply.__func__, in_axes=(None, 1), out_axes=1)(
    #         attn_weight, query
    #     )
    # )


if __name__ == "__main__":
    main()
