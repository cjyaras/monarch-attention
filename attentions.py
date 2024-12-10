"""Partially forked from https://github.com/google/flax/blob/main/flax/linen/attention.py"""

from functools import partial
from math import ceil, floor, sqrt
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax
from einshape import jax_einshape as einshape
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module
from flax.typing import Array, Dtype, PrecisionLike, PRNGKey

from configuration_vit import ModifiedViTConfig, ViTConfig
from projections import softmax, sparsemax
from solvers import low_rank_solve, monarch_solve
from utils import vmap_lift


def dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[Module] = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., Array] = jax.lax.dot_general,
    config: Optional[ViTConfig] = None,
):
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    assert isinstance(config, ModifiedViTConfig)

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

    if config.attention_temperature is not None:
        query = query / config.attention_temperature

    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum(
        "...qhd,...khd->...hqk",
        query,
        key,
        precision=precision,
        _dot_general=einsum_dot_general,
    )

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    projection_fn = softmax if config.attention_type == "softmax" else sparsemax

    # normalize the attention weights
    if force_fp32_for_softmax and dtype != jnp.float32:
        attn_weights = projection_fn(attn_weights.astype(jnp.float32), axis=-1)
    else:
        attn_weights = projection_fn(attn_weights, axis=-1).astype(dtype)

    # Save some intermediates
    if module:
        module.sow("intermediates", "a", attn_weights)
        module.sow("intermediates", "q", einshape("...qhd->...hqd", query))
        module.sow("intermediates", "k", einshape("...khd->...hkd", key))

    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights


def efficient_attention_weights(
    query: Array,
    key: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    module: Optional[Module] = None,
    config: Optional[ViTConfig] = None,
):
    assert isinstance(config, ModifiedViTConfig)

    # Make sure other arguments are appropriate (bidirectional, inference mode, no bias, etc.)
    assert bias is None
    assert mask is None
    assert broadcast_dropout is True
    assert dropout_rng is None
    assert dropout_rate == 0.0
    assert deterministic is True

    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

    if config.attention_temperature is not None:
        query = query / config.attention_temperature

    query = einshape("...qhd->...hqd", query)
    key = einshape("...khd->...hkd", key)

    if module:
        module.sow("intermediates", "q", query)
        module.sow("intermediates", "k", key)

    n = query.shape[-2]

    if config.attention_type == "low-rank":
        solver = partial(
            low_rank_solve,
            num_steps=5,
            tx=optax.adam(0.1),
            rank=ceil(sqrt(n)),
            return_history=False,
        )
    else:
        solver = partial(
            monarch_solve,
            num_steps=10,
            tx=optax.sgd(1e5),
            block_size=floor(sqrt(n)),
            padding_type="pre",
            return_history=False,
        )

    attn_weights = vmap_lift(solver, query.ndim - 2, in_axes=(0, 0), out_axes=0)(
        query, key
    )
    # attn_weights_e2e = vmap_lift(attn_weights.get_e2e.__func__, query.ndim - 2, 0, 0)(
    #     attn_weights
    # )
    return attn_weights
