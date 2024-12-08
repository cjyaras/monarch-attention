from typing import Callable, Optional

import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module
from flax.typing import Array, Dtype, PrecisionLike, PRNGKey

from configuration_vit import ModifiedViTConfig, ViTConfig
from projections import softmax, sparsemax


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
    """Computes dot-product attention weights given query and key.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
      query: queries for calculating attention with shape of ``[batch...,
        q_length, num_heads, qk_depth_per_head]``.
      key: keys for calculating attention with shape of ``[batch..., kv_length,
        num_heads, qk_depth_per_head]``.
      bias: bias for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks. Attention weights are masked out if their
        corresponding mask value is ``False``.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: infer from inputs and params)
      precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.
      module: the Module that will sow the attention weights into the
        'intermediates' collection. Remember to mark 'intermediates' as mutable
        via ``mutable=['intermediates']`` in order to have that collection
        returned. If ``module`` is None, the attention weights will not be sowed.
      force_fp32_for_softmax: bool, whether to force the softmax to be computed in
        fp32. This is useful for mixed-precision training where higher precision
        is desired for numerical stability.
      einsum_dot_general: the dot_general to use in einsum.

    Returns:
      Output of shape ``[batch..., num_heads, q_length, kv_length]``.
    """
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
