"""Attention types shared by the experiments, and the hook that routes Hugging Face
attention layers through them.

Models whose config sets ``_attn_implementation = ATTN_IMPLEMENTATION`` call
``custom_attention_forward`` in every attention layer. Layers that have an
``attn_module`` attribute use it; all other layers (e.g. the BART decoder)
fall back to standard SDPA attention.
"""

from enum import StrEnum

import torch
import torch.nn as nn
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

from experiments.common.baselines import (
    Cosformer,
    LinearAttention,
    Linformer,
    Nystromformer,
    Performer,
    Softmax,
)
from ma.monarch_attention import MonarchAttention, PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    monarch_attention = "monarch-attention"
    monarch = "monarch-attention"  # alias, used by the DiT command-line scripts
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"
    linear_attention = "linear-attention"


ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch_attention: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
    AttentionType.linear_attention: LinearAttention,
}


class AttentionConfig:
    """Attention settings for the Hugging Face experiment configs (use as a mixin)."""

    def __init__(
        self,
        attention_type: AttentionType | dict[int, AttentionType] = AttentionType.softmax,
        enable_flash_attention: bool = False,
        num_steps: int | None = None,
        rank: int | None = None,
        block_size: int | None = None,
        pad_type: PadType = PadType.pre,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention_type = attention_type
        self.enable_flash_attention = enable_flash_attention  # Softmax
        self.num_steps = num_steps  # Monarch
        self.block_size = block_size  # Monarch
        self.pad_type = pad_type  # Monarch
        self.rank = rank  # Low-rank attention
        self._attn_implementation = ATTN_IMPLEMENTATION


def get_attn_module(config, layer_num: int | None = None) -> nn.Module:
    """Build the attention module for one layer from an experiment config."""
    attention_type = config.attention_type
    if isinstance(attention_type, dict):
        attention_type = attention_type[layer_num]

    match attention_type:
        case AttentionType.softmax:
            args = (config.enable_flash_attention,)
        case AttentionType.monarch_attention:
            impl = "triton" if torch.cuda.is_available() else "torch"
            args = (config.block_size, config.num_steps, config.pad_type, impl)
        case (
            AttentionType.linformer
            | AttentionType.performer
            | AttentionType.nystromformer
        ):
            args = (config.rank,)
        case AttentionType.cosformer | AttentionType.linear_attention:
            args = ()
        case _:
            raise ValueError(f"Invalid attention type: {attention_type}")

    return ATTENTION_TYPE_TO_MODULE[attention_type](*args)


def get_mixed_type(
    efficient_attn_layers,
    efficient_type: AttentionType,
    default_type: AttentionType = AttentionType.softmax,
    num_layers: int = 12,
) -> dict[int, AttentionType]:
    return {
        layer_num: efficient_type if layer_num in efficient_attn_layers else default_type
        for layer_num in range(num_layers)
    }


ATTN_IMPLEMENTATION = "custom"


def custom_attention_forward(module, query, key, value, attention_mask, **kwargs):
    attn_module = getattr(module, "attn_module", None)
    if attn_module is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )

    if attention_mask is not None:
        # (B, 1, Q, K) boolean mask from sdpa_mask -> (B, K) padding mask, 1 = keep
        attention_mask = attention_mask[:, 0, 0, :].to(query.dtype)

    attn_output = attn_module(query, key, value, attention_mask)
    return attn_output.transpose(1, 2).contiguous(), None


AttentionInterface.register(ATTN_IMPLEMENTATION, custom_attention_forward)
AttentionMaskInterface.register(ATTN_IMPLEMENTATION, sdpa_mask)
