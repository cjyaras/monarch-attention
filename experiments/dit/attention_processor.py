from typing import Dict, Optional

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from common.baselines import Cosformer, Linformer, Nystromformer, Performer, Softmax, LinearAttention
from ma.monarch_attention import MonarchAttention
from experiments.dit.config import AttentionType, EfficientAttnConfig

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
    AttentionType.linear_attention: LinearAttention,
}


def prepare_args(config: EfficientAttnConfig, layer_num: Optional[int] = None):
    if isinstance(config.attention_type, Dict):
        assert layer_num is not None
        attn_type = config.attention_type[layer_num]
    else:
        attn_type = config.attention_type

    match attn_type:
        case AttentionType.monarch:
            return (
                config.block_size,
                config.num_steps,
                config.pad_type,
            )

        case AttentionType.softmax:
            return (config.enable_flash_attention,)
        case AttentionType.linformer:
            return (config.rank,)

        case AttentionType.performer:
            return (config.rank,)

        case AttentionType.nystromformer:
            return (config.rank, config.num_attention_heads,)

        case AttentionType.cosformer:
            return ()
        
        case AttentionType.linear_attention:
            return ()

        case _:
            raise ValueError(
                f"Invalid attention type: {config.attention_type}"
            )


class EfficientAttnProcessor(AttnProcessor2_0):
    def __init__(self, config: EfficientAttnConfig, layer_num: Optional[int] = None):
        super().__init__()
        if isinstance(config.attention_type, Dict):
            attention_type = config.attention_type[layer_num]
            module = ATTENTION_TYPE_TO_MODULE[attention_type]
        else:
            module = ATTENTION_TYPE_TO_MODULE[config.attention_type]

        self.attn_module = module(*prepare_args(config, layer_num))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """
        Overwrite __call__ method to call self.attn_module() instead of F.scaled_dot_product_attention()
        """

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            # SOBA and baselines require (batch_size, sequence_length) shape mask
            assert attention_mask.shape == (batch_size, sequence_length)

            # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            # attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = self.attn_module(query, key, value, attention_mask)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
