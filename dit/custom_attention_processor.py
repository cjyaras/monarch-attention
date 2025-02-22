from typing import Optional

import torch

from .attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import logging

from common.soba import *
from common.baselines import *

ATTENTION_TYPE_TO_MODULE = {
        "soba": SobaMonarch,
        "linformer": Linformer,
        "performer": Performer,
        "nystromformer": Nystromformer,
        "cosformer": Cosformer
    }

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class EfficientAttnConfig:
    def __init__(self, 
        efficient_attention_type: str,
        rank: int = 8,
        block_size: Optional[int] = 14, 
        num_steps: Optional[int] = 3,
        pad_type: PadType = PadType.pre,
        init_type: InitType = InitType.eye,
        share_kv: bool = False,
        estimator_type: str = "pos",
        ortho_features: bool = True,
        conv_kernel_size: Optional[int] = None,
    ):
        
        self.efficient_attention_type = efficient_attention_type

        # SobaMonarch
        self.num_steps = num_steps
        self.block_size = block_size
        self.pad_type = pad_type
        self.init_type = init_type

        # Linformer
        # self.rank used as projection dim
        self.rank = rank
        self.share_kv = share_kv

        # Compute sequence length
        # TODO: how to do this? 

        # Performer
        # self.rank used as num_samples
        self.estimator_type = estimator_type
        self.ortho_features = ortho_features

        # Nystromformer
        # self.rank used as number of landmarks
        self.conv_kernel_size = conv_kernel_size

        # Cosformer: none



def prepare_args(config: EfficientAttnConfig):
     match config.efficient_attention_type:
        case "soba":
            return (
                config.block_size,
                config.num_steps,
                config.pad_type,
                config.init_type,
            )

        # Commenting this out for now b/c unsure how to compute sequence length before initializing model
        # case "linformer":
        #     return (config.rank, config.seq_len, config.share_kv)

        case "performer":
            return (config.rank, config.estimator_type, config.ortho_features)

        case "nystromformer":
            return (config.rank, config.num_attention_heads, config.conv_kernel_size)

        case "cosformer":
            return ()

        case _:
            raise ValueError(f"Invalid attention type: {config.efficient_attention_type}")



class EfficientAttnProcessor(AttnProcessor2_0):
    def __init__(self, config: EfficientAttnConfig):
        super().__init__()
        self.config = config
        module = ATTENTION_TYPE_TO_MODULE[config.efficient_attention_type]
        self.attn_module = module(*prepare_args(config))

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
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            # SOBA and baselines require (batch_size, sequence_length) shape mask
            assert attention_mask.shape == (batch_size, sequence_length) 

            #attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            #attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

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

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states