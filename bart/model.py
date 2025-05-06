# Code based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartDecoder,
    BartModel,
    BartEncoderLayer,
    BartAttention,
    BartLearnedPositionalEmbedding,
)
from transformers.utils.logging import ERROR, set_verbosity  # type: ignore

from common.baselines import (
    Cosformer,
    LinearAttention,
    Linformer,
    Nystromformer,
    Performer,
    Softmax,
)
from common.utils import get_device
from ma.monarch_attention import MonarchAttention
from bart.config import AttentionType, CustomBartConfig


ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch_attention: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
    AttentionType.linear_attention: LinearAttention,
}


def prepare_args(attention_type: AttentionType, config: CustomBartConfig) -> Tuple:

    match attention_type:

        case AttentionType.softmax:
            return (config.enable_flash_attention,)

        case AttentionType.monarch_attention:
            return (config.block_size, config.num_steps, config.pad_type)

        case (
            AttentionType.linformer
            | AttentionType.performer
            | AttentionType.nystromformer
        ):
            return (config.rank,)

        case AttentionType.cosformer | AttentionType.linear_attention:
            return ()

        case _:
            raise ValueError(f"Invalid attention type: {attention_type}")




class CustomBartAttention(BartAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layer_num: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[CustomBartConfig] = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            bias=bias,
            is_causal=is_causal,
            config=config,
        )
        
        if isinstance(config.attention_type, Dict):
            attention_type = config.attention_type[layer_num]
            module = ATTENTION_TYPE_TO_MODULE[attention_type]
            self.attn_module = module(*prepare_args(attention_type, config))
        else:
            module = ATTENTION_TYPE_TO_MODULE[config.attention_type]
            self.attn_module = module(*prepare_args(config.attention_type, config))


    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        assert layer_head_mask is None
        assert key_value_states is None
        assert past_key_value is None
        assert output_attentions is False


        if self.config.attention_type != AttentionType.softmax:
            assert not self.training


        # [CL, 05/05/2025]
        # For some reason torchtnt.utils.flops.FlopTensorDispatchMode
        # makes the shape of hidden_states (N, C) not (B,N,C)...
        # It can be circumvented when B=1 and unsqueezing.
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) 

        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz, self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        if attention_mask is not None and len(attention_mask.shape)==4:
            attention_mask = attention_mask[:,0,0,:]
            attention_mask = torch.where(attention_mask==0, 1.0, 0.0)


        attn_output = self.attn_module(
            query_states, key_states, value_states, attention_mask
        )
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        attn_weights_reshaped = None
        past_key_value = None
        return attn_output, attn_weights_reshaped, past_key_value





class CustomBartEncoderLayer(BartEncoderLayer):
    def __init__(self, config: CustomBartConfig, layer_num: int):
        super().__init__(config)
        self.self_attn = CustomBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            layer_num=layer_num,
            dropout=config.attention_dropout,
            config=config,
        )

    def forward(self, hidden_states, *args, **kwargs):
        res = super().forward(hidden_states, *args, **kwargs)
        return res



class CustomBartDecoder(BartDecoder):
    def __init__(self, config: CustomBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_decoder_position_embeddings,
            config.d_model,
        )
        self.post_init()


class CustomBartModel(BartModel):
    def __init__(
        self,
        config: CustomBartConfig,
    ):
        super().__init__(config)
        self.encoder.layers = nn.ModuleList([CustomBartEncoderLayer(config, i) for i in range(config.encoder_layers)])
        self.decoder = CustomBartDecoder(config, self.shared)
        self.post_init()



class CustomBartForConditionalGeneration(BartForConditionalGeneration):
    config_class=CustomBartConfig
    def __init__(self, config: CustomBartConfig):
        super().__init__(config)
        if config.use_original_bart:
            self.model = BartModel(config)
            self.model.decoder.embed_positions = BartLearnedPositionalEmbedding(
                config.max_decoder_position_embeddings,
                config.d_model,
            )
        else:
            self.model = CustomBartModel(config)
        self.post_init()


    @torch.no_grad()
    def resize_position_embeddings(self, new_max_position_embeddings: int):
        old_embeddings = self.model.encoder.embed_positions  # shared with decoder
        old_num_positions, embed_dim = old_embeddings.weight.shape
        
        # Create new position embeddings
        new_embeddings = BartLearnedPositionalEmbedding(new_max_position_embeddings, embed_dim)
        
        interpolated = torch.nn.functional.interpolate(
            old_embeddings.weight[2:,:].T.unsqueeze(0),
            new_max_position_embeddings,
            mode='linear',
            align_corners=True,
        ).squeeze(0).T

        new_embeddings.weight.copy_(
            torch.cat(
                [
                    old_embeddings.weight[:2,:],
                    interpolated
                ]
            )
        )
        
        # Replace embeddings in both encoder and decoder
        self.model.encoder.embed_positions = new_embeddings
        #self.model.decoder.embed_positions = new_embeddings
        
        # Update config
        self.config.max_position_embeddings = new_max_position_embeddings
        for mn, m in self.named_modules():
            if hasattr(m, 'config'):
                m.config.max_position_embeddings = new_max_position_embeddings



def get_model(config: CustomBartConfig) -> CustomBartForConditionalGeneration:
    device = get_device()
    model = CustomBartForConditionalGeneration.from_pretrained(
        "./bart/finetuned/output/", config=config
    )
    model = model.to(device)  # type: ignore
    model.eval()
    return model
