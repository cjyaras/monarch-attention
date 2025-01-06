"""Forked from https://huggingface.co/mtreviso/sparsemax-roberta/blob/main/sparse_roberta.py."""

# TODO: Allow attention_mask for efficient attention by modifying attention_mask in model

from math import sqrt
from typing import Literal, Optional, Tuple

import torch
from entmax import sparsemax
from torch import nn
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaSelfAttention,
)

AttentionType = Literal[
    "softmax", "sparsemax", "low-rank", "monarch", "block-diag-low-rank"
]

from sobalib.layers import BlockDiagLowRankMHA, LowRankMHA, MonarchMHA, PadType


class CustomRobertaConfig(RobertaConfig):
    def __init__(
        self,
        attention_type: AttentionType = "softmax",
        scale_attention_temperature: bool = False,
        efficient_attention_num_steps: Optional[int] = None,
        efficient_attention_step_size: Optional[float] = None,
        efficient_attention_rank: Optional[int] = None,
        efficient_attention_block_size: Optional[int] = None,
        efficient_attention_pad_type: PadType = "pre",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._attn_implementation = "sdpa"
        self.attention_type = attention_type
        self.scale_attention_temperature = scale_attention_temperature
        self.efficient_attention_num_steps = efficient_attention_num_steps
        self.efficient_attention_step_size = efficient_attention_step_size
        self.efficient_attention_rank = efficient_attention_rank
        self.efficient_attention_block_size = efficient_attention_block_size
        self.efficient_attention_pad_type = efficient_attention_pad_type


class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config: CustomRobertaConfig, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        assert self.position_embedding_type == "absolute"
        assert not self.is_decoder

        self.attention_type = config.attention_type

        if self.attention_type in ["low-rank", "monarch", "block-diag-low-rank"]:
            num_steps = config.efficient_attention_num_steps
            step_size = config.efficient_attention_step_size
            assert num_steps is not None and step_size is not None

            if self.attention_type == "low-rank":
                rank = config.efficient_attention_rank
                assert rank is not None
                self.efficient_attn = torch.compile(
                    LowRankMHA(
                        num_steps=num_steps,
                        step_size=step_size,
                        rank=rank,
                    ),
                    mode="reduce-overhead",
                )

            elif self.attention_type == "monarch":
                block_size = config.efficient_attention_block_size
                pad_type = config.efficient_attention_pad_type
                assert block_size is not None and pad_type is not None
                # self.efficient_attn = torch.compile(
                #     MonarchMHA(
                #         block_size=block_size,
                #         num_steps=num_steps,
                #         step_size=step_size,
                #         pad_type=pad_type,  # type: ignore
                #     ),
                #     # mode="reduce-overhead",
                # )
                self.efficient_attn = MonarchMHA(
                    block_size=block_size,
                    num_steps=num_steps,
                    step_size=step_size,
                    pad_type=pad_type,  # type: ignore
                )

            else:
                block_size = config.efficient_attention_block_size
                rank = config.efficient_attention_rank
                pad_type = config.efficient_attention_pad_type
                assert (
                    block_size is not None and pad_type is not None and rank is not None
                )
                self.efficient_attn = BlockDiagLowRankMHA(
                    block_size, rank, num_steps, step_size, pad_type  # type: ignore
                )

        if config.scale_attention_temperature:
            self.register_buffer(
                "attention_temperature",
                torch.full((self.num_attention_heads,), 1.0),
                persistent=True,
            )
        else:
            self.attention_temperature = None

        self.enable_flash_attention = config._attn_implementation == "sdpa"

    # do not try to load sparsity_per_head and n_tokens when loading from a checkpoint
    def load_state_dict(self, state_dict, **kwargs):
        if "sparsity_per_head" in state_dict:
            del state_dict["sparsity_per_head"]
        if "n_tokens" in state_dict:
            del state_dict["n_tokens"]
        super().load_state_dict(state_dict, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        assert attention_mask is None
        assert head_mask is None
        assert encoder_hidden_states is None
        assert encoder_attention_mask is None
        assert past_key_value is None
        assert output_attentions is False

        assert not self.training

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.attention_temperature is not None:
            query_layer = query_layer / self.attention_temperature[..., None, None]

        if self.attention_type == "softmax" and self.enable_flash_attention:
            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                head_mask,
                0.0,
                is_causal=False,
                scale=None,
            )
        elif self.attention_type in ["softmax", "sparsemax"]:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / sqrt(self.attention_head_size)
            attention_probs = (
                sparsemax(attention_scores, dim=-1)
                if self.attention_type == "sparsemax"
                else nn.functional.softmax(attention_scores, dim=-1)
            )
            assert isinstance(attention_probs, torch.Tensor)
            context_layer = torch.matmul(attention_probs, value_layer)
        elif self.attention_type in ["low-rank", "monarch", "block-diag-low-rank"]:
            context_layer = self.efficient_attn(query_layer, key_layer, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer,)


class CustomRobertaModel(RobertaModel):
    def __init__(self, config: CustomRobertaConfig, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        for layer in self.encoder.layer:
            layer.attention.self = CustomRobertaSelfAttention(config)
        self.post_init()


class CustomRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config: CustomRobertaConfig):
        super().__init__(config)
        self.roberta = CustomRobertaModel(config, add_pooling_layer=False)
        self.post_init()


class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config: CustomRobertaConfig):
        super().__init__(config)
        self.roberta = CustomRobertaModel(config)
        self.post_init()
