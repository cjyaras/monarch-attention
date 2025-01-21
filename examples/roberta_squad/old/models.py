from enum import StrEnum
from math import sqrt
from typing import List, Optional, Tuple

import torch
from common.utils import maybe_compile
from entmax import sparsemax
from torch import nn
from torch._prims_common import DeviceLikeType
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaForMaskedLM,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaSelfAttention,
)

from sobalib.layers import LowRankMHA, MonarchBlockDiagonalMHA, MonarchMHA, PadType

Tensor = torch.Tensor


class AttentionType(StrEnum):
    softmax = "softmax"
    sparsemax = "sparsemax"
    low_rank = "low-rank"
    monarch = "monarch"
    monarch_block_diagonal = "monarch-block-diagonal"
    hybrid = "hybrid"


class CustomRobertaConfig(RobertaConfig):
    def __init__(
        self,
        attention_type: AttentionType = AttentionType.softmax,
        enable_flash_attention: bool = False,
        scale_attention_temperature: bool = False,
        efficient_attention_num_steps: Optional[int] = None,
        efficient_attention_step_size: Optional[float] = None,
        efficient_attention_rank: Optional[int] = None,
        efficient_attention_block_size: Optional[int] = None,
        efficient_attention_pad_type: PadType = PadType.post,
        hybrid_attention_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Set _attn_implementation to eager to override attention_mask logic in RobertaModel
        self._attn_implementation = "eager"
        self.attention_type = attention_type
        self.scale_attention_temperature = scale_attention_temperature
        self.enable_flash_attention = enable_flash_attention
        self.efficient_attention_num_steps = efficient_attention_num_steps
        self.efficient_attention_step_size = efficient_attention_step_size
        self.efficient_attention_rank = efficient_attention_rank
        self.efficient_attention_block_size = efficient_attention_block_size
        self.efficient_attention_pad_type = efficient_attention_pad_type
        self.hybrid_attention_layers = hybrid_attention_layers


def get_config() -> CustomRobertaConfig:
    config = CustomRobertaConfig.from_pretrained("deepset/roberta-base-squad2")
    assert isinstance(config, CustomRobertaConfig)
    return config


class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(
        self, layer_num: int, config: CustomRobertaConfig, position_embedding_type=None
    ):
        super().__init__(config, position_embedding_type=position_embedding_type)
        assert self.position_embedding_type == "absolute"
        assert not self.is_decoder

        self.attention_type = config.attention_type

        if self.attention_type in [
            AttentionType.low_rank,
            AttentionType.monarch,
            AttentionType.monarch_block_diagonal,
        ]:
            num_steps = config.efficient_attention_num_steps
            step_size = config.efficient_attention_step_size
            assert num_steps is not None and step_size is not None

            if self.attention_type == AttentionType.low_rank:
                rank = config.efficient_attention_rank
                assert rank is not None
                self.efficient_attn = LowRankMHA(
                    num_steps=num_steps,
                    step_size=step_size,
                    rank=rank,
                )

            elif self.attention_type == AttentionType.monarch:
                block_size = config.efficient_attention_block_size
                pad_type = config.efficient_attention_pad_type
                assert block_size is not None and pad_type is not None
                self.efficient_attn = MonarchMHA(
                    block_size=block_size,
                    num_steps=num_steps,
                    step_size=step_size,
                    pad_type=pad_type,
                )

            elif self.attention_type == AttentionType.monarch_block_diagonal:
                block_size = config.efficient_attention_block_size
                pad_type = config.efficient_attention_pad_type
                assert block_size is not None and pad_type is not None
                self.efficient_attn = MonarchBlockDiagonalMHA(
                    block_size=block_size,
                    num_steps=num_steps,
                    step_size=step_size,
                    pad_type=pad_type,
                )

            else:
                raise ValueError(f"Invalid attention type: {self.attention_type}")

        if self.attention_type == AttentionType.hybrid:
            num_steps = config.efficient_attention_num_steps
            step_size = config.efficient_attention_step_size
            assert num_steps is not None and step_size is not None
            block_size = config.efficient_attention_block_size
            pad_type = config.efficient_attention_pad_type
            hybrid_layers = config.hybrid_attention_layers
            assert (
                block_size is not None
                and pad_type is not None
                and hybrid_layers is not None
            )

            if layer_num in hybrid_layers:
                self.efficient_attn = MonarchMHA(
                    block_size=block_size,
                    num_steps=num_steps,
                    step_size=step_size,
                    pad_type=pad_type,
                )
            else:
                self.efficient_attn = None

        if self.attention_type in [AttentionType.softmax, AttentionType.sparsemax]:
            self.efficient_attn = None

        if config.scale_attention_temperature:
            self.register_buffer(
                "attention_temperature",
                torch.full((self.num_attention_heads,), 1.0),
                persistent=True,
            )
        else:
            self.attention_temperature = None

        if self.efficient_attn is not None:
            maybe_compile(self.efficient_attn)

        self.enable_flash_attention = config.enable_flash_attention

    # do not try to load sparsity_per_head and n_tokens when loading from a checkpoint
    def load_state_dict(self, state_dict, **kwargs):
        if "sparsity_per_head" in state_dict:
            del state_dict["sparsity_per_head"]
        if "n_tokens" in state_dict:
            del state_dict["n_tokens"]
        super().load_state_dict(state_dict, **kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor]:

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

        if self.attention_type == AttentionType.softmax and self.enable_flash_attention:
            extended_attention_mask = (
                _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, dtype=query_layer.dtype
                )
                if attention_mask is not None
                else None
            )
            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                extended_attention_mask,
                0.0,
                is_causal=False,
                scale=None,
            )

        elif self.attention_type in [
            AttentionType.softmax,
            AttentionType.sparsemax,
        ] or (
            self.attention_type == AttentionType.hybrid and self.efficient_attn is None
        ):
            extended_attention_mask = (
                (
                    (1.0 - attention_mask[:, None, None, :])
                    * torch.finfo(query_layer.dtype).min
                )
                if attention_mask is not None
                else None
            )
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / sqrt(self.attention_head_size)
            if extended_attention_mask is not None:
                attention_scores = attention_scores + extended_attention_mask
            attention_probs = (
                sparsemax(attention_scores, dim=-1)
                if (
                    self.attention_type == AttentionType.sparsemax
                    or self.attention_type == AttentionType.hybrid
                )
                else nn.functional.softmax(attention_scores, dim=-1)
            )
            assert isinstance(attention_probs, Tensor)
            context_layer = torch.matmul(attention_probs, value_layer)

        else:
            assert self.efficient_attn is not None
            context_layer = self.efficient_attn(
                query_layer, key_layer, value_layer, attention_mask
            )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer,)


class CustomRobertaAttention(RobertaAttention):
    def __init__(
        self, layer_num: int, config: CustomRobertaConfig, position_embedding_type=None
    ):
        super().__init__(config, position_embedding_type)
        self.self = CustomRobertaSelfAttention(
            layer_num, config, position_embedding_type=position_embedding_type
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output, None)
        return outputs  # type: ignore


class CustomRobertaModel(RobertaModel):
    def __init__(self, config: CustomRobertaConfig, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        for layer_num, layer in enumerate(self.encoder.layer):
            layer.attention = CustomRobertaAttention(layer_num, config)
        self.post_init()

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int]
    ) -> Tensor:

        assert not self.config.is_decoder
        return attention_mask  # Delegate the exact computation to the attention layer

        # dtype = self.dtype
        # assert len(input_shape) == 2
        # batch_size, seq_length = input_shape

        # if (
        #     self.config.attention_type == AttentionType.softmax
        #     and self.config.enable_flash_attention
        # ):
        #     extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
        #         attention_mask, dtype, tgt_len=seq_length
        #     )
        #     return extended_attention_mask

        # if self.config.attention_type in [
        #     AttentionType.softmax,
        #     AttentionType.sparsemax,
        # ]:
        #     extended_attention_mask = attention_mask[:, None, None, :]
        #     extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
        #         dtype
        #     ).min
        #     return extended_attention_mask

        # # For efficient attention, we just want to return the original attention mask
        # return attention_mask


class CustomRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config: CustomRobertaConfig):
        super().__init__(config)
        self.roberta = CustomRobertaModel(config, add_pooling_layer=False)
        self.post_init()


class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config: CustomRobertaConfig):
        super().__init__(config)
        self.roberta = CustomRobertaModel(config, add_pooling_layer=False)
        self.post_init()


class CustomRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    def __init__(self, config: CustomRobertaConfig):
        super().__init__(config)
        self.roberta = CustomRobertaModel(config, add_pooling_layer=False)
        self.post_init()


def get_model(
    config: CustomRobertaConfig, device: DeviceLikeType
) -> CustomRobertaForQuestionAnswering:
    model = CustomRobertaForQuestionAnswering.from_pretrained(
        "deepset/roberta-base-squad2", config=config
    )
    if config.scale_attention_temperature:
        model.load_state_dict(
            torch.load("roberta/sparsemax_temperature.pt", weights_only=True),
            strict=False,
        )
    model = model.to(device)  # type: ignore
    model.eval()
    return model
