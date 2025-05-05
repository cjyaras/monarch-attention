from typing import Dict, Optional, Tuple

import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaForQuestionAnswering,
    RobertaModel,
    RobertaSelfAttention,
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
from roberta.config import AttentionType, CustomRobertaConfig

set_verbosity(ERROR)

Tensor = torch.Tensor

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch_attention: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
    AttentionType.linear_attention: LinearAttention,
}


def prepare_args(attention_type: AttentionType, config: CustomRobertaConfig) -> Tuple:

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


class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(
        self, layer_num: int, config: CustomRobertaConfig, position_embedding_type=None
    ):
        super().__init__(config, position_embedding_type=position_embedding_type)
        assert self.position_embedding_type == "absolute"
        assert not self.is_decoder

        if isinstance(config.attention_type, Dict):
            attention_type = config.attention_type[layer_num]
            module = ATTENTION_TYPE_TO_MODULE[attention_type]
            self.attn_module = module(*prepare_args(attention_type, config))
        else:
            module = ATTENTION_TYPE_TO_MODULE[config.attention_type]
            self.attn_module = module(*prepare_args(config.attention_type, config))

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

        context_layer = self.attn_module(
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


class CustomRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    def __init__(self, config: CustomRobertaConfig):
        super().__init__(config)
        self.roberta = CustomRobertaModel(config, add_pooling_layer=False)
        self.post_init()


def get_model(config: CustomRobertaConfig) -> CustomRobertaForQuestionAnswering:
    device = get_device()
    model = CustomRobertaForQuestionAnswering.from_pretrained(
        "csarron/roberta-base-squad-v1", config=config
    )
    # model = CustomRobertaForQuestionAnswering.from_pretrained(
    #     "deepset/roberta-base-squad2", config=config
    # )
    model = model.to(device)  # type: ignore
    model.eval()
    return model
