from typing import Optional, Tuple

import torch
from common.baselines import Softmax, Sparsemax
from common.utils import maybe_compile
from config import AttentionType, CustomRobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaForMaskedLM,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaSelfAttention,
)

from sobalib.layers import SobaMonarch

Tensor = torch.Tensor

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.sparsemax: Sparsemax,
    AttentionType.soba_monarch: SobaMonarch,
}


def prepare_args(config: CustomRobertaConfig, layer_num: int) -> Tuple:

    if config.attention_type == AttentionType.softmax:
        return (config.enable_flash_attention,)

    elif config.attention_type == AttentionType.sparsemax:
        return ()

    elif config.attention_type == AttentionType.soba_monarch:
        return (config.block_size, config.num_steps, config.step_size, config.pad_type)

    elif config.attention_type == AttentionType.hybrid:
        assert config.hybrid_attention_layers is not None
        return (
            (config.block_size, config.num_steps, config.step_size, config.pad_type)
            if layer_num in config.hybrid_attention_layers
            else ()
        )

    else:
        raise ValueError(f"Invalid attention type: {config.attention_type}")


class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(
        self, layer_num: int, config: CustomRobertaConfig, position_embedding_type=None
    ):
        super().__init__(config, position_embedding_type=position_embedding_type)
        assert self.position_embedding_type == "absolute"
        assert not self.is_decoder

        if config.attention_type == AttentionType.hybrid:
            assert config.hybrid_attention_layers is not None
            module = ATTENTION_TYPE_TO_MODULE[
                (
                    AttentionType.soba_monarch
                    if layer_num in config.hybrid_attention_layers
                    else AttentionType.sparsemax
                )
            ]
        else:
            module = ATTENTION_TYPE_TO_MODULE[config.attention_type]

        self.attn_module = module(*prepare_args(config, layer_num))
        maybe_compile(self.attn_module)

        if config.scale_attention_temperature:
            self.register_buffer(
                "attention_temperature",
                torch.full((self.num_attention_heads,), 1.0),
                persistent=True,
            )
        else:
            self.attention_temperature = None

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


def get_model(config: CustomRobertaConfig) -> CustomRobertaForQuestionAnswering:
    model = CustomRobertaForQuestionAnswering.from_pretrained(
        "deepset/roberta-base-squad2", config=config
    )
    if config.scale_attention_temperature:
        model.load_state_dict(
            torch.load("roberta_squad/sparsemax_temperature.pt", weights_only=True),
            strict=False,
        )
    model.eval()
    return model
