from typing import Optional, Tuple

import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaForMaskedLM,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaSelfAttention,
)
from transformers.utils.logging import ERROR, set_verbosity

from common.baselines import Softmax, Sparsemax
from common.soba import SobaMonarch
from common.utils import get_device, maybe_compile
from roberta.config import AttentionType, CustomRobertaConfig

set_verbosity(ERROR)

Tensor = torch.Tensor

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.sparsemax: Sparsemax,
    AttentionType.soba_monarch: SobaMonarch,
}


def prepare_args(config: CustomRobertaConfig) -> Tuple:

    match config.attention_type:

        case AttentionType.softmax:
            return (config.enable_flash_attention,)

        case AttentionType.sparsemax:
            return (config.num_attention_heads,)

        case AttentionType.soba_monarch:
            return (
                config.block_size,
                config.num_steps,
                config.num_attention_heads,
                config.pad_type,
            )

        case _:
            raise ValueError(f"Invalid attention type: {config.attention_type}")


class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(
        self, layer_num: int, config: CustomRobertaConfig, position_embedding_type=None
    ):
        super().__init__(config, position_embedding_type=position_embedding_type)
        assert self.position_embedding_type == "absolute"
        assert not self.is_decoder

        module = ATTENTION_TYPE_TO_MODULE[config.attention_type]

        self.attn_module = module(*prepare_args(config))
        maybe_compile(self.attn_module)

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
    device = get_device()
    model = CustomRobertaForQuestionAnswering.from_pretrained(
        "csarron/roberta-base-squad-v1",
        config=config,  # deepset/roberta-base-squad2", config=config
    )
    model = model.to(device)  # type: ignore
    model.eval()

    if config.attn_module_save_path is not None:
        model.load_state_dict(
            torch.load(config.attn_module_save_path, weights_only=True), strict=False
        )

    return model
