"""Forked from https://huggingface.co/mtreviso/sparsemax-roberta/blob/main/sparse_roberta.py."""

import math
from typing import Optional, Tuple

import torch
from entmax import (  # Ensure this is installed: pip install entmax
    entmax_bisect,
    sparsemax,
)
from torch import nn
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaModel
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaClassificationHead,
    RobertaForSequenceClassification,
    RobertaSelfAttention,
    RobertaSelfOutput,
)

try:
    from entmax_triton import entmax_triton  # type: ignore
except ImportError:

    def entmax_triton(x, alpha=1.5, n_iter=10, fast_math=False):
        # give a warning if entmax_triton is not found
        print("entmax_triton not found! Using entmax_bisect instead.")
        return entmax_bisect(x, alpha=alpha, n_iter=n_iter)


class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)

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
        mixed_query_layer = self.query(hidden_states)

        assert attention_mask is None
        assert head_mask is None
        assert encoder_hidden_states is None
        assert encoder_attention_mask is None
        assert past_key_value is None
        assert output_attentions is False

        assert self.position_embedding_type == "absolute"
        assert not self.training

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # type: ignore
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply sparse attention
        attention_probs = sparsemax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)  # type: ignore
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs  # type: ignore


class CustomRobertaAttention(RobertaAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = CustomRobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)


class CustomRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.roberta.encoder.layer:
            layer.attention = CustomRobertaAttention(config)


class CustomRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.encoder.layer:
            layer.attention = CustomRobertaAttention(config)


class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = CustomRobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()


def get_custom_model(
    model_name_or_path, initial_alpha=2.0, use_triton_entmax=False, from_scratch=False
):
    config = RobertaConfig.from_pretrained(model_name_or_path)
    config.initial_alpha = initial_alpha
    config.use_triton_entmax = use_triton_entmax
    # load from a pretrained checkpoint or start from scratch
    if from_scratch:
        print("Training from scratch...")
        print("Config:", config)
        model = CustomRobertaForMaskedLM._from_config(config)
    else:
        print("Loading from pretrained checkpoint...")
        print("Config:", config)
        model = CustomRobertaForMaskedLM.from_pretrained(
            model_name_or_path, config=config
        )
        # test if alpha and use_triton_entmax are set correctly
        assert model.roberta.encoder.layer[0].attention.self.alpha == initial_alpha
        assert (
            model.roberta.encoder.layer[0].attention.self.use_triton_entmax
            == use_triton_entmax
        )
    return model
