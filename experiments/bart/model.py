# Code based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartDecoder,
    BartModel,
    BartLearnedPositionalEmbedding,
)
from transformers.utils.logging import ERROR, set_verbosity  # type: ignore

from experiments.common.baselines import (
    Cosformer,
    LinearAttention,
    Linformer,
    Nystromformer,
    Performer,
    Softmax,
)
from experiments.common.utils import get_device
from ma.monarch_attention import MonarchAttention
from experiments.bart.config import AttentionType, CustomBartConfig


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




def get_attn_module(layer_num: int, config: CustomBartConfig) -> nn.Module:
    if isinstance(config.attention_type, Dict):
        attention_type = config.attention_type[layer_num]
    else:
        attention_type = config.attention_type
    module = ATTENTION_TYPE_TO_MODULE[attention_type]
    return module(*prepare_args(attention_type, config))


class CustomBartDecoder(BartDecoder):
    def __init__(self, config: CustomBartConfig):
        super().__init__(config)
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_decoder_position_embeddings,
            config.d_model,
        )
        self.post_init()

    def forward(self, *args, encoder_hidden_states=None, **kwargs):
        if encoder_hidden_states is not None and encoder_hidden_states.ndim == 2:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
        return super().forward(*args, encoder_hidden_states=encoder_hidden_states, **kwargs)


class CustomBartModel(BartModel):
    def __init__(
        self,
        config: CustomBartConfig,
    ):
        super().__init__(config)
        for layer_num, layer in enumerate(self.encoder.layers):
            layer.self_attn.attn_module = get_attn_module(layer_num, config)
        self.decoder = CustomBartDecoder(config)
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



def get_model(
    config: CustomBartConfig,
    model_checkpoint_path:str = "./bart/finetuned/output/",
) -> CustomBartForConditionalGeneration:
    device = get_device()
    model = CustomBartForConditionalGeneration.from_pretrained(
        model_checkpoint_path, config=config
    )
    model = model.to(device)  # type: ignore
    model.eval()
    return model
