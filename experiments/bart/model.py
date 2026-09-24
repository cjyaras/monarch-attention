# Code based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py

import torch
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartDecoder,
    BartModel,
    BartLearnedPositionalEmbedding,
)

from experiments.common.attention import get_attn_module
from experiments.common.utils import get_device
from experiments.bart.config import CustomBartConfig


class CustomBartDecoder(BartDecoder):
    def __init__(self, config: CustomBartConfig):
        super().__init__(config)
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
        for layer_num, layer in enumerate(self.encoder.layers):
            layer.self_attn.attn_module = get_attn_module(config, layer_num)
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
        for m in self.modules():
            if hasattr(m, 'config'):
                m.config.max_position_embeddings = new_max_position_embeddings


def get_model(
    config: CustomBartConfig,
    model_checkpoint_path:str = "experiments/bart/finetuned/output/",
) -> CustomBartForConditionalGeneration:
    device = get_device()
    model = CustomBartForConditionalGeneration.from_pretrained(
        model_checkpoint_path, config=config
    )
    model = model.to(device)  # type: ignore
    model.eval()
    return model
