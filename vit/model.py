from typing import Dict, Optional, Tuple, Union

import torch
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification,
    ViTModel,
    ViTSelfAttention,
)
from transformers.utils.logging import ERROR, set_verbosity  # type: ignore

from common.baselines import Cosformer, Linformer, Nystromformer, Performer, Softmax
from common.utils import get_device, maybe_compile
from ma.monarch_attention import MonarchAttention
from vit.config import AttentionType, CustomViTConfig

set_verbosity(ERROR)

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
}


def prepare_args(attention_type: AttentionType, config: CustomViTConfig) -> Tuple:

    match attention_type:

        case AttentionType.softmax:
            return (config.enable_flash_attention,)

        case AttentionType.monarch:
            return (
                config.block_size,
                config.num_steps,
                config.pad_type,
            )

        case AttentionType.linformer:
            return (config.rank, config.seq_len, config.share_kv)

        case AttentionType.performer:
            return (config.rank, config.estimator_type, config.ortho_features)

        case AttentionType.nystromformer:
            return (config.rank, config.num_attention_heads, config.conv_kernel_size)

        case AttentionType.cosformer:
            return ()

        case _:
            raise ValueError(f"Invalid attention type: {attention_type}")


class CustomViTSelfAttention(ViTSelfAttention):

    def __init__(self, layer_num: int, config: CustomViTConfig):
        super().__init__(config)

        if isinstance(config.attention_type, Dict):
            attention_type = config.attention_type[layer_num]
            module = ATTENTION_TYPE_TO_MODULE[attention_type]
            self.attn_module = module(*prepare_args(attention_type, config))
        else:
            module = ATTENTION_TYPE_TO_MODULE[config.attention_type]
            self.attn_module = module(*prepare_args(config.attention_type, config))

        maybe_compile(self.attn_module)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        assert head_mask is None
        assert not output_attentions

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = self.attn_module(query_layer, key_layer, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None  # type: ignore


class CustomViTModel(ViTModel):

    def __init__(
        self,
        config: CustomViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__(config, add_pooling_layer, use_mask_token)
        for layer_num, layer in enumerate(self.encoder.layer):
            layer.attention.attention = CustomViTSelfAttention(layer_num, config)  # type: ignore
        self.post_init()


class CustomViTForImageClassification(ViTForImageClassification):

    def __init__(self, config: CustomViTConfig):
        super().__init__(config)
        self.vit = CustomViTModel(config, add_pooling_layer=False)
        self.post_init()


def get_model(config: CustomViTConfig) -> CustomViTForImageClassification:
    device = get_device()
    model = CustomViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
    )
    model = model.to(device)  # type: ignore
    model.eval()
    return model
