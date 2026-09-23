from typing import Dict, Tuple

import torch
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification,
    ViTModel,
)
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
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
from experiments.vit.config import AttentionType, CustomViTConfig

set_verbosity(ERROR)

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch_attention: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
    AttentionType.linear_attention: LinearAttention,
}


def prepare_args(attention_type: AttentionType, config: CustomViTConfig) -> Tuple:

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


def get_attn_module(layer_num: int, config: CustomViTConfig) -> torch.nn.Module:
    if isinstance(config.attention_type, Dict):
        attention_type = config.attention_type[layer_num]
    else:
        attention_type = config.attention_type
    module = ATTENTION_TYPE_TO_MODULE[attention_type]
    return module(*prepare_args(attention_type, config))


class CustomViTModel(ViTModel):

    def __init__(
        self,
        config: CustomViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__(config, add_pooling_layer, use_mask_token)
        for layer_num, layer in enumerate(self.layers):
            layer.attention.attn_module = get_attn_module(layer_num, config)  # type: ignore
        self.post_init()


# Transformers only renames old-format checkpoint weights for its own classes.
register_checkpoint_conversion_mapping(
    "CustomViTModel", get_checkpoint_conversion_mapping("ViTModel")
)


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
