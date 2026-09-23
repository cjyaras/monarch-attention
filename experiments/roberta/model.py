from typing import Dict, Tuple

import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaForQuestionAnswering,
    RobertaModel,
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
from experiments.roberta.config import AttentionType, CustomRobertaConfig

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


def get_attn_module(layer_num: int, config: CustomRobertaConfig) -> torch.nn.Module:
    if isinstance(config.attention_type, Dict):
        attention_type = config.attention_type[layer_num]
    else:
        attention_type = config.attention_type
    module = ATTENTION_TYPE_TO_MODULE[attention_type]
    return module(*prepare_args(attention_type, config))


class CustomRobertaModel(RobertaModel):
    def __init__(self, config: CustomRobertaConfig, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        assert not config.is_decoder
        for layer_num, layer in enumerate(self.encoder.layer):
            layer.attention.self.attn_module = get_attn_module(layer_num, config)
        self.post_init()


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
