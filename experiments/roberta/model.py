from transformers.models.roberta.modeling_roberta import (
    RobertaForQuestionAnswering,
    RobertaModel,
)
from transformers.utils.logging import ERROR, set_verbosity  # type: ignore

from experiments.common.attention import get_attn_module
from experiments.common.utils import get_device
from experiments.roberta.config import CustomRobertaConfig

set_verbosity(ERROR)


class CustomRobertaModel(RobertaModel):
    def __init__(self, config: CustomRobertaConfig, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        assert not config.is_decoder
        for layer_num, layer in enumerate(self.encoder.layer):
            layer.attention.self.attn_module = get_attn_module(config, layer_num)
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
