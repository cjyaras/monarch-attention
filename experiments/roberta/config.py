from transformers.models.roberta.configuration_roberta import RobertaConfig

from experiments.common.attention import AttentionConfig


class CustomRobertaConfig(AttentionConfig, RobertaConfig):
    # An explicit __init__ stops transformers from generating one that skips the mixin
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def get_config() -> CustomRobertaConfig:
    config = CustomRobertaConfig.from_pretrained("csarron/roberta-base-squad-v1")
    assert isinstance(config, CustomRobertaConfig)
    return config
