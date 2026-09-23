from transformers.models.vit.configuration_vit import ViTConfig

from experiments.common.attention import AttentionConfig


class CustomViTConfig(AttentionConfig, ViTConfig):
    # An explicit __init__ stops transformers from generating one that skips the mixin
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def get_config() -> CustomViTConfig:
    config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
    assert isinstance(config, CustomViTConfig)
    return config
