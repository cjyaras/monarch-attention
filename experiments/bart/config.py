from transformers.models.bart.configuration_bart import BartConfig

from experiments.common.attention import AttentionConfig


class CustomBartConfig(AttentionConfig, BartConfig):
    model_type = "custom_bart"

    def __init__(
        self,
        enable_flash_attention: bool = True,
        max_decoder_position_embeddings: int = 1024,
        use_original_bart: bool = False,
        **kwargs,
    ):
        super().__init__(enable_flash_attention=enable_flash_attention, **kwargs)
        self.max_decoder_position_embeddings = max_decoder_position_embeddings
        self.use_original_bart = use_original_bart


def get_config() -> CustomBartConfig:
    config = CustomBartConfig.from_pretrained(
        "facebook/bart-base",
        max_position_embeddings=8192,
    )
    assert isinstance(config, CustomBartConfig)
    return config
