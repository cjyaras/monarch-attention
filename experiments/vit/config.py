from enum import StrEnum

from transformers.models.vit.configuration_vit import ViTConfig

from ma.monarch_attention import PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    monarch_attention = "monarch-attention"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"
    linear_attention = "linear-attention"


class CustomViTConfig(ViTConfig):
    def __init__(
        self,
        attention_type: (
            AttentionType | dict[int, AttentionType]
        ) = AttentionType.softmax,
        enable_flash_attention: bool = False,
        num_steps: int | None = None,
        rank: int | None = None,
        block_size: int | None = None,
        pad_type: PadType = PadType.pre,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention_type = attention_type

        # Softmax
        self.enable_flash_attention = enable_flash_attention

        # Monarch
        self.num_steps = num_steps
        self.block_size = block_size
        self.pad_type = pad_type

        # Low-rank attention
        self.rank = rank


def get_config() -> CustomViTConfig:
    config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
    assert isinstance(config, CustomViTConfig)
    return config
