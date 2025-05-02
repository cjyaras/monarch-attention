from enum import StrEnum

from transformers.models.roberta.configuration_roberta import RobertaConfig

from ma.monarch_attention import PadType
from roberta.data import MAX_LENGTH


class AttentionType(StrEnum):
    softmax = "softmax"
    monarch_attention = "monarch-attention"
    hybrid = "hybrid"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"
    linear_attention = "linear-attention"


class CustomRobertaConfig(RobertaConfig):
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

        # Set _attn_implementation to eager to override attention_mask logic in RobertaModel
        self._attn_implementation = "eager"


def get_config() -> CustomRobertaConfig:
    # config = CustomRobertaConfig.from_pretrained("deepset/roberta-base-squad2")
    config = CustomRobertaConfig.from_pretrained("csarron/roberta-base-squad-v1")
    assert isinstance(config, CustomRobertaConfig)
    return config
