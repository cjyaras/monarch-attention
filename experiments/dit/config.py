from enum import StrEnum
from typing import Dict, Optional, Union

from ma.monarch_attention import PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    monarch = "monarch"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"
    linear_attention = "linear-attention"


class EfficientAttnConfig:
    def __init__(
        self,
        attention_type: Union[AttentionType, Dict[int, AttentionType]],
        enable_flash_attention: bool = False,
        rank: int | None = 32,
        block_size: Optional[int] = 16,
        num_steps: Optional[int] = 3,
        pad_type: PadType = PadType.pre,
        num_attention_heads: int = 16
    ):

        self.attention_type = attention_type

        # Softmax
        self.enable_flash_attention = enable_flash_attention

        # Monarch
        self.num_steps = num_steps
        self.block_size = block_size
        self.pad_type = pad_type

        # Low-rank attention
        self.rank = rank
        self.num_attention_heads = num_attention_heads


def get_config(attention_type: AttentionType, **kwargs):
    return EfficientAttnConfig(attention_type, **kwargs)
