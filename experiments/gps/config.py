from dataclasses import dataclass
from enum import StrEnum

from ma.monarch_attention import PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    monarch_attention = "monarch-attention"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"
    linear_attention = "linear-attention"


@dataclass
class CustomGPSConfig:
    input_dims: int = 932
    output_dims: int = 5
    hidden_dims: int = 96
    num_layers: int = 1
    num_heads: int = 2
    attention_type: AttentionType = AttentionType.softmax
    enable_flash_attention: bool = False
    num_steps: int | None = None
    rank: int | None = None
    block_size: int | None = None
    pad_type: PadType = PadType.post


def get_config() -> CustomGPSConfig:
    return CustomGPSConfig()
