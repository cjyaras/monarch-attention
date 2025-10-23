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
    input_dims: int = 14
    edge_attr_dims: int = 2
    output_dims: int = 21
    hidden_dims: int = 128
    pe_dims: int = 16
    num_layers: int = 4
    num_heads: int = 4
    dropout_p: float = 0.0
    attention_type: AttentionType = AttentionType.softmax
    enable_flash_attention: bool = False
    num_steps: int | None = None
    rank: int | None = None
    block_size: int | None = None
    pad_type: PadType = PadType.post


def get_config() -> CustomGPSConfig:
    return CustomGPSConfig()
