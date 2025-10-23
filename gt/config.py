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
class CustomGTConfig:
    input_dims: int
    output_dims: int
    hidden_dims: int = 96
    pos_enc_dims: int = 16
    num_layers: int = 4
    num_heads: int = 8
    dropout_p: float = 0.5
    attn_dropout_p: float = 0.5
    attention_type: AttentionType = AttentionType.softmax
    enable_flash_attention: bool = False
    num_steps: int | None = None
    rank: int | None = None
    block_size: int | None = None
    pad_type: PadType = PadType.post


def get_config(input_dims: int, output_dims: int) -> CustomGTConfig:
    return CustomGTConfig(input_dims=input_dims, output_dims=output_dims)
