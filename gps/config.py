from enum import StrEnum


class AttentionType(StrEnum):
    softmax = "softmax"
    monarch_attention = "monarch-attention"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"
    linear_attention = "linear-attention"


class CustomGPSConfig:
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_layers: int,
        num_heads: int,
        attention_type: AttentionType = AttentionType.softmax,
        enable_flash_attention: bool = False,
        num_steps: int | None = None,
        rank: int | None = None,
        block_size: int | None = None,
        pad_type: str | None = None,
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
