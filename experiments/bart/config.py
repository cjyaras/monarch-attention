from enum import StrEnum
from typing import Dict, Optional, Union

from transformers.models.bart.configuration_bart import BartConfig

from ma.monarch_attention import PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    monarch_attention = "monarch-attention"
    hybrid = "hybrid"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"
    linear_attention = "linear-attention"


class CustomBartConfig(BartConfig):
    model_type = "custom_bart"
    def __init__(
        self,
        attention_type: Union[
            AttentionType, Dict[int, AttentionType]
        ] = AttentionType.softmax,
        enable_flash_attention: bool = True,
        num_steps: Optional[int] = None,
        rank: Optional[int] = None,
        block_size: Optional[int] = None,
        pad_type: PadType = PadType.pre,
        share_kv: bool = False,
        estimator_type: str = "pos",
        ortho_features: bool = True,
        conv_kernel_size: Optional[int] = None,
        max_decoder_position_embeddings: int = 1024,
        use_original_bart: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Training-related arguments
        self.max_decoder_position_embeddings = max_decoder_position_embeddings
        self.use_original_bart = use_original_bart

        self.attention_type = attention_type

        # Softmax
        self.enable_flash_attention = enable_flash_attention

        # Monarch
        self.num_steps = num_steps
        self.block_size = block_size
        self.pad_type = pad_type

        # Linformer
        # self.rank used as projection dim
        self.rank = rank
        self.share_kv = share_kv

        # Performer
        # self.rank used as num_samples
        self.estimator_type = estimator_type
        self.ortho_features = ortho_features

        # Nystromformer
        # self.rank used as number of landmarks
        self.conv_kernel_size = conv_kernel_size

        # Cosformer: none


def get_config() -> CustomBartConfig:
    config = CustomBartConfig.from_pretrained(
        "facebook/bart-base",
        max_position_embeddings=8192,
    )
    assert isinstance(config, CustomBartConfig)
    return config
