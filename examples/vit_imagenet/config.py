from enum import StrEnum
from typing import Optional

from transformers.models.vit.configuration_vit import ViTConfig

from sobalib.layers import PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    sparsemax = "sparsemax"
    low_rank = "low-rank"
    monarch = "monarch"
    monarch_block_diagonal = "monarch-block-diagonal"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"


class CustomViTConfig(ViTConfig):
    def __init__(
        self,
        attention_type: AttentionType = AttentionType.softmax,
        enable_flash_attention: bool = False,
        scale_attention_temperature: bool = False,
        efficient_attention_num_steps: Optional[int] = None,
        efficient_attention_step_size: Optional[float] = None,
        efficient_attention_rank: Optional[int] = None,
        efficient_attention_block_size: Optional[int] = None,
        efficient_attention_pad_type: PadType = PadType.pre,
        share_kv: bool = False,
        estimator_type: str = "pos",
        ortho_features: bool = True,
        conv_kernel_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention_type = attention_type
        self.enable_flash_attention = enable_flash_attention
        self.scale_attention_temperature = scale_attention_temperature
        self.efficient_attention_num_steps = efficient_attention_num_steps
        self.efficient_attention_step_size = efficient_attention_step_size
        self.efficient_attention_rank = efficient_attention_rank
        self.efficient_attention_block_size = efficient_attention_block_size
        self.efficient_attention_pad_type = efficient_attention_pad_type

        # Linformer
        # self.efficient_attention_rank used as projection dim
        self.share_kv = share_kv

        # Performer
        # self.efficient_attention_rank used as num_samples
        self.estimator_type = estimator_type
        self.ortho_features = ortho_features

        # Nystromformer
        # self.efficient_attention_rank used as number of landmarks
        self.conv_kernel_size = conv_kernel_size

        # Cosformer: none


def get_config() -> CustomViTConfig:
    config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
    assert isinstance(config, CustomViTConfig)
    return config
