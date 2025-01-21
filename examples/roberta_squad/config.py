from enum import StrEnum
from typing import List, Optional

from transformers.models.roberta.configuration_roberta import RobertaConfig

from sobalib.layers import PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    sparsemax = "sparsemax"
    low_rank = "low-rank"
    monarch = "monarch"
    monarch_block_diagonal = "monarch-block-diagonal"
    hybrid = "hybrid"


class CustomRobertaConfig(RobertaConfig):
    def __init__(
        self,
        attention_type: AttentionType = AttentionType.softmax,
        enable_flash_attention: bool = False,
        scale_attention_temperature: bool = False,
        efficient_attention_num_steps: Optional[int] = None,
        efficient_attention_step_size: Optional[float] = None,
        efficient_attention_rank: Optional[int] = None,
        efficient_attention_block_size: Optional[int] = None,
        efficient_attention_pad_type: PadType = PadType.post,
        hybrid_attention_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Set _attn_implementation to eager to override attention_mask logic in RobertaModel
        self._attn_implementation = "eager"
        self.attention_type = attention_type
        self.scale_attention_temperature = scale_attention_temperature
        self.enable_flash_attention = enable_flash_attention
        self.efficient_attention_num_steps = efficient_attention_num_steps
        self.efficient_attention_step_size = efficient_attention_step_size
        self.efficient_attention_rank = efficient_attention_rank
        self.efficient_attention_block_size = efficient_attention_block_size
        self.efficient_attention_pad_type = efficient_attention_pad_type
        self.hybrid_attention_layers = hybrid_attention_layers


def get_config() -> CustomRobertaConfig:
    config = CustomRobertaConfig.from_pretrained("deepset/roberta-base-squad2")
    assert isinstance(config, CustomRobertaConfig)
    return config
