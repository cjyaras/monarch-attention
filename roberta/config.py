from enum import StrEnum
from typing import List, Optional

from transformers.models.roberta.configuration_roberta import RobertaConfig

from common.soba import PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    sparsemax = "sparsemax"
    soba_monarch = "soba-monarch"
    hybrid = "hybrid"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"


class CustomRobertaConfig(RobertaConfig):
    def __init__(
        self,
        attention_type: AttentionType = AttentionType.softmax,
        attn_module_save_path: Optional[str] = None,
        enable_flash_attention: bool = False,
        num_steps: Optional[int] = None,
        block_size: Optional[int] = None,
        pad_type: PadType = PadType.pre,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention_type = attention_type
        self.attn_module_save_path = attn_module_save_path

        # Softmax
        self.enable_flash_attention = enable_flash_attention

        # Sparsemax: none

        # SobaMonarch
        self.num_steps = num_steps
        self.block_size = block_size
        self.pad_type = pad_type

        # Hybrid
        self.hybrid_attention_layers: List[int] = []

        # Set _attn_implementation to eager to override attention_mask logic in RobertaModel
        self._attn_implementation = "eager"


def get_config() -> CustomRobertaConfig:
    config = CustomRobertaConfig.from_pretrained("deepset/roberta-base-squad2")
    assert isinstance(config, CustomRobertaConfig)
    return config
