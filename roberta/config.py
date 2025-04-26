from enum import StrEnum
from typing import Dict, List, Optional, Union

from transformers.models.roberta.configuration_roberta import RobertaConfig

from common.soba import PadType
from roberta.data import MAX_LENGTH


class AttentionType(StrEnum):
    softmax = "softmax"
    soba_monarch = "soba-monarch"
    hybrid = "hybrid"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"


class CustomRobertaConfig(RobertaConfig):
    def __init__(
        self,
        attention_type: Union[
            AttentionType, Dict[int, AttentionType]
        ] = AttentionType.softmax,
        enable_flash_attention: bool = False,
        num_steps: Optional[int] = None,
        rank: Optional[int] = None,
        block_size: Optional[int] = None,
        pad_type: PadType = PadType.pre,
        share_kv: bool = False,
        estimator_type: str = "pos",
        ortho_features: bool = True,
        conv_kernel_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention_type = attention_type

        # Softmax
        self.enable_flash_attention = enable_flash_attention

        # SobaMonarch
        self.num_steps = num_steps
        self.block_size = block_size
        self.pad_type = pad_type

        # Linformer
        # self.rank used as projection dim
        self.rank = rank
        self.share_kv = share_kv

        # Compute sequence length
        self.seq_len = MAX_LENGTH

        # Performer
        # self.rank used as num_samples
        self.estimator_type = estimator_type
        self.ortho_features = ortho_features

        # Nystromformer
        # self.rank used as number of landmarks
        self.conv_kernel_size = conv_kernel_size

        # Set _attn_implementation to eager to override attention_mask logic in RobertaModel
        self._attn_implementation = "eager"


def get_config() -> CustomRobertaConfig:
    config = CustomRobertaConfig.from_pretrained("deepset/roberta-base-squad2")
    assert isinstance(config, CustomRobertaConfig)
    return config
