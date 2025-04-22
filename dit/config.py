from typing import Optional, Union, Dict
from enum import StrEnum

from common.soba import PadType, InitType


class AttentionType(StrEnum):
    softmax = "softmax"
    soba_monarch = "soba-monarch"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"

class EfficientAttnConfig:
    def __init__(self, 
        efficient_attention_type: Union[AttentionType, Dict[int, AttentionType]],
        enable_flash_attention: bool = False,
        rank: int = 8,
        block_size: Optional[int] = 16, 
        num_steps: Optional[int] = 3,
        pad_type: PadType = PadType.pre,
        init_type: InitType = InitType.eye,
        share_kv: bool = False,
        seq_len: int = 16**2,
        estimator_type: str = "pos",
        ortho_features: bool = True,
        num_attention_heads: int = 16,
        conv_kernel_size: Optional[int] = None,
        module_device = None
    ):
        
        self.efficient_attention_type = efficient_attention_type
        self.module_device = module_device

        # Softmax
        self.enable_flash_attention = enable_flash_attention

        # SobaMonarch
        self.num_steps = num_steps
        self.block_size = block_size
        self.pad_type = pad_type
        self.init_type = init_type

        # Linformer
        # self.rank used as projection dim
        self.rank = rank
        self.share_kv = share_kv
        self.seq_len = seq_len

        # Performer
        # self.rank used as num_samples
        self.estimator_type = estimator_type
        self.ortho_features = ortho_features

        # Nystromformer
        # self.rank used as number of landmarks
        self.num_attention_heads = num_attention_heads
        self.conv_kernel_size = conv_kernel_size

        # Cosformer: none


def get_config(efficient_attention_type: AttentionType, **kwargs):
    return EfficientAttnConfig(efficient_attention_type, **kwargs)