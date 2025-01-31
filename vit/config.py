from enum import StrEnum
from typing import Optional

from transformers.models.vit.configuration_vit import ViTConfig

from common.soba import PadType


class AttentionType(StrEnum):
    softmax = "softmax"
    sparsemax = "sparsemax"
    soba_monarch = "soba-monarch"
    linformer = "linformer"
    performer = "performer"
    nystromformer = "nystromformer"
    cosformer = "cosformer"


class CustomViTConfig(ViTConfig):
    def __init__(
        self,
        attention_type: AttentionType = AttentionType.softmax,
        attn_module_save_path: Optional[str] = None,
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
        self.attn_module_save_path = attn_module_save_path

        # Softmax
        self.enable_flash_attention = enable_flash_attention

        # Sparsemax: none

        # SobaMonarch
        self.num_steps = num_steps
        self.block_size = block_size
        self.pad_type = pad_type

        # Linformer
        # self.rank used as projection dim
        self.rank = rank
        self.share_kv = share_kv

        # Compute sequence length
        assert isinstance(self.image_size, int) and isinstance(self.patch_size, int)
        self.seq_len = (
            self.image_size // self.patch_size
        ) ** 2 + 1  # TODO: cleaner way to get sequence length here?

        # Performer
        # self.rank used as num_samples
        self.estimator_type = estimator_type
        self.ortho_features = ortho_features

        # Nystromformer
        # self.rank used as number of landmarks
        self.conv_kernel_size = conv_kernel_size

        # Cosformer: none


def get_config() -> CustomViTConfig:
    config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
    assert isinstance(config, CustomViTConfig)
    return config
