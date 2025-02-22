from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .model import DiTTransformer2DModel
from .custom_attention import CustomBasicTransformerBlock
from .custom_attention_processor import EfficientAttnConfig

class CustomDiTTransformer2DModel(DiTTransformer2DModel):
    """
    A custom 2D Transformer model that takes in efficient attention approximations
    """

    def __init__(
        self,
        efficient_attention_config: EfficientAttnConfig,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = True,
        sample_size: int = 32,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5
    ):
        
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            norm_num_groups,
            attention_bias,
            sample_size,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            norm_eps
        )

        assert efficient_attention_config.efficient_attention_type in ["softmax", "soba", "linformer", "cosformer", "nystromformer", "performer"]

        # Change to custom Transformer blocks which allows for efficient attention modules
        self.transformer_blocks = nn.ModuleList(
            [
                CustomBasicTransformerBlock(
                    efficient_attention_config,
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps
                )
                for _ in range(self.config.num_layers)
            ]
        )