from enum import StrEnum
from math import sqrt
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from common.utils import maybe_compile
from entmax import sparsemax
from torch._prims_common import DeviceLikeType
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification,
    ViTModel,
    ViTSelfAttention,
)

from sobalib.layers import LowRankMHA, MonarchBlockDiagonalMHA, MonarchMHA, PadType
from baselines import Linformer, Performer, Nystromformer, Cosformer


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
        estimator_type: str = 'pos',
        ortho_features: bool = True, 
        conv_kernel_size: int = None,
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


class CustomViTSelfAttention(ViTSelfAttention):

    def __init__(self, config: CustomViTConfig):
        super().__init__(config)
        self.attention_type = config.attention_type

        if self.attention_type in [
            AttentionType.softmax,
            AttentionType.sparsemax
        ]:
            pass

        elif self.attention_type in [
            AttentionType.low_rank,
            AttentionType.monarch,
            AttentionType.monarch_block_diagonal,
        ]:
            num_steps = config.efficient_attention_num_steps
            step_size = config.efficient_attention_step_size
            assert num_steps is not None and step_size is not None

            if self.attention_type == AttentionType.low_rank:
                rank = config.efficient_attention_rank
                assert rank is not None
                self.efficient_attn = LowRankMHA(
                    num_steps=num_steps,
                    step_size=step_size,
                    rank=rank,
                )

            elif self.attention_type == AttentionType.monarch:
                block_size = config.efficient_attention_block_size
                pad_type = config.efficient_attention_pad_type
                assert block_size is not None and pad_type is not None
                self.efficient_attn = MonarchMHA(
                    block_size=block_size,
                    num_steps=num_steps,
                    step_size=step_size,
                    pad_type=pad_type,
                )

            elif self.attention_type == AttentionType.monarch_block_diagonal:
                block_size = config.efficient_attention_block_size
                pad_type = config.efficient_attention_pad_type
                assert block_size is not None and pad_type is not None
                self.efficient_attn = MonarchBlockDiagonalMHA(
                    block_size=block_size,
                    num_steps=num_steps,
                    step_size=step_size,
                    pad_type=pad_type,
                )

            maybe_compile(self.efficient_attn)

        elif self.attention_type in [
            AttentionType.linformer, 
            AttentionType.performer,
            AttentionType.nystromformer,
            AttentionType.cosformer
        ]:
            assert isinstance(config.image_size, int) and isinstance(config.patch_size, int)
            seq_len = (config.image_size // config.patch_size)**2 + 1 # TODO: cleaner way to get sequence length here?

            if self.attention_type == AttentionType.linformer:
                proj_dim = config.efficient_attention_rank
                share_kv = config.share_kv
                self.efficient_attn = Linformer(
                    proj_dim=proj_dim,
                    seq_len=seq_len,
                    share_kv=share_kv
                )
            
            elif self.attention_type == AttentionType.performer:
                num_samples = config.efficient_attention_rank
                estimator_type = config.estimator_type
                ortho_features = config.ortho_features
                self.efficient_attn = Performer(
                    num_samples=num_samples,
                    estimator_type=estimator_type,
                    ortho_features=ortho_features
                )
            
            elif self.attention_type == AttentionType.nystromformer:
                num_landmarks = config.efficient_attention_rank
                conv_kernel_size = config.conv_kernel_size
                num_heads = config.num_attention_heads
                self.efficient_attn = Nystromformer(
                    num_landmarks=num_landmarks,
                    num_heads=num_heads,
                    conv_kernel_size=conv_kernel_size
                )

            elif self.attention_type == AttentionType.cosformer:
                self.efficient_attn = Cosformer()
                

            maybe_compile(self.efficient_attn)
        
        else:
                raise ValueError(f"Invalid attention type: {self.attention_type}")

        

        if config.scale_attention_temperature:
            self.register_buffer(
                "attention_temperature",
                torch.full((self.num_attention_heads,), 1.0),
                persistent=True,
            )
        else:
            self.attention_temperature = None

        self.enable_flash_attention = config.enable_flash_attention

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        assert head_mask is None
        assert not output_attentions

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.attention_temperature is not None:
            query_layer = query_layer / self.attention_temperature[..., None, None]

        if self.attention_type == AttentionType.softmax and self.enable_flash_attention:
            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                head_mask,
                0.0,
                is_causal=False,
                scale=None,
            )

        elif self.attention_type in [AttentionType.softmax, AttentionType.sparsemax]:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / sqrt(self.attention_head_size)
            attention_probs = (
                sparsemax(attention_scores, dim=-1)
                if self.attention_type == AttentionType.sparsemax
                else nn.functional.softmax(attention_scores, dim=-1)
            )
            assert isinstance(attention_probs, torch.Tensor)
            context_layer = torch.matmul(attention_probs, value_layer)

        # TODO: Add baselines forward logic here
        else:
            context_layer = self.efficient_attn(query_layer, key_layer, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None  # type: ignore


class CustomViTModel(ViTModel):

    def __init__(
        self,
        config: CustomViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__(config, add_pooling_layer, use_mask_token)
        for layer in self.encoder.layer:
            layer.attention.attention = CustomViTSelfAttention(config)
        self.post_init()


class CustomViTForImageClassification(ViTForImageClassification):

    def __init__(self, config: CustomViTConfig):
        super().__init__(config)
        self.vit = CustomViTModel(config, add_pooling_layer=False)
        self.post_init()


def get_model(
    config: CustomViTConfig, device: DeviceLikeType
) -> CustomViTForImageClassification:
    model = CustomViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
    )
    if config.scale_attention_temperature:
        model.load_state_dict(
            torch.load("vit/sparsemax_temperature.pt", weights_only=True), strict=False
        )
    model = model.to(device)  # type: ignore
    model.eval()
    return model
