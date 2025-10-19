from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from common.baselines import (
    Cosformer,
    LinearAttention,
    Linformer,
    Nystromformer,
    Performer,
    Softmax,
)
from common.utils import get_device
from gps.config import AttentionType, CustomGPSConfig
from ma.monarch_attention import MonarchAttention

Tensor = torch.Tensor

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch_attention: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
    AttentionType.linear_attention: LinearAttention,
}


def prepare_args(attention_type: AttentionType, config: CustomGPSConfig) -> Tuple:

    match attention_type:

        case AttentionType.softmax:
            return (config.enable_flash_attention,)

        case AttentionType.monarch_attention:
            return (config.block_size, config.num_steps, config.pad_type)

        case (
            AttentionType.linformer
            | AttentionType.performer
            | AttentionType.nystromformer
        ):
            return (config.rank,)

        case AttentionType.cosformer | AttentionType.linear_attention:
            return ()

        case _:
            raise ValueError(f"Invalid attention type: {attention_type}")


class GPSAttention(nn.Module):

    def __init__(self, config: CustomGPSConfig):
        super().__init__()
        self.query = nn.Linear(config.hidden_dims, config.hidden_dims)
        self.key = nn.Linear(config.hidden_dims, config.hidden_dims)
        self.value = nn.Linear(config.hidden_dims, config.hidden_dims)
        self.output = nn.Linear(config.hidden_dims, config.hidden_dims)

        module = ATTENTION_TYPE_TO_MODULE[config.attention_type]

        self.attn_module = module(*prepare_args(config.attention_type, config))
        self.hidden_dims = config.hidden_dims
        self.num_heads = config.num_heads
        self.head_dims = config.hidden_dims // config.num_heads

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dims)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: Tensor) -> Tensor:
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        attn_output = self.attn_module(query, key, value)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_attn_output_shape = attn_output.size()[:-2] + (self.hidden_dims,)
        attn_output = attn_output.view(*new_attn_output_shape)
        attn_hidden_states = self.output(attn_output)
        return attn_hidden_states


class GPSLayer(nn.Module):

    def __init__(self, config: CustomGPSConfig):
        super().__init__()
        self.mpnn = gnn.GCNConv(config.hidden_dims, config.hidden_dims)

        self.attn = GPSAttention(config)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dims, 2 * config.hidden_dims),
            nn.GELU(),
            nn.Linear(2 * config.hidden_dims, config.hidden_dims),
        )

    def forward(self, hidden_states: Tensor, edge_index: Tensor) -> Tensor:
        mpnn_hidden_states = self.mpnn(hidden_states, edge_index) + hidden_states
        attn_hidden_states = (
            self.attn(hidden_states.unsqueeze(0)).squeeze(0) + hidden_states
        )
        hidden_states = mpnn_hidden_states + attn_hidden_states
        hidden_states = self.ffn(hidden_states) + hidden_states
        return hidden_states


class GPSModel(nn.Module):

    def __init__(self, config: CustomGPSConfig):
        super().__init__()
        self.encoder = nn.Linear(config.input_dims, config.hidden_dims)
        self.backbone = nn.ModuleList(
            [GPSLayer(config) for _ in range(config.num_layers)]
        )
        self.head = nn.Linear(config.hidden_dims, config.output_dims)

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        hidden_states = self.encoder(node_features)
        for layer in self.backbone:
            hidden_states = layer(hidden_states, edge_index)
        output = self.head(hidden_states)
        return output


def get_model(config: CustomGPSConfig, pretrained: bool = True) -> GPSModel:
    device = get_device()
    model = GPSModel(config)
    if pretrained:
        model.load_state_dict(torch.load("gps/gps_model.pt", map_location=device))
    model = model.to(device)
    model.train()
    return model
