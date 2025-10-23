from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

from common.baselines import (
    Cosformer,
    LinearAttention,
    Linformer,
    Nystromformer,
    Performer,
    Softmax,
)
from common.utils import get_device
from gps_lrgb.config import AttentionType, CustomGPSConfig
from ma.monarch_attention import MonarchAttention

PRETRAINED_PATH = "gps_lrgb/gps_model.pt"

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch_attention: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
    AttentionType.linear_attention: LinearAttention,
}


def prepare_args(attention_type: AttentionType, config: CustomGPSConfig) -> tuple:

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

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        attn_output = self.attn_module(
            query, key, value, attention_mask=attention_mask.int()
        )
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_attn_output_shape = attn_output.size()[:-2] + (self.hidden_dims,)
        attn_output = attn_output.view(*new_attn_output_shape)
        hidden_states = self.output(attn_output)
        return hidden_states


class CustomGPSConv(GPSConv):

    def __init__(self, conv: Optional[MessagePassing], config: CustomGPSConfig):
        super().__init__(
            channels=config.hidden_dims, conv=conv, dropout=config.dropout_p
        )
        self.attn = GPSAttention(config)
        self.attn_type = config.attention_type.value

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)
        h = self.attn(h, attention_mask=~mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out


class CustomGPSModel(torch.nn.Module):

    def __init__(self, config: CustomGPSConfig):
        super().__init__()

        channels = config.hidden_dims
        pe_dim = config.pe_dims
        num_layers = config.num_layers

        self.node_emb = Linear(config.input_dims, channels - pe_dim)
        self.pe_lin = Linear(config.pe_dims, pe_dim)
        self.pe_norm = BatchNorm1d(config.pe_dims)
        self.edge_emb = Linear(config.edge_attr_dims, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = CustomGPSConv(conv=GINEConv(nn), config=config)
            self.convs.append(conv)

        self.head = Linear(channels, config.output_dims)

        # self.mlp = Sequential(
        #     Linear(channels, channels // 2),
        #     ReLU(),
        #     Linear(channels // 2, channels // 4),
        #     ReLU(),
        #     Linear(channels // 4, 1),
        # )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        # x = global_add_pool(x, batch)
        # return self.mlp(x)
        return self.head(x)


def get_model(config: CustomGPSConfig, pretrained: bool = True) -> CustomGPSModel:
    device = get_device()
    model = CustomGPSModel(config)
    if pretrained:
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
    model = model.to(device)
    return model
