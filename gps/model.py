import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from common.baselines import Softmax

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


class GPSAttention(nn.Module):

    def __init__(self, hidden_dims: int, num_heads: int, attention_type: str):
        super().__init__()
        self.query = nn.Linear(hidden_dims, hidden_dims)
        self.key = nn.Linear(hidden_dims, hidden_dims)
        self.value = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, hidden_dims)

        self.attn_module = Softmax()
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.head_dims = hidden_dims // num_heads

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

    def __init__(self, hidden_dims: int, num_heads: int):
        super().__init__()
        self.mpnn = gnn.GCNConv(hidden_dims, hidden_dims)
        self.attn = GPSAttention(hidden_dims, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dims, 2 * hidden_dims),
            nn.GELU(),
            nn.Linear(2 * hidden_dims, hidden_dims),
        )

    def forward(self, hidden_states: Tensor, edge_index: Tensor) -> Tensor:
        mpnn_hidden_states = self.mpnn(hidden_states, edge_index)
        attn_hidden_states = self.attn(hidden_states)
        hidden_states = hidden_states + mpnn_hidden_states + attn_hidden_states
        hidden_states = hidden_states + self.ffn(hidden_states)
        return hidden_states


class GPSModel(nn.Module):

    def __init__(
        self, input_dims: int, hidden_dims: int, output_dims: int, num_heads: int
    ):
        super().__init__()
        self.encoder = nn.Linear(input_dims, hidden_dims)
        self.head = nn.Linear(hidden_dims, output_dims)
        self.backbone = GPSLayer(hidden_dims, num_heads)

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        pass
