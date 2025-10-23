from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.baselines import (
    Cosformer,
    LinearAttention,
    Linformer,
    Nystromformer,
    Performer,
    Softmax,
)
from ma.monarch_attention import MonarchAttention
from tokengt.config import POS_EMB_DIMS, AttentionType, CustomTokenGTConfig
from tokengt.data import Data

PRETRAINED_PATH = "tokengt/token_gt_model.pt"

ATTENTION_TYPE_TO_MODULE = {
    AttentionType.softmax: Softmax,
    AttentionType.monarch_attention: MonarchAttention,
    AttentionType.linformer: Linformer,
    AttentionType.performer: Performer,
    AttentionType.nystromformer: Nystromformer,
    AttentionType.cosformer: Cosformer,
    AttentionType.linear_attention: LinearAttention,
}


def prepare_args(attention_type: AttentionType, config: CustomTokenGTConfig) -> Tuple:

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


class TokenGTEmbedding(nn.Module):

    def __init__(self, config: CustomTokenGTConfig):
        super().__init__()
        type_emb_dims = config.input_dims

        # edge features are a learnable parameter
        self.edge_features = nn.Parameter(torch.randn(1, config.input_dims))

        # the type of token: node or edge
        self.token_type_embedding = nn.Parameter(torch.randn(2, type_emb_dims))

        # linear projection to hidden dimensions
        self.projection = nn.Linear(
            config.input_dims + 2 * POS_EMB_DIMS + type_emb_dims,
            config.hidden_dims,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        positional_encoding: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # node_features: (num_nodes, feature_dims)
        # positional_encoding: (num_nodes, pos_emb_dims)
        # edge_index: (2, num_edges)

        # node embeddings are constructed from node features + 2 * positional encodings + type embedding
        node_token_type_emb = self.token_type_embedding[0]  # (type_emb_dims,)
        node_embeddings = torch.cat(
            [
                node_features,
                positional_encoding,
                positional_encoding,
                node_token_type_emb.unsqueeze(0).expand(node_features.size(0), -1),
            ],
            dim=-1,
        )  # (num_nodes, feature_dims + 2 * pos_emb_dims + type_emb_dims)

        edge_token_type_emb = self.token_type_embedding[1]  # (type_emb_dims,)
        edge_embeddings = torch.cat(
            [
                self.edge_features.expand(edge_index.size(1), -1),
                torch.cat(
                    [
                        positional_encoding[edge_index[0]],
                        positional_encoding[edge_index[1]],
                    ],
                    dim=-1,
                ),
                edge_token_type_emb.unsqueeze(0).expand(edge_index.size(1), -1),
            ],
            dim=-1,
        )  # (num_edges, feature_dims + 2 * pos_emb_dims + type_emb_dims)

        embeddings = torch.cat(
            [node_embeddings, edge_embeddings], dim=0
        )  # (num_nodes + num_edges, ...)
        embeddings = self.projection(embeddings)  # (num_nodes + num_edges, hidden_dims)
        return embeddings


class TokenGTSelfAttention(nn.Module):

    def __init__(self, config: CustomTokenGTConfig):
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

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dims)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        attn_output = self.attn_module(query, key, value)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(*hidden_states.shape)
        attn_hidden_states = self.output(attn_output)
        return attn_hidden_states


class EncoderBlock(nn.Module):

    def __init__(self, config: CustomTokenGTConfig):
        super().__init__()
        self.self_attn = TokenGTSelfAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dims, 2 * config.hidden_dims),
            nn.GELU(),
            nn.Linear(2 * config.hidden_dims, config.hidden_dims),
        )
        self.norm1 = nn.RMSNorm(config.hidden_dims)
        self.norm2 = nn.RMSNorm(config.hidden_dims)
        self.dropout_p = config.dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            F.dropout(
                self.self_attn(self.norm1(x).unsqueeze(0)).squeeze(0),
                p=self.dropout_p,
                training=self.training,
            )
            + x
        )
        x = (
            F.dropout(self.ffn(self.norm2(x)), p=self.dropout_p, training=self.training)
            + x
        )
        return x


class TokenGTEncoder(nn.Module):

    def __init__(self, config: CustomTokenGTConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TokenGTModel(nn.Module):
    def __init__(self, config: CustomTokenGTConfig):
        super().__init__()
        self.embedding = TokenGTEmbedding(config)
        self.encoder = TokenGTEncoder(config)
        self.head = nn.Linear(config.hidden_dims, config.output_dims)

    def forward(
        self,
        node_features: torch.Tensor,
        positional_encoding: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = node_features.size(0)
        x = self.embedding(node_features, positional_encoding, edge_index)
        x = self.encoder(x)[:num_nodes]
        x = self.head(x)
        return x


def get_model(config: CustomTokenGTConfig, pretrained: bool = False) -> TokenGTModel:
    model = TokenGTModel(config)
    if pretrained:
        state_dict = torch.load(PRETRAINED_PATH)
        model.load_state_dict(state_dict)
    return model
