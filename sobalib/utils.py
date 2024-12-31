from math import inf, sqrt

import torch
from sparsemax import Sparsemax
from torch.nn.functional import softmax

Tensor = torch.Tensor


def calibrate_sparsemax_temperature(
    query: Tensor, key: Tensor, attention_temperature_vals: Tensor
) -> Tensor:
    # query: [num_examples, num_layers, num_heads, seq_len, dim_per_head]
    # key: [num_examples, num_layers, num_heads, seq_len, dim_per_head]
    # return: [num_layers, num_heads]
    sparsemax = Sparsemax()
    attn_weights = query @ key.transpose(-1, -2) / sqrt(query.size(-1))
    softmax_attn_weights = softmax(attn_weights, dim=-1)[..., None, :, :]
    sparsemax_attn_weights = sparsemax(
        attn_weights[..., None, :, :] / attention_temperature_vals[:, None, None]
    )
    attn_weights_diff = torch.flatten(
        softmax_attn_weights - sparsemax_attn_weights, start_dim=-2
    )
    differences = torch.linalg.norm(attn_weights_diff, ord=inf, dim=-1)
    optimal_temperature_idx = differences.mean(dim=0).min(dim=-1)[1]
    optimal_temperature = attention_temperature_vals[optimal_temperature_idx]
    return optimal_temperature
