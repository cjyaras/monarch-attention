from math import inf, sqrt
from typing import List

import torch
from entmax import sparsemax
from torch.nn.functional import softmax
from tqdm.auto import tqdm

Tensor = torch.Tensor


def calibrate_sparsemax_temperature(
    query_list: List[Tensor], key_list: List[Tensor], attention_temperature_vals: Tensor
) -> Tensor:
    """
    query_list: List of [num_layers, num_heads, seq_len, dim_per_head]
    key_list: List of [num_layers, num_heads, seq_len, dim_per_head]
    attention_temperature_vals: [num_temperatures]

    returns: [num_layers, num_heads]
    """
    num_layers, num_heads, seq_len, dim_per_head = query_list[0].shape
    differences = torch.zeros(
        num_layers,
        num_heads,
        len(attention_temperature_vals),
        device=query_list[0].device,
    )

    for query, key in tqdm(zip(query_list, key_list)):
        attn_weights = query @ key.transpose(-1, -2) / sqrt(query.size(-1))
        softmax_attn_weights = softmax(attn_weights, dim=-1)[..., None, :, :]
        sparsemax_attn_weights = sparsemax(
            attn_weights[..., None, :, :] / attention_temperature_vals[:, None, None]
        )
        attn_weights_diff = torch.flatten(
            softmax_attn_weights - sparsemax_attn_weights, start_dim=-2
        )
        differences += torch.linalg.norm(attn_weights_diff, ord=inf, dim=-1)
    optimal_temperature_idx = differences.min(dim=-1)[1]
    optimal_temperature = attention_temperature_vals[optimal_temperature_idx]
    return optimal_temperature
