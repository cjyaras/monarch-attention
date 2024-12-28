from math import inf, sqrt

import matplotlib.pyplot as plt
import torch
from sparsemax import Sparsemax
from torch.nn.functional import softmax

Tensor = torch.Tensor


def calibrate_sparsemax_temperature(query: Tensor, key: Tensor):
    # query: [total_num_heads, seq_len, dim_per_head]
    # key: [total_num_heads, seq_len, dim_per_head]
    sparsemax = Sparsemax()
    attention_temperature_vals = torch.logspace(0, 2, 100)
    attn_weights = query @ key.transpose(-1, -2) / sqrt(query.size(-1))
    softmax_attn_weights = softmax(attn_weights, dim=-1)[:, None, :, :]
    sparsemax_attn_weights = sparsemax(
        attn_weights[:, None, :, :] / attention_temperature_vals[None, :, None, None]
    )
    attn_weights_diff = softmax_attn_weights - sparsemax_attn_weights
    attn_weights_diff = attn_weights_diff.reshape(
        query.shape[0], attention_temperature_vals.shape[0], -1
    )
    differences = torch.linalg.norm(attn_weights_diff, ord=inf, dim=-1)
    plt.tight_layout()
    plt.semilogx(attention_temperature_vals, differences.T, alpha=0.5)
    plt.show()
