from math import inf
from typing import List

import torch
from common.baselines import Softmax, Sparsemax
from common.utils import get_device
from config import get_config
from extract import extract_query_key
from tqdm.auto import tqdm

Tensor = torch.Tensor

NUM_SAMPLES = 128
BATCH_SIZE = 4
SEARCH_RANGE = (1.0, 50.0)
SEARCH_STEPS = 50
ORD = 2


@torch.no_grad()
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
    search_size = len(attention_temperature_vals)
    differences = torch.zeros(
        search_size,
        num_layers * num_heads,
        device=query_list[0].device,
    )

    softmax = Softmax()
    sparsemax = Sparsemax()

    for i in tqdm(range(len(query_list))):
        query = query_list[i].reshape(1, num_layers * num_heads, seq_len, dim_per_head)
        key = key_list[i].reshape(1, num_layers * num_heads, seq_len, dim_per_head)
        softmax_attn_probs = softmax.get_matrix(query, key)
        sparsemax_attn_probs = sparsemax.get_matrix(
            query / attention_temperature_vals[:, None, None, None],
            key / torch.ones_like(attention_temperature_vals[:, None, None, None]),
        )
        attn_weights_diff = torch.flatten(
            softmax_attn_probs - sparsemax_attn_probs, start_dim=-2
        )
        differences += torch.linalg.norm(attn_weights_diff, ord=ORD, dim=-1)

    optimal_temperature_idx = differences.min(dim=0)[1]
    optimal_temperature = attention_temperature_vals[
        optimal_temperature_idx.reshape(num_layers, num_heads)
    ]
    return optimal_temperature


@torch.no_grad()
def main():
    config = get_config()
    device = get_device()

    all_query, all_key = extract_query_key(config, NUM_SAMPLES, batch_size=BATCH_SIZE)

    optimal_temperature = calibrate_sparsemax_temperature(
        all_query, all_key, torch.linspace(*SEARCH_RANGE, SEARCH_STEPS).to(device)
    )

    torch.save(
        {
            f"vit.encoder.layer.{i}.attention.attention.attention_temperature": optimal_temperature[
                i
            ]
            for i in range(len(optimal_temperature))
        },
        "vit_imagenet/sparsemax_temperature.pt",
    )


if __name__ == "__main__":
    main()
