from typing import Dict, List, Tuple

import torch
from vit.models import CustomViTForImageClassification

Tensor = torch.Tensor
from math import inf, sqrt

from entmax import sparsemax
from torch.nn.functional import softmax
from tqdm import tqdm


def _register_qk_hook(
    model: CustomViTForImageClassification,
    all_layer_intermediates: List[Dict[str, Tensor]],
):
    layers = model.vit.encoder.layer

    for layer_idx in range(len(layers)):
        attn_layer = layers[layer_idx].attention.attention

        def query_hook(_layer_idx):
            def hook(module, input, output):
                all_layer_intermediates[_layer_idx].update(
                    {"query": attn_layer.transpose_for_scores(output)}
                )

            return hook

        def key_hook(_layer_idx):
            def hook(module, input, output):
                all_layer_intermediates[_layer_idx].update(
                    {"key": attn_layer.transpose_for_scores(output)}
                )

            return hook

        attn_layer.query.register_forward_hook(query_hook(layer_idx))
        attn_layer.key.register_forward_hook(key_hook(layer_idx))


def extract_qk(
    model: CustomViTForImageClassification, inputs: Dict[str, Dict]
) -> Tuple[Tensor, Tensor]:
    all_layer_intermediates = [{} for _ in range(len(model.vit.encoder.layer))]
    _register_qk_hook(model, all_layer_intermediates)

    with torch.no_grad():
        model(**inputs["image"])

    query = torch.stack(
        [
            layer_intermediates["query"]
            for layer_intermediates in all_layer_intermediates
        ],
        dim=0,
    ).transpose(1, 0)

    key = torch.stack(
        [layer_intermediates["key"] for layer_intermediates in all_layer_intermediates],
        dim=0,
    ).transpose(1, 0)

    return query, key


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
