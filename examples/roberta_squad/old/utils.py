from typing import Dict, List, Tuple

import torch
from models import CustomRobertaForQuestionAnswering

Tensor = torch.Tensor


def _register_qk_hook(
    model: CustomRobertaForQuestionAnswering,
    all_layer_intermediates: List[Dict[str, torch.Tensor]],
):
    layers = model.roberta.encoder.layer

    for layer_idx in range(len(layers)):
        attn_layer = layers[layer_idx].attention.self

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


@torch.no_grad()
def extract_qk(
    model: CustomRobertaForQuestionAnswering, inputs: Dict[str, Tensor]
) -> Tuple[Tensor, Tensor]:
    all_layer_intermediates = [{} for _ in range(len(model.roberta.encoder.layer))]
    _register_qk_hook(model, all_layer_intermediates)

    model(**inputs)

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
