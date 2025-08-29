from typing import Dict, List, Optional, Tuple

import torch

from dit.config import AttentionType
from dit.model import CustomDiTTransformer2DModel
from dit.pipeline import get_pipeline

Tensor = torch.Tensor


def _register_qk_hook(
    model: CustomDiTTransformer2DModel,
    all_layer_intermediates: List[Dict[str, List[Tensor]]],
):
    layers = model.transformer_blocks

    for layer_idx in range(len(layers)):
        attn_layer = layers[layer_idx].attn1.processor.attn_module
        #attn_layer = layers[layer_idx].attention.attention.attn_module  # type: ignore

        def qk_hook(_layer_idx):
            def hook(module, input, output):
                query, key, _, _ = input
                all_layer_intermediates[_layer_idx]["query"].append(query)
                all_layer_intermediates[_layer_idx]["key"].append(key)

            return hook

        attn_layer.register_forward_hook(qk_hook(layer_idx))  # type: ignore


@torch.no_grad()
def extract_query_key(
    attn_type: AttentionType,
    words: list,
    seed: int = 33,
    num_inference_steps: int = 1,
) -> Tuple[Tensor, Tensor]:

    pipe = get_pipeline(attn_type)

    all_layer_intermediates = [
        {"query": [], "key": []} for _ in range(pipe.transformer.config.num_layers)
    ]
    _register_qk_hook(pipe.transformer, all_layer_intermediates)

    class_ids = pipe.get_label_ids(words)
    pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, generator=torch.manual_seed(seed))

    query = torch.stack(
        [
            torch.cat(layer_intermediates["query"])
            for layer_intermediates in all_layer_intermediates
        ]
    ).transpose(1, 0)

    key = torch.stack(
        [
            torch.cat(layer_intermediates["key"])
            for layer_intermediates in all_layer_intermediates
        ]
    ).transpose(1, 0)

    return query, key
