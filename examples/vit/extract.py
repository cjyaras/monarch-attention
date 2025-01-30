from typing import Dict, List, Optional, Tuple

import torch
from vit.config import CustomViTConfig
from vit.data import get_dataset
from vit.evaluation import CustomImageClassificationEvaluator
from vit.metric import TopKAccuracy
from vit.model import CustomViTForImageClassification
from vit.pipeline import get_pipeline

Tensor = torch.Tensor


def _register_qk_hook(
    model: CustomViTForImageClassification,
    all_layer_intermediates: List[Dict[str, List[Tensor]]],
):
    layers = model.vit.encoder.layer

    for layer_idx in range(len(layers)):
        attn_layer = layers[layer_idx].attention.attention.attn_module

        def qk_hook(_layer_idx):
            def hook(module, input, output):
                query, key, _ = input
                all_layer_intermediates[_layer_idx]["query"].append(query)
                all_layer_intermediates[_layer_idx]["key"].append(key)

            return hook

        attn_layer.register_forward_hook(qk_hook(layer_idx))


@torch.no_grad()
def extract_query_key(
    config: CustomViTConfig, num_samples: Optional[int] = None, batch_size: int = 1
) -> Tuple[Tensor, Tensor]:

    dataset = get_dataset(num_samples=num_samples)
    evaluator = CustomImageClassificationEvaluator(top_k=1)
    metric = TopKAccuracy()

    pipe = get_pipeline(config, batch_size=batch_size)

    all_layer_intermediates = [
        {"query": [], "key": []} for _ in range(config.num_hidden_layers)
    ]
    _register_qk_hook(pipe.model, all_layer_intermediates)

    evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset,
        metric=metric,
        label_mapping=pipe.model.config.label2id,  # type: ignore
    )

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
