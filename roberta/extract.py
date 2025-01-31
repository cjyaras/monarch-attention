from typing import Dict, List, Optional, Tuple

import torch
from roberta.config import CustomRobertaConfig
from roberta.data import get_dataset
from roberta.evaluation import CustomQuestionAnsweringEvaluator
from roberta.model import CustomRobertaForQuestionAnswering
from roberta.pipeline import get_pipeline

Tensor = torch.Tensor


def _register_qk_hook(
    model: CustomRobertaForQuestionAnswering,
    all_layer_intermediates: List[Dict[str, List[Tensor]]],
):
    layers = model.roberta.encoder.layer

    for layer_idx in range(len(layers)):
        attn_layer = layers[layer_idx].attention.self.attn_module

        def qk_hook(_layer_idx):
            def hook(module, input, output):
                query, key, _, attention_mask = input
                all_layer_intermediates[_layer_idx]["query"].append(query)
                all_layer_intermediates[_layer_idx]["key"].append(key)
                all_layer_intermediates[_layer_idx]["attention_mask"].append(
                    attention_mask
                )

            return hook

        attn_layer.register_forward_hook(qk_hook(layer_idx))


@torch.no_grad()
def extract_query_key_mask(
    config: CustomRobertaConfig,
    num_samples: Optional[int] = None,
    batch_size: int = 1,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:

    dataset = get_dataset(num_samples=num_samples)
    evaluator = CustomQuestionAnsweringEvaluator()

    pipe = get_pipeline(config, batch_size=batch_size)

    all_layer_intermediates = [
        {"query": [], "key": [], "attention_mask": []}
        for _ in range(config.num_hidden_layers)
    ]
    _register_qk_hook(pipe.model, all_layer_intermediates)

    evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset,
        metric="squad_v2",
        squad_v2_format=True,
    )

    query = list(
        torch.unbind(
            torch.stack(
                [
                    torch.cat(layer_intermediates["query"])
                    for layer_intermediates in all_layer_intermediates
                ]
            ).transpose(1, 0)
        )
    )

    key = list(
        torch.unbind(
            torch.stack(
                [
                    torch.cat(layer_intermediates["key"])
                    for layer_intermediates in all_layer_intermediates
                ]
            ).transpose(1, 0)
        )
    )

    attention_mask = list(
        torch.unbind(torch.cat(all_layer_intermediates[0]["attention_mask"]))
    )

    return query, key, attention_mask
