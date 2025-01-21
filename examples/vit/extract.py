from typing import Dict, List, Tuple

import torch
from common.data import IterableKeyDataset
from common.utils import get_device
from config import get_config
from data import get_dataset
from datasets import IterableDataset
from model import CustomViTConfig, CustomViTForImageClassification, get_model
from pipeline import get_pipeline
from tqdm import tqdm

Tensor = torch.Tensor


def _register_qk_hook(
    model: CustomViTForImageClassification,
    all_layer_intermediates: List[Dict[str, List[Tensor]]],
):
    layers = model.vit.encoder.layer

    for layer_idx in range(len(layers)):
        attn_layer = layers[layer_idx].attention.attention

        def query_hook(_layer_idx):
            def hook(module, input, output):
                all_layer_intermediates[_layer_idx]["query"].append(
                    attn_layer.transpose_for_scores(output)
                )

            return hook

        def key_hook(_layer_idx):
            def hook(module, input, output):
                all_layer_intermediates[_layer_idx]["key"].append(
                    attn_layer.transpose_for_scores(output)
                )

            return hook

        attn_layer.query.register_forward_hook(query_hook(layer_idx))
        attn_layer.key.register_forward_hook(key_hook(layer_idx))


@torch.no_grad()
def extract_query_key(
    config: CustomViTConfig, dataset: IterableDataset
) -> Tuple[List[Tensor], List[Tensor]]:

    pipe = get_pipeline(config)

    all_layer_intermediates = [{"query": [], "key": []} for _ in range(12)]
    _register_qk_hook(pipe.model, all_layer_intermediates)

    result = pipe(IterableKeyDataset(dataset, "image"))  # type: ignore
    assert result is not None

    [_ for _ in tqdm(result)]

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

    return query, key
