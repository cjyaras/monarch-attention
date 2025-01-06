from math import sqrt
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from entmax import sparsemax
from transformers import AutoImageProcessor
from vit.models import CustomViTConfig, CustomViTForImageClassification

from sobalib.layers import MonarchMHA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("imagenet-1k", split="validation", streaming=True)
assert isinstance(ds, IterableDataset)
ds_examples = ds.take(1)
images = [item["image"] for item in ds_examples]
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)

config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
model = CustomViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)
model = model.to(device)  # type: ignore


def register_qk_hook(
    model: CustomViTForImageClassification,
    all_layer_intermediates: List[Dict[str, torch.Tensor]],
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


all_layer_intermediates = [{} for _ in range(config.num_hidden_layers)]
register_qk_hook(model, all_layer_intermediates)

inputs = image_processor(images=images, return_tensors="pt").to(device)

with torch.no_grad():
    model(**inputs)

query = torch.stack(
    [layer_intermediates["query"] for layer_intermediates in all_layer_intermediates],
    dim=0,
).transpose(1, 0)
key = torch.stack(
    [layer_intermediates["key"] for layer_intermediates in all_layer_intermediates],
    dim=0,
).transpose(1, 0)

with torch.no_grad():
    layer, head = 5, 5
    # layer, head = 0, 10

    # query = query[0, layer, head][1:][None, None]
    # key = key[0, layer, head][1:][None, None]

    query = query[0, layer, head][None, None]
    key = key[0, layer, head][None, None]

    attn_scores = (query @ key.transpose(-1, -2) / sqrt(query.shape[-1]))[0, 0]
    efficient_attn = MonarchMHA(14, 3, 1e5, "pre")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sparsemax(attn_scores / 10.0))  # type: ignore
    ax[1].imshow(efficient_attn.get_matrix(query / 10.0, key)[0, 0])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()  # type: ignore
    plt.show()
