from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from transformers import AutoImageProcessor
from vit.configuration_vit import ModifiedViTConfig, ViTConfig
from vit.modeling_vit import ViTForImageClassification

from sobalib.utils import calibrate_sparsemax_temperature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("imagenet-1k", split="validation", streaming=True)
assert isinstance(ds, IterableDataset)
ds_examples = ds.take(1)
images = [item["image"] for item in ds_examples]
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)

base_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
config = ModifiedViTConfig.from_dict(base_config.to_dict())
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
).to(
    device  # type: ignore
)


def register_qk_hook(
    model: nn.Module, all_layer_intermediates: List[Dict[str, torch.Tensor]]
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
optimal_temperature = calibrate_sparsemax_temperature(
    query, key, torch.linspace(1, 50, 50).to(device)
)
np.savetxt("vit/optimal_temperature.txt", optimal_temperature.cpu().numpy())
