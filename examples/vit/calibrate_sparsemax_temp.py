from math import sqrt
from typing import Dict, List

import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor
from vit.configuration_vit import ModifiedViTConfig, ViTConfig
from vit.modeling_vit import ViTForImageClassification

from sobalib.utils import calibrate_sparsemax_temperature

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://farm7.staticflickr.com/6139/6023621033_e4534f0655_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)  # type: ignore

base_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
config = ModifiedViTConfig.from_dict(base_config.to_dict())

processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
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

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    model(**inputs)

all_query = torch.cat(
    [all_layer_intermediates[layer]["query"][0, :] for layer in range(1)], dim=0
)
all_key = torch.cat(
    [all_layer_intermediates[layer]["key"][0, :] for layer in range(1)], dim=0
)
calibrate_sparsemax_temperature(all_query, all_key)
