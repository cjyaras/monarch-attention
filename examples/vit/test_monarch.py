from time import time

import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from torchtnt.utils.flops import FlopTensorDispatchMode
from transformers import AutoImageProcessor
from vit.configuration_vit import ModifiedViTConfig, ViTConfig
from vit.modeling_vit import ViTForImageClassification

ds = load_dataset("imagenet-1k", split="validation", streaming=True)
assert isinstance(ds, IterableDataset)
ds_examples = ds.take(5)
images = [item["image"] for item in ds_examples]
labels = [item["label"] for item in ds_examples]
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)
inputs = image_processor(images=images, return_tensors="pt")
base_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
k = 5

# Softmax attention
config = ModifiedViTConfig.from_dict(base_config.to_dict())
config.attention_type = "softmax"
config._attn_implementation = "eager"
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)

with torch.no_grad():
    model(**inputs)
    start = time()
    model(**inputs)
    print(f"Softmax attention time: {time() - start:.2f}s")

with FlopTensorDispatchMode(model) as ftdm:
    with torch.no_grad():
        logits = model(**inputs).logits
        softmax_labels = torch.topk(logits, k, dim=-1).indices
    softmax_flops = ftdm.flop_counts["vit.encoder.layer.0.attention"]["bmm.default"]

# Monarch sparsemax attention
config = ModifiedViTConfig.from_dict(base_config.to_dict())
assert isinstance(config, ModifiedViTConfig)
config.attention_type = "monarch"
config.attention_temperature = 10.0
config.efficient_attention_num_steps = 2
config.efficient_attention_step_size = 4e5
config.efficient_attention_block_size = 14
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)

with torch.no_grad():
    model(**inputs)
    start = time()
    model(**inputs)
    print(f"Monarch attention time: {time() - start:.2f}s")
    print()

with FlopTensorDispatchMode(model) as ftdm:
    with torch.no_grad():
        logits = model(**inputs).logits
        monarch_labels = torch.topk(logits, k, dim=-1).indices
    monarch_flops = ftdm.flop_counts["vit.encoder.layer.0.attention"]["bmm.default"]

for i in range(len(images)):
    softmax_string_labels = [
        model.config.id2label[int(softmax_labels[i, j].item())] for j in range(k)
    ]
    monarch_string_labels = [
        model.config.id2label[int(monarch_labels[i, j].item())] for j in range(k)
    ]
    print(
        f"Softmax: {softmax_string_labels} | Monarch: {monarch_string_labels} | True: {model.config.id2label[labels[i]]}"
    )
    print()

print(
    f"Monarch to Softmax FLOP ratio: {monarch_flops:0.2e}/{softmax_flops:0.2e} ({monarch_flops / softmax_flops * 100:0.2f}%)"
)
