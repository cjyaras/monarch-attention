from time import time

import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from torchtnt.utils.flops import FlopTensorDispatchMode
from transformers import AutoImageProcessor
from transformers.utils import logging
from vit.models import CustomViTConfig, CustomViTForImageClassification

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("imagenet-1k", split="validation", streaming=True)
assert isinstance(ds, IterableDataset)
ds_examples = ds.take(5)
images = [item["image"] for item in ds_examples]
labels = [item["label"] for item in ds_examples]
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)
inputs = image_processor(images=images, return_tensors="pt").to(device)

base_config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
k = 5

# Softmax attention
config = CustomViTConfig.from_dict(base_config.to_dict())
config.attention_type = "softmax"
config._attn_implementation = "eager"
model = CustomViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)
model = model.to(device)  # type: ignore

with torch.no_grad():
    model(**inputs)

if torch.cuda.is_available():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        model(**inputs)
        start_event.record()  # type: ignore
        model(**inputs)
        end_event.record()  # type: ignore
        torch.cuda.synchronize()
        print(f"Softmax attention time: {start_event.elapsed_time(end_event):.2f}ms")
else:

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

# Sparsemax attention
config = CustomViTConfig.from_dict(base_config.to_dict())
assert isinstance(config, CustomViTConfig)
config.attention_type = "sparsemax"
config.scale_attention_temperature = True
model = CustomViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)
model.load_state_dict(
    torch.load("vit/sparsemax_temperature.pt", weights_only=True), strict=False
)
model = model.to(device)  # type: ignore

with torch.no_grad():
    logits = model(**inputs).logits
    sparsemax_labels = torch.topk(logits, k, dim=-1).indices

# Monarch sparsemax attention
config = CustomViTConfig.from_dict(base_config.to_dict())
assert isinstance(config, CustomViTConfig)
config.attention_type = "monarch"
config.scale_attention_temperature = True
config.efficient_attention_num_steps = 3
config.efficient_attention_step_size = 1e5
config.efficient_attention_block_size = 14
model = CustomViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)
model.load_state_dict(
    torch.load("vit/sparsemax_temperature.pt", weights_only=True), strict=False
)
model = model.to(device)  # type: ignore

with torch.no_grad():
    model(**inputs)

if torch.cuda.is_available():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        model(**inputs)
        start_event.record()  # type: ignore
        model(**inputs)
        end_event.record()  # type: ignore
        torch.cuda.synchronize()
        print(f"Monarch attention time: {start_event.elapsed_time(end_event):.2f}ms")
else:
    with torch.no_grad():
        model(**inputs)
        start = time()
        model(**inputs)
        print(f"Monarch attention time: {time() - start:.2f}s")

with FlopTensorDispatchMode(model) as ftdm:
    with torch.no_grad():
        logits = model(**inputs).logits
        monarch_labels = torch.topk(logits, k, dim=-1).indices
    monarch_flops = ftdm.flop_counts["vit.encoder.layer.0.attention"]["bmm.default"]

for i in range(len(images)):
    softmax_string_labels = [
        model.config.id2label[int(softmax_labels[i, j].item())] for j in range(k)
    ]
    sparsemax_string_labels = [
        model.config.id2label[int(sparsemax_labels[i, j].item())] for j in range(k)
    ]
    monarch_string_labels = [
        model.config.id2label[int(monarch_labels[i, j].item())] for j in range(k)
    ]
    print(
        f"Softmax: {softmax_string_labels}\nSparsemax: {sparsemax_string_labels}\nMonarch: {monarch_string_labels}\nTrue: {model.config.id2label[labels[i]]}"
    )
    print()

print(
    f"Monarch to Softmax FLOP ratio: {monarch_flops:0.2e}/{softmax_flops:0.2e} ({monarch_flops / softmax_flops * 100:0.2f}%)"
)
