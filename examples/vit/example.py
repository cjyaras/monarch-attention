from time import time
from typing import List

import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from torchtnt.utils.flops import FlopTensorDispatchMode
from transformers import ViTImageProcessor, pipeline
from transformers.utils import logging
from vit.models import CustomViTConfig, CustomViTForImageClassification

logging.set_verbosity(logging.CRITICAL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("imagenet-1k", split="validation", streaming=True)
assert isinstance(ds, IterableDataset)
ds_examples = ds.take(5)
images = [item["image"] for item in ds_examples]
labels = [item["label"] for item in ds_examples]

image_processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)

base_config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")

top_k = 3

# Softmax

config = CustomViTConfig.from_dict(base_config.to_dict())
config.attention_type = "softmax"
config._attn_implementation = "eager"
model = CustomViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)
model = model.to(device)  # type: ignore
classifier = pipeline(
    "image-classification", model=model, image_processor=image_processor, device=device
)

with torch.no_grad():
    classifier(images)

with FlopTensorDispatchMode(model) as ftdm:

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()  # type: ignore
            softmax_outputs = classifier(images)
            end_event.record()  # type: ignore
            torch.cuda.synchronize()
            print(
                f"Softmax attention time: {start_event.elapsed_time(end_event):.2f}ms"
            )

    else:

        with torch.no_grad():
            start = time()
            softmax_outputs = classifier(images)
            print(f"Softmax attention time: {time() - start:.2f}s")

    softmax_flops = ftdm.flop_counts["vit.encoder.layer.0.attention"]["bmm.default"]

# Monarch sparsemax

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

classifier = pipeline(
    "image-classification", model=model, image_processor=image_processor, device=device
)

with torch.no_grad():
    classifier(images)

with FlopTensorDispatchMode(model) as ftdm:

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()  # type: ignore
            monarch_outputs = classifier(images)
            end_event.record()  # type: ignore
            torch.cuda.synchronize()
            print(
                f"Monarch attention time: {start_event.elapsed_time(end_event):.2f}ms"
            )

    else:

        with torch.no_grad():
            start = time()
            monarch_outputs = classifier(images)
            print(f"Monarch attention time: {time() - start:.2f}s")

    monarch_flops = ftdm.flop_counts["vit.encoder.layer.0.attention"]["bmm.default"]

assert isinstance(softmax_outputs, List)
assert isinstance(monarch_outputs, List)

for i in range(len(images)):
    print("Softmax:")
    print(softmax_outputs[i][:top_k])
    print("Monarch:")
    print(monarch_outputs[i][:top_k])
    print(f"True: {model.config.id2label[labels[i]]}")
    print()

print(
    f"Monarch to Softmax FLOP ratio: {monarch_flops:0.2e}/{softmax_flops:0.2e} ({monarch_flops / softmax_flops * 100:0.2f}%)"
)
