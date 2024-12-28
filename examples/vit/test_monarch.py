import requests
import torch
from PIL import Image
from torchtnt.utils.flops import FlopTensorDispatchMode
from transformers import AutoImageProcessor

from examples.vit.configuration_vit import ModifiedViTConfig, ViTConfig
from examples.vit.modeling_vit import ViTForImageClassification

image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://farm7.staticflickr.com/6139/6023621033_e4534f0655_z.jpg",
    "https://farm4.staticflickr.com/3699/8943306698_ca46820139_z.jpg",
    "https://farm3.staticflickr.com/2826/9688908056_22512acdaf_z.jpg",
    "https://farm2.staticflickr.com/1349/4610248959_30c464a5b6_z.jpg",
    "https://farm7.staticflickr.com/6140/5926597200_ae3122bcaa_z.jpg",
]
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]  # type: ignore
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)
inputs = image_processor(images=images, return_tensors="pt")

base_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")

# Softmax attention
config = ModifiedViTConfig.from_dict(base_config.to_dict())
config.attention_type = "softmax"
config._attn_implementation = "eager"
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)
with FlopTensorDispatchMode(model) as ftdm:
    with torch.no_grad():
        logits = model(**inputs).logits
        softmax_labels = logits.argmax(-1)
    softmax_flops = ftdm.flop_counts["vit.encoder.layer.0.attention"]["bmm.default"]

# Monarch sparsemax attention
config = ModifiedViTConfig.from_dict(base_config.to_dict())
config.attention_type = "monarch"
config.attention_temperature = 10.0
config.efficient_attention_num_steps = 2
config.efficient_attention_step_size = 4e5
config.efficient_attention_block_size = 14
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)

with FlopTensorDispatchMode(model) as ftdm:
    with torch.no_grad():
        logits = model(**inputs).logits
        monarch_labels = logits.argmax(-1)
    monarch_flops = ftdm.flop_counts["vit.encoder.layer.0.attention"]["bmm.default"]

for i in range(len(image_urls)):
    print(f"URL: {image_urls[i]}")
    print(
        f"Softmax: {model.config.id2label[softmax_labels[i].item()]} | Monarch: {model.config.id2label[monarch_labels[i].item()]}"
    )
    print()

print(
    f"Monarch to Softmax FLOP ratio: {monarch_flops:0.2e}/{softmax_flops:0.2e} ({monarch_flops / softmax_flops * 100:0.2f}%)"
)
