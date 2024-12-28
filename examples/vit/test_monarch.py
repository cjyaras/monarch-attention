import requests
import torch
from PIL import Image
from torchtnt.utils.flops import FlopTensorDispatchMode
from transformers import AutoImageProcessor

from examples.vit.configuration_vit import ModifiedViTConfig, ViTConfig
from examples.vit.modeling_vit import ViTForImageClassification

# url = "https://farm7.staticflickr.com/6140/5926597200_ae3122bcaa_z.jpg"
url = "https://farm3.staticflickr.com/2826/9688908056_22512acdaf_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)  # type: ignore

base_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
config = ModifiedViTConfig.from_dict(base_config.to_dict())

assert isinstance(config, ModifiedViTConfig)
# config.attention_type = "low-rank"
# config.attention_type = "sparsemax"
config.attention_type = "monarch"
config.attention_temperature = 10.0
config.efficient_attention_num_steps = 2
config.efficient_attention_step_size = 4e5
config.efficient_attention_block_size = 14
config.efficient_attention_rank = 14

processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)


inputs = processor(images=image, return_tensors="pt")

with FlopTensorDispatchMode(model) as ftdm:
    with torch.no_grad():
        logits = model(**inputs).logits
    print(ftdm.flop_counts["vit.encoder.layer.0.attention"])

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
