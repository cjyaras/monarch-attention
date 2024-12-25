import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor

from configuration_vit import ModifiedViTConfig, ViTConfig
from modeling_vit import ViTForImageClassification

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)  # type: ignore

base_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
config = ModifiedViTConfig.from_dict(base_config.to_dict())
assert isinstance(config, ModifiedViTConfig)
config.attention_type = "monarch"
config.attention_temperature = 10.0
config.efficient_attention_num_steps = 5
config.efficient_attention_step_size = 5e5
config.efficient_attention_block_size = 16

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=config
)


inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
