import jax
import requests
from flax.traverse_util import flatten_dict
from PIL import Image
from transformers import AutoImageProcessor

from configuration_vit import ModifiedViTConfig, ViTConfig
from modeling_flax_vit import (
    FlaxSequenceClassifierOutput,
    FlaxViTForImageClassification,
)

image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://farm7.staticflickr.com/6139/6023621033_e4534f0655_z.jpg",
    "https://farm4.staticflickr.com/3699/8943306698_ca46820139_z.jpg",
    "https://farm3.staticflickr.com/2826/9688908056_22512acdaf_z.jpg",
    "https://farm2.staticflickr.com/1349/4610248959_30c464a5b6_z.jpg",
    "https://farm7.staticflickr.com/6140/5926597200_ae3122bcaa_z.jpg",
]

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = image_urls[0]
image = Image.open(requests.get(url, stream=True).raw)  # type: ignore

base_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")

config = ModifiedViTConfig.from_dict(base_config.to_dict())
config.attention_type = "softmax"
# config.attention_type = "monarch"
# config.attention_temperature = 10.0
config.return_intermediates = False

image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
)
model = FlaxViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    config=config,
)
assert isinstance(model, FlaxViTForImageClassification)

inputs = image_processor(images=image, return_tensors="np")
# outputs, intermediates = model(**inputs)
outputs = model(**inputs)
assert isinstance(outputs, FlaxSequenceClassifierOutput)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
