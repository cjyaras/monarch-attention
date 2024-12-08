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

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)  # type: ignore

base_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")

config = ModifiedViTConfig.from_dict(base_config.to_dict())
config.attention_type = "sparsemax"
config.attention_temperature = 10.0
config.output_intermediates = True

image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
)
model = FlaxViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    config=config,
)
assert isinstance(model, FlaxViTForImageClassification)

inputs = image_processor(images=image, return_tensors="np")
outputs, intermediates = model(**inputs)
print(flatten_dict(intermediates).keys())
assert isinstance(outputs, FlaxSequenceClassifierOutput)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
