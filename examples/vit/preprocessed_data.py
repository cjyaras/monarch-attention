from typing import Optional

import torch
from common.data import dataset_from_iterable
from vit.data import get_dataset
from vit.processor import get_processor


def get_preprocessed_dataset(num_samples: Optional[int] = None):
    dataset = dataset_from_iterable(get_dataset(num_samples))
    processor = get_processor()

    def transform(example_batch):
        inputs = processor(
            [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt"
        )
        inputs["labels"] = torch.tensor(example_batch["label"])
        return inputs

    preprocessed_dataset = dataset.with_transform(transform)
    return preprocessed_dataset


# model_name_or_path = "google/vit-base-patch16-224-in21k"
# processor = ViTImageProcessor.from_pretrained(model_name_or_path)


# ds = load_dataset("beans")


# def transform(example_batch):
#     # Take a list of PIL images and turn them to pixel values
#     inputs = processor([x for x in example_batch["image"]], return_tensors="pt")

#     # Don't forget to include the labels!
#     inputs["labels"] = example_batch["labels"]
#     return inputs


# prepared_ds = ds.with_transform(transform)
# print(prepared_ds["validation"][0:4]["pixel_values"].shape)
