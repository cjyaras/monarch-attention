from typing import Optional

import torch
from datasets import Dataset, IterableDataset, load_dataset

from common.data import dataset_from_iterable
from common.utils import get_device, move
from vit.processor import get_processor


def get_dataset(num_samples: Optional[int] = None, split="validation") -> Dataset:
    dataset = load_dataset("imagenet-1k", split=split, streaming=True)
    assert isinstance(dataset, IterableDataset)
    dataset = dataset.take(num_samples) if num_samples is not None else dataset
    dataset = dataset_from_iterable(dataset)
    return dataset


def get_processed_dataset(
    num_samples: Optional[int] = None, split="validation"
) -> Dataset:
    device = get_device()
    dataset = get_dataset(num_samples, split)
    processor = get_processor()

    def transform(example_batch):
        inputs = processor(
            [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt"
        )
        inputs["labels"] = torch.tensor(example_batch["label"])
        return move(inputs, device)

    processed_dataset = dataset.with_transform(transform)
    return processed_dataset
