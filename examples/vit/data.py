from typing import Optional

from datasets import IterableDataset, load_dataset


def get_dataset(num_samples: Optional[int] = None) -> IterableDataset:
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    assert isinstance(dataset, IterableDataset)
    dataset = dataset.take(num_samples) if num_samples is not None else dataset
    return dataset
