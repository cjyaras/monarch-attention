from typing import Optional

from datasets import Dataset, load_dataset


MAX_LENGTH = 384
DOC_STRIDE = 128


def get_dataset(
    num_samples: Optional[int] = None, split: str = "validation"
) -> Dataset:
    dataset = load_dataset("kmfoda/booksum", split=split)
    assert isinstance(dataset, Dataset)
    dataset = dataset.select(range(num_samples)) if num_samples is not None else dataset
    return dataset
