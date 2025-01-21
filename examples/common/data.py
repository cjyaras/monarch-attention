from functools import partial

from datasets import Dataset, IterableDataset


class IterableKeyDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, key: str):
        self.dataset = dataset
        self.key = key

    def __iter__(self):
        for item in self.dataset:
            yield item[self.key]


def dataset_from_iterable(iterable_dataset: IterableDataset) -> Dataset:

    def gen_from_iterable_dataset(_iterable_dataset):
        yield from _iterable_dataset

    dataset = Dataset.from_generator(
        partial(gen_from_iterable_dataset, iterable_dataset),
        features=iterable_dataset.features,
    )
    assert isinstance(dataset, Dataset)
    return dataset
