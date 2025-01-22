from functools import partial

from datasets import Dataset, IterableDataset


def dataset_from_iterable(iterable_dataset: IterableDataset) -> Dataset:

    def gen_from_iterable_dataset(_iterable_dataset):
        yield from _iterable_dataset

    dataset = Dataset.from_generator(
        partial(gen_from_iterable_dataset, iterable_dataset),
        features=iterable_dataset.features,
    )
    assert isinstance(dataset, Dataset)
    return dataset
