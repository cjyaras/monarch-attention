from typing import Optional

from datasets import Dataset, load_dataset

from experiments.common.utils import get_device, move
from experiments.roberta.processor import get_processor

MAX_LENGTH = 384
DOC_STRIDE = 128


def get_dataset(
    num_samples: Optional[int] = None, split: str = "validation"
) -> Dataset:
    dataset = load_dataset("kmfoda/booksum", split=split)
    assert isinstance(dataset, Dataset)
    dataset = dataset.select(range(num_samples)) if num_samples is not None else dataset
    return dataset


# def get_processed_dataset(
#    num_samples: Optional[int] = None, split: str = "validation"
# ) -> Dataset:
#    device = get_device()
#    dataset = get_dataset(num_samples, split=split)
#    processor = get_processor()
#
#    pad_on_right = processor.padding_side == "right"
#
#    def transform(example_batch):
#        example_batch["question"] = [q.lstrip() for q in example_batch["question"]]
#
#        inputs = processor(
#            example_batch["question" if pad_on_right else "context"],
#            example_batch["context" if pad_on_right else "question"],
#            truncation="only_second" if pad_on_right else "only_first",
#            max_length=MAX_LENGTH,
#            stride=DOC_STRIDE,
#            padding="max_length",
#            return_tensors="pt",
#        )
#
#        return move(inputs, device)
#
#    processed_dataset = dataset.with_transform(transform)
#    return processed_dataset
