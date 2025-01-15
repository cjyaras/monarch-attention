from enum import StrEnum
from typing import Optional

import datasets
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast

MAX_LENGTH = 512


class GlueTaskName(StrEnum):
    cola = "cola"
    mnli = "mnli"
    mrpc = "mrpc"
    qnli = "qnli"
    qqp = "qqp"
    rte = "rte"
    sst2 = "sst2"
    stsb = "stsb"


GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}


def glue_dataset(
    task_name: GlueTaskName,
    num_samples: Optional[int] = None,
    min_length: Optional[int] = None,
) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        "nyu-mll/glue",
        task_name,
        split=(
            "validation_matched" if task_name == GlueTaskName.mnli else "validation"
        ),
    )
    assert isinstance(dataset, datasets.Dataset)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]

    def length_of(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        length = len(tokenizer(*texts)["input_ids"])
        return length

    dataset = dataset.filter(lambda example: length_of(example) <= MAX_LENGTH)

    if min_length is not None:
        dataset = dataset.filter(lambda example: length_of(example) >= min_length)

    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    def preprocess_fn(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        result = tokenizer(
            *texts,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        result["labels"] = example["label"]
        return result

    dataset = dataset.map(preprocess_fn, batched=True, remove_columns=dataset.column_names)  # type: ignore
    return dataset


if __name__ == "__main__":

    task_name = GlueTaskName.stsb
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def length_of(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        length = len(tokenizer(*texts)["input_ids"])
        return length

    dataset = datasets.load_dataset(
        "nyu-mll/glue",
        task_name,
        split=(
            "validation_matched" if task_name == GlueTaskName.mnli else "validation"
        ),
    )
    plt.hist([length_of(example) for example in dataset], bins=50)
    plt.show()


def glue_dataloader(
    task_name: GlueTaskName,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    min_length: Optional[int] = None,
) -> DataLoader:

    dataset = datasets.load_dataset(
        "nyu-mll/glue",
        task_name,
        split=(
            "validation_matched" if task_name == GlueTaskName.mnli else "validation"
        ),
    )
    assert isinstance(dataset, datasets.Dataset)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]

    def length_of(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        return len(tokenizer(*texts)["input_ids"])  # type: ignore

    dataset = dataset.filter(lambda example: length_of(example) <= MAX_LENGTH)

    if min_length is not None:
        dataset = dataset.filter(lambda example: length_of(example) >= min_length)

    def preprocess_fn(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        result = tokenizer(
            *texts,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        result["labels"] = example["label"]
        return result

    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    dataset = dataset.map(preprocess_fn, batched=True, remove_columns=dataset.column_names)  # type: ignore

    def collate_fn(examples):
        input_ids = torch.stack(
            [torch.tensor(example["input_ids"]) for example in examples], dim=0
        )
        attention_mask = torch.stack(
            [torch.tensor(example["attention_mask"]) for example in examples], dim=0
        )
        labels = torch.stack(
            [torch.tensor(example["labels"]) for example in examples], dim=0
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)  # type: ignore
    return dataloader
