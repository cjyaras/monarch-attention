from enum import StrEnum

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast


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


def glue_dataloader(
    task_name: GlueTaskName, batch_size: int = 1, streaming: bool = True
) -> DataLoader:

    dataset = load_dataset(
        "nyu-mll/glue",
        task_name,
        split=(
            "validation_matched" if task_name == GlueTaskName.mnli else "validation"
        ),
        streaming=streaming,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]

    def preprocess_fn(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        result = tokenizer(
            *texts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        result["labels"] = example["label"]
        return result

    def collate_fn(examples):
        input_ids = torch.cat([example["input_ids"] for example in examples])
        attention_mask = torch.cat([example["attention_mask"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(preprocess_fn, batched=not streaming, remove_columns=dataset.column_names)  # type: ignore

    assert isinstance(dataset, Dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader
