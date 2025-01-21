from typing import Optional

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

MAX_LENGTH = 384
DOC_STRIDE = 128
USE_SQUARE_V2 = True
MODEL_NAME = "deepset/roberta-base-squad2"


def squad_dataloader(batch_size: int = 1, num_samples: Optional[int] = None):
    dataset = datasets.load_dataset(
        "squad_v2" if USE_SQUARE_V2 else "squad", split="validation"
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    pad_on_right = tokenizer.padding_side == "right"

    def preprocess_fn(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    assert isinstance(dataset, datasets.Dataset)

    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,  # type: ignore
    )

    def collate_fn(examples):
        # TODO: Need to put back the other columns later
        input_ids = torch.stack(
            [torch.tensor(example["input_ids"]) for example in examples], dim=0
        )
        attention_mask = torch.stack(
            [torch.tensor(example["attention_mask"]) for example in examples], dim=0
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)  # type: ignore
    return dataloader


def main():
    dataloader = squad_dataloader(batch_size=2)
    print(next(iter(dataloader)))


if __name__ == "__main__":
    main()
