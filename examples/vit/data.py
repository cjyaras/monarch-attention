from typing import Optional

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessorFast


def imagenet_dataloader(
    batch_size: int = 1, streaming: bool = True, num_samples: Optional[int] = None
) -> DataLoader:
    dataset = datasets.load_dataset(
        "imagenet-1k", split="validation", streaming=streaming
    )
    image_processor = ViTImageProcessorFast.from_pretrained(
        "google/vit-base-patch16-224"
    )

    def preprocess_fn(example):
        image = example["image"].convert("RGB")
        result = image_processor(image, return_tensors="pt")
        result["labels"] = example["label"]
        return result

    if num_samples is not None:
        if streaming:
            assert isinstance(dataset, datasets.IterableDataset)
            dataset = dataset.take(num_samples)
        else:
            assert isinstance(dataset, datasets.Dataset)
            dataset = dataset.select(range(num_samples))

    dataset = dataset.map(
        preprocess_fn, batched=not streaming, remove_columns=dataset.column_names  # type: ignore
    )

    def collate_fn(examples):
        pixel_values = torch.cat([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    assert isinstance(dataset, Dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader
