import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor


def imagenet_dataloader(batch_size: int = 1):
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    image_processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224", use_fast=True
    )
    dataset = dataset.map(
        lambda example: {
            "image": image_processor(
                example["image"].convert("RGB"), return_tensors="pt"
            ),
            "label": example["label"],
        }
    )

    def collate_fn(examples):
        return {
            "image": {
                "pixel_values": torch.concatenate(
                    [example["image"]["pixel_values"] for example in examples]
                )
            },
            "label": torch.tensor([example["label"] for example in examples]),
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)  # type: ignore
    return dataloader
