import os
from glob import glob
from typing import Optional

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, constants, hf_hub_download, snapshot_download

from experiments.common.utils import get_device, move
from experiments.vit.processor import get_processor

REPO_ID = "ILSVRC/imagenet-1k"


def get_shards(num_samples: Optional[int], split: str) -> list[str]:
    """Local paths of the first parquet shards covering `num_samples` rows (all if None).

    Shards are downloaded into the Hugging Face cache; offline, only cached shards are used.
    """
    if constants.HF_HUB_OFFLINE:
        snapshot = snapshot_download(REPO_ID, repo_type="dataset")
        names = [
            os.path.relpath(path, snapshot)
            for path in glob(os.path.join(snapshot, "data", f"{split}-*.parquet"))
        ]
    else:
        names = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        names = [name for name in names if name.startswith(f"data/{split}-")]

    shards, num_rows = [], 0
    for name in sorted(names):
        shard = hf_hub_download(REPO_ID, name, repo_type="dataset")
        shards.append(shard)
        num_rows += pq.ParquetFile(shard).metadata.num_rows
        if num_samples is not None and num_rows >= num_samples:
            return shards
    if num_samples is not None:
        raise ValueError(
            f"Only {num_rows} {split} samples available, need {num_samples}"
        )
    return shards


def get_dataset(
    num_samples: Optional[int] = None, split: str = "validation"
) -> Dataset:
    # Read local shards instead of streaming: a streamed read still in flight at
    # exit deadlocks interpreter shutdown in pyarrow's IO threads.
    dataset = load_dataset(
        "parquet", data_files={split: get_shards(num_samples, split)}, split=split
    )
    assert isinstance(dataset, Dataset)
    dataset = dataset.select(range(num_samples)) if num_samples is not None else dataset
    return dataset


def get_processed_dataset(
    num_samples: Optional[int] = None, split: str = "validation"
) -> Dataset:
    device = get_device()
    dataset = get_dataset(num_samples, split)
    processor = get_processor()

    def transform(example_batch):
        inputs = processor(
            [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt"
        )
        return move(inputs, device)

    processed_dataset = dataset.with_transform(transform)
    return processed_dataset
