from typing import Optional

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import HfFileSystem

from experiments.common.utils import get_device, move
from experiments.vit.processor import get_processor

REPO_ID = "ILSVRC/imagenet-1k"


def get_shards(num_samples: Optional[int], split: str) -> list[str]:
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{REPO_ID}/data/{split}-*.parquet"))
    if num_samples is None:
        return shards
    needed, num_rows = [], 0
    for shard in shards:
        needed.append(shard)
        with fs.open(shard, "rb") as f:
            num_rows += pq.ParquetFile(f).metadata.num_rows
        if num_rows >= num_samples:
            break
    return needed


def get_dataset(
    num_samples: Optional[int] = None, split: str = "validation"
) -> Dataset:
    # Download only the shards needed instead of streaming: a streamed read still
    # in flight at exit deadlocks interpreter shutdown in pyarrow's IO threads.
    data_files = [
        shard.removeprefix(f"datasets/{REPO_ID}/")
        for shard in get_shards(num_samples, split)
    ]
    dataset = load_dataset(
        REPO_ID,
        data_files={split: data_files},
        split=split,
        verification_mode="no_checks",  # only a subset of the files is loaded
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
