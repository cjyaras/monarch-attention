from contextlib import contextmanager
from typing import Dict, List, TypeVar

import torch
from transformers.image_processing_base import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding

T = TypeVar("T")

Tensor = torch.Tensor


def move(obj: T, device) -> T:
    if isinstance(obj, (Tensor, BatchFeature, BatchEncoding)):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, List):
        return [move(v, device) for v in obj]  # type: ignore
    elif isinstance(obj, Dict):
        return {k: move(v, device) for k, v in obj.items()}  # type: ignore
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batches(dataset, batch_size: int):
    """Yield consecutive slices of a datasets.Dataset as dicts of columns."""
    for start in range(0, len(dataset), batch_size):
        yield dataset[start : start + batch_size]


def attention_bmm_flops(model, module_names: list[str], run) -> int:
    """bmm FLOPs spent inside the named attention modules while calling `run()`."""
    from torchtnt.utils.flops import FlopTensorDispatchMode

    with FlopTensorDispatchMode(model) as ftdm:
        run()
        return sum(ftdm.flop_counts[name]["bmm.default"] for name in module_names)


@contextmanager
def capture_attention_inputs(attn_modules):
    """Record the query, key and padding mask (1 = keep) passed to each attention module.

    Yields one dict of lists per module; use `stack_captured` to combine them.
    """
    records = [{"query": [], "key": [], "attention_mask": []} for _ in attn_modules]

    def make_hook(record):
        def hook(module, args, output):
            query, key, _, attention_mask = args
            if attention_mask is None:
                attention_mask = query.new_ones(query.shape[0], query.shape[2])
            record["query"].append(query)
            record["key"].append(key)
            record["attention_mask"].append(attention_mask)

        return hook

    handles = [m.register_forward_hook(make_hook(r)) for m, r in zip(attn_modules, records)]
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def stack_captured(records, name: str) -> Tensor:
    """(num_samples, num_layers, ...) tensor of one captured input."""
    return torch.stack([torch.cat(record[name]) for record in records]).transpose(1, 0)
