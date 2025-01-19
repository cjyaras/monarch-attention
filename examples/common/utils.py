from math import inf, sqrt
from time import time
from typing import Any, Dict, List, TypeVar

import torch
import torch.nn as nn
from entmax import sparsemax
from torch._prims_common import DeviceLikeType
from torch.nn.functional import softmax
from tqdm.auto import tqdm

T = TypeVar("T")

Tensor = torch.Tensor


def get_device() -> DeviceLikeType:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_compile(module: nn.Module):
    if torch.cuda.is_available():
        module.compile()


def move(obj: T, device: DeviceLikeType) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}  # type: ignore
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def benchmark_time(model: nn.Module, inputs: Dict[str, Tensor]) -> float:
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # type: ignore
        model(**inputs)
        end_event.record()  # type: ignore
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)
    else:
        start = time()
        model(**inputs)
        total_ms = (time() - start) * 1000

    return total_ms


def benchmark_flops(
    model: nn.Module, inputs: Dict[str, Tensor], module_name: str
) -> float:
    from torchtnt.utils.flops import FlopTensorDispatchMode

    with FlopTensorDispatchMode(model) as ftdm:
        model(**inputs)
        flops_counts = ftdm.flop_counts[module_name]["bmm.default"]

    return flops_counts


@torch.no_grad()
def calibrate_sparsemax_temperature(
    query_list: List[Tensor], key_list: List[Tensor], attention_temperature_vals: Tensor
) -> Tensor:
    """
    query_list: List of [num_layers, num_heads, seq_len, dim_per_head]
    key_list: List of [num_layers, num_heads, seq_len, dim_per_head]
    attention_temperature_vals: [num_temperatures]

    returns: [num_layers, num_heads]
    """
    num_layers, num_heads, seq_len, dim_per_head = query_list[0].shape
    differences = torch.zeros(
        num_layers,
        num_heads,
        len(attention_temperature_vals),
        device=query_list[0].device,
    )

    for query, key in tqdm(zip(query_list, key_list)):
        attn_weights = query @ key.transpose(-1, -2) / sqrt(query.size(-1))
        softmax_attn_weights = softmax(attn_weights, dim=-1)[..., None, :, :]
        sparsemax_attn_weights = sparsemax(
            attn_weights[..., None, :, :] / attention_temperature_vals[:, None, None]
        )
        attn_weights_diff = torch.flatten(
            softmax_attn_weights - sparsemax_attn_weights, start_dim=-2
        )
        differences += torch.linalg.norm(attn_weights_diff, ord=inf, dim=-1)

    optimal_temperature_idx = differences.min(dim=-1)[1]
    optimal_temperature = attention_temperature_vals[optimal_temperature_idx]
    return optimal_temperature
