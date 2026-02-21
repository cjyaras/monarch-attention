from collections.abc import Callable
from enum import StrEnum

import torch
import torch.nn as nn

from ma.ma_torch import monarch_attention_torch

Tensor = torch.Tensor

MonarchAttentionFn = Callable[
    [Tensor, Tensor, Tensor, Tensor | None, int, int, bool], Tensor
]

_IMPLEMENTATIONS: dict[str, MonarchAttentionFn] = {}


def register_impl(name: str, fn: MonarchAttentionFn) -> None:
    _IMPLEMENTATIONS[name] = fn


register_impl("torch", monarch_attention_torch)

try:
    from ma.ma_triton import monarch_attention_triton

    register_impl("triton", monarch_attention_triton)
except ImportError:
    pass


class PadType(StrEnum):
    pre = "pre"
    post = "post"


class MonarchAttention(nn.Module):

    def __init__(self, block_size, num_steps, pad_type, impl="torch"):
        super().__init__()
        self.block_size = block_size
        self.num_steps = num_steps
        self.pad_type = pad_type

        if impl not in _IMPLEMENTATIONS:
            available = ", ".join(sorted(_IMPLEMENTATIONS))
            raise ValueError(f"Unknown impl {impl!r}. Available: {available}")
        self._impl_fn = _IMPLEMENTATIONS[impl]

    def forward(self, query, key, value, attention_mask=None):
        return self._impl_fn(
            query,
            key,
            value,
            attention_mask,
            self.num_steps,
            self.block_size,
            self.pad_type == PadType.pre,
        )

    def get_matrix(self, query, key, attention_mask=None):
        batch_size, num_heads, seq_len, _ = query.shape
        value = torch.eye(seq_len, device=query.device).expand(
            batch_size, num_heads, seq_len, seq_len
        )
        return self.forward(query, key, value, attention_mask)
