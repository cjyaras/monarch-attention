from enum import StrEnum

import torch
import torch.nn as nn

from ma.ma_torch import monarch_attention_torch

try:
    from ma.ma_triton import monarch_attention_triton
except ModuleNotFoundError as e:  # Triton isn't available on every platform
    if e.name != "triton":
        raise
    monarch_attention_triton = None

IMPLEMENTATIONS = {"torch": monarch_attention_torch, "triton": monarch_attention_triton}


class PadType(StrEnum):
    pre = "pre"
    post = "post"


class MonarchAttention(nn.Module):
    """Sub-quadratic approximation of softmax attention with Monarch matrices.

    Args:
        block_size: Monarch block size B; sequences are padded to a multiple of it.
        num_steps: number of alternating optimization steps T.
        pad_type: whether padding goes before (`pre`) or after (`post`) the sequence.
        impl: `"torch"` (reference) or `"triton"` (fused CUDA kernels).

    `forward(query, key, value, attention_mask=None)` takes tensors of shape
    (batch, heads, seq_len, head_dim) and an optional (batch, seq_len) mask with
    1 for tokens to keep, and returns (batch, heads, seq_len, head_dim).
    """

    def __init__(self, block_size: int, num_steps: int, pad_type: PadType, impl: str = "torch"):
        super().__init__()
        if IMPLEMENTATIONS.get(impl) is None:
            available = ", ".join(name for name, fn in IMPLEMENTATIONS.items() if fn)
            raise ValueError(f"Unknown or unavailable impl {impl!r}. Available: {available}")
        self.block_size = block_size
        self.num_steps = num_steps
        self.pad_type = pad_type
        self._impl_fn = IMPLEMENTATIONS[impl]

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
        """The (batch, heads, seq_len, seq_len) attention matrix this module applies."""
        batch_size, num_heads, seq_len, _ = query.shape
        value = torch.eye(seq_len, device=query.device, dtype=query.dtype).expand(
            batch_size, num_heads, seq_len, seq_len
        )
        return self.forward(query, key, value, attention_mask)
