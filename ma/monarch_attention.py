from enum import StrEnum

import torch
import torch.nn as nn

from ma.ma_torch import monarch_attention_torch
from ma.ma_triton import monarch_attention_triton

Tensor = torch.Tensor


class PadType(StrEnum):
    pre = "pre"
    post = "post"


class MonarchAttention(nn.Module):

    def __init__(
        self,
        block_size: int,
        num_steps: int,
        pad_type: PadType,
        impl: str | None = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_steps = num_steps
        self.pad_type = pad_type

        # Triton version is giving incorrect results at the moment, setting to torch implementation
        impl = "torch"

        # if impl is None:
        #     impl = "triton" if torch.cuda.is_available() else "torch"

        self.impl = impl

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return (
            monarch_attention_triton
            if self.impl == "triton"
            else monarch_attention_torch
        )(
            query,
            key,
            value,
            attention_mask,
            self.num_steps,
            self.block_size,
            self.pad_type == PadType.pre,
        )

    def get_matrix(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape
        value = torch.eye(seq_len, device=query.device).expand(
            batch_size, num_heads, seq_len, seq_len
        )
        return self.forward(query, key, value, attention_mask)
