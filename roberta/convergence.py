import os

import matplotlib.pyplot as plt
import torch

from common.baselines import (
    Cosformer,
    LinearAttention,
    Nystromformer,
    Performer,
    Softmax,
)
from ma.monarch_attention import MonarchAttention, PadType
from roberta.config import get_config
from roberta.extract import extract_query_key_mask
from roberta.model import AttentionType

Tensor = torch.Tensor

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans Mono",
        "font.size": 12,
    }
)


@torch.no_grad()
def main():
    query = torch.load("roberta/query.pt")
    key = torch.load("roberta/key.pt")
    attention_mask = torch.load("roberta/attention_mask.pt")

    layer, head = 5, 5

    seq_len = torch.sum(attention_mask[0])

    query = query[0, layer, head, :seq_len][None, None, ...]
    key = key[0, layer, head, :seq_len][None, None, ...]



if __name__ == "__main__":
    main()
