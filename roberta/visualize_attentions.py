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
    torch.manual_seed(0)
    if not os.path.exists("roberta/query.pt"):
        config = get_config()
        config.attention_type = AttentionType.softmax
        query, key, attention_mask = extract_query_key_mask(
            config,
            num_samples=4,
            batch_size=4,
            split="validation",
        )
        torch.save(query, "roberta/query.pt")
        torch.save(key, "roberta/key.pt")
        torch.save(attention_mask, "roberta/attention_mask.pt")

    query = torch.load("roberta/query.pt")
    key = torch.load("roberta/key.pt")
    attention_mask = torch.load("roberta/attention_mask.pt")

    # layer = 1
    # head = 5

    # layer = 3
    # head = 3

    # layer, head = 2, 10

    layer, head = 5, 5

    seq_len = torch.sum(attention_mask[0])

    query = query[0, layer, head, :seq_len][None, None, ...]
    key = key[0, layer, head, :seq_len][None, None, ...]

    attention_maps = []

    model_names = [
        "cosformer",
        "linear-attention",
        "performer",
        "nystromformer",
        "monarch-attention",
        "softmax",
    ]

    for model in [
        Cosformer(),
        LinearAttention(),
        Performer(64),
        Nystromformer(32),
        MonarchAttention(32, 2, PadType.pre),
        Softmax(),
    ]:
        attention_maps.append(model.get_matrix(query, key)[0, 0])

    fig, ax = plt.subplots(1, len(attention_maps), figsize=(len(attention_maps) * 4, 4))

    for i, m in enumerate(attention_maps):
        ax[i].imshow(m.cpu().numpy()[25:75, 25:75], cmap="cividis")
        ax[i].set_title(model_names[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    fig.savefig("figures/roberta_attention_maps.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
