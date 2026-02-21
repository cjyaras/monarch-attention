import os

import matplotlib.pyplot as plt
import torch

from experiments.common.baselines import (
    Cosformer,
    LinearAttention,
    Nystromformer,
    Performer,
    Softmax,
)
from ma.monarch_attention import MonarchAttention, PadType
from experiments.vit.config import get_config
from experiments.vit.extract import extract_query_key
from experiments.vit.model import AttentionType

Tensor = torch.Tensor

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans Mono",
        "font.size": 12,
    }
)


@torch.no_grad()
def main():
    if not os.path.exists("experiments/vit/query.pt"):
        config = get_config()
        config.attention_type = AttentionType.softmax
        query, key = extract_query_key(
            config,
            num_samples=1,
            batch_size=1,
            split="validation",
        )
        torch.save(query, "experiments/vit/query.pt")
        torch.save(key, "experiments/vit/key.pt")

    query = torch.load("experiments/vit/query.pt")
    key = torch.load("experiments/vit/key.pt")

    layer, head = 3, 3

    query = query[0, layer, head, 1:][None, None, ...]
    key = key[0, layer, head, 1:][None, None, ...]

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
        Nystromformer(14),
        MonarchAttention(14, 2, PadType.pre),
        Softmax(),
    ]:
        attention_maps.append(model.get_matrix(query, key)[0, 0])

    fig, ax = plt.subplots(1, len(attention_maps), figsize=(len(attention_maps) * 4, 4))

    for i, m in enumerate(attention_maps):
        ax[i].imshow(m.cpu().numpy(), cmap="GnBu")
        ax[i].set_title(model_names[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    fig.savefig("experiments/figures/attention_maps.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
