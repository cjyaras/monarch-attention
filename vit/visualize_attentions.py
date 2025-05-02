import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common.baselines import Softmax
from ma.ma_history import monarch_attention_history, monarch_matrix
from vit.config import get_config
from vit.extract import extract_query_key
from vit.model import AttentionType

Tensor = torch.Tensor


def softmax_attention(query: Tensor, key: Tensor) -> Tensor:
    return F.softmax(query @ key.transpose(-2, -1) / query.shape[-1] ** 0.5, dim=-1)


@torch.no_grad()
def main():
    if not os.path.exists("vit/query.pt"):
        config = get_config()
        config.attention_type = AttentionType.softmax
        query, key = extract_query_key(
            config,
            num_samples=1,
            batch_size=1,
            split="validation",
        )
        torch.save(query, "vit/query.pt")
        torch.save(key, "vit/key.pt")

    query = torch.load("vit/query.pt")
    key = torch.load("vit/key.pt")

    # layer = 1
    # head = 5

    # layer = 3
    # head = 3

    # layer, head = 2, 10

    layer, head = 4, 2

    query = query[0, layer, head, 1:50]
    key = key[0, layer, head, 1:50]

    softmax_matrix = softmax_attention(query, key)
    monarch_matrix_history = monarch_attention_history(query, key, T=1, B=7)

    fig, ax = plt.subplots(1, len(monarch_matrix_history) + 1)
    for i, m in enumerate(monarch_matrix_history):
        ax[i].imshow(m.cpu().numpy())
        ax[i].set_title(f"Monarch (step {i+1})")
        ax[i].axis("off")
    ax[-1].imshow(softmax_matrix.cpu().numpy())
    ax[-1].set_title(f"Softmax")
    ax[-1].axis("off")
    fig.savefig("figures/monarch_history.pdf", bbox_inches="tight")

    plt.imshow(monarch_matrix_history[0].cpu().numpy())


if __name__ == "__main__":
    main()
