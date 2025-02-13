import os

import matplotlib.pyplot as plt
import torch

from common.baselines import Softmax
from common.soba import InitType, PadType, SobaMonarch
from vit.config import get_config
from vit.extract import extract_query_key
from vit.model import AttentionType


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

    layer = 1
    head = 5

    softmax = Softmax()
    soba_monarch = SobaMonarch(14, 2, PadType.pre, InitType.eye)

    query = query[[0], layer]
    key = key[[0], layer]

    softmax_matrix = softmax.get_matrix(query, key)[0, head]
    soba_monarch_matrix = soba_monarch.get_matrix(query, key)[0, head]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(softmax_matrix)
    ax[1].imshow(soba_monarch_matrix)
    plt.show()


if __name__ == "__main__":
    main()
