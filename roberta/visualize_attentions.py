import os

import matplotlib.pyplot as plt
import torch

from common.baselines import Softmax
from common.soba import InitType, PadType, SobaMonarch
from roberta.config import get_config
from roberta.extract import extract_query_key_mask
from roberta.model import AttentionType


@torch.no_grad()
def main():
    if not os.path.exists("vit/query.pt"):
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

    layer = 5
    head = 0

    softmax = Softmax()
    soba_monarch = SobaMonarch(24, 3, PadType.pre, InitType.ones)

    query = query[[0], layer]
    key = key[[0], layer]
    attention_mask = attention_mask[[0]]

    seq_len = torch.sum(attention_mask)

    softmax_matrix = softmax.get_matrix(query, key, attention_mask)[0, head]
    soba_monarch_matrix = soba_monarch.get_matrix(query, key, attention_mask)[0, head]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(softmax_matrix[:seq_len, :seq_len])
    ax[1].imshow(soba_monarch_matrix[:seq_len, :seq_len])
    plt.show()


if __name__ == "__main__":
    main()
