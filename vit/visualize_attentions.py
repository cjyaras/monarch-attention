import os
from math import sqrt

import matplotlib.pyplot as plt
import torch

from common.baselines import Softmax
from common.soba import PadType, SobaMonarch
from vit.config import get_config
from vit.extract import extract_query_key
from vit.model import AttentionType

Tensor = torch.Tensor


def softmax_varational_objective(P: Tensor, Z: Tensor) -> Tensor:
    return torch.sum(torch.special.xlogy(P, P)) - torch.sum(P * Z)


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

    layer = 5
    head = 0

    query = query[0:1, layer, head : head + 1, 1:]
    key = key[0:1, layer, head : head + 1, 1:]

    # attention_mask = torch.tensor([[1] * (14 * 7) + [0] * (14 * 7)])
    attention_mask = None
    softmax_matrix = Softmax().get_matrix(query, key, attention_mask=attention_mask)
    soba_monarch_matrix = SobaMonarch(14, 2, PadType.pre).get_matrix(
        query, key, attention_mask=attention_mask
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(softmax_matrix[0, 0])
    ax[1].imshow(soba_monarch_matrix[0, 0])
    plt.show()

    # softmax_matrix = Softmax().get_matrix(query, key)
    # soba_monarch_matrix = SobaMonarch(14, 4, PadType.pre).get_matrix(
    #     query, key, return_history=True
    # )
    # ones_soba_monarch_matrix = SobaMonarch(
    #     14, 4, PadType.pre, InitType.ones
    # ).get_matrix(query, key, return_history=True)

    # pre_attn_scores = query @ key.transpose(-2, -1) / sqrt(query.shape[-1])

    # softmax_value = float(softmax_varational_objective(softmax_matrix, pre_attn_scores))

    # eye_soba_monarch_value = torch.vmap(
    #     softmax_varational_objective, in_dims=(0, None)
    # )(eye_soba_monarch_matrix, pre_attn_scores)

    # plt.plot(eye_soba_monarch_value, label="Eye")
    # plt.plot(ones_soba_monarch_value, label="Ones")
    # plt.axhline(softmax_value, color="red", label="Softmax")
    # plt.legend()
    # plt.show()

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(softmax_matrix)
    # ax[1].imshow(eye_soba_monarch_matrix[-1])
    # ax[1].set_title(f"Eye")
    # ax[2].imshow(ones_soba_monarch_matrix[-1])
    # ax[2].set_title(f"Ones")
    # plt.show()


if __name__ == "__main__":
    main()
