import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from experiments.common.attention import AttentionType, get_mixed_type
from experiments.dit.model import NUM_LAYERS
from experiments.common.baselines import Softmax, Nystromformer
from experiments.dit.extract import extract_query_key


@torch.no_grad()
def main():
    if not os.path.exists("experiments/dit/query.pt"):
        efficient_attention_type = AttentionType.softmax
        query, key = extract_query_key(
            efficient_attention_type,
            words=["triceratops"],
            seed=33,
            num_inference_steps=1,
        )
        torch.save(query, "experiments/dit/query.pt")
        torch.save(key, "experiments/dit/key.pt")

    if not os.path.exists("experiments/dit/query_first_half_nystrom.pt"):
        attn_type = get_mixed_type(
            range(14), AttentionType.nystromformer, num_layers=NUM_LAYERS
        )
        query, key = extract_query_key(
            attn_type, words=["triceratops"], seed=0, num_inference_steps=1
        )
        torch.save(query, "experiments/dit/query_first_half_nystrom.pt")
        torch.save(key, "experiments/dit/key_first_half_nystrom.pt")

    layers = np.arange(14, 28)
    heads = np.arange(0, 16)

    # Look across heads in certain layer
    for layer in layers:
        query = torch.load("experiments/dit/query.pt")
        key = torch.load("experiments/dit/key.pt")

        query_nystrom = torch.load("experiments/dit/query_first_half_nystrom.pt")
        key_nystrom = torch.load("experiments/dit/key_first_half_nystrom.pt")

        query = query[[0], layer]
        key = key[[0], layer]

        query_nystrom = query_nystrom[[0], layer]
        key_nystrom = key_nystrom[[0], layer]

        softmax = Softmax()
        nystrom = Nystromformer(32, 16)

        for head in heads:
            softmax_nystrom_matrix = (
                softmax.get_matrix(query_nystrom, key_nystrom)[0, head]
                .detach()
                .cpu()
                .numpy()
            )
            nystrom_matrix = (
                nystrom.get_matrix(query_nystrom, key_nystrom)[0, head]
                .detach()
                .cpu()
                .numpy()
            )
            softmax_matrix = (
                softmax.get_matrix(query, key)[0, head].detach().cpu().numpy()
            )

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(softmax_matrix)
            ax[0].set_title("Softmax (original)")

            ax[1].imshow(softmax_nystrom_matrix)
            ax[1].set_title("Softmax (first half nystrom)")

            ax[2].imshow(nystrom_matrix)
            ax[2].set_title("Nystromformer")

            fig.suptitle(
                "Layer " + str(layer) + " head " + str(head) + " attention matrices"
            )
            plt.savefig(
                "experiments/dit/nystrom_attns/layer_"
                + str(layer)
                + "_head_"
                + str(head)
                + "_attentions.png"
            )
            plt.close()


if __name__ == "__main__":
    main()
