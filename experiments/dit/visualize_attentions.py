from typing import List, Dict
import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from experiments.common.baselines import Softmax, Nystromformer
from ma.monarch_attention import MonarchAttention, PadType
from experiments.dit.extract import extract_query_key
from experiments.dit.config import AttentionType


def generate_attn_dict(attn_type: AttentionType, layers_to_replace: List, num_layers: int = 28) -> Dict:
    assert max(layers_to_replace) <= num_layers

    attn_dict = {}
    for i in range(1, num_layers + 1):
        attn_dict[i] = attn_type if i in layers_to_replace else AttentionType.softmax

    return attn_dict

@torch.no_grad()
def main():
    if not os.path.exists("experiments/dit/query.pt"):
        efficient_attention_type = AttentionType.softmax
        query, key = extract_query_key(
            efficient_attention_type,
            words=["triceratops"],
            seed=33,
            num_inference_steps=1
        )
        torch.save(query, "experiments/dit/query.pt")
        torch.save(key, "experiments/dit/key.pt")

    if not os.path.exists("dit/query_first_half_nystrom.pt"):
        attn_type = generate_attn_dict(AttentionType.nystromformer, list(range(1, 15)))
        query, key = extract_query_key(
            attn_type,
            words=["triceratops"],
            seed=0,
            num_inference_steps=1
        )
        torch.save(query, "dit/query_first_half_nystrom.pt")
        torch.save(key, "dit/key_first_half_nystrom.pt")

    layers = np.arange(14, 28)
    heads = np.arange(0, 16)

    # Look across heads in certain layer
    for layer in layers:
        query = torch.load("experiments/dit/query.pt")
        key = torch.load("experiments/dit/key.pt")

        query_nystrom = torch.load("dit/query_first_half_nystrom.pt")
        key_nystrom = torch.load("dit/key_first_half_nystrom.pt")

        query = query[[0], layer]
        key = key[[0], layer]

        query_nystrom = query_nystrom[[0], layer]
        key_nystrom = key_nystrom[[0], layer]

        softmax = Softmax()
        #monarch = MonarchAttention(16, 3, PadType.pre)
        nystrom = Nystromformer(32, 16)

        for head in heads:
            softmax_nystrom_matrix = softmax.get_matrix(query_nystrom, key_nystrom)[0, head].detach().cpu().numpy()
            #monarch_matrix = monarch.get_matrix(query, key)[0, head].detach().cpu().numpy()
            nystrom_matrix = nystrom.get_matrix(query_nystrom, key_nystrom)[0, head].detach().cpu().numpy()
            softmax_matrix = softmax.get_matrix(query, key)[0, head].detach().cpu().numpy()

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(softmax_matrix)
            ax[0].set_title('Softmax (original)')

            ax[1].imshow(softmax_nystrom_matrix)
            ax[1].set_title('Softmax (first half nystrom)')

            #ax[1].imshow(monarch_matrix)
            #ax[1].set_title('Monarch')

            ax[2].imshow(nystrom_matrix)
            ax[2].set_title('Nystromformer')

            fig.suptitle('Layer ' + str(layer) + ' head ' + str(head) + ' attention matrices')
            plt.savefig('./dit/nystrom_attns/layer_' + str(layer) + '_head_' + str(head) + '_attentions.png')
            plt.close()


if __name__ == "__main__":
    main()
