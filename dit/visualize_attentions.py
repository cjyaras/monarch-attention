import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from common.baselines import Softmax
from common.soba import InitType, PadType, SobaMonarch
from dit.config import get_config
from dit.extract import extract_query_key
from dit.config import AttentionType


@torch.no_grad()
def main():
    if not os.path.exists("dit/query.pt"):
        efficient_attention_type = AttentionType.softmax
        query, key = extract_query_key(
            efficient_attention_type,
            words=["triceratops"],
            seed=33,
            num_inference_steps=1
        )
        torch.save(query, "dit/query.pt")
        torch.save(key, "dit/key.pt")

    layers = np.arange(0, 28)
    heads = np.arange(5, 6)

    # Look across heads in certain layer
    for layer in layers:
        query = torch.load("dit/query.pt")
        key = torch.load("dit/key.pt")

        query = query[[0], layer]
        key = key[[0], layer]

        softmax = Softmax()
        soba_monarch_ones = SobaMonarch(16, 3, PadType.pre, InitType.ones)
        soba_monarch_eye = SobaMonarch(16, 3, PadType.pre, InitType.eye)

        for head in heads:
            softmax_matrix = softmax.get_matrix(query, key)[0, head]
            soba_monarch_ones_matrix = soba_monarch_ones.get_matrix(query, key)[0, head]
            soba_monarch_eye_matrix = soba_monarch_eye.get_matrix(query, key)[0, head]

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(softmax_matrix)
            ax[0].set_title('Softmax')

            ax[1].imshow(soba_monarch_ones_matrix)
            ax[1].set_title('SOBA (ones init)')

            ax[2].imshow(soba_monarch_eye_matrix)
            ax[2].set_title('SOBA (Identity init)')

            fig.suptitle('Layer ' + str(layer) + ' head ' + str(head) + ' attention matrices')
            plt.show()


if __name__ == "__main__":
    main()
