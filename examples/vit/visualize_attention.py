from math import sqrt

import matplotlib.pyplot as plt
import torch
from common.utils import get_device, move
from entmax import sparsemax
from vit.data import imagenet_dataloader
from vit.models import get_config, get_model
from vit.utils import extract_qk

from sobalib.layers import MonarchMHA, PadType


@torch.no_grad()
def main():
    device = get_device()
    inputs = move(next(iter(imagenet_dataloader())), device)

    config = get_config()
    model = get_model(config, device)

    query, key = extract_qk(model, inputs)

    layer, head = 0, 3
    query = query[0, layer, head][None, None]
    key = key[0, layer, head][None, None]

    attn_scores = (query @ key.transpose(-1, -2) / sqrt(query.shape[-1]))[0, 0]
    efficient_attn = MonarchMHA(14, 10, 2.5, PadType.pre)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sparsemax(attn_scores / 10.0))  # type: ignore
    ax[1].imshow(efficient_attn.get_matrix(query / 10.0, key)[0, 0])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()  # type: ignore
    plt.show()


if __name__ == "__main__":
    main()
