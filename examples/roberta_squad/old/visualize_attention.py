from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from common.utils import get_device, move
from data import squad_dataloader
from entmax import sparsemax
from models import get_config, get_model
from utils import extract_qk

from sobalib.layers import LowRankMHA, MonarchBlockDiagonalMHA, MonarchMHA, PadType


@torch.no_grad()
def main():
    device = get_device()

    examples = list(squad_dataloader(batch_size=1, num_samples=2))
    inputs = move(examples[1], device)
    config = get_config()
    model = get_model(config, device)

    sparsemax_temps = torch.load("roberta/sparsemax_temperature.pt", weights_only=True)
    query, key = extract_qk(model, inputs)

    layer, head = 5, 5

    attention_mask = inputs["attention_mask"]
    print(attention_mask.sum(-1))
    attention_mask_inf = (1 - attention_mask[:, None, None, :]) * -1e9

    query = (
        query[0, layer, head][None, None]
        / sparsemax_temps[
            f"roberta.encoder.layer.{layer}.attention.self.attention_temperature"
        ][head]
    )
    key = key[0, layer, head][None, None]

    attn_scores = (
        query @ key.transpose(-1, -2) / sqrt(query.shape[-1]) + attention_mask_inf
    )[0, 0]
    efficient_attn = MonarchBlockDiagonalMHA(16, 20, 2.5, PadType.pre)
    # efficient_attn = LowRankMHA(10, 5, 1.0)

    # print(torch.linalg.svdvals(sparsemax(attn_scores)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sparsemax(attn_scores))  # type: ignore
    ax[1].imshow(
        efficient_attn.get_matrix(query, key, attention_mask=attention_mask)[0, 0]
    )
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()  # type: ignore
    plt.show()


if __name__ == "__main__":
    main()
