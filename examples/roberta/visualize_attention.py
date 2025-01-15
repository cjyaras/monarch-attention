from math import sqrt

import matplotlib.pyplot as plt
import torch
from common.utils import get_device, move
from entmax import sparsemax
from roberta.data import GlueTaskName, glue_dataloader
from roberta.models import get_config, get_model
from roberta.utils import extract_qk

from sobalib.layers import LowRankMHA, MonarchMHA, PadType


@torch.no_grad()
def main():
    device = get_device()
    inputs = move(next(iter(glue_dataloader(GlueTaskName.cola))), device)
    config = get_config()
    model = get_model(GlueTaskName.cola, config, device)

    attention_mask = inputs["attention_mask"]
    print(attention_mask)

    attention_mask_inf = (1 - attention_mask[:, None, None, :]) * -1e9

    query, key = extract_qk(model, inputs)

    # layer, head = 0, 3
    layer, head = 5, 5
    query = query[0, layer, head][None, None]
    key = key[0, layer, head][None, None]

    attn_scores = (
        query @ key.transpose(-1, -2) / sqrt(query.shape[-1]) + attention_mask_inf
    )[0, 0]
    efficient_attn = MonarchMHA(4, 50, 0.1, PadType.post)
    # efficient_attn = LowRankMHA(100, 50, 1.0)

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
