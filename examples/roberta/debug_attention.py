from math import sqrt

import matplotlib.pyplot as plt
import torch
from common.utils import get_device, move
from data import squad_dataloader
from entmax import sparsemax
from models import get_config, get_model
from utils import extract_qk

from sobalib.layers import LowRankMHA, MonarchBlockDiagonalMHA, MonarchMHA, PadType


def loss_fn(mat, target):
    residual = mat - target
    return 1 / 2 * torch.mean(torch.square(residual), dim=(-1, -2))


@torch.no_grad()
def main():
    device = get_device()

    examples = list(squad_dataloader(batch_size=1, num_samples=2))
    inputs = move(examples[1], device)
    config = get_config()
    model = get_model(config, device)

    sparsemax_temps = torch.load("roberta/sparsemax_temperature.pt", weights_only=True)
    query, key = extract_qk(model, inputs)

    layer, head = 2, 10

    attention_mask = inputs["attention_mask"]
    num_valid_idx = int(torch.sum(attention_mask, dim=-1))
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
    )[0, 0][:num_valid_idx, :num_valid_idx]
    attn_probs = sparsemax(attn_scores)

    true_loss_val = loss_fn(attn_probs, attn_scores).item()

    for learning_rate in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        num_blocks = 16
        num_steps = 40
        efficient_attn = MonarchMHA(num_blocks, num_steps, learning_rate, PadType.post)
        efficient_attn_matrix = efficient_attn.get_matrix(
            query, key, attention_mask=attention_mask, return_history=True
        )[0, 0][..., :num_valid_idx, :num_valid_idx]

        loss_vals = loss_fn(efficient_attn_matrix, attn_scores)
        plt.plot(loss_vals, label=f"learning_rate={learning_rate}", alpha=0.6)
        plt.legend()

    plt.axhline(true_loss_val, color="red", linestyle="--")
    plt.show()
    # exit()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(attn_probs)  # type: ignore
    ax[1].imshow(efficient_attn_matrix[-1])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()  # type: ignore
    plt.show()


if __name__ == "__main__":
    main()
