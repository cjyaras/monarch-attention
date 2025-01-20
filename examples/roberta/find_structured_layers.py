from math import sqrt

import matplotlib.pyplot as plt
import torch
from common.utils import get_device, move
from data import squad_dataloader
from entmax import sparsemax
from models import get_config, get_model
from tqdm.auto import tqdm
from utils import extract_qk

from sobalib.layers import MonarchMHA, PadType


def loss_fn(probs, scores):
    residual = probs - scores
    return 1 / 2 * torch.sum(torch.square(residual), dim=(-1, -2))


@torch.no_grad()
def main():
    device = get_device()

    examples = list(squad_dataloader(batch_size=1, num_samples=2))
    inputs = move(examples[1], device)
    config = get_config()
    model = get_model(config, device)

    # obj = torch.load("roberta/saved_query_key.pt", weights_only=True)
    # all_query = obj["all_query"]
    # all_key = obj["all_key"]
    # attention_mask = obj["attention_mask"]

    sparsemax_temps = torch.load("roberta/sparsemax_temperature.pt", weights_only=True)
    all_query, all_key = extract_qk(model, inputs)
    attention_mask = inputs["attention_mask"]

    # torch.save(
    #     {"all_query": all_query, "all_key": all_key, "attention_mask": attention_mask},
    #     "roberta/saved_query_key.pt",
    # )

    attention_mask_inf = (1 - attention_mask[:, None, None, :]) * -1e9
    num_valid_idx = int(torch.sum(attention_mask, dim=-1))

    def f(layer, head):
        query = (
            all_query[0, layer, head][None, None]
            / sparsemax_temps[
                f"roberta.encoder.layer.{layer}.attention.self.attention_temperature"
            ][head]
        )
        key = all_key[0, layer, head][None, None]

        attn_scores = (
            query @ key.transpose(-1, -2) / sqrt(query.shape[-1]) + attention_mask_inf
        )[0, 0][:num_valid_idx, :num_valid_idx]
        attn_probs = sparsemax(attn_scores)

        min_loss = loss_fn(attn_probs, attn_scores).item()

        learning_rate = 5.0
        num_steps = 10
        num_blocks = 16
        efficient_attn = MonarchMHA(num_blocks, num_steps, learning_rate, PadType.post)
        efficient_attn_matrix = efficient_attn.get_matrix(
            query, key, attention_mask=attention_mask, return_history=True
        )[0, 0][..., :num_valid_idx, :num_valid_idx]
        # return torch.max(torch.abs(efficient_attn_matrix - attn_probs))
        # return (
        #     torch.linalg.norm(efficient_attn_matrix - attn_probs)
        #     / torch.linalg.norm(attn_probs)
        #     * 100
        # )
        efficient_attn_matrix = efficient_attn_matrix[[0, -1]]

        init_loss, final_loss = torch.unbind(
            loss_fn(efficient_attn_matrix, attn_scores)
        )

        # return (final_loss - min_loss) / min_loss * 100

        return (final_loss - min_loss) / (init_loss - min_loss) * 100

    tolerated_error = 10

    error_vals = []
    for layer in tqdm(range(12)):
        for head in range(12):
            error_vals.append(f(layer, head))

    error_vals = torch.tensor(error_vals)

    print(torch.mean(1.0 * (error_vals < tolerated_error)))

    plt.plot(torch.arange(144) / 12, error_vals, "o")
    plt.axhline(
        tolerated_error, color="red", linestyle="--", label=f"{tolerated_error}% error"
    )
    plt.xlabel("Layer")
    plt.ylabel("Error (%)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
