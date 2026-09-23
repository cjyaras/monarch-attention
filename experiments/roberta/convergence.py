
import matplotlib.pyplot as plt
import torch


Tensor = torch.Tensor

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans Mono",
        "font.size": 12,
    }
)


@torch.no_grad()
def main():
    query = torch.load("experiments/roberta/query.pt")
    key = torch.load("experiments/roberta/key.pt")
    attention_mask = torch.load("experiments/roberta/attention_mask.pt")

    layer, head = 5, 5

    seq_len = torch.sum(attention_mask[0])

    query = query[0, layer, head, :seq_len][None, None, ...]
    key = key[0, layer, head, :seq_len][None, None, ...]



if __name__ == "__main__":
    main()
