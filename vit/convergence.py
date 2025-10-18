from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F

from ma.ma_history import monarch_attention_history

Tensor = torch.Tensor


def softmax_attention(query: Tensor, key: Tensor) -> Tensor:
    return F.softmax(query @ key.transpose(-2, -1) / query.shape[-1] ** 0.5, dim=-1)


@torch.no_grad()
def main():

    query = torch.load("vit/query.pt")
    key = torch.load("vit/key.pt")

    np.save("vit/query.npy", query.cpu().numpy())
    np.save("vit/key.npy", key.cpu().numpy())

    layer, head = 0, 5

    q = query[0, layer, head, 1:]
    k = key[0, layer, head, 1:]
    d = q.shape[-1]

    def f(a):
        return torch.sum(a * (q @ k.T)) / sqrt(d) - torch.sum(torch.xlogy(a, a))

    print(f(softmax_attention(q, k)))
    print([f(a) for a in monarch_attention_history(q, k, 5, 14)])


if __name__ == "__main__":
    main()
