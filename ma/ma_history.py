from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange

Tensor = torch.Tensor


def monarch_matrix(L: Tensor, R: Tensor) -> Tensor:
    out = torch.einsum("jkl,kji->ljki", L, R)
    return rearrange(out, "l j k i -> (l j) (k i)")


def monarch_attention_history(q: Tensor, k: Tensor, T: int, B: int) -> list[Tensor]:
    N, D = q.shape
    M = N // B

    q = q / sqrt(D)

    qb = rearrange(q, "(l j) v -> j l v", j=B)
    kb = rearrange(k, "(k i) v -> k i v", i=B)

    L = torch.stack(B * [torch.eye(M, device=q.device)])

    history = []

    # Alternating maximization for L, R
    for t in range(T):
        # R update
        aR = torch.einsum("jkl,jlv->kjv", L, qb)
        bR = torch.einsum("kjv,kiv->kji", aR, kb)
        cR = torch.einsum("jkl->kj", L)
        R = F.softmax(bR / cR[:, :, None], dim=2)

        history.append(monarch_matrix(L, R))

        # L update
        aL = torch.einsum("kji,kiv->jkv", R, kb)
        bL = torch.einsum("jkv,jlv->jkl", aL, qb)
        cL = torch.einsum("kji->jk", R * torch.log(R))
        L = F.softmax(bL - cL[:, :, None], dim=1)

        history.append(monarch_matrix(L, R))

    return history
