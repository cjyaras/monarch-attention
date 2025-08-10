from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
xlogy = torch.special.xlogy


def al_cl_ref(ar, k, cr, sm_scale, mask, eps=1e-12):
    r_hat = sm_scale * (ar @ k.transpose(-1, -2)).to(torch.float)
    r_hat = r_hat / (cr[..., :, None] + eps)
    r_hat = r_hat + torch.where(mask[..., None, :], 0.0, -float("inf"))
    r_hat = torch.exp(
        r_hat - torch.clamp(torch.max(r_hat, dim=-1, keepdim=True).values, min=eps)
    )
    r = r_hat / (torch.sum(r_hat, dim=-1, keepdim=True) + eps)
    r = torch.clamp(r, min=torch.finfo(r.dtype).tiny)

    cl = torch.sum(xlogy(r, r), dim=-1).transpose(-1, -2)
    al = sm_scale * (r.to(k.dtype) @ k).transpose(-2, -3)

    return al, cl


def ar_cr_ref(al, q, cl, mask_t):
    l_hat = (al @ q.transpose(-1, -2)).to(torch.float)
    l_hat = l_hat - cl[..., :, None]
    l = F.softmax(l_hat, dim=-2)
    l = mask_t[..., None, :] * l

    cr = torch.sum(l, dim=-1).transpose(-1, -2)
    ar = (l.to(q.dtype) @ q).transpose(-2, -3)

    return ar, cr


def al_y_cl_ref(ar, k, v, cr, sm_scale, mask, eps=1e-12):
    r_hat = sm_scale * (ar @ k.transpose(-1, -2)).to(torch.float)
    r_hat = r_hat / (cr[..., :, None] + eps)
    r_hat = r_hat + torch.where(mask[..., None, :], 0.0, -float("inf"))
    r_hat = torch.exp(
        r_hat - torch.clamp(torch.max(r_hat, dim=-1, keepdim=True).values, min=eps)
    )
    r = r_hat / (torch.sum(r_hat, dim=-1, keepdim=True) + eps)
    r = torch.clamp(r, min=torch.finfo(r.dtype).tiny)

    cl = torch.sum(xlogy(r, r), dim=-1).transpose(-1, -2)
    al = sm_scale * (r.to(k.dtype) @ k).transpose(-2, -3)
    y = (r.to(v.dtype) @ v).transpose(-2, -3)

    return al, y, cl


def z_ref(al, q, cl, y):
    l_hat = (q @ al.transpose(-1, -2)).to(torch.float)
    l_hat = l_hat - cl[..., None, :]
    l = F.softmax(l_hat, dim=-1)

    z = (l.to(y.dtype) @ y).transpose(-2, -3).contiguous()

    return z


def monarch_attention_torch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
) -> Tensor:
    E, H, N, D = q.shape
    _, _, _, Dv = v.shape
    M = (N + B - 1) // B
    N_padded = M * B

    sm_scale = 1 / sqrt(D)

    pad_t = (N_padded - N, 0) if pre_pad else (0, N_padded - N)
    pad_t_2d = (0, 0) + pad_t

    q = F.pad(q, pad_t_2d).view(E, H, M, B, D)
    k = F.pad(k, pad_t_2d).view(E, H, M, B, D)
    v = F.pad(v, pad_t_2d).view(E, H, M, B, Dv)

    ar = q
    cr = torch.ones(E, H, M, B, device=q.device, dtype=torch.float)
    q = q.transpose(-2, -3)

    pad_offset = N_padded - N if pre_pad else 0
    range_n = torch.arange(M * B).view(M, B).to(q.device)
    mask = range_n >= pad_offset if pre_pad else range_n < N

    if attn_mask is not None:
        attn_mask = F.pad(attn_mask, pad_t).view(E, 1, M, B)
        mask = torch.logical_and(mask, attn_mask)

    for _ in range(T - 1):
        al, cl = al_cl_ref(ar, k, cr, sm_scale, mask)
        ar, cr = ar_cr_ref(al, q, cl, mask.mT)

    al, y, cl = al_y_cl_ref(ar, k, v, cr, sm_scale, mask)
    z = z_ref(al, q, cl, y)
    z = z.view(E, H, N_padded, Dv)

    return z[..., N_padded - N :, :] if pre_pad else z[..., :N, :]
