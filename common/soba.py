from enum import StrEnum
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
xlogy = torch.special.xlogy


class PadType(StrEnum):
    pre = "pre"
    post = "post"


def al_cl_ref(ar, k, cr, sm_scale, mask, eps=1e-12):
    r_hat = sm_scale * (ar @ k.transpose(-1, -2)).to(torch.float)
    r_hat = r_hat / (cr[..., :, None] + eps)
    r_hat = r_hat + torch.where(mask[..., None, :], 0.0, -float("inf"))
    r_hat = torch.exp(
        r_hat - torch.clamp(torch.max(r_hat, dim=-1, keepdim=True).values, min=eps)
    )
    r = r_hat / (torch.sum(r_hat, dim=-1, keepdim=True) + eps)

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


def soba_monarch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
) -> Tensor:
    # check_inputs(q, k, v)
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


class SobaMonarch(nn.Module):

    def __init__(self, block_size: int, num_steps: int, pad_type: PadType):
        super().__init__()
        self.block_size = block_size
        self.num_steps = num_steps
        self.pad_type = pad_type

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return soba_monarch(
            query,
            key,
            value,
            attention_mask,
            self.num_steps,
            self.block_size,
            self.pad_type == PadType.pre,
        )

    def get_matrix(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape
        value = torch.eye(seq_len, device=query.device).expand(
            batch_size, num_heads, seq_len, seq_len
        )
        return self.forward(query, key, value, attention_mask)


# from enum import StrEnum
# from math import ceil, prod, sqrt
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat

# Tensor = torch.Tensor


# class PadType(StrEnum):
#     pre = "pre"
#     post = "post"


# # rewriting functions here to prep for aten implementation


# def monarch_matrix(L: Tensor, R: Tensor, pad_amount: int, pad_type: PadType) -> Tensor:
#     out = torch.einsum("...jlk,...kji->...ljki", L, R)
#     out = rearrange(out, "... l j k i -> ... (l j) (k i)")

#     match pad_type:
#         case PadType.pre:
#             return out[..., pad_amount:, pad_amount:]
#         case PadType.post:
#             return out[..., : -pad_amount or None, : -pad_amount or None]


# def _monarch_multiply_padded(L: Tensor, R: Tensor, x: Tensor) -> Tensor:
#     *batch_shape, n, d = x.shape
#     *_, m, b, _ = R.shape
#     batch_size = prod(batch_shape)

#     X = x.reshape(batch_size * m, b, d)
#     Y = torch.bmm(R.reshape(batch_size * m, b, b), X)
#     Y = (
#         Y.reshape(batch_size, m, b, d)
#         .transpose(-3, -2)
#         .contiguous()
#         .reshape(batch_size * b, m, d)
#     )
#     Z = torch.bmm(L.reshape(batch_size * b, m, m), Y)
#     z = (
#         Z.reshape(batch_size, b, m, d)
#         .transpose(-3, -2)
#         .contiguous()
#         .reshape(*batch_shape, n, d)
#     )

#     return z


# def monarch_multiply(
#     L: Tensor, R: Tensor, x: Tensor, pad_amount: int, pad_type: PadType
# ) -> Tensor:

#     match pad_type:
#         case PadType.pre:
#             pad_t = (0, 0) + (pad_amount, 0)
#         case PadType.post:
#             pad_t = (0, 0) + (0, pad_amount)

#     x_padded = F.pad(x, pad_t)
#     z_padded = _monarch_multiply_padded(L, R, x_padded)

#     match pad_type:
#         case PadType.pre:
#             return z_padded[..., pad_amount:, :]
#         case PadType.post:
#             return z_padded[..., : -pad_amount or None, :]


# class SobaMonarch(nn.Module):

#     def __init__(self, block_size: int, num_steps: int, pad_type: PadType):
#         super().__init__()
#         self.block_size = block_size
#         self.num_steps = num_steps
#         self.pad_type = pad_type

#     def get_matrix(
#         self,
#         query: Tensor,
#         key: Tensor,
#         attention_mask: Optional[Tensor] = None,
#         return_history: bool = False,
#     ) -> Tensor:
#         seq_len, head_dim = query.shape[-2:]
#         query = query / sqrt(head_dim)
#         pad_amount = self._get_pad_amount(seq_len)
#         valid_mask = self._get_valid_mask(attention_mask)
#         left, right = self._get_factors(query, key, valid_mask, return_history)
#         matrix = self._get_matrix_from_factors(left, right, pad_amount)
#         return matrix

#     def forward(
#         self,
#         query: Tensor,
#         key: Tensor,
#         value: Tensor,
#         attention_mask: Optional[Tensor] = None,
#     ) -> Tensor:
#         head_dim = query.shape[-1]
#         query = query / sqrt(head_dim)
#         valid_mask = self._get_valid_mask(attention_mask)
#         left, right = self._get_factors(query, key, valid_mask, False)
#         return monarch_multiply(
#             left, right, value, self._get_pad_amount(query.shape[-2]), self.pad_type
#         )

#     def _get_num_blocks(self, seq_len: int) -> int:
#         num_blocks = ceil(seq_len / self.block_size)
#         return num_blocks

#     def _get_pad_amount(self, seq_len: int) -> int:
#         num_blocks = self._get_num_blocks(seq_len)
#         seq_len_padded = self.block_size * num_blocks
#         pad_amount = seq_len_padded - seq_len
#         return pad_amount

#     # def _multiply(self, left: Tensor, right: Tensor, inputs: Tensor) -> Tensor:
#     #     seq_len = inputs.shape[-2]
#     #     pad_amount = self._get_pad_amount(seq_len)

#     #     match self.pad_type:
#     #         case PadType.pre:
#     #             pad_t = (0, 0) + (pad_amount, 0)
#     #         case PadType.post:
#     #             pad_t = (0, 0) + (0, pad_amount)
#     #         case _:
#     #             raise ValueError("Invalid pad_type")

#     #     x = F.pad(inputs, pad_t)
#     #     X = rearrange(x, "... (k i) v -> ... k i v", i=self.block_size)
#     #     Y = torch.einsum("...kji,...kiv->...kjv", right, X)
#     #     Z = torch.einsum("...jlk,...kjv->...ljv", left, Y)
#     #     z = rearrange(Z, "... l j v -> ... (l j) v")

#     #     match self.pad_type:
#     #         case PadType.pre:
#     #             return z[..., pad_amount:, :]
#     #         case PadType.post:
#     #             return z[..., : -pad_amount or None, :]
#     #         case _:
#     #             raise ValueError("Invalid pad_type")

#     def _get_matrix_from_factors(
#         self, left: Tensor, right: Tensor, pad_amount: int
#     ) -> Tensor:
#         return monarch_matrix(left, right, pad_amount, self.pad_type)

#     def _get_valid_mask(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
#         if attention_mask is not None:
#             attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")
#         return attention_mask

#     def _create_right_mask(
#         self,
#         right_shape: Tuple[int, ...],
#         valid_mask: Optional[Tensor],
#         pad_amount: int,
#         device,
#     ) -> Tensor:
#         right_mask = torch.ones(right_shape, device=device, dtype=torch.bool)
#         right_mask_flat = rearrange(right_mask, "... k j i -> ... j (k i)")

#         match self.pad_type:
#             case PadType.pre:
#                 right_mask_flat[..., :pad_amount] = False
#                 if valid_mask is not None:
#                     right_mask_flat[..., pad_amount:] = valid_mask.bool()
#             case PadType.post:
#                 right_mask_flat[..., -pad_amount or right_mask_flat.shape[-1] :] = False
#                 if valid_mask is not None:
#                     right_mask_flat[..., : -pad_amount or None] = valid_mask.bool()
#             case _:
#                 raise ValueError("Invalid pad_type")

#         right_mask = rearrange(
#             right_mask_flat, "... j (k i) -> ... k j i", i=self.block_size
#         )
#         return right_mask

#     def _get_factors(
#         self,
#         query: Tensor,
#         key: Tensor,
#         valid_mask: Optional[Tensor],
#         return_history: bool,
#     ) -> Tuple[Tensor, Tensor]:
#         seq_len = query.shape[-2]
#         batch_shape = query.shape[:-2]
#         pad_amount = self._get_pad_amount(seq_len)
#         num_blocks = self._get_num_blocks(seq_len)
#         block_size = self.block_size

#         if valid_mask is not None:
#             query = query * valid_mask.transpose(-1, -2)
#             key = key * valid_mask.transpose(-1, -2)

#         pad_t = (0, 0) + (
#             (pad_amount, 0) if self.pad_type == PadType.pre else (0, pad_amount)
#         )
#         query = F.pad(query, pad_t)
#         key = F.pad(key, pad_t)
#         query = rearrange(query, "... (l j) v -> ... l j v", j=block_size)
#         key = rearrange(key, "... (k i) v -> ... k i v", i=block_size)

#         left_shape = batch_shape + (block_size, num_blocks, num_blocks)
#         right_shape = batch_shape + (num_blocks, block_size, block_size)

#         right_mask = self._create_right_mask(
#             right_shape, valid_mask, pad_amount, query.device
#         )

#         left = torch.eye(num_blocks, device=query.device).expand(left_shape)

#         if return_history:
#             left_history = []
#             right_history = []

#         for step in range(self.num_steps):

#             if step % 2 == 0:
#                 beta = torch.einsum("...jlk,...ljv,...kiv->...kji", left, query, key)
#                 tau = repeat(torch.sum(left, dim=-2), "... j k -> ... k j 1")
#                 right = F.softmax(
#                     torch.where(right_mask, beta / tau, torch.finfo(beta.dtype).min),
#                     dim=-1,
#                 )

#             else:
#                 alpha = repeat(
#                     torch.sum(torch.special.xlogy(right, right), dim=-1),
#                     "... k j -> ... j 1 k",
#                 )
#                 beta = torch.einsum("...kji,...ljv,...kiv->...jlk", right, query, key)
#                 left = F.softmax(beta - alpha, dim=-1)

#             if return_history:
#                 left_history.append(left)
#                 right_history.append(right)

#         if return_history:
#             left = torch.stack(left_history)
#             right = torch.stack(right_history)

#         return left, right
