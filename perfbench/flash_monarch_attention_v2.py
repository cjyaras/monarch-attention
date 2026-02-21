from math import sqrt

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


Tensor = torch.Tensor
xlogy = torch.special.xlogy


@triton.jit
def xlogx(x):
    return tl.where(x == 0.0, 0.0, x * tl.log(x))


@triton.jit
def _ayc_kernel(
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_b,
    stride_q_d,
    k_ptr,
    stride_k_e,
    stride_k_h,
    stride_k_m,
    stride_k_b,
    stride_k_d,
    v_ptr,
    stride_v_e,
    stride_v_h,
    stride_v_m,
    stride_v_b,
    stride_v_d,
    a_ptr,
    stride_a_e,
    stride_a_h,
    stride_a_m,
    stride_a_b,
    stride_a_d,
    y_ptr,
    stride_y_e,
    stride_y_h,
    stride_y_m,
    stride_y_b,
    stride_y_d,
    c_ptr,
    stride_c_e,
    stride_c_h,
    stride_c_m,
    stride_c_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    H: int,
    M: int,
    B: int,
    D: int,
    N: int,
    sm_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
):
    idx_ehm = tl.program_id(0)
    idx_eh = idx_ehm // M
    idx_e = idx_eh // H
    idx_h = idx_eh % H
    idx_m = idx_ehm % M

    pad_offset = M * B - N if PRE_PAD else 0

    range_b = tl.arange(0, BLOCK_B)
    range_d = tl.arange(0, BLOCK_D)
    range_n = B * idx_m + range_b

    mask_b = range_b < B
    k_mask_b = mask_b & (range_n >= pad_offset if PRE_PAD else range_n < N)
    mask_d = range_d < D

    # Load optional mask
    if HAS_ATTN_MASK:
        mask_block_ptr = (
            mask_ptr
            + stride_mask_e * idx_e
            + stride_mask_m * idx_m
            + stride_mask_b * (range_b - pad_offset)
        )
        valid_token_mask = tl.load(mask_block_ptr, mask=k_mask_b, other=0)
        k_mask_b = k_mask_b & valid_token_mask

    # Load q
    q_block_ptr = (
        q_ptr
        + stride_q_e * idx_e
        + stride_q_h * idx_h
        + stride_q_m * idx_m
        + (stride_q_b * (range_b - pad_offset)[:, None] + stride_q_d * range_d[None, :])
    )
    q = tl.load(q_block_ptr, mask=mask_b[:, None] & mask_d[None, :], other=0.0)

    # Load k
    k_block_ptr = (
        k_ptr
        + stride_k_e * idx_e
        + stride_k_h * idx_h
        + stride_k_m * idx_m
        + (stride_k_b * (range_b - pad_offset)[:, None] + stride_k_d * range_d[None, :])
    )
    k = tl.load(k_block_ptr, mask=k_mask_b[:, None] & mask_d[None, :], other=0.0)

    # Compute r
    r = sm_scale * tl.dot(q, tl.trans(k))
    r = r + tl.where(k_mask_b[None, :], 0.0, float("-inf"))
    r = tl.exp(r - tl.maximum(tl.max(r, axis=1, keep_dims=True), 0.0))
    r = r / (tl.sum(r, axis=1, keep_dims=True) + 1e-12)

    # Store c
    c = tl.sum(xlogx(r), axis=1)
    c_block_ptr = (
        c_ptr
        + stride_c_e * idx_e
        + stride_c_h * idx_h
        + stride_c_m * idx_m
        + (stride_c_b * range_b)
    )
    tl.store(c_block_ptr, c, mask=mask_b)

    # Store a
    a = (sm_scale * tl.dot(r.to(k.dtype), k)).to(q.dtype)
    a_block_ptr = (
        a_ptr
        + stride_a_e * idx_e
        + stride_a_h * idx_h
        + stride_a_m * idx_m
        + (stride_a_b * range_b[:, None] + stride_a_d * range_d[None, :])
    )
    tl.store(a_block_ptr, a, mask=mask_b[:, None] & mask_d[None, :])

    # Load v
    v_block_ptr = (
        v_ptr
        + stride_v_e * idx_e
        + stride_v_h * idx_h
        + stride_v_m * idx_m
        + (stride_v_b * (range_b - pad_offset)[:, None] + stride_v_d * range_d[None, :])
    )
    v = tl.load(v_block_ptr, mask=k_mask_b[:, None] & mask_d[None, :], other=0.0)

    # Store y
    y = tl.dot(r.to(v.dtype), v).to(q.dtype)
    y_block_ptr = (
        y_ptr
        + stride_y_e * idx_e
        + stride_y_h * idx_h
        + stride_y_m * idx_m
        + (stride_y_b * range_b[:, None] + stride_y_d * range_d[None, :])
    )
    tl.store(y_block_ptr, y, mask=mask_b[:, None] & mask_d[None, :])


@triton.jit
def _z_kernel(
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_b,
    stride_q_d,
    a_ptr,
    stride_a_e,
    stride_a_h,
    stride_a_m,
    stride_a_b,
    stride_a_d,
    y_ptr,
    stride_y_e,
    stride_y_h,
    stride_y_m,
    stride_y_b,
    stride_y_d,
    c_ptr,
    stride_c_e,
    stride_c_h,
    stride_c_m,
    stride_c_b,
    z_ptr,
    stride_z_e,
    stride_z_h,
    stride_z_m,
    stride_z_b,
    stride_z_d,
    H: int,
    M: int,
    B: int,
    D: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
):
    idx_ehb = tl.program_id(0)
    idx_eh = idx_ehb // B
    idx_e = idx_eh // H
    idx_h = idx_eh % H
    idx_b = idx_ehb % B

    pad_offset = M * B - N if PRE_PAD else 0

    range_m = tl.arange(0, BLOCK_M)
    range_d = tl.arange(0, BLOCK_D)
    range_n = idx_b + B * range_m

    mask_m = range_m < M
    q_mask_m = mask_m & (range_n >= pad_offset if PRE_PAD else range_n < N)
    mask_d = range_d < D

    # Load q
    q_block_ptr = (
        q_ptr
        + stride_q_e * idx_e
        + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * range_m[:, None] + stride_q_d * range_d[None, :])
    )
    q = tl.load(q_block_ptr, mask=q_mask_m[:, None] & mask_d[None, :], other=0.0)

    # Load a
    a_block_ptr = (
        a_ptr
        + stride_a_e * idx_e
        + stride_a_h * idx_h
        + stride_a_b * idx_b
        + (stride_a_m * range_m[:, None] + stride_a_d * range_d[None, :])
    )
    a = tl.load(a_block_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Load c
    c_block_ptr = (
        c_ptr
        + stride_c_e * idx_e
        + stride_c_h * idx_h
        + stride_c_b * idx_b
        + (stride_c_m * range_m)
    )
    c = tl.load(c_block_ptr, mask=mask_m, other=0.0)

    # Attention matrix
    l = tl.dot(q, tl.trans(a))
    l = l - c[None, :]
    l = l + tl.where(mask_m[None, :], 0.0, float("-inf"))
    l = tl.exp(l - tl.max(l, axis=1, keep_dims=True))
    l = l / tl.sum(l, axis=1, keep_dims=True)

    # Load y
    y_block_ptr = (
        y_ptr
        + stride_y_e * idx_e
        + stride_y_h * idx_h
        + stride_y_b * idx_b
        + (stride_y_m * range_m[:, None] + stride_y_d * range_d[None, :])
    )
    y = tl.load(y_block_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Store z
    z = tl.dot(l.to(y.dtype), y).to(q.dtype)
    z_block_ptr = (
        z_ptr
        + stride_z_e * idx_e
        + stride_z_h * idx_h
        + stride_z_b * (idx_b - pad_offset)
        + (stride_z_m * range_m[:, None] + stride_z_d * range_d[None, :])
    )
    tl.store(z_block_ptr, z, mask=q_mask_m[:, None] & mask_d[None, :])


def flash_monarch_attention_v2(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
) -> Tensor:
    check_inputs(q, k, v)
    assert T == 1
    E, H, N, D = q.shape
    M = triton.cdiv(N, B)

    HMBDN = (H, M, B, D, N)

    grid_ehm = (E * H * M,)
    grid_ehb = (E * H * B,)

    BLOCK_B = max(triton.next_power_of_2(B), 16)
    BLOCK_M = max(triton.next_power_of_2(M), 16)
    BLOCK_D = max(triton.next_power_of_2(D), 16)

    sm_scale = 1 / sqrt(D)

    q_strides = (q.stride(0), q.stride(1), B * q.stride(2), q.stride(2), q.stride(3))
    k_strides = (k.stride(0), k.stride(1), B * k.stride(2), k.stride(2), k.stride(3))
    v_strides = (v.stride(0), v.stride(1), B * v.stride(2), v.stride(2), v.stride(3))

    a = torch.empty(E, H, M, B, D, device=q.device, dtype=q.dtype)
    a_strides = (a.stride(0), a.stride(1), a.stride(2), a.stride(3), a.stride(4))

    y = torch.empty_like(a)
    y_strides = (y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4))

    c = torch.empty(E, H, M, B, device=q.device, dtype=torch.float)
    c_strides = (c.stride(0), c.stride(1), c.stride(2), c.stride(3))

    attn_mask_strides = (
        (attn_mask.stride(0), B * attn_mask.stride(1), attn_mask.stride(1))
        if attn_mask is not None
        else (0, 0, 0)
    )

    _ayc_kernel[grid_ehm](
        q,
        *q_strides,
        k,
        *k_strides,
        v,
        *v_strides,
        a,
        *a_strides,
        y,
        *y_strides,
        c,
        *c_strides,
        attn_mask,
        *attn_mask_strides,
        *HMBDN,
        sm_scale=sm_scale,
        HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
        BLOCK_B=BLOCK_B,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
        PRE_PAD=pre_pad,  # type: ignore
    )

    z = torch.empty_like(q)
    z_strides = (
        z.stride(0),
        z.stride(1),
        B * z.stride(2),
        z.stride(2),
        z.stride(3),
    )

    _z_kernel[grid_ehb](
        q,
        *q_strides,
        a,
        *a_strides,
        y,
        *y_strides,
        c,
        *c_strides,
        z,
        *z_strides,
        *HMBDN,
        BLOCK_M=BLOCK_M,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
        PRE_PAD=pre_pad,  # type: ignore
    )

    return z


def al_cl_ref(ar, k, cr, sm_scale, mask, eps=1e-8):
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


def al_y_cl_ref(ar, k, v, cr, sm_scale, mask, eps=1e-8):
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


def flash_monarch_attention_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
) -> Tensor:
    E, H, N, D = q.shape
    M = triton.cdiv(N, B)
    N_padded = M * B

    sm_scale = 1 / sqrt(D)

    pad_t = (N_padded - N, 0) if pre_pad else (0, N_padded - N)
    pad_t_2d = (0, 0) + pad_t

    q = F.pad(q, pad_t_2d).view(E, H, M, B, D)
    k = F.pad(k, pad_t_2d).view(E, H, M, B, D)
    v = F.pad(v, pad_t_2d).view(E, H, M, B, D)

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
    z = z.view(E, H, N_padded, D)

    return z[..., N_padded - N :, :] if pre_pad else z[..., :N, :]


def test():
    torch.manual_seed(1)
    E, H, N, D = 1, 1, 16 * 16, 16
    q = torch.randn(E, H, N, D, dtype=torch.float16).cuda()
    k = torch.randn(E, H, N, D, dtype=torch.float16).cuda()
    v = torch.randn(E, H, N, D, dtype=torch.float16).cuda()

    # attn_mask = None
    attn_mask = torch.tensor([[True] * (256 - 64) + [False] * 64]).bool().cuda()

    T, B = 1, 18
    pre_pad = True

    o1 = flash_monarch_attention_v2(q, k, v, attn_mask, T, B, pre_pad=pre_pad)
    o2 = flash_monarch_attention_reference(q, k, v, attn_mask, T, B, pre_pad=pre_pad)

    print(o1)
    print()
    print(o2)


if __name__ == "__main__":
    test()
