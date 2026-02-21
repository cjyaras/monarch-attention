from math import sqrt

import torch
import triton
import triton.language as tl


Tensor = torch.Tensor


@triton.jit
def xlogx(x):
    return tl.where(x == 0, 0.0, x * tl.log(x))


@triton.jit
def _al_cl_kernel(
    ar_ptr,
    stride_ar_e,
    stride_ar_h,
    stride_ar_m,
    stride_ar_b,
    stride_ar_d,
    k_ptr,
    stride_k_e,
    stride_k_h,
    stride_k_m,
    stride_k_b,
    stride_k_d,
    cr_ptr,
    stride_cr_e,
    stride_cr_h,
    stride_cr_m,
    stride_cr_b,
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    H: int,
    M: int,
    B: int,
    D: int,
    N: int,
    is_first_call: int,
    sm_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    EPS: tl.constexpr,
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
    pad_mask_b = mask_b & ((range_n >= pad_offset) if PRE_PAD else (range_n < N))
    k_mask_b = pad_mask_b
    mask_d = range_d < D

    if HAS_ATTN_MASK:
        mask_block_ptr = (
            mask_ptr
            + stride_mask_e * idx_e
            + stride_mask_m * idx_m
            + stride_mask_b * (range_b - pad_offset)
        )
        valid_token_mask = tl.load(
            mask_block_ptr,
            mask=pad_mask_b,
            other=0,
        )
        k_mask_b = pad_mask_b & valid_token_mask

    # Load ar
    ar_block_ptr = (
        ar_ptr
        + stride_ar_e * idx_e
        + stride_ar_h * idx_h
        + stride_ar_m * idx_m
        + (
            stride_ar_b * (range_b - (pad_offset if is_first_call else 0))[:, None]
            + stride_ar_d * range_d[None, :]
        )
    )
    ar = tl.load(
        ar_block_ptr,
        mask=(pad_mask_b if is_first_call else mask_b)[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load k
    k_block_ptr = (
        k_ptr
        + stride_k_e * idx_e
        + stride_k_h * idx_h
        + stride_k_m * idx_m
        + (stride_k_b * (range_b - pad_offset)[:, None] + stride_k_d * range_d[None, :])
    )
    k = tl.load(
        k_block_ptr,
        mask=k_mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load cr
    cr_block_ptr = (
        cr_ptr
        + stride_cr_e * idx_e
        + stride_cr_h * idx_h
        + stride_cr_m * idx_m
        + (stride_cr_b * range_b)
    )
    cr = tl.load(cr_block_ptr, mask=mask_b, other=1.0)

    # Attention matrix
    r = sm_scale * tl.dot(ar, tl.trans(k))
    r = r / (cr[:, None] + EPS)
    r = r + tl.where(k_mask_b[None, :], 0.0, float("-inf"))
    r = tl.exp(r - tl.clamp(tl.max(r, axis=1, keep_dims=True), EPS, float("inf")))
    r = r / (tl.sum(r, axis=1, keep_dims=True) + EPS)

    # Store cl
    cl = tl.sum(xlogx(r), axis=1)
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_m * idx_m
        + (stride_cl_b * range_b)
    )
    tl.store(cl_block_ptr, cl, mask=mask_b)

    # Store al
    al = (sm_scale * tl.dot(r.to(k.dtype), k)).to(ar.dtype)
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_m * idx_m
        + (stride_al_b * range_b[:, None] + stride_al_d * range_d[None, :])
    )
    tl.store(
        al_block_ptr,
        al,
        mask=mask_b[:, None] & mask_d[None, :],
    )


@triton.jit
def _ar_cr_kernel(
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_b,
    stride_q_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    ar_ptr,
    stride_ar_e,
    stride_ar_h,
    stride_ar_m,
    stride_ar_b,
    stride_ar_d,
    cr_ptr,
    stride_cr_e,
    stride_cr_h,
    stride_cr_m,
    stride_cr_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    H: int,
    M: int,
    B: int,
    D: int,
    N: int,
    HAS_ATTN_MASK: tl.constexpr,
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

    if HAS_ATTN_MASK:
        mask_block_ptr = (
            mask_ptr
            + stride_mask_e * idx_e
            + stride_mask_b * (idx_b - pad_offset)
            + stride_mask_m * range_m
        )
        valid_token_mask = tl.load(
            mask_block_ptr,
            mask=q_mask_m,
            other=0,
        )
        q_mask_m = q_mask_m & valid_token_mask

    # Load al
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_b * idx_b
        + (stride_al_m * range_m[:, None] + stride_al_d * range_d[None, :])
    )
    al = tl.load(
        al_block_ptr,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load q
    q_block_ptr = (
        q_ptr
        + stride_q_e * idx_e
        + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * range_m[:, None] + stride_q_d * range_d[None, :])
    )
    q = tl.load(
        q_block_ptr,
        mask=q_mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load cl
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_b * idx_b
        + (stride_cl_m * range_m)
    )
    cl = tl.load(cl_block_ptr, mask=mask_m, other=0.0)

    # Attention matrix
    l = tl.dot(al, tl.trans(q))
    l = l - cl[:, None]
    l = l + tl.where(mask_m[:, None], 0.0, float("-inf"))
    l = tl.exp(l - tl.max(l, axis=0, keep_dims=True))
    l = l / tl.sum(l, axis=0, keep_dims=True)
    l = q_mask_m[None, :] * l

    # Store cr
    cr = tl.sum(l, axis=1)
    cr_block_ptr = (
        cr_ptr
        + stride_cr_e * idx_e
        + stride_cr_h * idx_h
        + stride_cr_b * idx_b
        + (stride_cr_m * range_m)
    )
    tl.store(cr_block_ptr, cr, mask=mask_m)

    # Store ar
    ar = tl.dot(l.to(q.dtype), q).to(al.dtype)
    ar_block_ptr = (
        ar_ptr
        + stride_ar_e * idx_e
        + stride_ar_h * idx_h
        + stride_ar_b * idx_b
        + (stride_ar_m * range_m[:, None] + stride_ar_d * range_d[None, :])
    )
    tl.store(
        ar_block_ptr,
        ar,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _al_y_cl_kernel(
    ar_ptr,
    stride_ar_e,
    stride_ar_h,
    stride_ar_m,
    stride_ar_b,
    stride_ar_d,
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
    cr_ptr,
    stride_cr_e,
    stride_cr_h,
    stride_cr_m,
    stride_cr_b,
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    y_ptr,
    stride_y_e,
    stride_y_h,
    stride_y_m,
    stride_y_b,
    stride_y_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    H: int,
    M: int,
    B: int,
    D: int,
    N: int,
    is_first_call: int,
    sm_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    EPS: tl.constexpr,
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
    pad_mask_b = mask_b & ((range_n >= pad_offset) if PRE_PAD else range_n < N)
    k_mask_b = pad_mask_b
    mask_d = range_d < D

    if HAS_ATTN_MASK:
        mask_block_ptr = (
            mask_ptr
            + stride_mask_e * idx_e
            + stride_mask_m * idx_m
            + stride_mask_b * (range_b - pad_offset)
        )
        valid_token_mask = tl.load(
            mask_block_ptr,
            mask=pad_mask_b,
            other=0,
        )
        k_mask_b = pad_mask_b & valid_token_mask

    # Load ar
    ar_block_ptr = (
        ar_ptr
        + stride_ar_e * idx_e
        + stride_ar_h * idx_h
        + stride_ar_m * idx_m
        + (
            stride_ar_b * (range_b - (pad_offset if is_first_call else 0))[:, None]
            + stride_ar_d * range_d[None, :]
        )
    )
    ar = tl.load(
        ar_block_ptr,
        mask=(pad_mask_b if is_first_call else mask_b)[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load k
    k_block_ptr = (
        k_ptr
        + stride_k_e * idx_e
        + stride_k_h * idx_h
        + stride_k_m * idx_m
        + (stride_k_b * (range_b - pad_offset)[:, None] + stride_k_d * range_d[None, :])
    )
    k = tl.load(
        k_block_ptr,
        mask=k_mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load cr
    cr_block_ptr = (
        cr_ptr
        + stride_cr_e * idx_e
        + stride_cr_h * idx_h
        + stride_cr_m * idx_m
        + (stride_cr_b * range_b)
    )
    cr = tl.load(cr_block_ptr, mask=mask_b, other=1.0)

    # Attention matrix
    r = sm_scale * tl.dot(ar, tl.trans(k))
    r = r / (cr[:, None] + EPS)
    r = r + tl.where(k_mask_b[None, :], 0.0, float("-inf"))
    r = tl.exp(r - tl.clamp(tl.max(r, axis=1, keep_dims=True), EPS, float("inf")))
    r = r / (tl.sum(r, axis=1, keep_dims=True) + EPS)

    # Store cl
    cl = tl.sum(xlogx(r), axis=1)
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_m * idx_m
        + (stride_cl_b * range_b)
    )
    tl.store(cl_block_ptr, cl, mask=mask_b)

    # Store al
    al = (sm_scale * tl.dot(r.to(k.dtype), k)).to(ar.dtype)
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_m * idx_m
        + (stride_al_b * range_b[:, None] + stride_al_d * range_d[None, :])
    )
    tl.store(
        al_block_ptr,
        al,
        mask=mask_b[:, None] & mask_d[None, :],
    )

    # Load v
    v_block_ptr = (
        v_ptr
        + stride_v_e * idx_e
        + stride_v_h * idx_h
        + stride_v_m * idx_m
        + (stride_v_b * (range_b - pad_offset)[:, None] + stride_v_d * range_d[None, :])
    )
    v = tl.load(
        v_block_ptr,
        mask=k_mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Store y
    y = tl.dot(r.to(v.dtype), v).to(ar.dtype)
    y_block_ptr = (
        y_ptr
        + stride_y_e * idx_e
        + stride_y_h * idx_h
        + stride_y_m * idx_m
        + (stride_y_b * range_b[:, None] + stride_y_d * range_d[None, :])
    )
    tl.store(
        y_block_ptr,
        y,
        mask=mask_b[:, None] & mask_d[None, :],
    )


@triton.jit
def _z_kernel(
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_b,
    stride_q_d,
    y_ptr,
    stride_y_e,
    stride_y_h,
    stride_y_m,
    stride_y_b,
    stride_y_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
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

    # Load al
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_b * idx_b
        + (stride_al_m * range_m[:, None] + stride_al_d * range_d[None, :])
    )
    al = tl.load(
        al_block_ptr,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load q
    q_block_ptr = (
        q_ptr
        + stride_q_e * idx_e
        + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * range_m[:, None] + stride_q_d * range_d[None, :])
    )
    q = tl.load(
        q_block_ptr,
        mask=q_mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load cl
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_b * idx_b
        + (stride_cl_m * range_m)
    )
    cl = tl.load(cl_block_ptr, mask=mask_m, other=0.0)

    # Attention matrix
    l = tl.dot(q, tl.trans(al))
    l = l - cl[None, :]
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
    y = tl.load(
        y_block_ptr,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Store z
    z = tl.dot(l.to(y.dtype), y).to(al.dtype)
    z_block_ptr = (
        z_ptr
        + stride_z_e * idx_e
        + stride_z_h * idx_h
        + stride_z_b * (idx_b - pad_offset)
        + (stride_z_m * range_m[:, None] + stride_z_d * range_d[None, :])
    )
    tl.store(
        z_block_ptr,
        z,
        mask=q_mask_m[:, None] & mask_d[None, :],
    )


def monarch_attention_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
    eps: float = 0.0,
) -> Tensor:
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

    ar = torch.empty(E, H, M, B, D, device=q.device, dtype=q.dtype)
    al = torch.empty_like(ar)

    ar_strides = (ar.stride(0), ar.stride(1), ar.stride(2), ar.stride(3), ar.stride(4))
    al_strides = (al.stride(0), al.stride(1), al.stride(2), al.stride(3), al.stride(4))

    cr = torch.ones(E, H, M, B, device=q.device, dtype=torch.float)
    cl = torch.empty_like(cr)

    cr_strides = (cr.stride(0), cr.stride(1), cr.stride(2), cr.stride(3))
    cl_strides = (cl.stride(0), cl.stride(1), cl.stride(2), cl.stride(3))

    attn_mask_strides = (
        (attn_mask.stride(0), B * attn_mask.stride(1), attn_mask.stride(1))
        if attn_mask is not None
        else (0, 0, 0)
    )

    for t in range(T - 1):
        is_first_call = t == 0
        _ar = q if is_first_call else ar
        _ar_strides = q_strides if is_first_call else ar_strides
        _al_cl_kernel[grid_ehm](
            _ar,
            *_ar_strides,
            k,
            *k_strides,
            cr,
            *cr_strides,
            al,
            *al_strides,
            cl,
            *cl_strides,
            attn_mask,
            *attn_mask_strides,
            *HMBDN,
            is_first_call=is_first_call,
            sm_scale=sm_scale,
            HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
            BLOCK_B=BLOCK_B,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
            EPS=eps,  # type: ignore
        )

        _ar_cr_kernel[grid_ehb](
            al,
            *al_strides,
            q,
            *q_strides,
            cl,
            *cl_strides,
            ar,
            *ar_strides,
            cr,
            *cr_strides,
            attn_mask,
            *attn_mask_strides,
            *HMBDN,
            HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
            BLOCK_M=BLOCK_M,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
        )

    y = torch.empty_like(al)
    y_strides = (y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4))

    is_first_call_y = T == 1
    _ar_y = q if is_first_call_y else ar
    _ar_y_strides = q_strides if is_first_call_y else ar_strides

    _al_y_cl_kernel[grid_ehm](
        _ar_y,
        *_ar_y_strides,
        k,
        *k_strides,
        v,
        *v_strides,
        cr,
        *cr_strides,
        al,
        *al_strides,
        y,
        *y_strides,
        cl,
        *cl_strides,
        attn_mask,
        *attn_mask_strides,
        *HMBDN,
        is_first_call=is_first_call_y,
        sm_scale=sm_scale,
        HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
        BLOCK_B=BLOCK_B,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
        PRE_PAD=pre_pad,  # type: ignore
        EPS=eps,  # type: ignore
    )

    z = torch.empty_like(v)
    z_strides = (z.stride(0), z.stride(1), B * z.stride(2), z.stride(2), z.stride(3))

    _z_kernel[grid_ehb](
        al,
        *al_strides,
        q,
        *q_strides,
        y,
        *y_strides,
        cl,
        *cl_strides,
        z,
        *z_strides,
        *HMBDN,
        BLOCK_M=BLOCK_M,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
        PRE_PAD=pre_pad,  # type: ignore
    )

    return z
