from math import e, log2, sqrt

import torch
import triton
import triton.language as tl


Tensor = torch.Tensor


def _config(n: int) -> dict:
    """Launch configuration for a softmax over n entries (keys of a block, or
    blocks), tuned on an A100 for batch sizes 1 and 8.

    Each program takes BLOCK_R query rows, up to 128 (fewer only if that pads
    less than 128 would), and streams over the n entries in chunks of BLOCK_C.
    """
    rows = min(max(triton.next_power_of_2(n), 16), 128)
    if n > 128 and triton.cdiv(n, 128) * 128 - n > n // 8:
        rows = 64
    return dict(
        BLOCK_R=rows,
        BLOCK_C=min(rows, 32),
        num_warps=2 if n <= 32 else 4,
        num_stages=2,
    )


@triton.jit
def _strides_mb(H, M, B, D):
    """Strides (e, h, m, b, d) of a contiguous (E, H, M, B, D) tensor."""
    return H * M * B * D, M * B * D, B * D, D, 1


@triton.jit
def _strides_bm(H, M, B, D):
    """Strides (e, h, m, b, d) of a contiguous (E, H, B, M, D) tensor."""
    return H * B * M * D, B * M * D, D, M * D, 1


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
    v_ptr,
    stride_v_e,
    stride_v_h,
    stride_v_m,
    stride_v_b,
    stride_v_d,
    cr_ptr,
    al_ptr,
    y_ptr,
    cl_ptr,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    H: int,
    M: int,
    B: int,
    D: int,
    N: int,
    IS_FIRST_CALL: tl.constexpr,
    qk_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    # Intermediate buffers: al, y, cl (and lse) are laid out (E, H, B, M, ...) so
    # that _z_kernel and _ar_cr_kernel read them contiguously; ar and cr are
    # (E, H, M, B, ...) for _al_cl_kernel
    stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d = _strides_bm(
        H, M, B, D
    )
    stride_y_e, stride_y_h, stride_y_m, stride_y_b, stride_y_d = _strides_bm(H, M, B, D)
    stride_cr_e, stride_cr_h, stride_cr_m, stride_cr_b, _ = _strides_mb(H, M, B, 1)
    stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b, _ = _strides_bm(H, M, B, 1)
    # One program per (batch, head, block m, chunk of BLOCK_R query rows). It
    # streams over the block's keys in chunks of BLOCK_C with an online softmax.
    idx_ehm = tl.program_id(0)
    idx_eh = idx_ehm // M
    idx_e = idx_eh // H
    idx_h = idx_eh % H
    idx_m = idx_ehm % M

    pad_offset = M * B - N if PRE_PAD else 0

    range_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    range_d = tl.arange(0, BLOCK_D)
    range_n_r = B * idx_m + range_r

    mask_r = range_r < B
    pad_mask_r = mask_r & ((range_n_r >= pad_offset) if PRE_PAD else range_n_r < N)
    mask_d = range_d < D

    # Load ar
    ar_block_ptr = (
        ar_ptr
        + stride_ar_e * idx_e
        + stride_ar_h * idx_h
        + stride_ar_m * idx_m
        + (
            stride_ar_b * (range_r - (pad_offset if IS_FIRST_CALL else 0))[:, None]
            + stride_ar_d * range_d[None, :]
        )
    )
    ar = tl.load(
        ar_block_ptr,
        mask=(pad_mask_r if IS_FIRST_CALL else mask_r)[:, None] & mask_d[None, :],
        other=0.0,
    )
    if not IS_FIRST_CALL:  # cr is all ones before the first _ar_cr_kernel
        cr_block_ptr = (
            cr_ptr
            + stride_cr_e * idx_e
            + stride_cr_h * idx_h
            + stride_cr_m * idx_m
            + (stride_cr_b * range_r)
        )
        cr = tl.load(cr_block_ptr, mask=mask_r, other=1.0)

    # Running row max, normalizer sum(exp2(s - max)) and entropy numerator
    # sum(exp2(s - max) (s - max)), rescaled whenever the max grows
    row_max = tl.full([BLOCK_R], float("-inf"), tl.float32)
    denom = tl.zeros([BLOCK_R], tl.float32)
    ent = tl.zeros([BLOCK_R], tl.float32)
    acc_al = tl.zeros([BLOCK_R, BLOCK_D], tl.float32)
    acc_y = tl.zeros([BLOCK_R, BLOCK_D], tl.float32)

    for start in range(0, B, BLOCK_C):
        range_c = start + tl.arange(0, BLOCK_C)
        range_n_c = B * idx_m + range_c
        mask_c = range_c < B
        k_mask_c = mask_c & ((range_n_c >= pad_offset) if PRE_PAD else range_n_c < N)
        if HAS_ATTN_MASK:
            mask_block_ptr = (
                mask_ptr
                + stride_mask_e * idx_e
                + stride_mask_m * idx_m
                + stride_mask_b * (range_c - pad_offset)
            )
            k_mask_c = k_mask_c & tl.load(mask_block_ptr, mask=k_mask_c, other=0)

        # Load k
        k_block_ptr = (
            k_ptr
            + stride_k_e * idx_e
            + stride_k_h * idx_h
            + stride_k_m * idx_m
            + (
                stride_k_b * (range_c - pad_offset)[:, None]
                + stride_k_d * range_d[None, :]
            )
        )
        k = tl.load(k_block_ptr, mask=k_mask_c[:, None] & mask_d[None, :], other=0.0)

        # Base-2 logits s, and p = exp2(s - max) for the chunk
        s = qk_scale * tl.dot(ar, tl.trans(k))
        if not IS_FIRST_CALL:
            s = s / cr[:, None]
        s = tl.where(k_mask_c[None, :], s, float("-inf"))
        new_max = tl.maximum(row_max, tl.max(s, axis=1))
        # Rows whose keys so far are all masked keep a max of -inf; shift by 0
        shift = tl.where(new_max == float("-inf"), 0.0, new_max)
        alpha = tl.exp2(row_max - shift)
        p = tl.exp2(s - shift[:, None])
        s_shifted = tl.where(k_mask_c[None, :], s - shift[:, None], 0.0)
        ent_shift = tl.where(denom > 0, denom * (row_max - shift), 0.0)
        ent = alpha * (ent + ent_shift) + tl.sum(p * s_shifted, axis=1)
        denom = alpha * denom + tl.sum(p, axis=1)
        row_max = new_max
        acc_al = alpha[:, None] * acc_al + tl.dot(p.to(k.dtype), k)

        if COMPUTE_Y:
            v_block_ptr = (
                v_ptr
                + stride_v_e * idx_e
                + stride_v_h * idx_h
                + stride_v_m * idx_m
                + (
                    stride_v_b * (range_c - pad_offset)[:, None]
                    + stride_v_d * range_d[None, :]
                )
            )
            v = tl.load(
                v_block_ptr, mask=k_mask_c[:, None] & mask_d[None, :], other=0.0
            )
            acc_y = alpha[:, None] * acc_y + tl.dot(p.to(v.dtype), v)

    # r = p / denom. A block whose keys are all masked (denom = 0) has no
    # attention weights and gets cl = inf, which gives it zero weight when
    # queries choose between blocks. cl = sum(r log2 r) = ent / denom - log2(denom).
    has_keys = denom > 0
    inv_denom = tl.where(has_keys, 1.0 / denom, 0.0)
    cl = tl.where(has_keys, ent * inv_denom - tl.log2(denom), float("inf"))
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_m * idx_m
        + (stride_cl_b * range_r)
    )
    tl.store(cl_block_ptr, cl, mask=mask_r)

    # Store al
    al = (qk_scale * acc_al * inv_denom[:, None]).to(ar.dtype)
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_m * idx_m
        + (stride_al_b * range_r[:, None] + stride_al_d * range_d[None, :])
    )
    tl.store(al_block_ptr, al, mask=mask_r[:, None] & mask_d[None, :])

    if COMPUTE_Y:
        y = (acc_y * inv_denom[:, None]).to(ar.dtype)
        y_block_ptr = (
            y_ptr
            + stride_y_e * idx_e
            + stride_y_h * idx_h
            + stride_y_m * idx_m
            + (stride_y_b * range_r[:, None] + stride_y_d * range_d[None, :])
        )
        tl.store(y_block_ptr, y, mask=mask_r[:, None] & mask_d[None, :])


@triton.jit
def _ar_cr_kernel(
    al_ptr,
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_b,
    stride_q_d,
    cl_ptr,
    lse_ptr,  # same layout as cl
    ar_ptr,
    cr_ptr,
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
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    ALL_BLOCKS: tl.constexpr,
):
    # Intermediate buffers: al, y, cl (and lse) are laid out (E, H, B, M, ...) so
    # that _z_kernel and _ar_cr_kernel read them contiguously; ar and cr are
    # (E, H, M, B, ...) for _al_cl_kernel
    stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d = _strides_bm(
        H, M, B, D
    )
    stride_ar_e, stride_ar_h, stride_ar_m, stride_ar_b, stride_ar_d = _strides_mb(
        H, M, B, D
    )
    stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b, _ = _strides_bm(H, M, B, 1)
    stride_cr_e, stride_cr_h, stride_cr_m, stride_cr_b, _ = _strides_mb(H, M, B, 1)
    # One program per (batch, head, position b, chunk of BLOCK_R blocks). It
    # streams over the queries in chunks of BLOCK_C. Each query's softmax over
    # blocks is normalized directly if the program holds all blocks (ALL_BLOCKS),
    # and otherwise by its log-sum-exp from _z_kernel(COMPUTE_Z=False).
    idx_ehb = tl.program_id(0)
    idx_eh = idx_ehb // B
    idx_e = idx_eh // H
    idx_h = idx_eh % H
    idx_b = idx_ehb % B

    pad_offset = M * B - N if PRE_PAD else 0

    range_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    range_d = tl.arange(0, BLOCK_D)
    mask_r = range_r < M
    mask_d = range_d < D

    # Load al and cl
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_b * idx_b
        + (stride_al_m * range_r[:, None] + stride_al_d * range_d[None, :])
    )
    al = tl.load(al_block_ptr, mask=mask_r[:, None] & mask_d[None, :], other=0.0)
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_b * idx_b
        + (stride_cl_m * range_r)
    )
    cl = tl.load(cl_block_ptr, mask=mask_r, other=0.0)

    acc_cr = tl.zeros([BLOCK_R], tl.float32)
    acc_ar = tl.zeros([BLOCK_R, BLOCK_D], tl.float32)

    for start in range(0, M, BLOCK_C):
        range_c = start + tl.arange(0, BLOCK_C)
        range_n = idx_b + B * range_c
        mask_c = range_c < M
        q_mask_c = mask_c & (range_n >= pad_offset if PRE_PAD else range_n < N)
        if HAS_ATTN_MASK:
            mask_block_ptr = (
                mask_ptr
                + stride_mask_e * idx_e
                + stride_mask_b * (idx_b - pad_offset)
                + stride_mask_m * range_c
            )
            q_mask_c = q_mask_c & tl.load(mask_block_ptr, mask=q_mask_c, other=0)

        # Load q and its log-sum-exp
        q_block_ptr = (
            q_ptr
            + stride_q_e * idx_e
            + stride_q_h * idx_h
            + stride_q_b * (idx_b - pad_offset)
            + (stride_q_m * range_c[:, None] + stride_q_d * range_d[None, :])
        )
        q = tl.load(q_block_ptr, mask=q_mask_c[:, None] & mask_d[None, :], other=0.0)

        # Attention matrix, normalized over blocks (rows); masked queries get 0
        l = tl.dot(al, tl.trans(q)) - cl[:, None]
        if ALL_BLOCKS:
            l = tl.where(mask_r[:, None], l, float("-inf"))
            col_max = tl.max(l, axis=0)
            l = tl.exp2(l - tl.where(col_max == float("-inf"), 0.0, col_max)[None, :])
            l = l * (1.0 / tl.sum(l, axis=0))[None, :]
        else:
            lse_block_ptr = (
                lse_ptr
                + stride_cl_e * idx_e
                + stride_cl_h * idx_h
                + stride_cl_b * idx_b
                + (stride_cl_m * range_c)
            )
            lse = tl.load(lse_block_ptr, mask=mask_c, other=0.0)
            l = tl.exp2(l - lse[None, :])
        l = tl.where(mask_r[:, None] & q_mask_c[None, :], l, 0.0)
        acc_cr += tl.sum(l, axis=1)
        acc_ar += tl.dot(l.to(q.dtype), q)

    # Store cr and ar
    cr_block_ptr = (
        cr_ptr
        + stride_cr_e * idx_e
        + stride_cr_h * idx_h
        + stride_cr_b * idx_b
        + (stride_cr_m * range_r)
    )
    tl.store(cr_block_ptr, acc_cr, mask=mask_r)
    ar_block_ptr = (
        ar_ptr
        + stride_ar_e * idx_e
        + stride_ar_h * idx_h
        + stride_ar_b * idx_b
        + (stride_ar_m * range_r[:, None] + stride_ar_d * range_d[None, :])
    )
    tl.store(ar_block_ptr, acc_ar.to(al.dtype), mask=mask_r[:, None] & mask_d[None, :])


@triton.jit
def _z_kernel(
    al_ptr,
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_b,
    stride_q_d,
    y_ptr,
    cl_ptr,
    lse_ptr,  # same layout as cl
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
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    COMPUTE_Z: tl.constexpr,
):
    # Intermediate buffers: al, y, cl (and lse) are laid out (E, H, B, M, ...) so
    # that _z_kernel and _ar_cr_kernel read them contiguously; ar and cr are
    # (E, H, M, B, ...) for _al_cl_kernel
    stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d = _strides_bm(
        H, M, B, D
    )
    stride_y_e, stride_y_h, stride_y_m, stride_y_b, stride_y_d = _strides_bm(H, M, B, D)
    stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b, _ = _strides_bm(H, M, B, 1)
    # One program per (batch, head, position b, chunk of BLOCK_R query blocks).
    # It streams over the blocks' al, cl and y in chunks of BLOCK_C with an
    # online softmax. With COMPUTE_Z=False it only stores each query's
    # log-sum-exp over blocks, for _ar_cr_kernel.
    idx_ehb = tl.program_id(0)
    idx_eh = idx_ehb // B
    idx_e = idx_eh // H
    idx_h = idx_eh % H
    idx_b = idx_ehb % B

    pad_offset = M * B - N if PRE_PAD else 0

    range_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    range_d = tl.arange(0, BLOCK_D)
    range_n = idx_b + B * range_r

    mask_r = range_r < M
    q_mask_r = mask_r & (range_n >= pad_offset if PRE_PAD else range_n < N)
    mask_d = range_d < D

    # Load q
    q_block_ptr = (
        q_ptr
        + stride_q_e * idx_e
        + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * range_r[:, None] + stride_q_d * range_d[None, :])
    )
    q = tl.load(q_block_ptr, mask=q_mask_r[:, None] & mask_d[None, :], other=0.0)

    row_max = tl.full([BLOCK_R], float("-inf"), tl.float32)
    denom = tl.zeros([BLOCK_R], tl.float32)
    acc = tl.zeros([BLOCK_R, BLOCK_D], tl.float32)

    for start in range(0, M, BLOCK_C):
        range_c = start + tl.arange(0, BLOCK_C)
        mask_c = range_c < M

        # Load al, cl and y
        al_block_ptr = (
            al_ptr
            + stride_al_e * idx_e
            + stride_al_h * idx_h
            + stride_al_b * idx_b
            + (stride_al_m * range_c[:, None] + stride_al_d * range_d[None, :])
        )
        al = tl.load(al_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        cl_block_ptr = (
            cl_ptr
            + stride_cl_e * idx_e
            + stride_cl_h * idx_h
            + stride_cl_b * idx_b
            + (stride_cl_m * range_c)
        )
        cl = tl.load(cl_block_ptr, mask=mask_c, other=0.0)
        # Blocks whose keys are all masked have cl = inf, so s = -inf
        s = tl.dot(q, tl.trans(al)) - cl[None, :]
        s = tl.where(mask_c[None, :], s, float("-inf"))
        new_max = tl.maximum(row_max, tl.max(s, axis=1))
        shift = tl.where(new_max == float("-inf"), 0.0, new_max)
        alpha = tl.exp2(row_max - shift)
        p = tl.exp2(s - shift[:, None])
        denom = alpha * denom + tl.sum(p, axis=1)
        row_max = new_max
        if COMPUTE_Z:
            y_block_ptr = (
                y_ptr
                + stride_y_e * idx_e
                + stride_y_h * idx_h
                + stride_y_b * idx_b
                + (stride_y_m * range_c[:, None] + stride_y_d * range_d[None, :])
            )
            y = tl.load(y_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
            acc = alpha[:, None] * acc + tl.dot(p.to(y.dtype), y)

    if COMPUTE_Z:
        z = (acc * (1.0 / denom)[:, None]).to(q.dtype)
        z_block_ptr = (
            z_ptr
            + stride_z_e * idx_e
            + stride_z_h * idx_h
            + stride_z_b * (idx_b - pad_offset)
            + (stride_z_m * range_r[:, None] + stride_z_d * range_d[None, :])
        )
        tl.store(z_block_ptr, z, mask=q_mask_r[:, None] & mask_d[None, :])
    else:
        lse_block_ptr = (
            lse_ptr
            + stride_cl_e * idx_e
            + stride_cl_h * idx_h
            + stride_cl_b * idx_b
            + (stride_cl_m * range_r)
        )
        tl.store(lse_block_ptr, row_max + tl.log2(denom), mask=mask_r)


def monarch_attention_triton(
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

    HMBDN = (H, M, B, D, N)

    # _al_cl_kernel: softmax over B keys, _ar_cr_kernel and _z_kernel: softmax
    # over M blocks
    config_b = _config(B)
    config_m = _config(M)
    grid_al_cl = (E * H * M, triton.cdiv(B, config_b["BLOCK_R"]))
    grid_z = (E * H * B, triton.cdiv(M, config_m["BLOCK_R"]))

    BLOCK_D = max(triton.next_power_of_2(D), 16)

    # The kernels work in base 2 (exp2 and log2 are native instructions): logits
    # are scaled by log2(e), and so are al, cl and lse.
    qk_scale = log2(e) / sqrt(D)

    q_strides = (q.stride(0), q.stride(1), B * q.stride(2), q.stride(2), q.stride(3))
    k_strides = (k.stride(0), k.stride(1), B * k.stride(2), k.stride(2), k.stride(3))
    v_strides = (v.stride(0), v.stride(1), B * v.stride(2), v.stride(2), v.stride(3))

    # Intermediate buffers, contiguous in the layouts the kernels assume
    # (_strides_bm, _strides_mb)
    al = torch.empty(E, H, B, M, D, device=q.device, dtype=q.dtype)
    y = torch.empty_like(al)
    cl = torch.empty(E, H, B, M, device=q.device, dtype=torch.float)
    # Only needed for T > 1: ar, cr and each query's log-sum-exp over blocks
    ar = torch.empty(E, H, M, B, D, device=q.device, dtype=q.dtype) if T > 1 else None
    ar_strides = (H * M * B * D, M * B * D, B * D, D, 1)
    cr = torch.empty(E, H, M, B, device=q.device, dtype=torch.float) if T > 1 else None
    # _ar_cr_kernel needs each query's log-sum-exp over blocks unless one
    # program holds all of them
    all_blocks = config_m["BLOCK_R"] >= M
    lse = torch.empty_like(cl) if T > 1 and not all_blocks else None

    z = torch.empty_like(v)
    z_strides = (z.stride(0), z.stride(1), B * z.stride(2), z.stride(2), z.stride(3))

    attn_mask_strides = (
        (attn_mask.stride(0), B * attn_mask.stride(1), attn_mask.stride(1))
        if attn_mask is not None
        else (0, 0, 0)
    )

    for t in range(T - 1):
        is_first_call = t == 0
        _ar = q if is_first_call else ar
        _ar_strides = q_strides if is_first_call else ar_strides
        _al_cl_kernel[grid_al_cl](
            _ar,
            *_ar_strides,
            k,
            *k_strides,
            v,
            *v_strides,
            cr,
            al,
            y,
            cl,
            attn_mask,
            *attn_mask_strides,
            *HMBDN,
            IS_FIRST_CALL=is_first_call,  # type: ignore
            qk_scale=qk_scale,
            HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
            COMPUTE_Y=False,  # type: ignore
            **config_b,
        )

        if not all_blocks:
            _z_kernel[grid_z](
                al,
                q,
                *q_strides,
                y,
                cl,
                lse,
                z,
                *z_strides,
                *HMBDN,
                BLOCK_D=BLOCK_D,  # type: ignore
                PRE_PAD=pre_pad,  # type: ignore
                COMPUTE_Z=False,  # type: ignore
                **config_m,
            )

        _ar_cr_kernel[grid_z](
            al,
            q,
            *q_strides,
            cl,
            lse,
            ar,
            cr,
            attn_mask,
            *attn_mask_strides,
            *HMBDN,
            HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
            ALL_BLOCKS=all_blocks,  # type: ignore
            **config_m,
        )

    is_first_call_y = T == 1
    _ar_y = q if is_first_call_y else ar
    _ar_y_strides = q_strides if is_first_call_y else ar_strides

    _al_cl_kernel[grid_al_cl](
        _ar_y,
        *_ar_y_strides,
        k,
        *k_strides,
        v,
        *v_strides,
        cr,
        al,
        y,
        cl,
        attn_mask,
        *attn_mask_strides,
        *HMBDN,
        IS_FIRST_CALL=is_first_call_y,  # type: ignore
        qk_scale=qk_scale,
        HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
        PRE_PAD=pre_pad,  # type: ignore
        COMPUTE_Y=True,  # type: ignore
        **config_b,
    )

    _z_kernel[grid_z](
        al,
        q,
        *q_strides,
        y,
        cl,
        lse,
        z,
        *z_strides,
        *HMBDN,
        BLOCK_D=BLOCK_D,  # type: ignore
        PRE_PAD=pre_pad,  # type: ignore
        COMPUTE_Z=True,  # type: ignore
        **config_m,
    )

    return z
