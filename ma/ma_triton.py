from functools import cache
from math import e, log2, sqrt

import torch
import triton
import triton.language as tl
from triton import knobs
from torch.library import triton_op, wrap_triton


Tensor = torch.Tensor


@cache
def _config(n: int, blocks: bool) -> dict:
    """Launch configuration for a softmax over n entries: the keys of a block
    (_al_cl_kernel) or the blocks (`blocks`: _ar_cr_kernel, _z_kernel), tuned on
    an A100 SXM and an H100 for batch sizes 1 and 8.

    Each program takes BLOCK_R query rows and streams over the n entries in
    chunks of BLOCK_C. Rows are capped at 128 (64 if 128 would pad much more,
    and 64 for the keys of blocks of 90 to 255: the smaller tiles let more warps
    hide memory latency). For n >= 90, deeper pipelining keeps more loads in
    flight: 3 stages for 128-row tiles, 4 for 64-row tiles.
    """
    rows = min(max(triton.next_power_of_2(n), 16), 128)
    if n > 128 and triton.cdiv(n, 128) * 128 - n > n // 8:
        rows = 64
    if not blocks and 90 <= n < 256:
        rows = 64
    return dict(
        BLOCK_R=rows,
        BLOCK_C=min(rows, 64 if rows == 128 and n >= 128 else 32),
        num_warps=2 if n <= 32 else 4,
        num_stages=2 if n < 90 else (3 if rows == 128 else 4),
    )


@cache
def _z_config(n: int) -> dict:
    """_z_kernel's configuration: _config's, but from 90 to 255 blocks with at
    most 168 registers per thread, which fits a third program per SM (+2-7% on
    an A100 SXM and an H100). Caps help only some shapes, so not elsewhere."""
    config = _config(n, blocks=True)
    if 90 <= n < 256:
        rows = config["BLOCK_R"]
        config = config | dict(
            BLOCK_C=32 if rows == 128 else 64, num_stages=3, maxnreg=168
        )
    return config


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
    al_scale_ptr,
    y_scale_ptr,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    H: int,
    M: int,
    B: int,
    D: int,
    N: int,
    b0: int,  # first position of the group (see _sliced)
    Bg: int,  # positions in the group
    NUM_CHUNKS: tl.constexpr,  # of BLOCK_R rows; a constant, so // and % are cheap
    IS_FIRST_CALL: tl.constexpr,
    QK_SCALE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    FP8: tl.constexpr,  # al and y are FP8 with per-row scales (cl's layout)
    COMPUTE_Y: tl.constexpr,
):
    # Intermediate buffers: al, y, cl (and lse) are laid out (E, H, B, M, ...) so
    # that _z_kernel and _ar_cr_kernel read them contiguously; ar and cr are
    # (E, H, M, B, ...) for _al_cl_kernel
    stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d = _strides_bm(
        H, M, Bg, D
    )
    stride_y_e, stride_y_h, stride_y_m, stride_y_b, stride_y_d = _strides_bm(
        H, M, Bg, D
    )
    stride_cr_e, stride_cr_h, stride_cr_m, stride_cr_b, _ = _strides_mb(H, M, Bg, 1)
    stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b, _ = _strides_bm(H, M, Bg, 1)
    # One program per (batch, head, block m, chunk of BLOCK_R query rows). It
    # streams over the block's keys in chunks of BLOCK_C with an online softmax.
    # A block's chunks are consecutive programs, so they share its keys in L2.
    idx_ehm = tl.program_id(0) // NUM_CHUNKS
    idx_chunk = tl.program_id(0) % NUM_CHUNKS
    idx_eh = idx_ehm // M
    idx_e = (idx_eh // H).to(tl.int64)  # 64-bit offsets for inputs over 2^31 elements
    idx_h = (idx_eh % H).to(tl.int64)
    idx_m = idx_ehm % M

    pad_offset = M * B - N if PRE_PAD else 0

    range_r = b0 + idx_chunk * BLOCK_R + tl.arange(0, BLOCK_R)  # positions
    local_r = range_r - b0  # positions within the group
    range_d = tl.arange(0, BLOCK_D)
    range_n_r = B * idx_m + range_r

    mask_r = local_r < Bg
    pad_mask_r = mask_r & ((range_n_r >= pad_offset) if PRE_PAD else range_n_r < N)
    mask_d = range_d < D

    # Load ar
    ar_block_ptr = (
        ar_ptr
        + stride_ar_e * idx_e
        + stride_ar_h * idx_h
        + stride_ar_m * idx_m
        + (
            stride_ar_b * (range_r - (pad_offset if IS_FIRST_CALL else b0))[:, None]
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
            + (stride_cr_b * local_r)
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
        s = QK_SCALE * tl.dot(ar, tl.trans(k))
        if not IS_FIRST_CALL:
            # Positions without valid queries have cr = 0 (and s = 0): as in
            # the reference, add a tiny epsilon so padded tokens stay finite
            s = s / (cr[:, None] + 1e-12)
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
        + (stride_cl_b * local_r)
    )
    tl.store(cl_block_ptr, cl, mask=mask_r)

    # Store al (and y): FP8 rows divided by their max magnitude / 448, the
    # largest FP8 e4m3 value, plus that scale per row
    al = QK_SCALE * acc_al * inv_denom[:, None]
    scale_ptr_offset = (
        stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_m * idx_m
        + (stride_cl_b * local_r)
    )
    if FP8:
        al_scale = tl.maximum(tl.max(tl.abs(al), axis=1), 1e-30) / 448.0
        tl.store(al_scale_ptr + scale_ptr_offset, al_scale, mask=mask_r)
        al = al / al_scale[:, None]
    al = al.to(al_ptr.dtype.element_ty)
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_m * idx_m
        + (stride_al_b * local_r[:, None] + stride_al_d * range_d[None, :])
    )
    tl.store(al_block_ptr, al, mask=mask_r[:, None] & mask_d[None, :])

    if COMPUTE_Y:
        y = acc_y * inv_denom[:, None]
        if FP8:
            y_scale = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-30) / 448.0
            tl.store(y_scale_ptr + scale_ptr_offset, y_scale, mask=mask_r)
            y = y / y_scale[:, None]
        y = y.to(y_ptr.dtype.element_ty)
        y_block_ptr = (
            y_ptr
            + stride_y_e * idx_e
            + stride_y_h * idx_h
            + stride_y_m * idx_m
            + (stride_y_b * local_r[:, None] + stride_y_d * range_d[None, :])
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
    al_scale_ptr,
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
    b0: int,  # first position of the group (see _sliced)
    Bg: int,  # positions in the group
    NUM_CHUNKS: tl.constexpr,  # of BLOCK_R rows; a constant, so // and % are cheap
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    FP8: tl.constexpr,  # al and y are FP8 with per-row scales (cl's layout)
    ALL_BLOCKS: tl.constexpr,
):
    # Intermediate buffers: al, y, cl (and lse) are laid out (E, H, B, M, ...) so
    # that _z_kernel and _ar_cr_kernel read them contiguously; ar and cr are
    # (E, H, M, B, ...) for _al_cl_kernel
    stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d = _strides_bm(
        H, M, Bg, D
    )
    stride_ar_e, stride_ar_h, stride_ar_m, stride_ar_b, stride_ar_d = _strides_mb(
        H, M, Bg, D
    )
    stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b, _ = _strides_bm(H, M, Bg, 1)
    stride_cr_e, stride_cr_h, stride_cr_m, stride_cr_b, _ = _strides_mb(H, M, Bg, 1)
    # One program per (batch, head, position b, chunk of BLOCK_R blocks). It
    # streams over the queries in chunks of BLOCK_C. Each query's softmax over
    # blocks is normalized directly if the program holds all blocks (ALL_BLOCKS),
    # and otherwise by its log-sum-exp from _z_kernel(COMPUTE_Z=False). A
    # position's chunks are consecutive programs, so they share its queries in L2.
    idx_ehb = tl.program_id(0) // NUM_CHUNKS
    idx_chunk = tl.program_id(0) % NUM_CHUNKS
    idx_eh = idx_ehb // Bg
    idx_e = (idx_eh // H).to(tl.int64)  # 64-bit offsets for inputs over 2^31 elements
    idx_h = (idx_eh % H).to(tl.int64)
    idx_bl = idx_ehb % Bg  # position within the group
    idx_b = b0 + idx_bl

    pad_offset = M * B - N if PRE_PAD else 0

    range_r = idx_chunk * BLOCK_R + tl.arange(0, BLOCK_R)
    range_d = tl.arange(0, BLOCK_D)
    mask_r = range_r < M
    mask_d = range_d < D

    # Load al and cl
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_b * idx_bl
        + (stride_al_m * range_r[:, None] + stride_al_d * range_d[None, :])
    )
    al = tl.load(al_block_ptr, mask=mask_r[:, None] & mask_d[None, :], other=0.0)
    if FP8:
        al_scale_ptrs = (
            al_scale_ptr
            + stride_cl_e * idx_e
            + stride_cl_h * idx_h
            + stride_cl_b * idx_bl
            + (stride_cl_m * range_r)
        )
        al_scale = tl.load(al_scale_ptrs, mask=mask_r, other=0.0)
        al = (
            al.to(q_ptr.dtype.element_ty) * al_scale.to(q_ptr.dtype.element_ty)[:, None]
        )
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_b * idx_bl
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
                + stride_cl_b * idx_bl
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
        + stride_cr_b * idx_bl
        + (stride_cr_m * range_r)
    )
    tl.store(cr_block_ptr, acc_cr, mask=mask_r)
    ar_block_ptr = (
        ar_ptr
        + stride_ar_e * idx_e
        + stride_ar_h * idx_h
        + stride_ar_b * idx_bl
        + (stride_ar_m * range_r[:, None] + stride_ar_d * range_d[None, :])
    )
    tl.store(
        ar_block_ptr,
        acc_ar.to(ar_ptr.dtype.element_ty),
        mask=mask_r[:, None] & mask_d[None, :],
    )


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
    al_scale_ptr,
    y_scale_ptr,
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
    b0: int,  # first position of the group (see _sliced)
    Bg: int,  # positions in the group
    NUM_CHUNKS: tl.constexpr,  # of BLOCK_R rows; a constant, so // and % are cheap
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    FP8: tl.constexpr,  # al and y are FP8 with per-row scales (cl's layout)
    COMPUTE_Z: tl.constexpr,
):
    # Intermediate buffers: al, y, cl (and lse) are laid out (E, H, B, M, ...) so
    # that _z_kernel and _ar_cr_kernel read them contiguously; ar and cr are
    # (E, H, M, B, ...) for _al_cl_kernel
    stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d = _strides_bm(
        H, M, Bg, D
    )
    stride_y_e, stride_y_h, stride_y_m, stride_y_b, stride_y_d = _strides_bm(
        H, M, Bg, D
    )
    stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b, _ = _strides_bm(H, M, Bg, 1)
    # One program per (batch, head, position b, chunk of BLOCK_R query blocks).
    # It streams over the blocks' al, cl and y in chunks of BLOCK_C with an
    # online softmax. With COMPUTE_Z=False it only stores each query's
    # log-sum-exp over blocks, for _ar_cr_kernel. A position's chunks are
    # consecutive programs, so they share its al, cl and y in L2.
    idx_ehb = tl.program_id(0) // NUM_CHUNKS
    idx_chunk = tl.program_id(0) % NUM_CHUNKS
    idx_eh = idx_ehb // Bg
    idx_e = (idx_eh // H).to(tl.int64)  # 64-bit offsets for inputs over 2^31 elements
    idx_h = (idx_eh % H).to(tl.int64)
    idx_bl = idx_ehb % Bg  # position within the group
    idx_b = b0 + idx_bl

    pad_offset = M * B - N if PRE_PAD else 0

    range_r = idx_chunk * BLOCK_R + tl.arange(0, BLOCK_R)
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
            + stride_al_b * idx_bl
            + (stride_al_m * range_c[:, None] + stride_al_d * range_d[None, :])
        )
        al = tl.load(al_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        scale_ptr_offset = (
            stride_cl_e * idx_e
            + stride_cl_h * idx_h
            + stride_cl_b * idx_bl
            + (stride_cl_m * range_c)
        )
        if FP8:
            al_scale = tl.load(al_scale_ptr + scale_ptr_offset, mask=mask_c, other=0.0)
            al = al.to(q.dtype) * al_scale.to(q.dtype)[:, None]
        cl_block_ptr = (
            cl_ptr
            + stride_cl_e * idx_e
            + stride_cl_h * idx_h
            + stride_cl_b * idx_bl
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
                + stride_y_b * idx_bl
                + (stride_y_m * range_c[:, None] + stride_y_d * range_d[None, :])
            )
            y = tl.load(y_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
            if FP8:
                y_scale = tl.load(
                    y_scale_ptr + scale_ptr_offset, mask=mask_c, other=0.0
                )
                y = y.to(q.dtype) * y_scale.to(q.dtype)[:, None]
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
            + stride_cl_b * idx_bl
            + (stride_cl_m * range_r)
        )
        tl.store(lse_block_ptr, row_max + tl.log2(denom), mask=mask_r)


def _monarch_attention(
    q, k, v, attn_mask, T, B, pre_pad, launch, z, b0=0, Bg=None, fp8=False
) -> None:
    """The kernel launches, writing the output into z; `launch` wraps each
    kernel (wrap_triton in the op). Only positions b0 to b0 + Bg of each block
    are computed (default: all), which bounds the intermediate buffers."""
    E, H, N, D = q.shape
    M = triton.cdiv(N, B)
    Bg = B if Bg is None else Bg

    HMBDN = (H, M, B, D, N, b0, Bg)

    # _al_cl_kernel: softmax over B keys, _ar_cr_kernel and _z_kernel: softmax
    # over M blocks
    config_b = _config(B, blocks=False)
    # Rows are positions, so a narrow group of positions takes fewer per program
    rows_b = min(config_b["BLOCK_R"], max(triton.next_power_of_2(Bg), 16))
    config_b = config_b | dict(BLOCK_R=rows_b)
    config_m = _config(M, blocks=True)
    config_z = _z_config(M)  # same tiles as config_m
    chunks_b = triton.cdiv(Bg, config_b["BLOCK_R"])
    chunks_m = triton.cdiv(M, config_m["BLOCK_R"])
    grid_al_cl = (E * H * M * chunks_b, 1)
    grid_z = (E * H * Bg * chunks_m, 1)

    BLOCK_D = max(triton.next_power_of_2(D), 16)

    # The kernels work in base 2 (exp2 and log2 are native instructions): logits
    # are scaled by log2(e), and so are al, cl and lse.
    qk_scale = log2(e) / sqrt(D)

    q_strides = (q.stride(0), q.stride(1), B * q.stride(2), q.stride(2), q.stride(3))
    k_strides = (k.stride(0), k.stride(1), B * k.stride(2), k.stride(2), k.stride(3))
    v_strides = (v.stride(0), v.stride(1), B * v.stride(2), v.stride(2), v.stride(3))

    # Intermediate buffers, contiguous in the layouts the kernels assume
    # (_strides_bm, _strides_mb)
    stored = torch.float8_e4m3fn if fp8 else q.dtype  # al and y
    al = torch.empty(E, H, Bg, M, D, device=q.device, dtype=stored)
    y = torch.empty_like(al)
    cl = torch.empty(E, H, Bg, M, device=q.device, dtype=torch.float)
    # FP8 al and y: one scale per row, in cl's layout
    al_scale = torch.empty_like(cl) if fp8 else None
    y_scale = torch.empty_like(cl) if fp8 else None
    # Only needed for T > 1: ar, cr and each query's log-sum-exp over blocks
    ar = torch.empty(E, H, M, Bg, D, device=q.device, dtype=q.dtype) if T > 1 else None
    ar_strides = (H * M * Bg * D, M * Bg * D, Bg * D, D, 1)
    cr = torch.empty(E, H, M, Bg, device=q.device, dtype=torch.float) if T > 1 else None
    # _ar_cr_kernel needs each query's log-sum-exp over blocks unless one
    # program holds all of them
    all_blocks = config_m["BLOCK_R"] >= M
    lse = torch.empty_like(cl) if T > 1 and not all_blocks else None

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
        launch(_al_cl_kernel)[grid_al_cl](
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
            al_scale,
            y_scale,
            attn_mask,
            *attn_mask_strides,
            *HMBDN,
            NUM_CHUNKS=chunks_b,  # type: ignore
            IS_FIRST_CALL=is_first_call,  # type: ignore
            QK_SCALE=qk_scale,  # type: ignore
            HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
            FP8=fp8,  # type: ignore
            COMPUTE_Y=False,  # type: ignore
            **config_b,
        )

        if not all_blocks:
            launch(_z_kernel)[grid_z](
                al,
                q,
                *q_strides,
                y,
                cl,
                lse,
                al_scale,
                y_scale,
                z,
                *z_strides,
                *HMBDN,
                NUM_CHUNKS=chunks_m,  # type: ignore
                BLOCK_D=BLOCK_D,  # type: ignore
                PRE_PAD=pre_pad,  # type: ignore
                FP8=fp8,  # type: ignore
                COMPUTE_Z=False,  # type: ignore
                **config_z,
            )

        launch(_ar_cr_kernel)[grid_z](
            al,
            q,
            *q_strides,
            cl,
            lse,
            al_scale,
            ar,
            cr,
            attn_mask,
            *attn_mask_strides,
            *HMBDN,
            NUM_CHUNKS=chunks_m,  # type: ignore
            HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
            FP8=fp8,  # type: ignore
            ALL_BLOCKS=all_blocks,  # type: ignore
            **config_m,
        )

    is_first_call_y = T == 1
    _ar_y = q if is_first_call_y else ar
    _ar_y_strides = q_strides if is_first_call_y else ar_strides

    launch(_al_cl_kernel)[grid_al_cl](
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
        al_scale,
        y_scale,
        attn_mask,
        *attn_mask_strides,
        *HMBDN,
        NUM_CHUNKS=chunks_b,  # type: ignore
        IS_FIRST_CALL=is_first_call_y,  # type: ignore
        QK_SCALE=qk_scale,  # type: ignore
        HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
        PRE_PAD=pre_pad,  # type: ignore
        FP8=fp8,  # type: ignore
        COMPUTE_Y=True,  # type: ignore
        **config_b,
    )

    launch(_z_kernel)[grid_z](
        al,
        q,
        *q_strides,
        y,
        cl,
        lse,
        al_scale,
        y_scale,
        z,
        *z_strides,
        *HMBDN,
        NUM_CHUNKS=chunks_m,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
        PRE_PAD=pre_pad,  # type: ignore
        FP8=fp8,  # type: ignore
        COMPUTE_Z=True,  # type: ignore
        **config_z,
    )


MAX_WORKSPACE = 256 * 2**20  # bytes of intermediate buffers per call, by default


def _workspace_per_head(q, T, B, fp8=False) -> int:
    """Bytes of intermediate buffers for one (batch element, head)."""
    M = triton.cdiv(q.shape[2], B)
    tokens, D, size = M * B, q.shape[3], q.element_size()
    stored = 1 if fp8 else size  # al and y
    per_head = 2 * tokens * D * stored + 4 * tokens  # al, y, cl
    if fp8:
        per_head += 2 * 4 * tokens  # their scales
    if T > 1:
        per_head += tokens * D * size + 2 * 4 * tokens  # ar, cr, lse
    return per_head


def _sliced(
    q, k, v, attn_mask, T, B, pre_pad, max_workspace, fp8, make_launch
) -> Tensor:
    """Run the kernels on groups whose intermediate buffers fit in max_workspace
    bytes (0: no limit), one group at a time: batch elements, else heads of one
    batch element, else positions of one head. Every position b of a block only
    needs position b of the other blocks' intermediates, so positions split too
    (each group re-reads all keys and values)."""
    E, H = q.shape[:2]
    z = torch.empty_like(v)
    per_head = _workspace_per_head(q, T, B, fp8)
    heads = max_workspace // per_head if max_workspace else E * H
    everything = (slice(None), slice(None), 0, B)
    if heads >= E * H:
        groups = [everything]
    elif heads >= H:  # whole batch elements (the mask is per batch element)
        per = heads // H
        groups = [(slice(e, e + per), slice(None), 0, B) for e in range(0, E, per)]
    elif heads >= 1:  # heads of one batch element
        groups = [
            (slice(e, e + 1), slice(h, h + heads), 0, B)
            for e in range(E)
            for h in range(0, H, heads)
        ]
    else:  # positions of one head; multiples of 16 fill the kernels' tiles
        per = max(max_workspace * B // per_head, 1)
        per = per // 16 * 16 if per >= 16 else per
        groups = [
            (slice(e, e + 1), slice(h, h + 1), b0, min(per, B - b0))
            for e in range(E)
            for h in range(H)
            for b0 in range(0, B, per)
        ]
    for i, (batch, head, b0, Bg) in enumerate(groups):
        mask = None if attn_mask is None else attn_mask[batch]
        _monarch_attention(
            q[batch, head],
            k[batch, head],
            v[batch, head],
            mask,
            T,
            B,
            pre_pad,
            make_launch(i),
            z[batch, head],
            b0,
            Bg,
            fp8,
        )
    return z


# A PyTorch custom op, so that torch.compile can trace the kernel launches (and
# CUDA graphs can capture them). Eager calls skip it: the dispatcher adds ~40 us.
@triton_op("ma::monarch_attention_triton", mutates_args=())
def _monarch_attention_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
    max_workspace: int,
    fp8: bool,
) -> Tensor:
    return _sliced(
        q, k, v, attn_mask, T, B, pre_pad, max_workspace, fp8, lambda _: wrap_triton
    )


def monarch_attention_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
    max_workspace: int | None = MAX_WORKSPACE,
    fp8: bool = False,
) -> Tensor:
    """MonarchAttention with the Triton kernels. The intermediate buffers take
    ~2x the output's memory; with max_workspace (bytes; None: no limit), heads
    and batch elements are processed in groups whose buffers fit in it. With
    fp8, the buffers al and y are FP8 e4m3 with per-row scales (half the
    memory and less traffic; needs a GPU with FP8 support, e.g. an H100)."""
    max_workspace = max_workspace or 0
    if torch.compiler.is_compiling():
        return _monarch_attention_op(
            q, k, v, attn_mask, T, B, pre_pad, max_workspace, fp8
        )
    # Everything Triton specializes the kernels on (shapes, strides, dtypes,
    # pointer alignment) is a function of this key and the group number
    key = (
        q.device,
        q.dtype,
        q.shape,
        q.stride(),
        k.stride(),
        v.stride(),
        T,
        B,
        pre_pad,
        max_workspace,
        fp8,
    )
    key += (q.data_ptr() % 16, k.data_ptr() % 16, v.data_ptr() % 16)
    if attn_mask is not None:
        key += (attn_mask.dtype, attn_mask.shape, attn_mask.stride())
        key += (attn_mask.data_ptr() % 16,)
    return _sliced(
        q,
        k,
        v,
        attn_mask,
        T,
        B,
        pre_pad,
        max_workspace,
        fp8,
        lambda group: _CachedLaunch(key + (group,), q.device.index),
    )


# Compiled kernel and names of its keyword arguments, per (key, launch number)
_LAUNCH_CACHE: dict = {}


class _CachedLaunch:
    """Launches the kernels of one monarch_attention_triton call.

    The first call with a given key goes through Triton's JIT launcher, which
    costs ~20 us of host time per launch; later calls launch the compiled
    kernels directly.
    """

    def __init__(self, key, device_index):
        self.key = key
        self.count = 0
        self.hooks = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
        # e.g. a profiler: go through Triton's launcher, which calls the hooks
        self.hooked = any(hook.calls for hook in self.hooks)
        self.stream = torch._C._cuda_getCurrentRawStream(device_index)

    def __call__(self, kernel):
        self.kernel = kernel
        self.count += 1
        return self

    def __getitem__(self, grid):
        self.grid = grid
        return self._run

    def _run(self, *args, **kwargs):
        entry = _LAUNCH_CACHE.get((self.key, self.count))
        if entry is None or self.hooked:
            compiled = self.kernel[self.grid](*args, **kwargs)
            names = self.kernel.arg_names[len(args) :]
            _LAUNCH_CACHE[(self.key, self.count)] = compiled, names
            return
        compiled, names = entry
        grid = self.grid
        compiled.run(
            grid[0],
            grid[1],
            1,
            self.stream,
            compiled.function,
            compiled.packed_metadata,
            None,
            *self.hooks,
            *args,
            *(kwargs[name] for name in names),
        )
