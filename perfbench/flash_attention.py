from math import sqrt

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from utils import check_inputs

Tensor = torch.Tensor


def _get_flash_attn_kernel_autotune_config():
    return [
        # triton.Config({'BLOCK_N': BLOCK, 'BLOCK_M': BLOCK}, num_stages=num_stages, num_warps=num_warps)
        # for BLOCK in [32, 64, 128, 256]
        # for num_stages in [1]
        # for num_warps in [2, 4, 8]
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=1, num_warps=8)
    ]


@triton.autotune(configs=_get_flash_attn_kernel_autotune_config(), key=["H", "N", "D"])
@triton.jit
def _flash_attn_kernel(
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    k_ptr,
    stride_k_e,
    stride_k_h,
    stride_k_n,
    stride_k_d,
    v_ptr,
    stride_v_e,
    stride_v_h,
    stride_v_n,
    stride_v_d,
    o_ptr,
    stride_o_e,
    stride_o_h,
    stride_o_m,
    stride_o_d,
    mask_ptr,
    stride_mask_e,
    stride_mask_n,
    H: int,
    N: int,
    D: int,
    sm_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Compute indices
    idx_eh = tl.program_id(0)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    idx_m_block = tl.program_id(1)
    start_m = BLOCK_M * idx_m_block
    start_m = tl.multiple_of(start_m, BLOCK_M)

    # Load q
    q_block_ptr = (
        q_ptr
        + stride_q_e * idx_e
        + stride_q_h * idx_h
        + (
            stride_q_m * (start_m + tl.arange(0, BLOCK_M)[:, None])
            + stride_q_d * tl.arange(0, BLOCK_D)[None, :]
        )
    )
    q = tl.load(
        q_block_ptr,
        mask=(start_m + tl.arange(0, BLOCK_M)[:, None] < N)
        & (tl.arange(0, BLOCK_D)[None, :] < D),
        other=0.0,
    )

    # Initialize statistics and accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load k
        k_block_ptr = (
            k_ptr
            + stride_k_e * idx_e
            + stride_k_h * idx_h
            + (
                stride_k_n * (start_n + tl.arange(0, BLOCK_N)[:, None])
                + stride_k_d * tl.arange(0, BLOCK_D)[None, :]
            )
        )
        k = tl.load(
            k_block_ptr,
            mask=((start_n + tl.arange(0, BLOCK_N)[:, None]) < N)
            & (tl.arange(0, BLOCK_D)[None, :] < D),
            other=0.0,
        )

        # Attention matrix
        qk = tl.dot(q, tl.trans(k))
        qk = qk + tl.where(
            (start_n + tl.arange(0, BLOCK_N)[None, :]) < N, 0.0, float("-inf")
        )

        if HAS_ATTN_MASK:
            mask_block_ptr = (
                mask_ptr
                + stride_mask_e * idx_e
                + stride_mask_n * (start_n + tl.arange(0, BLOCK_N)[None, :])
            )
            mask = tl.load(
                mask_block_ptr,
                mask=(start_n + tl.arange(0, BLOCK_N)[None, :] < N),
                other=0,
            )
            qk = sm_scale * qk + tl.where(mask, 0.0, float("-inf"))
            m_ij = tl.maximum(tl.max(qk, axis=1), m_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(sm_scale * qk, axis=1), m_i)
            p = tl.exp(sm_scale * qk - m_ij[:, None])

        alpha = tl.exp(m_i - m_ij)
        o = alpha[:, None] * o

        # Load v
        v_block_ptr = (
            v_ptr
            + stride_v_e * idx_e
            + stride_v_h * idx_h
            + (
                stride_v_n * (start_n + tl.arange(0, BLOCK_N)[:, None])
                + stride_v_d * tl.arange(0, BLOCK_D)[None, :]
            )
        )
        v = tl.load(
            v_block_ptr,
            mask=((start_n + tl.arange(0, BLOCK_N)[:, None]) < N)
            & (tl.arange(0, BLOCK_D)[None, :] < D),
            other=0.0,
        )

        # Accumulate output
        o = o + tl.dot(p.to(v.dtype), v)

        # Update statistics
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    # Store output
    o = o / l_i[:, None]
    o_block_ptr = (
        o_ptr
        + stride_o_e * idx_e
        + stride_o_h * idx_h
        + (
            stride_o_m * (start_m + tl.arange(0, BLOCK_M)[:, None])
            + stride_o_d * tl.arange(0, BLOCK_D)[None, :]
        )
    )
    tl.store(
        o_block_ptr,
        o.to(q.dtype),
        mask=(start_m + tl.arange(0, BLOCK_M)[:, None] < N)
        & (tl.arange(0, BLOCK_D)[None, :] < D),
    )


def flash_attention(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None
) -> Tensor:
    check_inputs(q, k, v)
    E, H, N, D = q.shape
    grid = lambda meta: (E * H, triton.cdiv(N, meta["BLOCK_M"]))
    BLOCK_D = max(triton.next_power_of_2(D), 16)
    sm_scale = 1 / sqrt(D)
    attn_mask_strides = (
        (attn_mask.stride(0), attn_mask.stride(1)) if attn_mask is not None else (0, 0)
    )

    o = torch.empty_like(q)
    _flash_attn_kernel[grid](
        q,
        *q.stride(),
        k,
        *k.stride(),
        v,
        *v.stride(),
        o,
        *o.stride(),
        attn_mask,
        *attn_mask_strides,
        H,
        N,
        D,
        sm_scale,
        HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
    )
    return o


def test():
    torch.manual_seed(0)
    E, H, N, D = 8, 12, 256, 128
    attn_mask = torch.tensor([[True] * 128 + [True] * 128]).bool().cuda()
    # attn_mask = None
    q = torch.randn(E, H, N, D, dtype=torch.float16).cuda()
    k = torch.randn(E, H, N, D, dtype=torch.float16).cuda()
    v = torch.randn(E, H, N, D, dtype=torch.float16).cuda()

    o1 = flash_attention(q, k, v, attn_mask)
    o2 = F.scaled_dot_product_attention(
        q, k, v, attn_mask[:, None, None, :] if attn_mask is not None else None
    )
    print(o1[0, 0, :5, :5])
    print(o2[0, 0, :5, :5])


if __name__ == "__main__":
    test()
