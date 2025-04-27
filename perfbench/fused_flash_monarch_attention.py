from math import sqrt

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

Tensor = torch.Tensor


@triton.jit
def flash_monarch_attn_kernel(
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
    o_ptr,
    stride_o_e,
    stride_o_h,
    stride_o_m,
    stride_o_b,
    stride_o_d,
    H: int,
    sm_scale: float,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx_eh = tl.program_id(0)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    q_block_ptr = (
        q_ptr
        + stride_q_e * idx_e
        + stride_q_h * idx_h
        + (
            stride_q_m * tl.arange(0, BLOCK_M)[:, None, None]
            + stride_q_b * tl.arange(0, BLOCK_B)[None, :, None]
            + stride_q_d * tl.arange(0, BLOCK_D)[None, None, :]
        )
    )
    q = tl.load(q_block_ptr)
    q_t = tl.permute(q, (1, 0, 2))

    k_block_ptr = (
        k_ptr
        + stride_k_e * idx_e
        + stride_k_h * idx_h
        + (
            stride_k_m * tl.arange(0, BLOCK_M)[:, None, None]
            + stride_k_b * tl.arange(0, BLOCK_B)[None, :, None]
            + stride_k_d * tl.arange(0, BLOCK_D)[None, None, :]
        )
    )
    k = tl.load(k_block_ptr)

    v_block_ptr = (
        v_ptr
        + stride_v_e * idx_e
        + stride_v_h * idx_h
        + (
            stride_v_m * tl.arange(0, BLOCK_M)[:, None, None]
            + stride_v_b * tl.arange(0, BLOCK_B)[None, :, None]
            + stride_v_d * tl.arange(0, BLOCK_D)[None, None, :]
        )
    )
    v = tl.load(v_block_ptr)

    r = sm_scale * tl.dot(q, tl.permute(k, (0, 2, 1)))
    r = tl.exp(r - tl.max(r, axis=2, keep_dims=True))
    r = r / tl.sum(r, axis=2, keep_dims=True)

    cl = tl.sum(r * tl.log(r), axis=2, keep_dims=True)

    a = (sm_scale * tl.dot(r.to(k.dtype), k)).to(k.dtype)

    a_t = tl.permute(a, (1, 0, 2))

    l = tl.dot(a_t, tl.permute(q_t, (0, 2, 1)))
    l = l - cl
    l = tl.exp(l - tl.max(l, axis=1, keep_dims=True))
    l = l / tl.sum(l, axis=1, keep_dims=True)

    cr = tl.sum(l, axis=2, keep_dims=True)

    a_t = tl.dot(l.to(q.dtype), q).to(q.dtype)

    a = tl.permute(a_t, (1, 0, 2))

    r = sm_scale * tl.dot(a, tl.permute(k, (0, 2, 1)))

    r = tl.exp(r - tl.max(r, axis=2, keep_dims=True))
    r = r / tl.sum(r, axis=2, keep_dims=True)
    r = r / cr

    cl = tl.sum(r * tl.log(r), axis=2, keep_dims=True)

    a = (sm_scale * tl.dot(r.to(k.dtype), k)).to(k.dtype)
    y = tl.dot(r.to(v.dtype), v).to(v.dtype)

    a_t = tl.permute(a, (1, 0, 2))
    y_t = tl.permute(y, (1, 0, 2))

    l = tl.dot(q_t, tl.permute(a_t, (0, 2, 1)))
    l = l - cl
    l = tl.exp(l - tl.max(l, axis=2, keep_dims=True))
    l = l / tl.sum(l, axis=2, keep_dims=True)

    z = tl.dot(l.to(y_t.dtype), y_t).to(y_t.dtype)

    o_block_ptr = (
        o_ptr
        + stride_o_e * idx_e
        + stride_o_h * idx_h
        + (
            stride_o_b * tl.arange(0, BLOCK_B)[:, None, None]
            + stride_o_m * tl.arange(0, BLOCK_M)[None, :, None]
            + stride_o_d * tl.arange(0, BLOCK_D)[None, :, :]
        )
    )

    tl.store(o_block_ptr, z)

def fused_flash_monarch_attention(q: Tensor, k: Tensor, v: Tensor, B: int) -> Tensor:
    E, H, N, D = q.shape
    M = triton.cdiv(N, B)

    grid = (E * H,)

    BLOCK_B = max(triton.next_power_of_2(B), 16)
    BLOCK_M = max(triton.next_power_of_2(M), 16)
    BLOCK_D = max(triton.next_power_of_2(D), 16)

    o = torch.empty_like(q)

    sm_scale = 1 / sqrt(D)

    q_strides = (q.stride(0), q.stride(1), B * q.stride(2), q.stride(2), q.stride(3))
    k_strides = (k.stride(0), k.stride(1), B * k.stride(2), k.stride(2), k.stride(3))
    v_strides = (v.stride(0), v.stride(1), B * v.stride(2), v.stride(2), v.stride(3))
    o_strides = (o.stride(0), o.stride(1), B * o.stride(2), o.stride(2), o.stride(3))

    flash_monarch_attn_kernel[grid](
        q,
        *q_strides,
        k,
        *k_strides,
        v,
        *v_strides,
        o,
        *o_strides,
        H,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
        num_warps=16  # type: ignore
    )

    return o


def test():
    torch.cuda.manual_seed(0)

    q = torch.randn(1, 12, 256, 64, dtype=torch.float16).cuda()
    k = torch.randn(1, 12, 256, 64, dtype=torch.float16).cuda()
    v = torch.randn(1, 12, 256, 64, dtype=torch.float16).cuda()

    fused_flash_monarch_attention(q, k, v, 16)


if __name__ == "__main__":
    test()
