import pytest
import torch

from ma.ma_torch import monarch_attention_torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _rand_qkv(E, H, N, D, dtype, device="cuda"):
    q = torch.randn(E, H, N, D, device=device, dtype=dtype)
    k = torch.randn(E, H, N, D, device=device, dtype=dtype)
    v = torch.randn(E, H, N, D, device=device, dtype=dtype)
    return q, k, v


def _make_block_safe_mask(E, N, B, device="cuda"):
    """Generate a random boolean mask with at least one True per block."""
    M = (N + B - 1) // B
    mask = torch.rand(E, N, device=device) > 0.3
    # Ensure at least one True per block to avoid all-masked softmax (NaN)
    for e in range(E):
        for m in range(M):
            start = m * B
            end = min(start + B, N)
            if not mask[e, start:end].any():
                mask[e, start] = True
    return mask


# Focused parameter combos — float16 gives tight agreement, bfloat16 needs
# divisible N and small T to stay within tolerance.
MATCH_PARAMS = [
    # (E, H, N, D, B, T, pre_pad, dtype)
    # float16 cases
    (1, 1, 16, 16, 4, 2, False, torch.float16),
    (1, 1, 16, 16, 4, 2, True, torch.float16),
    (1, 4, 16, 16, 8, 3, False, torch.float16),
    (1, 1, 19, 16, 4, 2, False, torch.float16),  # non-divisible N, post-pad
    (1, 1, 19, 16, 4, 2, True, torch.float16),  # non-divisible N, pre-pad
    (2, 4, 32, 32, 8, 2, False, torch.float16),
    (2, 1, 24, 16, 8, 3, False, torch.float16),
    # bfloat16 cases (divisible N only — bfloat16 has lower mantissa precision)
    (2, 4, 16, 32, 4, 2, False, torch.bfloat16),
    (1, 1, 16, 32, 4, 2, True, torch.bfloat16),
    (1, 1, 24, 16, 8, 2, False, torch.bfloat16),
]


@requires_cuda
@pytest.mark.parametrize("E,H,N,D,B,T,pre_pad,dtype", MATCH_PARAMS)
def test_torch_triton_match(E, H, N, D, B, T, pre_pad, dtype):
    from ma.ma_triton import monarch_attention_triton

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    q, k, v = _rand_qkv(E, H, N, D, dtype)

    out_torch = monarch_attention_torch(q, k, v, None, T, B, pre_pad)
    out_triton = monarch_attention_triton(q, k, v, None, T, B, pre_pad)

    assert out_torch.shape == out_triton.shape
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(out_torch, out_triton, atol=atol, rtol=atol), (
        f"max diff: {(out_torch - out_triton).abs().max().item()}"
    )


MASK_PARAMS = [
    # (E, H, N, D, B, T, pre_pad, dtype)
    (1, 1, 16, 16, 4, 2, False, torch.float16),
    (1, 1, 16, 16, 4, 2, True, torch.float16),
    (1, 1, 19, 16, 4, 2, False, torch.float16),
    (1, 4, 19, 16, 4, 3, True, torch.float16),
    (1, 1, 32, 16, 8, 2, False, torch.float16),
    (2, 4, 32, 32, 8, 2, False, torch.float16),
    (1, 1, 16, 16, 4, 2, True, torch.bfloat16),
    (2, 4, 16, 32, 4, 2, False, torch.bfloat16),
]


@requires_cuda
@pytest.mark.parametrize("E,H,N,D,B,T,pre_pad,dtype", MASK_PARAMS)
def test_torch_triton_match_with_attention_mask(E, H, N, D, B, T, pre_pad, dtype):
    from ma.ma_triton import monarch_attention_triton

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    q, k, v = _rand_qkv(E, H, N, D, dtype)
    attn_mask = _make_block_safe_mask(E, N, B)

    out_torch = monarch_attention_torch(q, k, v, attn_mask, T, B, pre_pad)
    out_triton = monarch_attention_triton(q, k, v, attn_mask, T, B, pre_pad)

    assert out_torch.shape == out_triton.shape
    assert not torch.isnan(out_torch).any(), "torch output contains NaNs"
    assert not torch.isnan(out_triton).any(), "triton output contains NaNs"
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(out_torch, out_triton, atol=atol, rtol=atol), (
        f"max diff: {(out_torch - out_triton).abs().max().item()}"
    )


T1_PARAMS = [
    # (E, H, N, D, B, pre_pad, dtype)
    (1, 1, 16, 16, 4, False, torch.float16),
    (2, 4, 19, 32, 8, True, torch.bfloat16),
]


@requires_cuda
@pytest.mark.parametrize("E,H,N,D,B,pre_pad,dtype", T1_PARAMS)
def test_torch_only_t1(E, H, N, D, B, pre_pad, dtype):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    q, k, v = _rand_qkv(E, H, N, D, dtype)

    out = monarch_attention_torch(q, k, v, None, T=1, B=B, pre_pad=pre_pad)

    assert out.shape == (E, H, N, D)
    assert not torch.isnan(out).any(), "Output contains NaNs"


T1_TRITON_PARAMS = [
    # (E, H, N, D, B, pre_pad, dtype)
    (1, 1, 16, 16, 4, False, torch.float16),
    (1, 1, 16, 16, 4, True, torch.float16),
    (2, 4, 32, 32, 8, False, torch.float16),
    (1, 1, 19, 16, 4, False, torch.float16),
    (1, 1, 19, 16, 4, True, torch.float16),
    (2, 4, 16, 32, 4, False, torch.bfloat16),
]


@requires_cuda
@pytest.mark.parametrize("E,H,N,D,B,pre_pad,dtype", T1_TRITON_PARAMS)
def test_torch_triton_match_t1(E, H, N, D, B, pre_pad, dtype):
    from ma.ma_triton import monarch_attention_triton

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    q, k, v = _rand_qkv(E, H, N, D, dtype)

    out_torch = monarch_attention_torch(q, k, v, None, T=1, B=B, pre_pad=pre_pad)
    out_triton = monarch_attention_triton(q, k, v, None, T=1, B=B, pre_pad=pre_pad)

    assert out_torch.shape == out_triton.shape
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(out_torch, out_triton, atol=atol, rtol=atol), (
        f"max diff: {(out_torch - out_triton).abs().max().item()}"
    )
