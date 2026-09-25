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
    # Blocks longer than one key chunk
    (1, 2, 2000, 32, 90, 2, False, torch.float16),
    # More blocks than one program holds: _ar_cr_kernel uses _z_kernel's log-sum-exp
    (1, 2, 4500, 32, 16, 2, True, torch.float16),
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


@requires_cuda
@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize("masked", [False, True])
def test_torch_compile(T, masked):
    from ma import MonarchAttention, PadType

    torch.manual_seed(0)
    q, k, v = _rand_qkv(2, 4, 300, 32, torch.float16)
    mask = _make_block_safe_mask(2, 300, 16) if masked else None
    attn = MonarchAttention(16, T, PadType.post, impl="triton")
    compiled = torch.compile(attn, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, mask), attn(q, k, v, mask))


@requires_cuda
@pytest.mark.parametrize("T", [1, 2])
def test_cached_launches(T):
    """Repeat calls launch the compiled kernels directly; inputs whose pointers
    are aligned differently must not reuse kernels specialized for alignment."""
    from ma.ma_triton import monarch_attention_triton

    torch.manual_seed(0)
    q, k, v = _rand_qkv(2, 4, 300, 32, torch.float16)
    mask = _make_block_safe_mask(2, 300, 16)
    first = monarch_attention_triton(q, k, v, mask, T, 16, False)
    torch.testing.assert_close(
        monarch_attention_triton(q, k, v, mask, T, 16, False), first
    )
    # Same shapes and strides, but not 16-byte aligned
    buffers = [
        torch.empty(t.numel() + 1, device="cuda", dtype=t.dtype) for t in (q, k, v)
    ]
    unaligned = [b[1:].view_as(t).copy_(t) for b, t in zip(buffers, (q, k, v))]
    for _ in range(2):
        torch.testing.assert_close(
            monarch_attention_triton(*unaligned, mask, T, 16, False), first
        )
        torch.testing.assert_close(
            monarch_attention_triton(q, k, v, mask, T, 16, False), first
        )


@requires_cuda
@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize("group", ["batch", "heads"])
def test_workspace_limit(T, group):
    """With a small max_workspace, heads or batch elements are processed in
    groups; the output must not change."""
    from ma.ma_triton import _workspace_per_head, monarch_attention_triton

    torch.manual_seed(0)
    E, H, N, D, B = 3, 4, 300, 32, 16
    q, k, v = _rand_qkv(E, H, N, D, torch.float16)
    mask = _make_block_safe_mask(E, N, B)
    full = monarch_attention_triton(q, k, v, mask, T, B, True, max_workspace=None)
    per_head = _workspace_per_head(q, T, B)
    # 2 of 3 batch elements per group, or 3 of 4 heads (uneven last groups)
    budget = 2 * H * per_head if group == "batch" else 3 * per_head
    for _ in range(2):  # the second call launches cached kernels
        out = monarch_attention_triton(q, k, v, mask, T, B, True, max_workspace=budget)
        assert torch.equal(out, full)


@requires_cuda
def test_workspace_limit_compiled():
    from ma import MonarchAttention, PadType
    from ma.ma_triton import _workspace_per_head

    torch.manual_seed(0)
    q, k, v = _rand_qkv(2, 4, 300, 32, torch.float16)
    budget = 3 * _workspace_per_head(q, 1, 16)
    attn = MonarchAttention(16, 1, PadType.post, impl="triton", max_workspace=budget)
    expected = MonarchAttention(16, 1, PadType.post, "triton", max_workspace=None)
    torch.testing.assert_close(
        torch.compile(attn, fullgraph=True)(q, k, v), expected(q, k, v)
    )


@requires_cuda
@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize("pre_pad", [False, True])
@pytest.mark.parametrize("positions", [5, 16, 48])
def test_workspace_limit_positions(T, pre_pad, positions):
    """A budget below one head's buffers splits each head's positions into
    groups (5: narrower than a tile, 48: an uneven last group of 32). Groups
    narrower than a tile use smaller tiles, which round differently."""
    from ma.ma_triton import _workspace_per_head, monarch_attention_triton

    torch.manual_seed(0)
    E, H, N, D, B = 2, 2, 1000, 32, 80
    q, k, v = _rand_qkv(E, H, N, D, torch.float16)
    mask = _make_block_safe_mask(E, N, B)
    full = monarch_attention_triton(q, k, v, mask, T, B, pre_pad, max_workspace=None)
    budget = _workspace_per_head(q, T, B) * positions // B
    for _ in range(2):  # the second call launches cached kernels
        out = monarch_attention_triton(
            q, k, v, mask, T, B, pre_pad, max_workspace=budget
        )
        torch.testing.assert_close(out, full, atol=1e-3, rtol=1e-3)


@requires_cuda
@pytest.mark.parametrize("T", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_padded_queries_finite(T, dtype):
    """Sequences shorter than a block leave positions without valid queries;
    their (unused) outputs must stay finite, as in the reference: models keep
    processing padded tokens, and a NaN there spreads to every token."""
    from ma.ma_triton import monarch_attention_triton

    torch.manual_seed(0)
    E, H, N, D, B = 3, 2, 256, 32, 96
    q, k, v = _rand_qkv(E, H, N, D, dtype)
    mask = (
        torch.arange(N, device="cuda")[None] < torch.tensor([[256], [150], [40]]).cuda()
    )
    out = monarch_attention_triton(q, k, v, mask, T, B, False)
    expected = monarch_attention_torch(
        q.float(), k.float(), v.float(), mask, T, B, False
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
