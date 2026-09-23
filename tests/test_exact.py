"""Monarch attention is exact softmax attention when the blocking is trivial:
a single block covering the sequence, or blocks of size 1."""

import pytest
import torch
import torch.nn.functional as F

from ma import MonarchAttention, PadType

E, H, N, D = 2, 3, 24, 8

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def impls(device):
    # The Triton kernels only run on CUDA
    return ["torch", "triton"] if device == "cuda" else ["torch"]


def tol(impl):
    # Triton's tl.dot computes float32 products in TF32
    return 1e-5 if impl == "torch" else 5e-3


def _qkv(device, dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    return [torch.randn(E, H, N, D, generator=g, dtype=dtype).to(device) for _ in range(3)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block_size", [1, N, 32])
@pytest.mark.parametrize("num_steps", [1, 2, 3])
@pytest.mark.parametrize("pad_type", list(PadType))
def test_exact_without_mask(device, block_size, num_steps, pad_type):
    q, k, v = _qkv(device)
    expected = F.scaled_dot_product_attention(q, k, v)
    for impl in impls(device):
        out = MonarchAttention(block_size, num_steps, pad_type, impl)(q, k, v)
        torch.testing.assert_close(out, expected, atol=tol(impl), rtol=tol(impl))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block_size", [N, 32])
@pytest.mark.parametrize("num_steps", [1, 3])
@pytest.mark.parametrize("pad_type", list(PadType))
@pytest.mark.parametrize("trailing_padding", [False, True])
def test_exact_with_mask_single_block(device, block_size, num_steps, pad_type, trailing_padding):
    q, k, v = _qkv(device)
    if trailing_padding:
        mask = torch.arange(N)[None] < torch.tensor([[17], [N]])
    else:
        mask = torch.rand(E, N, generator=torch.Generator().manual_seed(1)) > 0.3
        mask[:, 0] = True
    mask = mask.to(device)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
    keep = mask[:, None, :, None].expand_as(q)  # outputs for masked queries are unused
    for impl in impls(device):
        out = MonarchAttention(block_size, num_steps, pad_type, impl)(q, k, v, mask)
        torch.testing.assert_close(out[keep], expected[keep], atol=tol(impl), rtol=tol(impl))
