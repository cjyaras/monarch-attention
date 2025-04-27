import torch

Tensor = torch.Tensor


def check_inputs(q, k, v):
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [torch.float16, torch.bfloat16]
