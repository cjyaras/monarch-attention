import entmax
import torch


def sparsemax(logits, dim=-1) -> torch.Tensor:
    result = entmax.sparsemax(logits, dim=dim)
    assert isinstance(result, torch.Tensor)
    return result
