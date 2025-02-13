import entmax
import torch

Tensor = torch.Tensor


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    result = entmax.sparsemax(logits, dim=dim)
    assert isinstance(result, torch.Tensor)
    return result


def _compute_lambda(alpha: Tensor, beta: Tensor) -> Tensor:
    z = beta / alpha
    sort_idx = torch.argsort(beta, dim=-1, descending=True)
    z_sort = torch.take_along_dim(z, sort_idx, dim=-1)
    alpha_sort = torch.take_along_dim(alpha, sort_idx, dim=-1)
    beta_sort = torch.take_along_dim(beta, sort_idx, dim=-1)
    inv_alpha_cum_sum = torch.cumsum(1 / alpha_sort, dim=-1)
    z_cum_sum = torch.cumsum(z_sort, dim=-1)
    k = (
        torch.sum(1 + beta_sort * inv_alpha_cum_sum > z_cum_sum, dim=-1, keepdim=True)
        - 1
    )
    k = torch.maximum(k, torch.zeros_like(k))
    lam = (torch.take_along_dim(z_cum_sum, k, dim=-1) - 1) / torch.take_along_dim(
        inv_alpha_cum_sum, k, dim=-1
    )
    return lam


# (s)implex (c)onstrained (s)eparable (q)uadratic minimization
# Minimize 0.5 alpha * p^2 - beta * p where p is on simplex, alpha >= 0
def minimize_scsq(alpha: Tensor, beta: Tensor) -> Tensor:
    zero_mask = torch.isclose(alpha, torch.zeros_like(alpha))
    positive_mask = torch.logical_not(zero_mask)
    lam = _compute_lambda(
        torch.where(positive_mask, alpha, 1.0),
        torch.where(positive_mask, beta, torch.finfo(beta.dtype).min),
    )
    smallest_idx = torch.argmax(
        torch.where(zero_mask, beta, torch.finfo(beta.dtype).min),
        dim=-1,
        keepdim=True,
    )
    largest_beta = torch.take_along_dim(
        torch.where(zero_mask, beta, torch.finfo(beta.dtype).min), smallest_idx, dim=-1
    )
    lam = torch.maximum(lam, largest_beta)
    p = torch.maximum(
        (torch.where(positive_mask, beta, torch.finfo(beta.dtype).min) - lam)
        / torch.where(positive_mask, alpha, 1),
        torch.zeros_like(beta),
    )
    p = torch.scatter(
        p,
        dim=-1,
        index=smallest_idx,
        src=torch.maximum(
            1.0 - torch.sum(p, dim=-1, keepdim=True),
            torch.take_along_dim(p, smallest_idx, dim=-1),
        ),
    )
    return torch.clip(p, 0.0, 1.0)
