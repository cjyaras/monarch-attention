import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

from common.utils import get_device

Tensor = torch.Tensor


class Baseline(nn.Module):

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ):
        raise NotImplementedError()

    def get_matrix(
        self, query: Tensor, key: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape
        return self.forward(
            query,
            key,
            torch.eye(seq_len, device=query.device, dtype=query.dtype).expand(
                (batch_size, num_heads, seq_len, seq_len)
            ),
            attention_mask,
        )


class Softmax(Baseline):

    def __init__(self, use_flash_attention: bool = False):
        super().__init__()
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        head_dim = query.shape[-1]

        if self.use_flash_attention:
            attention_mask = (
                _prepare_4d_attention_mask_for_sdpa(attention_mask, dtype=query.dtype)
                if attention_mask is not None
                else None
            )
            return F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
            )

        else:
            attention_mask = (
                (
                    (1.0 - attention_mask[:, None, None, :])
                    * torch.finfo(query.dtype).min
                )
                if attention_mask is not None
                else None
            )
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(head_dim)

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = F.softmax(attention_scores, dim=-1)
            return torch.matmul(attention_probs, value)


class Linformer(Baseline):

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape

        if attention_mask is not None:
            key = attention_mask[:, None, :, None] * key
            value = attention_mask[:, None, :, None] * value

        R = torch.randn(self.rank, seq_len).to(query.device) / math.sqrt(self.rank)

        key = torch.matmul(R, key)
        value = torch.matmul(R, value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(head_dim)

        attention_probs = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_probs, value)


class KernelAttention(Baseline):

    def phi(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape
        query = query / math.sqrt(head_dim)

        phi_q = self.phi(query)
        phi_k = self.phi(key)

        mask = None
        if attention_mask is not None:
            # Input mask shape: (B, N) -> Output mask shape: (B, 1, N, 1)
            mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(dtype=phi_k.dtype)
            # Apply mask to phi_k: effectively zeros out contributions from padding tokens
            phi_k = phi_k * mask

        # --- Linear Attention Computation ---
        # 1. Compute K'^T @ V term (where K' = phi(K))
        # Einsum: 'bhnd,bhnv->bhdv' where d=d_phi, v=head_dim
        kv_term = torch.einsum("bhnd,bhnv->bhdv", phi_k, value)  # (B, H, d_phi, d)

        # 2. Compute Q' @ (K'^T @ V) term (where Q' = phi(Q))
        # Einsum: 'bhnd,bhdv->bhnv'
        output = torch.einsum("bhnd,bhdv->bhnv", phi_q, kv_term)  # (B, H, N, d)

        # --- Normalization ---
        # 1. Compute K'^T @ 1s term
        ones_val = torch.ones(
            batch_size, num_heads, seq_len, 1, device=value.device, dtype=value.dtype
        )
        if mask is not None:
            # Apply mask to ones as well, so padding tokens don't contribute to sum
            ones_val = ones_val * mask
        # Einsum: 'bhnd,bhnz->bhdz' where z=1
        k_one_term = torch.einsum(
            "bhnd,bhnz->bhdz", phi_k, ones_val
        )  # (B, H, d_phi, 1)

        # 2. Compute Q' @ (K'^T @ 1s) term
        # Einsum: 'bhnd,bhdz->bhnz'
        normalizer = torch.einsum("bhnd,bhdz->bhnz", phi_q, k_one_term)  # (B, H, N, 1)

        print(output)
        exit()

        # 3. Normalize output
        # Add epsilon to prevent division by zero
        output = output / (normalizer + 1e-6)

        return output


class Performer(KernelAttention):

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def phi(self, x: Tensor) -> Tensor:
        dim = x.shape[-1]
        h_out = torch.exp(-torch.linalg.norm(x, dim=-1) ** 2 / 2)
        omega = torch.randn(dim, self.rank).to(x.device)
        f_out = torch.exp(torch.matmul(x, omega))
        out = (1.0 / math.sqrt(self.rank)) * h_out.unsqueeze(-1) * f_out
        return out


class Cosformer(KernelAttention):

    def phi(self, x: Tensor) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x_relu = F.relu(x)

        # Sin/cos re-weighting
        idx = self._get_idx(seq_len).to(x.device)
        out = torch.cat(
            [
                x_relu * torch.cos(idx / seq_len),
                x_relu * torch.sin(idx / seq_len),
            ],
            dim=-1,
        )
        return out

    def _get_idx(self, seq_len):
        return (np.pi / 2) * torch.arange(1, seq_len + 1).reshape(1, 1, -1, 1)


class LinearAttention(KernelAttention):

    def phi(self, x: Tensor) -> Tensor:
        return 1 + F.elu(x)


# Implementation inspired by https://github.com/mlpen/Nystromformer/blob/main/code/attention_nystrom.py
class Nystromformer(Baseline):
    def __init__(
        self,
        num_landmarks: int,
        num_heads: int | None = None,
        conv_kernel_size: int | None = None,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks  # 'm' in paper
        if conv_kernel_size is not None:
            assert num_heads is not None
            self.conv_ = nn.Conv2d(
                in_channels=num_heads,
                out_channels=num_heads,
                kernel_size=conv_kernel_size,
                padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=num_heads,
            )
        else:
            self.conv_ = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        assert query.shape == key.shape and key.shape == value.shape
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Vanilla attention
        if self.num_landmarks == seq_len:
            attention_probs = self._compute_vanilla_attn_mat(query, key, attention_mask)
            out = torch.matmul(attention_probs, value)
        # Nystromformer attention
        else:
            F_tilde, A_tilde, B_tilde = self._compute_nystrom_factors(
                query, key, attention_mask
            )
            out = torch.matmul(
                F_tilde, torch.matmul(A_tilde, torch.matmul(B_tilde, value))
            )

            if self.conv_ is not None:
                out += self.conv_(value)

        return out

    def _compute_vanilla_attn_mat(
        self, query: Tensor, key: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = query.shape

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            head_dim
        )  # Compute attention scores
        out = F.softmax(attention_scores, dim=-1)
        if attention_mask is not None:  # Apply mask if needed
            assert attention_mask.shape == (batch_size, seq_len)
            valid_attention_mask = self._get_valid_mask(attention_mask)
            assert valid_attention_mask is not None
            out = self._renormalize(self._mask_vanilla_attn(out, valid_attention_mask))

        return out

    def _compute_nystrom_factors(
        self, query: Tensor, key: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Compute landmarks for queries, keys
        query_lm = self._compute_landmarks(query)
        key_lm = self._compute_landmarks(key)

        # Compute factors for approx attention
        F_tilde = F.softmax(
            torch.matmul(query, key_lm.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1
        )
        A_tilde = self._iterative_pinv(
            F.softmax(
                torch.matmul(query_lm, key_lm.transpose(-2, -1)) / math.sqrt(head_dim),
                dim=-1,
            )
        )
        B_tilde = F.softmax(
            torch.matmul(query_lm, key.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1
        )

        # Apply mask if needed
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)
            valid_attention_mask = self._get_valid_mask(attention_mask)
            assert valid_attention_mask is not None
            F_tilde, B_tilde = self._mask_nystrom_factors(
                F_tilde, B_tilde, valid_attention_mask
            )

            F_tilde = self._renormalize(F_tilde)
            B_tilde = self._renormalize(B_tilde)

        return F_tilde, A_tilde, B_tilde

    def _compute_landmarks(self, mat: Tensor) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = mat.shape
        assert self.num_landmarks < seq_len

        l = (
            seq_len // self.num_landmarks
        )  # Truncate last few tokens if not perfectly divisible
        mat_lm = torch.cat(
            [
                mat[..., j * l : (j + 1) * l, :].mean(dim=-2, keepdim=True)
                for j in range(self.num_landmarks)
            ],
            dim=-2,
        )

        return mat_lm

    def _iterative_pinv(self, mat: Tensor, n_iter: int = 6) -> Tensor:
        I = torch.eye(mat.shape[-1], device=mat.device)

        init_coef = torch.max(torch.sum(torch.abs(mat), dim=-2)) * torch.max(
            torch.sum(torch.abs(mat), dim=-1)
        )
        out = mat.transpose(-2, -1) / init_coef

        for _ in range(n_iter):
            term1 = 7 * I - torch.matmul(mat, out)
            term2 = 15 * I - torch.matmul(mat, torch.matmul(out, term1))
            term3 = 13 * I - torch.matmul(mat, torch.matmul(out, term2))
            out = 0.25 * torch.matmul(out, term3)

        return out

    def _get_valid_mask(self, mask: Tensor | None) -> Tensor | None:
        if mask is not None:
            mask = rearrange(mask, "b s -> b 1 1 s")
        return mask

    def _mask_vanilla_attn(self, probs: Tensor, mask: Tensor) -> Tensor:
        return mask * probs * mask.transpose(-2, -1)

    def _mask_nystrom_factors(
        self, F: Tensor, B: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        return (
            F * mask.transpose(-2, -1),
            B * mask,
        )  # B is already of shape (batch_size, num_heads, num_landmarks, seq_len)

    def _renormalize(self, mat: Tensor):
        return mat / torch.sum(mat, dim=-1, keepdim=True)
