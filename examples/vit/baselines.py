from math import sqrt
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class Linformer(nn.Module):
    def __init__(self, proj_dim: int, seq_len: int, share_kv: bool = False):
        super().__init__()

        self.proj_dim = proj_dim
        self.share_kv = share_kv

        self.E = torch.randn(proj_dim, seq_len)
        if not self.share_kv:
            self.F = torch.randn(proj_dim, seq_len)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        assert query.shape == key.shape and key.shape == value.shape
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Project keys and values onto lower dim
        key_proj = torch.matmul(self.E, key)
        value_proj = (
            torch.matmul(self.F, value)
            if not self.share_kv
            else torch.matmul(self.E, value)
        )

        # Attention
        attention_scores = torch.matmul(query, key_proj.transpose(-2, -1)) / sqrt(
            head_dim
        )
        attention_probs = F.softmax(attention_scores, dim=-1)

        return torch.matmul(attention_probs, value_proj)


class Performer(nn.Module):
    def __init__(
        self, num_samples: int, estimator_type: str = "pos", ortho_features: bool = True
    ):
        super().__init__()

        self.num_samples = num_samples  # Number of random features
        assert estimator_type in [
            "trig",
            "pos",
            "hyp-pos",
        ]  # Type of feature map to use to estimate softmax attn
        self.estimator_type = estimator_type
        self.ortho_features = ortho_features  # Flag to use orthogonal features or not

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        assert query.shape == key.shape and key.shape == value.shape
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Compute Q', K', and C from paper
        query_prime = self._make_softmax_kernel_features(query)
        key_prime = self._make_softmax_kernel_features(key)
        C = torch.cat([value, torch.ones(batch_size, num_heads, seq_len, 1)], dim=-1)

        # Intermediate outputs for bidirectional attention
        buf1 = torch.matmul(key_prime.transpose(-2, -1), C)
        buf2 = torch.matmul(query_prime, buf1)

        # Performer attention output
        buf3 = buf2[..., :head_dim]
        buf4 = buf2[..., head_dim:].squeeze()
        return torch.einsum(
            "...ij,...i->...ij", buf3, 1.0 / buf4
        )  # Same as torch.matmul( diag(buf4)^{-1}, buf3 )

    # Construct random features to estimate softmax
    def _make_softmax_kernel_features(self, mat: Tensor) -> Tensor:
        if self.estimator_type == "trig":
            h_out, f_out = self._trig_softmax_kernel_features(mat)
        elif self.estimator_type == "pos":
            h_out, f_out = self._pos_softmax_kernel_features(mat)
        elif self.estimator_type == "hyp-pos":
            h_out, f_out = self._hyp_pos_softmax_kernel_features(mat)

        return (1.0 / sqrt(self.num_samples)) * h_out.unsqueeze(-1) * f_out

    # cos/sin features
    def _trig_softmax_kernel_features(self, mat: Tensor) -> Tuple[Tensor, Tensor]:
        dim = mat.shape[-1]

        h_out = torch.exp((torch.linalg.norm(mat, dim=-1) ** 2) / 2)

        omega = self._sample_random_vectors(dim).to(mat.device)
        f1_out = torch.cos(torch.matmul(mat, omega))
        f2_out = torch.sin(torch.matmul(mat, omega))

        return h_out, torch.cat([f1_out, f2_out], dim=-1)

    # Positive features
    def _pos_softmax_kernel_features(self, mat: Tensor) -> Tuple[Tensor, Tensor]:
        dim = mat.shape[-1]

        h_out = torch.exp(-(torch.linalg.norm(mat, dim=-1) ** 2) / 2)

        omega = self._sample_random_vectors(dim).to(mat.device)
        f_out = torch.exp(torch.matmul(mat, omega))

        return h_out, f_out

    # Hyperbolic positive features
    def _hyp_pos_softmax_kernel_features(self, mat: Tensor) -> Tuple[Tensor, Tensor]:
        dim = mat.shape[-1]

        h_out = (1.0 / sqrt(2)) * torch.exp(-(torch.linalg.norm(mat, dim=-1) ** 2) / 2)

        omega = self._sample_random_vectors(dim).to(mat.device)
        f1_out = torch.exp(torch.matmul(mat, omega))
        f2_out = torch.exp(torch.matmul(-mat, omega))

        return h_out, torch.cat([f1_out, f2_out], dim=-1)

    # Generate iid random vectors
    def _sample_random_vectors(self, dim) -> Tensor:
        m = self.num_samples

        if self.ortho_features:  # Orthogonal features
            assert m <= dim
            return self._gram_schmidt(torch.randn(dim, m))

        return torch.randn(dim, m)

    # Gram-schmidt process for orthogonalization
    def _gram_schmidt(self, mat):
        def _proj(v, u):
            return ((u * v).sum() / (u * u).sum()) * u

        num_samples = mat.shape[-1]

        out = torch.zeros_like(mat, device=mat.device)
        out[..., 0] = mat[..., 0].clone()

        for i in range(1, num_samples):
            v = mat[..., i].clone()

            projs_v_ui = 0
            for j in range(i):
                u_j = out[..., j]
                projs_v_ui += _proj(v, u_j)

            out[..., i] = v - projs_v_ui

        return out


# Implementation inspired by https://github.com/mlpen/Nystromformer/blob/main/code/attention_nystrom.py
class Nystromformer(nn.Module):
    def __init__(
        self,
        num_landmarks: int,
        num_heads: Optional[int] = None,
        conv_kernel_size: Optional[int] = None,
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

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        assert query.shape == key.shape and key.shape == value.shape
        batch_size, num_heads, seq_len, head_dim = query.shape

        if self.num_landmarks == seq_len:  # Vanilla attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(
                head_dim
            )
            attention_probs = F.softmax(attention_scores, dim=-1)
            out = torch.matmul(attention_probs, value)
        else:  # Nystromformer attention
            query_lm = self._compute_landmarks(query)
            key_lm = self._compute_landmarks(key)

            F_tilde = F.softmax(
                torch.matmul(query, key_lm.transpose(-2, -1)) / sqrt(head_dim), dim=-1
            )
            A_tilde = self._iterative_pinv(
                F.softmax(
                    torch.matmul(query_lm, key_lm.transpose(-2, -1)) / sqrt(head_dim),
                    dim=-1,
                )
            )
            B_tilde = F.softmax(
                torch.matmul(query_lm, key.transpose(-2, -1)) / sqrt(head_dim), dim=-1
            )

            out = torch.matmul(
                F_tilde, torch.matmul(A_tilde, torch.matmul(B_tilde, value))
            )

        if self.conv_ is not None:
            out += self.conv_(value)

        return out

    def _compute_landmarks(self, mat: Tensor) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = mat.shape
        assert self.num_landmarks <= seq_len

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


# Implementation inspired by https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
class Cosformer(nn.Module):
    def __init__(self, eps: Optional[float] = 1e-6):
        super().__init__()
        self.eps = eps  # Tolerance for normalization (to avoid divide-by-zero)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        assert query.shape == key.shape and key.shape == value.shape
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Apply relu to queries, keys
        query_relu = F.relu(query)
        key_relu = F.relu(key)

        # Sin/cos re-weighting
        idx = self._get_idx(seq_len).to(query.device)
        query_cos_sin = torch.cat(
            [
                query_relu * torch.cos(idx / seq_len),
                query_relu * torch.sin(idx / seq_len),
            ],
            dim=-1,
        )  # (batch_size, num_heads, seq_len, 2*head_dim)
        key_cos_sin = torch.cat(
            [key_relu * torch.cos(idx / seq_len), key_relu * torch.sin(idx / seq_len)],
            dim=-1,
        )

        # Compute K^T * V first
        kv_cos_sin = torch.matmul(
            key_cos_sin.transpose(-2, -1), value
        )  # (batch_size, num_heads, 2*head_dim, head_dim)

        # Normalization factor:
        # - torch.einsum(): Multiply each row of Q_cos and Q_sin with sums of rows of K_cos and K_sin, resp.
        # - torch.clamp(): make min at least self.eps for numerical stability
        norm_ = 1.0 / torch.clamp(
            torch.einsum(
                "...hld,...hd->...hl", query_cos_sin, torch.sum(key_cos_sin, dim=-2)
            ),
            min=self.eps,
        )

        # Compute approximate attention output:
        # - Equivalent to torch.matmul(query_cos_sin, kv_cos_sin) * norm_.unsqueeze(-1)
        return torch.einsum(
            "...hld,...hdm,...hl->...hlm", query_cos_sin, kv_cos_sin, norm_
        )

    def _get_idx(self, seq_len):
        return (np.pi / 2) * torch.arange(1, seq_len + 1).reshape(1, 1, -1, 1)
