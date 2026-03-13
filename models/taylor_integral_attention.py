import torch
import torch.nn as nn


def taylor_kernel(x):
    """
    Second-order Taylor expansion kernel: φ(x) = [1, x, x⊗x / √2]
    Approximates exp(x) to linearize softmax attention.

    Input:  (..., D)
    Output: (..., 1 + D + D²)
    """
    D = x.shape[-1]
    x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2) / (2 ** 0.5)
    ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
    return torch.cat([ones, x, x2], dim=-1)


class TaylorIntegralAttention(nn.Module):
    """
    Linear Integral Attention combining:
    1. Taylor kernel for O(N) attention complexity
    2. Integral signal averaging for attention denoising

    Standard softmax attention computes softmax(QK^T)V in O(N²).
    Taylor kernel replaces softmax with φ(x) = [1, x, x⊗x/√2], enabling:
        output = φ(Q) · (φ(K)^T · V)  — O(N · D_φ · D_v) instead of O(N²·D)

    S signal groups are independently computed then averaged for denoising,
    inspired by the Integral Transformer (Kobyzev et al., 2025).
    """

    def __init__(self, embed_dim, num_heads, num_signals=2, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_signals = num_signals

        assert self.head_dim % num_signals == 0, (
            f"head_dim ({self.head_dim}) must be divisible by num_signals ({num_signals})"
        )
        self.signal_dim = self.head_dim // num_signals

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, N, C = query.size()
        M = key.size(1)

        # Project and reshape into (B, H, S, seq_len, signal_dim)
        q = self.q_proj(query).view(B, N, self.num_heads, self.num_signals, self.signal_dim)
        k = self.k_proj(key).view(B, M, self.num_heads, self.num_signals, self.signal_dim)
        q = q.permute(0, 2, 3, 1, 4)  # (B, H, S, N, d)
        k = k.permute(0, 2, 3, 1, 4)  # (B, H, S, M, d)

        # V uses full head_dim, shared across signals
        v = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        # v: (B, H, M, head_dim)

        # Scale Q before kernel (equivalent to dividing logits by sqrt(d))
        scale = self.signal_dim ** -0.5
        q = q * scale

        # Apply Taylor kernel: (..., d) -> (..., 1 + d + d²)
        q_phi = taylor_kernel(q)  # (B, H, S, N, D_phi)
        k_phi = taylor_kernel(k)  # (B, H, S, M, D_phi)

        # Apply mask by zeroing out masked K positions before aggregation
        if mask is not None:
            # mask shape: (B, 1, 1, M) or (B, H, N, M) — True = keep
            k_phi = k_phi * mask.unsqueeze(2).unsqueeze(-1).float()

        # --- Linear attention O(N): multiply K^T·V first ---
        # Expand V for signal dimension: (B, H, 1, M, head_dim)
        v_exp = v.unsqueeze(2).expand(-1, -1, self.num_signals, -1, -1)
        # KV: (B, H, S, D_phi, head_dim) = φ(K)^T @ V
        kv = torch.matmul(k_phi.transpose(-2, -1), v_exp)
        # Numerator: (B, H, S, N, head_dim) = φ(Q) @ KV
        numer = torch.matmul(q_phi, kv)

        # Denominator for normalization: φ(Q) @ Σ_j φ(K_j)
        k_sum = k_phi.sum(dim=-2, keepdim=True)  # (B, H, S, 1, D_phi)
        denom = torch.matmul(q_phi, k_sum.transpose(-2, -1))  # (B, H, S, N, 1)

        # Normalize each signal independently, then average (integral denoising)
        output_signals = numer / (denom + 1e-6)  # (B, H, S, N, head_dim)
        attn_output = output_signals.mean(dim=2)  # (B, H, N, head_dim)

        attn_output = self.dropout(attn_output)

        # Reshape: (B, H, N, head_dim) -> (B, N, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        output = self.out_proj(attn_output)

        return output, None  # No explicit N×N attention matrix in linear attention
