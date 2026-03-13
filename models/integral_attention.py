import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegralAttention(nn.Module):
    """
    Integral Attention mechanism from "Integral Transformer: Denoising Attention,
    Not Too Much Not Too Little" (Kobyzev et al., 2025).

    Denoises attention by averaging S logit signals before softmax:
        phi(X) = softmax( (1/S) * sum_{s=1}^{S} Q_s K_s^T )

    The hidden score dimension d_h = head_dim / S, keeping parameter count
    and efficiency comparable to standard multi-head attention.
    """

    def __init__(self, embed_dim, num_heads, num_signals=8, dropout=0.0):
        super(IntegralAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_signals = num_signals

        assert self.head_dim % num_signals == 0, (
            f"head_dim ({self.head_dim}) must be divisible by num_signals ({num_signals})"
        )
        self.signal_dim = self.head_dim // num_signals

        # Q and K projections: split into S signal groups within each head
        # Total Q/K projection size = num_heads * num_signals * signal_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, N, C = query.size()
        M = key.size(1)  # source sequence length (may differ from N)

        # Project Q, K, V
        # Shape: (B, N, num_heads, head_dim) -> split head_dim into (num_signals, signal_dim)
        q = self.q_proj(query).view(B, N, self.num_heads, self.num_signals, self.signal_dim)
        k = self.k_proj(key).view(B, M, self.num_heads, self.num_signals, self.signal_dim)
        v = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        # v: (B, num_heads, M, head_dim)

        # Rearrange q, k for batched matmul:
        # q: (B, num_heads, num_signals, N, signal_dim)
        # k: (B, num_heads, num_signals, M, signal_dim)
        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 1, 4)

        # Compute S logit signals and average them
        # Each signal: Q_s @ K_s^T -> (B, num_heads, num_signals, N, M)
        # Scale by 1/sqrt(signal_dim) for each signal
        scale = self.signal_dim ** -0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Average over signals: (B, num_heads, N, M)
        avg_logits = logits.mean(dim=2)

        # Apply mask if provided
        if mask is not None:
            avg_logits = avg_logits.masked_fill(mask == 0, float('-inf'))

        # Softmax over averaged logits (Eq. 7)
        attn_probs = F.softmax(avg_logits, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        # attn_probs: (B, num_heads, N, M) @ v: (B, num_heads, M, head_dim)
        attn_output = torch.matmul(attn_probs, v)
        # (B, num_heads, N, head_dim) -> (B, N, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        output = self.out_proj(attn_output)

        return output, attn_probs
