import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegralDiffAttention(nn.Module):
    """
    Integral-Differential Attention.

    Combines two denoising strategies:
      1. Differential: within each signal pair, subtract attention logits
         to cancel correlated noise.
      2. Integral: average the differential logits across all pairs
         to further smooth out residual noise.

    Given S signals (S must be even), form S/2 pairs.
    For each pair i:
        diff_i = Q_{2i} K_{2i}^T  -  λ_i · Q_{2i+1} K_{2i+1}^T

    Then average and softmax:
        Attn = softmax( (2/S) · Σ_{i=0}^{S/2-1} diff_i )

    Each λ_i is a learnable per-head, per-pair scalar (initialized to 0.8).
    """

    def __init__(self, embed_dim, num_heads, num_signals=8, dropout=0.0,
                 lambda_init=0.8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_signals % 2 == 0, "num_signals must be even for pairing"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_signals = num_signals
        self.num_pairs = num_signals // 2

        assert self.head_dim % num_signals == 0, (
            f"head_dim ({self.head_dim}) must be divisible by num_signals ({num_signals})"
        )
        self.signal_dim = self.head_dim // num_signals

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable λ per head per pair — initialized via reparameterisation:
        #   λ = exp(lambda_log) so it stays positive.
        # Init so that exp(lambda_log) ≈ lambda_init
        self.lambda_log = nn.Parameter(
            torch.full((num_heads, self.num_pairs), fill_value=torch.tensor(lambda_init).log().item())
        )

    def forward(self, query, key, value, mask=None):
        B, N, C = query.size()
        M = key.size(1)

        # Project & reshape into (B, seq, num_heads, num_signals, signal_dim)
        q = self.q_proj(query).view(B, N, self.num_heads, self.num_signals, self.signal_dim)
        k = self.k_proj(key).view(B, M, self.num_heads, self.num_signals, self.signal_dim)
        v = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        # v: (B, num_heads, M, head_dim)

        # Rearrange for batched matmul
        # q, k: (B, num_heads, num_signals, seq, signal_dim)
        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 1, 4)

        # Compute per-signal logits: (B, num_heads, num_signals, N, M)
        scale = self.signal_dim ** -0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Split into pairs along signal dim
        # (B, num_heads, num_pairs, 2, N, M)
        logits_pairs = logits.view(B, self.num_heads, self.num_pairs, 2, N, M)
        logits_pos = logits_pairs[:, :, :, 0]   # (B, H, P, N, M)
        logits_neg = logits_pairs[:, :, :, 1]   # (B, H, P, N, M)

        # λ: (H, P) → (1, H, P, 1, 1) for broadcasting
        lam = self.lambda_log.exp().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Differential: subtract within each pair
        diff_logits = logits_pos - lam * logits_neg  # (B, H, P, N, M)

        # Integral: average across pairs
        avg_logits = diff_logits.mean(dim=2)  # (B, H, N, M)

        if mask is not None:
            avg_logits = avg_logits.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(avg_logits, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # (B, H, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        output = self.out_proj(attn_output)

        return output, attn_probs
