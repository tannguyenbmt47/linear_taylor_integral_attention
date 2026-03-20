import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenSparse(nn.Module):
    """
    Chọn lọc token quan trọng nhất dựa trên attention score.
    Giữ lại top-k token, phần còn lại gộp thành 1 fusion token.
    """
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

    def forward(self, tokens, attention_x, attention_y):
        B, L, C = tokens.size()
        score = attention_x + attention_y  # (B, L)
        num_keep_token = math.ceil(L * self.sparse_ratio)
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        keep_policy = score_index[:, :num_keep_token]  # (B, k)
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))
        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        non_keep_score = score_sort[:, num_keep_token:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)  # (B, 1, C)
        return select_tokens, extra_token, score_mask

class TokenAggregation(nn.Module):
    """
    FineLIP Adaptive Token Refinement Module (ATRM).
    Học aggregation weights bằng attention-like mechanism.
    
    Công thức: W_ref = Softmax(W_q @ σ(X @ W_k)^T / τ)
    X' = W_ref @ X  (aggregate N tokens thành N' tokens)
    
    Constraint: ∑W_ref[i,j] = 1 (column-wise sum = 1)
    """
    def __init__(self, dim=512, agent_num=4, d_k=64):
        super().__init__()
        self.dim = dim
        self.agent_num = agent_num
        self.d_k = d_k
        
        # W_k: project từ d → d_k (reduce dimension)
        self.W_k = nn.Linear(dim, d_k, bias=False)
        
        # W_q: query projection (agent_num, d_k)
        # Mỗi row là query cho một agent token
        self.W_q = nn.Parameter(torch.randn(agent_num, d_k))
        nn.init.trunc_normal_(self.W_q, std=0.02)
        
        # Learnable temperature (sparse attention)
        self.tau = nn.Parameter(torch.tensor(1.0))
        
        self.gelu = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: (B, N, d) - N tokens, mỗi token d-dim
            
        Returns:
            x_refined: (B, agent_num, d) - agent_num refined tokens
        """
        B, N, d = x.shape
        
        # 1. Project tokens: σ(X @ W_k)
        # xk: (B, N, d_k)
        xk = self.gelu(self.W_k(x))
        
        # 2. Compute attention: W_q @ xk^T / τ
        # W_q: (agent_num, d_k)
        # xk: (B, N, d_k) → transpose to (B, d_k, N)
        # W_q @ xk^T: need to expand W_q to batch dimension first
        # (1, agent_num, d_k) @ (B, d_k, N) → (B, agent_num, N)
        W_q_expanded = self.W_q.unsqueeze(0)  # (1, agent_num, d_k)
        xk_transposed = xk.transpose(1, 2)  # (B, d_k, N)
        attn = torch.matmul(W_q_expanded, xk_transposed) / self.tau  # (B, agent_num, N)
        
        # 3. Softmax per token (column-wise): ∑W_ref[i,j]=1
        attn = F.softmax(attn, dim=-1)
        
        # 4. Aggregate: W_ref @ X
        # (B, agent_num, N) @ (B, N, d) → (B, agent_num, d)
        x_refined = torch.bmm(attn, x)
        
        return x_refined

class TokenAdaptionModule(nn.Module):
    """
    Simplified: ATRM only - aggregate tokens directly to agent_num without sparse selection
    
    Flow:
    196 image tokens → FineLIP ATRM aggregation → agent_num (40) refined tokens
    """
    def __init__(self, embed_dim=512, num_patches=196, agent_num=40, d_k=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.agent_num = agent_num
        
        # ATRM aggregation without sparse selection
        self.aggregation = TokenAggregation(dim=embed_dim, agent_num=agent_num, d_k=d_k)

    def forward(self, tokens, attention_x, attention_y):
        """
        Args:
            tokens: (B, N, d) - N image tokens (N=196)
            attention_x, attention_y: scores (not used in simplified version)
            
        Returns:
            agent_tokens: (B, agent_num, d) - refined agent tokens (agent_num=40)
            score_mask: (B, N) - dummy mask (all ones)
        """
        B, N, d = tokens.shape
        
        # Direct aggregation: 196 → agent_num
        agent_tokens = self.aggregation(tokens)
        # agent_tokens: (B, agent_num, d)
        
        # Dummy mask (not used anymore)
        score_mask = torch.ones(B, N, device=tokens.device)
        
        return agent_tokens, score_mask
