import torch
import torch.nn as nn
from models.integral_attention import IntegralAttention


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) per sample."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob
        return x * mask / keep_prob


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_signals=8, mlp_ratio=4.0,
                 dropout=0.0, attn_dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = IntegralAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_signals=num_signals,
            dropout=attn_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop_path(h)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeiTIntegralAttention(nn.Module):
    """
    DeiT with softmax-based Integral Attention (O(N²)).

    Uses the denoising mechanism from "Integral Transformer":
        phi(X) = softmax( (1/S) * sum_{s=1}^{S} Q_s K_s^T )

    Averaging S logit signals before softmax reduces noise in attention
    weights while keeping standard quadratic complexity.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_signals=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.1,
        use_dist_token=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_dist_token = use_dist_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + optional distillation token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_dist_token else None
        num_tokens = 2 if use_dist_token else 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer encoder blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_signals=num_signals,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head(s)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes) if use_dist_token else None

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.use_dist_token:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.use_dist_token:
            cls_out = self.head(x[:, 0])
            dist_out = self.head_dist(x[:, 1])
            if self.training:
                return cls_out, dist_out
            return (cls_out + dist_out) / 2
        else:
            return self.head(x[:, 0])


# ──────────────────── Factory functions ────────────────────

def deit_tiny_integral(num_classes=1000, **kwargs):
    return DeiTIntegralAttention(
        embed_dim=192, depth=12, num_heads=3, num_signals=8,
        num_classes=num_classes, **kwargs,
    )

def deit_small_integral(num_classes=1000, **kwargs):
    return DeiTIntegralAttention(
        embed_dim=384, depth=12, num_heads=6, num_signals=8,
        num_classes=num_classes, **kwargs,
    )

def deit_base_integral(num_classes=1000, **kwargs):
    return DeiTIntegralAttention(
        embed_dim=768, depth=12, num_heads=12, num_signals=8,
        num_classes=num_classes, **kwargs,
    )
