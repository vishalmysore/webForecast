"""Core TimesFM-2.5-style layers.

Vendored faithfully from FareedKhan-dev/timesfm-from-scratch (src/tsfm/layers.py)
so the published 70M weights load without modification. Do not "improve" these —
any deviation changes the state_dict and breaks weight loading.

ResidualBlock (MLP + linear skip, SiLU), RMSNorm (zero-init -> 1+scale),
RotaryEmbedding (half-split), PerDimScale (1/ln2 * softplus), MultiHeadAttention
(fused QKV, qk-norm, RoPE, per-dim scale, SDPA scale=1.0), TransformerBlock
(sandwich RMSNorm around attention and FFN).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=False):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.output_layer = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.residual_layer = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.output_layer(self.act(self.hidden_layer(x))) + self.residual_layer(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))  # (1 + scale)

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.scale.float())).to(dt)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_period=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_period = max_period

    def cos_sin(self, positions, dtype):
        half = self.head_dim // 2
        device = positions.device
        freq = self.max_period ** (2.0 * torch.arange(half, device=device).float() / self.head_dim)
        ang = positions.float()[:, None] / freq[None, :]
        return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)

    @staticmethod
    def apply(x, cos, sin):
        first, second = x.chunk(2, dim=-1)
        cos = cos[None, None]
        sin = sin[None, None]
        return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


class PerDimScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))
        self.r = 1.442695041 / math.sqrt(dim)  # 1/ln(2) / sqrt(d)

    def forward(self, x):
        return x * (self.r * F.softplus(self.scale))


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, self.h, self.hd = cfg.model_dim, cfg.num_heads, cfg.head_dim
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_eps)
        self.rope = RotaryEmbedding(self.hd, cfg.rope_max_period)
        self.per_dim = PerDimScale(self.hd)

    def forward(self, x, attn_mask, cos, sin):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, N, self.h, self.hd))
        k = self.k_norm(k.view(B, N, self.h, self.hd))
        v = v.view(B, N, self.h, self.hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, N, hd]
        q = self.rope.apply(q, cos, sin)
        k = self.rope.apply(k, cos, sin)
        q = self.per_dim(q)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        o = o.transpose(1, 2).reshape(B, N, D)
        return self.out(o)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.model_dim
        self.pre_attn_ln = RMSNorm(d, cfg.rms_eps)
        self.post_attn_ln = RMSNorm(d, cfg.rms_eps)
        self.attn = MultiHeadAttention(cfg)
        self.pre_ff_ln = RMSNorm(d, cfg.rms_eps)
        self.post_ff_ln = RMSNorm(d, cfg.rms_eps)
        self.ff0 = nn.Linear(d, d, bias=False)
        self.ff1 = nn.Linear(d, d, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask, cos, sin):
        a = self.attn(self.pre_attn_ln(x), attn_mask, cos, sin)
        x = x + self.post_attn_ln(self.drop(a))
        f = self.ff1(self.act(self.ff0(self.pre_ff_ln(x))))
        x = x + self.post_ff_ln(self.drop(f))
        return x
