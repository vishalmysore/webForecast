"""200M-parameter decoder-only patched time-series transformer.

Data flow (spec.md Section 2):

    [Raw lookback vector]
        -> Instance Normalization (RevIN-style, per-instance mean/std)
        -> 1D Patching + Linear projection  (P time-steps -> d_model)  == "embedding"
        -> N x LLaMA-style decoder blocks (RMSNorm, RoPE, causal MHA, SwiGLU)
        -> Final RMSNorm
        -> Linear forecast head            (d_model -> P time-steps)   == "lm_head"
        -> Denormalization                 (undo the instance norm)

The block internals deliberately mirror a LLaMA/Mistral graph so that the MLC
LLM (TVM Unity) toolchain can compile it to WebGPU without custom operators.
The only non-standard pieces are the patch projection at the input and the
forecast head at the output; both are plain nn.Linear layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TSConfig, DEFAULT


# ----------------------------------------------------------------------------
# LLaMA-style primitives
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute cos/sin for rotary position embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, Dh) ; cos,sin: (T, Dh)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: TSConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.n_kv_heads != self.n_heads:  # GQA: broadcast KV groups
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Causal self-attention over the patch sequence.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: TSConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: TSConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ----------------------------------------------------------------------------
# Time-series specific I/O
# ----------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Continuous replacement for nn.Embedding (spec 2.Patching Layer).

    Splits a lookback window into non-overlapping patches of `patch_size`
    time-steps and linearly projects each patch to d_model. This is the
    "payload injection" the frontend/worker talks about: instead of integer
    token IDs, we push continuous P-vectors straight into the block stack.
    """

    def __init__(self, cfg: TSConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.proj = nn.Linear(cfg.patch_size, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) normalized series, L divisible by patch_size
        B, L = x.shape
        assert L % self.patch_size == 0, "lookback length must be a multiple of patch_size"
        patches = x.view(B, L // self.patch_size, self.patch_size)  # (B, N, P)
        return self.proj(patches)                                   # (B, N, d_model)


class ForecastHead(nn.Module):
    """Continuous replacement for lm_head: d_model -> P next time-steps."""

    def __init__(self, cfg: TSConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.d_model, cfg.patch_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # (B, N, P)


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, cfg: TSConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg)
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.head = ForecastHead(cfg)
        self._rope = {}  # cache keyed by (seq_len, device, dtype)

    # -- RevIN-style instance normalization -------------------------------
    @staticmethod
    def instance_norm(x: torch.Tensor, eps: float = 1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std, mean, std

    @staticmethod
    def instance_denorm(x, mean, std):
        return x * std + mean

    def _rope_cache(self, seq_len, device, dtype):
        key = (seq_len, device, dtype)
        if key not in self._rope:
            self._rope[key] = build_rope_cache(
                seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
        return self._rope[key]

    def backbone(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Run normalized series through embed + blocks + final norm."""
        h = self.patch_embed(x_norm)                      # (B, N, d_model)
        cos, sin = self._rope_cache(h.size(1), h.device, h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.norm(h)                               # (B, N, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing forward for training.

        x: (B, L) raw series. Returns per-patch next-patch predictions
        (B, N, P) in the *original* scale. Patch i predicts patch i+1.
        """
        x_norm, mean, std = self.instance_norm(x)
        h = self.backbone(x_norm)
        pred_norm = self.head(h)                          # (B, N, P) next-patch
        return self.instance_denorm(pred_norm, mean, std)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, n_patches: int = None) -> torch.Tensor:
        """Autoregressive forecast (spec 4.4 postprocessing).

        x: (B, L) raw lookback. Returns (B, n_patches * P) future values in the
        original scale. Normalization statistics are frozen from the lookback
        window so the whole forecast shares one instance-norm frame.
        """
        cfg = self.cfg
        n_patches = n_patches or cfg.n_pred_patches
        self.eval()

        x_norm, mean, std = self.instance_norm(x)
        work = x_norm
        preds = []
        for _ in range(n_patches):
            # Keep the context within the trained window.
            ctx = work[:, -cfg.n_ctx_patches * cfg.patch_size:]
            h = self.backbone(ctx)
            next_patch = self.head(h[:, -1, :])           # (B, P) last position
            preds.append(next_patch)
            work = torch.cat([work, next_patch], dim=1)   # feed back (normalized)

        out_norm = torch.cat(preds, dim=1)                # (B, n_patches * P)
        return self.instance_denorm(out_norm, mean, std)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg = TSConfig()
    model = TimeSeriesTransformer(cfg)
    n = count_params(model)
    print(f"Config: d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"inter={cfg.intermediate_size} patch={cfg.patch_size}")
    print(f"Parameters: {n:,} (~{n/1e6:.1f}M)")

    # Smoke test: lookback of 64 patches -> forecast 8 patches.
    L = 64 * cfg.patch_size
    x = torch.randn(2, L)
    y = model(x)
    print("teacher-forcing out:", tuple(y.shape))
    f = model.generate(x, n_patches=cfg.n_pred_patches)
    print("forecast out:", tuple(f.shape), "(horizon =", cfg.horizon, "steps)")
