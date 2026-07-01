"""PatchedDecoder — the from-scratch TimesFM-style model.

Vendored faithfully from FareedKhan-dev/timesfm-from-scratch (src/tsfm/model.py)
so the published 70M checkpoint loads as-is. `encode` is the single forward the
browser runtime calls (exported to ONNX); the autoregressive `forecast` loop is
reproduced in JavaScript in docs/timeSeriesWorker.js.
"""

import torch
import torch.nn as nn

from layers import ResidualBlock, RotaryEmbedding, TransformerBlock
from revin import first_patch_stats


class PatchedDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_len
        self.tokenizer = ResidualBlock(2 * p, cfg.model_dim, cfg.model_dim, bias=True)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_max_period)
        self.head = ResidualBlock(cfg.model_dim, cfg.model_dim,
                                  cfg.horizon_len * cfg.num_outputs, bias=False)

    def _attn_mask(self, patch_valid):
        # arange-based causal + diagonal (avoids Trilu/EyeLike ops so the graph
        # exports cleanly to ONNX / onnxruntime-web WebGPU). Numerically identical
        # to tril | eye.
        B, N = patch_valid.shape
        dev = patch_valid.device
        ar = torch.arange(N, device=dev)
        causal = ar[:, None] >= ar[None, :]                    # lower-triangular incl. diag
        eye = ar[:, None] == ar[None, :]
        mask = causal[None, None] & patch_valid[:, None, None, :]
        return mask | eye[None, None]

    def encode(self, x, padding):
        """context x, padding: [B, L] -> normalized out [B, N, horizon, Q], (mu, sigma)."""
        cfg = self.cfg
        p = cfg.patch_len
        B, L = x.shape
        assert L % p == 0, "context length must be a multiple of patch_len"
        N = L // p
        xp = x.view(B, N, p)
        pp = padding.view(B, N, p)
        mu, sigma = first_patch_stats(xp, pp)
        xn = ((xp - mu[:, None, None]) / sigma[:, None, None]).clamp(-20.0, 20.0)
        xn = xn * (1.0 - pp)
        tok_in = torch.cat([xn, pp], dim=-1)
        h = self.tokenizer(tok_in)
        pos = torch.arange(N, device=x.device)
        cos, sin = self.rope.cos_sin(pos, h.dtype)
        patch_valid = pp.min(dim=-1).values < 0.5
        mask = self._attn_mask(patch_valid)
        for blk in self.layers:
            h = blk(h, mask, cos, sin)
        out = self.head(h).view(B, N, cfg.horizon_len, cfg.num_outputs)
        return out, (mu, sigma)

    def forward(self, x, padding):
        return self.encode(x, padding)

    @torch.no_grad()
    def _forecast_once(self, context, horizon, point_channel=5):
        cfg = self.cfg
        h_len = cfg.horizon_len
        x = context
        pad = torch.zeros_like(x)
        points, quants = [], []
        produced = 0
        while produced < horizon:
            out, (mu, sigma) = self.encode(x, pad)
            last = out[:, -1] * sigma[:, None, None] + mu[:, None, None]
            points.append(last[..., point_channel])
            quants.append(last)
            new = last[..., point_channel]
            x = torch.cat([x, new], dim=1)
            pad = torch.cat([pad, torch.zeros_like(new)], dim=1)
            produced += h_len
        return torch.cat(points, dim=1)[:, :horizon], torch.cat(quants, dim=1)[:, :horizon]

    @torch.no_grad()
    def forecast(self, context, horizon, point_channel=5):
        return self._forecast_once(context, horizon, point_channel)


def build_model(cfg):
    return PatchedDecoder(cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from config import small
    cfg = small()
    m = build_model(cfg)
    n = count_params(m)
    print(f"small: dim={cfg.model_dim} layers={cfg.num_layers} heads={cfg.num_heads} "
          f"patch={cfg.patch_len} horizon={cfg.horizon_len} -> {n:,} (~{n/1e6:.1f}M)")
    x = torch.randn(1, cfg.context_len)
    pad = torch.zeros(1, cfg.context_len)
    out, (mu, sigma) = m.encode(x, pad)
    print("encode out:", tuple(out.shape), "mu/sigma:", tuple(mu.shape))
    pt, q = m.forecast(x, horizon=128)
    print("forecast point:", tuple(pt.shape), "quantiles:", tuple(q.shape))
