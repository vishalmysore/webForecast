"""First-patch reversible instance normalization (RevIN), masking-aware.

Vendored from FareedKhan-dev/timesfm-from-scratch (src/tsfm/revin.py). Standardize
the whole context by the mean/std of the FIRST patch with >= min_valid non-padded
points (fallback: last patch), with sigma floored at ctx_floor * whole-context std
for stability.
"""

import torch


def first_patch_stats(xp, pp, min_valid=3, eps=1e-6, ctx_floor=0.3):
    """xp, pp: [B, N, p]; pp 1=pad/missing. Returns mu, sigma: [B]."""
    B, N, p = xp.shape
    valid = 1.0 - pp
    counts = valid.sum(-1)
    has = counts >= min_valid
    idx = torch.argmax(has.to(torch.int64), dim=1)
    none = ~has.any(dim=1)
    idx = torch.where(none, torch.full_like(idx, N - 1), idx)
    rows = torch.arange(B, device=xp.device)
    arr = xp[rows, idx]
    m = valid[rows, idx]
    n = m.sum(-1).clamp(min=1.0)
    mu = (arr * m).sum(-1) / n
    var = (((arr - mu[:, None]) * m) ** 2).sum(-1) / n
    sigma = var.clamp(min=0.0).sqrt()
    cnt = valid.sum(dim=(1, 2)).clamp(min=1.0)
    gmu = (xp * valid).sum(dim=(1, 2)) / cnt
    gvar = (((xp - gmu[:, None, None]) * valid) ** 2).sum(dim=(1, 2)) / cnt
    ctx_std = gvar.clamp(min=0.0).sqrt()
    sigma = torch.maximum(sigma, ctx_floor * ctx_std)
    sigma = torch.where(sigma < eps, torch.ones_like(sigma), sigma)
    return mu, sigma
