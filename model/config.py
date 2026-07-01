"""Central configuration for the 200M patched time-series transformer.

Every number here is derived from spec.md (Section 2, Architecture Specifications).
The parameter budget is dominated by the 12 transformer blocks:

    per-block attention  = 4 * d_model^2                = 4 * 1024^2  =  4,194,304
    per-block SwiGLU MLP = 3 * d_model * intermediate   = 3 * 1024*4096 = 12,582,912
    per-block total                                     = 16,777,216
    x 12 layers                                         = 201,326,592  (~200M)

The patch-projection in / forecast-head out layers add ~34K params (negligible),
so the model lands at ~201M parameters, matching the spec's ~200M target.
"""

from dataclasses import dataclass, asdict


@dataclass
class TSConfig:
    # --- Transformer core (LLaMA-structured, see spec 2.MLC Compatibility) ---
    d_model: int = 1024            # hidden dimension
    n_layers: int = 12             # number of decoder blocks
    n_heads: int = 16              # attention heads
    n_kv_heads: int = 16           # KV heads (== n_heads -> plain MHA; set <16 for GQA)
    intermediate_size: int = 4096  # SwiGLU inner dim (tuned for the 200M budget)
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # --- Time-series patching (spec 2.Patching Layer) ---
    patch_size: int = 16           # P: time-steps per patch (the "token")
    n_ctx_patches: int = 512       # N_patches: max patches in the context window
    n_pred_patches: int = 8        # patches produced per forecast call (horizon = P * this)

    # --- MLC / vocab bookkeeping (spec 2.MLC Compatibility, point 1) ---
    # vocab_size is a structural placeholder. MLC/LLaMA graphs require an
    # embedding table + lm_head sized to vocab_size; we keep it tiny because the
    # real "embedding" is the continuous patch projection, and the real "lm_head"
    # is the forecast head. Kept != 0 so the LLaMA export stays graph-compatible.
    vocab_size: int = 32

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    @property
    def horizon(self) -> int:
        return self.patch_size * self.n_pred_patches


DEFAULT = TSConfig()
