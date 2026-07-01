"""Model configuration for the from-scratch TimesFM-style patched decoder.

Faithful to FareedKhan-dev/timesfm-from-scratch (the reference this project now
tracks). The publicly downloadable checkpoint is the *small* / 70M tier, which
is the default here so the real weights load cleanly.

    tiny : model_dim 512,  10 layers, 16 heads  (~17M)
    small: model_dim 1024, 10 layers, 16 heads  (~68M)   <- HF weights
    base : model_dim 1280, 20 layers, 16 heads  (~203M)  Google's backbone size
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    model_dim: int = 1024          # small / 70M default
    num_layers: int = 10
    num_heads: int = 16
    patch_len: int = 32            # input patch length (points per "token")
    horizon_len: int = 128         # output block length per position
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    dropout: float = 0.0
    rope_max_period: float = 10000.0
    rms_eps: float = 1e-6
    context_len: int = 512         # inference context (multiple of patch_len -> 16 patches)

    @property
    def head_dim(self) -> int:
        assert self.model_dim % self.num_heads == 0
        return self.model_dim // self.num_heads

    @property
    def num_outputs(self) -> int:
        return 1 + len(self.quantiles)  # channel 0 = point/mean, 1..9 = quantiles


def tiny() -> ModelConfig:
    return ModelConfig(model_dim=512, num_layers=10, num_heads=16)


def small() -> ModelConfig:
    return ModelConfig(model_dim=1024, num_layers=10, num_heads=16)


def base() -> ModelConfig:
    return ModelConfig(model_dim=1280, num_layers=20, num_heads=16)


SIZES = {"tiny": tiny, "small": small, "base": base}
DEFAULT = small()
