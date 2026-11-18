"""Minimal tensor BM3D components used by the noisy-clean optimizer."""

from .config import BM3DConfig, RuntimeConfig
from .oracle import BM3DMultiplierOracle
from .precomputed import BM3DWienerPrecomputed

__all__ = [
    "BM3DConfig",
    "BM3DMultiplierOracle",
    "BM3DWienerPrecomputed",
    "RuntimeConfig",
]
