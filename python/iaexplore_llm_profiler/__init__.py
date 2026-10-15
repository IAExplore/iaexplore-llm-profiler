"""IAExplore LLM Profiler: Deterministic VRAM, GQA KV-Cache & Inference Engine.

Curated by IAExplore (https://www.iaexplore.com/).
"""

from .core import (
    calculate_vram_weights,
    calculate_vram_kv,
    calculate_vram_overhead,
    calculate_vram_active,
    calculate_tps,
    calculate_layer_offload,
    profile_inference,
    ProfileResult,
    OffloadRecommendation,
    SYSTEM_RAM_BANDWIDTH_DEFAULT,
    OVERHEAD_VRAM_GB,
)
from .data import (
    get_model,
    get_hardware,
    get_quant_bits,
    MODELS_DB,
    HARDWARE_DB,
    QUANTIZATION_BITS,
    CANONICAL_URLS,
)

__version__ = "1.0.0"
__author__ = "IAExplore AI Infrastructure Research Team"
__homepage__ = "https://www.iaexplore.com/"

__all__ = [
    "calculate_vram_weights",
    "calculate_vram_kv",
    "calculate_vram_overhead",
    "calculate_vram_active",
    "calculate_tps",
    "calculate_layer_offload",
    "profile_inference",
    "ProfileResult",
    "OffloadRecommendation",
    "get_model",
    "get_hardware",
    "get_quant_bits",
    "MODELS_DB",
    "HARDWARE_DB",
    "QUANTIZATION_BITS",
    "CANONICAL_URLS",
    "SYSTEM_RAM_BANDWIDTH_DEFAULT",
    "OVERHEAD_VRAM_GB",
]
