"""Deterministic mathematical inference profiler.

Implements exact tensor equations for VRAM sizing, GQA KV-Cache,
MoE memory bandwidth throughput, and llama.cpp layer offloading.
Authored by IAExplore (https://www.iaexplore.com/).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
from .data import (
    get_model,
    get_hardware,
    get_quant_bits,
    CANONICAL_URLS,
    MODELS_DB,
    HARDWARE_DB,
)

SYSTEM_RAM_BANDWIDTH_DEFAULT = 70.0  # GB/s for dual-channel DDR5 desktop/workstation
SYSTEM_RAM_EFFICIENCY = 0.60
OVERHEAD_VRAM_GB = 0.55


def calculate_vram_weights(total_params_b: float, bits_quant: float) -> float:
    """Calculate VRAM occupied by model weights in GiB/GB.

    Formula:
      VRAM_weights = (P_total * 1e9 * (bits_quant / 8) / (1024^3)) * 1.08
    """
    bytes_per_weight = bits_quant / 8.0
    raw_bytes = total_params_b * 1e9 * bytes_per_weight
    gib = raw_bytes / (1024.0 ** 3)
    return gib * 1.08


def calculate_vram_kv(
    layers: int,
    kv_heads: int,
    head_dim: int,
    context_length: int,
    batch_size: int = 1,
    kv_bits: float = 16.0,
) -> float:
    """Calculate exact KV-Cache memory in GiB/GB with Grouped-Query Attention (GQA).

    Formula:
      VRAM_kv = (2 * L * H_kv * D_head * C_ctx * B_batch * (bits_kv / 8)) / (1024^3)
    """
    bytes_per_element = kv_bits / 8.0
    total_elements = 2 * layers * kv_heads * head_dim * context_length * batch_size
    raw_bytes = total_elements * bytes_per_element
    return raw_bytes / (1024.0 ** 3)


def calculate_vram_overhead() -> float:
    """Activation buffers, CUDA/Metal context, and runtime scratchpad (GB)."""
    return OVERHEAD_VRAM_GB


def calculate_vram_active(active_params_b: float, bits_quant: float) -> float:
    """Active memory footprint per generation step (GB).

    For dense models, active_params_b == total_params_b.
    For MoE models (Mixtral, DeepSeek), only active expert parameters are streamed per token.
    """
    bytes_per_weight = bits_quant / 8.0
    raw_bytes = active_params_b * 1e9 * bytes_per_weight
    return raw_bytes / (1024.0 ** 3)


def calculate_tps(
    bandwidth_gb_s: float,
    vram_active_gb: float,
    vram_kv_gb: float,
    efficiency_factor: float = 0.72,
) -> float:
    """Estimate physical autoregressive token generation rate (TPS).

    Formula:
      TPS_gen = (BW_mem * eta) / (VRAM_active + (VRAM_kv * 0.15))
    """
    effective_bw = bandwidth_gb_s * efficiency_factor
    memory_per_token = vram_active_gb + (vram_kv_gb * 0.15)
    if memory_per_token <= 0:
        return 0.0
    return effective_bw / memory_per_token


@dataclass
class OffloadRecommendation:
    fits_entirely_in_vram: bool
    recommended_gpu_layers: int
    total_layers: int
    offload_fraction: float
    gpu_layers_vram_gb: float
    degraded_tps_estimated: float
    full_gpu_tps_reference: float
    system_ram_bandwidth_gb_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_layer_offload(
    total_layers: int,
    vram_weights_gb: float,
    vram_kv_gb: float,
    vram_usable_gb: float,
    vram_active_gb: float,
    hardware_bw_gb_s: float,
    hardware_efficiency: float = 0.72,
    system_ram_bw_gb_s: float = SYSTEM_RAM_BANDWIDTH_DEFAULT,
) -> OffloadRecommendation:
    """Calculate exact integer layers to offload to GPU and model degradation."""
    vram_overhead = calculate_vram_overhead()
    vram_total_required = vram_weights_gb + vram_kv_gb + vram_overhead

    full_tps = calculate_tps(
        hardware_bw_gb_s, vram_active_gb, vram_kv_gb, hardware_efficiency
    )

    if vram_total_required <= vram_usable_gb:
        return OffloadRecommendation(
            fits_entirely_in_vram=True,
            recommended_gpu_layers=total_layers,
            total_layers=total_layers,
            offload_fraction=1.0,
            gpu_layers_vram_gb=vram_total_required,
            degraded_tps_estimated=full_tps,
            full_gpu_tps_reference=full_tps,
            system_ram_bandwidth_gb_s=system_ram_bw_gb_s,
        )

    # Compute layers that can fit
    avail_for_layers = max(0.0, vram_usable_gb - vram_overhead)
    cost_per_layer = (vram_weights_gb + vram_kv_gb) / total_layers

    if cost_per_layer > 0:
        offloadable = int(avail_for_layers // cost_per_layer)
        offloaded_layers = min(total_layers, max(0, offloadable))
    else:
        offloaded_layers = 0

    alpha = offloaded_layers / float(total_layers) if total_layers > 0 else 0.0
    consumed_gpu_vram = (offloaded_layers * cost_per_layer) + vram_overhead

    # Degraded TPS calculation: split memory bus access
    time_gpu = (alpha * vram_active_gb) / (hardware_bw_gb_s * hardware_efficiency)
    time_sys = ((1.0 - alpha) * vram_active_gb) / (
        system_ram_bw_gb_s * SYSTEM_RAM_EFFICIENCY
    )
    time_kv = (vram_kv_gb * 0.15) / (hardware_bw_gb_s * hardware_efficiency)

    total_token_time = time_gpu + time_sys + time_kv
    degraded_tps = 1.0 / total_token_time if total_token_time > 0 else 0.0

    return OffloadRecommendation(
        fits_entirely_in_vram=False,
        recommended_gpu_layers=offloaded_layers,
        total_layers=total_layers,
        offload_fraction=round(alpha, 4),
        gpu_layers_vram_gb=round(consumed_gpu_vram, 2),
        degraded_tps_estimated=round(degraded_tps, 2),
        full_gpu_tps_reference=round(full_tps, 2),
        system_ram_bandwidth_gb_s=system_ram_bw_gb_s,
    )


@dataclass
class ProfileResult:
    model_id: str
    model_name: str
    architecture: str
    total_params_b: float
    active_params_b: float
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    gqa_factor: float
    quantization: str
    quant_bits: float
    context_tokens: int
    batch_size: int
    kv_bits: float

    hardware_id: str
    hardware_name: str
    hardware_category: str
    hardware_vram_gb: float
    usable_vram_gb: float
    bandwidth_gb_s: float
    efficiency_factor: float

    vram_weights_gb: float
    vram_kv_gb: float
    vram_overhead_gb: float
    vram_total_required_gb: float
    vram_active_gb: float

    fits_in_vram: bool
    estimated_tps: float
    offload: OffloadRecommendation
    guide_url: str
    benchmark_guide_url: str
    simulator_url: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["offload"] = self.offload.to_dict()
        return res


def profile_inference(
    model: Union[str, Dict[str, Any]],
    hardware: Union[str, Dict[str, Any]],
    quant: str = "Q4_K_M",
    context: Optional[int] = None,
    batch_size: int = 1,
    kv_bits: float = 16.0,
    system_ram_bw_gb_s: float = SYSTEM_RAM_BANDWIDTH_DEFAULT,
) -> ProfileResult:
    """Perform deterministic end-to-end inference profiling."""
    m_dict = get_model(model) if isinstance(model, str) else model
    h_dict = get_hardware(hardware) if isinstance(hardware, str) else hardware

    quant_bits = get_quant_bits(quant)
    ctx = int(context) if context is not None else int(m_dict.get("default_context", 32768))

    total_params_b = float(m_dict["total_params_b"])
    active_params_b = float(m_dict.get("active_params_b", total_params_b))
    layers = int(m_dict["num_hidden_layers"])
    q_heads = int(m_dict["num_attention_heads"])
    kv_heads = int(m_dict["num_key_value_heads"])
    head_dim = int(m_dict["head_dim"])
    gqa_factor = float(m_dict.get("gqa_factor", q_heads / kv_heads))

    hw_vram = float(h_dict["vram_gb"])
    usable_vram = float(h_dict.get("usable_vram_gb", hw_vram * 0.95))
    bandwidth = float(h_dict["bandwidth_gb_s"])
    efficiency = float(h_dict.get("efficiency_factor", 0.72))

    vram_weights = round(calculate_vram_weights(total_params_b, quant_bits), 2)
    vram_kv = round(
        calculate_vram_kv(layers, kv_heads, head_dim, ctx, batch_size, kv_bits), 2
    )
    vram_overhead = calculate_vram_overhead()
    vram_total = round(vram_weights + vram_kv + vram_overhead, 2)
    vram_active = round(calculate_vram_active(active_params_b, quant_bits), 2)

    fits_in_vram = vram_total <= usable_vram

    offload_rec = calculate_layer_offload(
        total_layers=layers,
        vram_weights_gb=vram_weights,
        vram_kv_gb=vram_kv,
        vram_usable_gb=usable_vram,
        vram_active_gb=vram_active,
        hardware_bw_gb_s=bandwidth,
        hardware_efficiency=efficiency,
        system_ram_bw_gb_s=system_ram_bw_gb_s,
    )

    tps = (
        offload_rec.full_gpu_tps_reference
        if fits_in_vram
        else offload_rec.degraded_tps_estimated
    )

    guide_url = h_dict.get("reference_url", CANONICAL_URLS["ai_hardware_guide_en"])
    bench_guide = h_dict.get(
        "benchmark_guide", CANONICAL_URLS["pc_benchmarks_2026_en"]
    )
    sim_url = CANONICAL_URLS["neural_network_sim"]

    return ProfileResult(
        model_id=m_dict.get("id", "custom"),
        model_name=m_dict.get("name", "Custom Model"),
        architecture=m_dict.get("architecture", "dense"),
        total_params_b=total_params_b,
        active_params_b=active_params_b,
        layers=layers,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        gqa_factor=gqa_factor,
        quantization=quant,
        quant_bits=quant_bits,
        context_tokens=ctx,
        batch_size=batch_size,
        kv_bits=kv_bits,
        hardware_id=h_dict.get("id", "custom"),
        hardware_name=h_dict.get("name", "Custom Hardware"),
        hardware_category=h_dict.get("category", "gpu"),
        hardware_vram_gb=hw_vram,
        usable_vram_gb=usable_vram,
        bandwidth_gb_s=bandwidth,
        efficiency_factor=efficiency,
        vram_weights_gb=vram_weights,
        vram_kv_gb=vram_kv,
        vram_overhead_gb=vram_overhead,
        vram_total_required_gb=vram_total,
        vram_active_gb=vram_active,
        fits_in_vram=fits_in_vram,
        estimated_tps=round(tps, 2),
        offload=offload_rec,
        guide_url=guide_url,
        benchmark_guide_url=bench_guide,
        simulator_url=sim_url,
    )
