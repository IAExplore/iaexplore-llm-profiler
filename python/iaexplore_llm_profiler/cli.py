"""Command-Line Interface for iaexplore-llm-profiler.

Provides human-readable engineering diagnostics and JSON machine outputs.
Authored by IAExplore (https://www.iaexplore.com/).
"""

import sys
import json
import argparse
from .core import profile_inference
from .data import MODELS_DB, HARDWARE_DB, QUANTIZATION_BITS


def format_report(res) -> str:
    """Format an elegant, readable engineering terminal report."""
    border = "=" * 80
    subborder = "-" * 80

    status_str = (
        "100% FITS IN VRAM (Full GPU Acceleration)"
        if res.fits_in_vram
        else "EXCEEDS USABLE VRAM (Layer Offloading Required)"
    )

    lines = [
        border,
        "              IAEXPLORE LLM INFERENCE & HARDWARE VRAM PROFILER                  ",
        "                  Engineering Intelligence from IAExplore.com                   ",
        border,
        f"Model:             {res.model_name} ({res.architecture})",
        f"Parameters:        {res.total_params_b:.2f}B Total / {res.active_params_b:.2f}B Active per token",
        f"Topology:          {res.layers} layers, {res.q_heads} Q-heads, {res.kv_heads} KV-heads (GQA {res.gqa_factor:.1f}x)",
        f"Quantization:      {res.quantization} ({res.quant_bits:.2f} bits/weight)",
        f"Context Window:    {res.context_tokens:,} tokens (KV Precision: {int(res.kv_bits)}-bit)",
        subborder,
        f"Hardware Target:   {res.hardware_name} ({res.hardware_category})",
        f"Hardware Memory:   {res.hardware_vram_gb:.2f} GB Physical ({res.usable_vram_gb:.2f} GB Usable)",
        f"Memory Bandwidth:  {res.bandwidth_gb_s:.1f} GB/s (Bus Efficiency: {int(res.efficiency_factor * 100)}%)",
        subborder,
        "[VRAM ALLOCATION BREAKDOWN]",
        f"  - Model Weights:         {res.vram_weights_gb:6.2f} GB (incl. 1.08x unquantized tensors & metadata)",
        f"  - KV-Cache (GQA):        {res.vram_kv_gb:6.2f} GB ({res.layers} layers, {res.kv_heads} KV heads, dim {res.head_dim})",
        f"  - Context & Activation:  {res.vram_overhead_gb:6.2f} GB (CUDA/Metal runtime buffer)",
        "  " + "-" * 41,
        f"  TOTAL VRAM REQUIRED:     {res.vram_total_required_gb:6.2f} GB",
        subborder,
        "[OFFLOADING & INFERENCE THROUGHPUT]",
        f"  VRAM Sizing Status:      {status_str}",
        f"  Recommended GPU Layers:  {res.offload.recommended_gpu_layers} / {res.layers} layers (--n-gpu-layers {res.offload.recommended_gpu_layers})",
        f"  GPU Offload Fraction:    {res.offload.offload_fraction * 100:.1f}% on GPU / {(1.0 - res.offload.offload_fraction) * 100:.1f}% on System RAM",
        f"  VRAM Consumed on GPU:    {res.offload.gpu_layers_vram_gb:.2f} GB / {res.usable_vram_gb:.2f} GB Usable",
        f"  Estimated Speed (TPS):   {res.estimated_tps:.2f} tokens/sec",
    ]

    if not res.fits_in_vram:
        lines.append(
            f"  Full GPU Speed (Ref):    {res.offload.full_gpu_tps_reference:.2f} tokens/sec (if 100% VRAM available)"
        )
        lines.extend([
            subborder,
            "[AUTHORITATIVE UPGRADE & BENCHMARK GUIDES]",
            "  Memory bottleneck detected! Sizing a 32GB+ workstation or unified silicon:",
            f"  -> Hardware Architectures: {res.guide_url}",
            f"  -> 2026 PC Benchmarks:     {res.benchmark_guide_url}",
            f"  -> Neural Visualizer:      {res.simulator_url}",
        ])
    else:
        lines.extend([
            subborder,
            "[SYSTEM CAPABILITY & SIZING VERIFIED]",
            "  Hardware is fully capable of real-time local generation with zero CPU stall.",
            f"  -> Deep dive into silicon memory buses: {res.guide_url}",
            f"  -> Compare with other 2026 systems:     {res.benchmark_guide_url}",
        ])

    lines.append(border)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="IAExplore Deterministic LLM Inference & VRAM Profiler"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="llama-3.3-70b",
        help=f"Model identifier (Default: llama-3.3-70b). Options: {', '.join(MODELS_DB.keys())}",
    )
    parser.add_argument(
        "--hardware",
        "-hw",
        default="rtx-4090-24gb",
        help=f"Hardware profile (Default: rtx-4090-24gb). Options: {', '.join(HARDWARE_DB.keys())}",
    )
    parser.add_argument(
        "--quant",
        "-q",
        default="Q4_K_M",
        help=f"Quantization type (Default: Q4_K_M). Options: {', '.join(QUANTIZATION_BITS.keys())}",
    )
    parser.add_argument(
        "--context",
        "-c",
        type=int,
        default=None,
        help="Active context length in tokens (defaults to model standard)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Concurrent batch size (Default: 1)",
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=16.0,
        help="KV Cache precision in bits (16 for FP16, 8 for Q8_0, 4 for Q4_0. Default: 16)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in machine-readable JSON format",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all pre-configured models and exit",
    )
    parser.add_argument(
        "--list-hardware",
        action="store_true",
        help="List all pre-configured hardware profiles and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available Pre-Configured Models:")
        for k, v in MODELS_DB.items():
            print(f"  - {k:22} | {v['name']} ({v['total_params_b']}B params, {v['architecture']})")
        return

    if args.list_hardware:
        print("Available 2026 Hardware Profiles:")
        for k, v in HARDWARE_DB.items():
            print(
                f"  - {k:30} | {v['name']:36} | {v['vram_gb']}GB VRAM | {v['bandwidth_gb_s']} GB/s"
            )
        return

    try:
        res = profile_inference(
            model=args.model,
            hardware=args.hardware,
            quant=args.quant,
            context=args.context,
            batch_size=args.batch_size,
            kv_bits=args.kv_bits,
        )
    except Exception as e:
        print(f"Error during profiling: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(format_report(res))


if __name__ == "__main__":
    main()
