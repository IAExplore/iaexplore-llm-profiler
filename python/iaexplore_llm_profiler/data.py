"""Embedded technical specifications for 2026 LLMs and AI Hardware.

Provides out-of-the-box verified parameters without requiring external file reads.
Curated by IAExplore (https://www.iaexplore.com/).
"""

from typing import Dict, Any

# Effective bits per weight for common quantization formats
QUANTIZATION_BITS: Dict[str, float] = {
    "FP16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q8_1": 9.0,
    "Q6_K": 6.56,
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.25,
    "Q5_0": 5.5,
    "Q5_1": 6.0,
    "Q4_K_M": 4.5,
    "Q4_K_S": 4.25,
    "Q4_0": 4.5,
    "Q4_1": 5.0,
    "Q3_K_L": 3.8,
    "Q3_K_M": 3.5,
    "Q3_K_S": 3.25,
    "Q2_K": 2.65,
    "IQ4_XS": 4.25,
    "IQ3_M": 3.5,
    "IQ2_XS": 2.35,
}

MODELS_DB: Dict[str, Dict[str, Any]] = {
    "llama-3.3-70b": {
        "id": "llama-3.3-70b",
        "name": "Meta Llama 3.3 70B Instruct",
        "family": "Llama 3.3",
        "architecture": "dense",
        "total_params_b": 70.6,
        "active_params_b": 70.6,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 8192,
        "vocab_size": 128256,
        "max_context_length": 131072,
        "default_context": 32768,
        "gqa_factor": 8.0,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html"
    },
    "llama-3.1-8b": {
        "id": "llama-3.1-8b",
        "name": "Meta Llama 3.1 8B Instruct",
        "family": "Llama 3.1",
        "architecture": "dense",
        "total_params_b": 8.03,
        "active_params_b": 8.03,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 128256,
        "max_context_length": 131072,
        "default_context": 8192,
        "gqa_factor": 4.0,
        "reference_url": "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html"
    },
    "deepseek-r1-14b": {
        "id": "deepseek-r1-14b",
        "name": "DeepSeek R1 Distill Qwen 14B",
        "family": "DeepSeek R1",
        "architecture": "dense",
        "total_params_b": 14.8,
        "active_params_b": 14.8,
        "num_hidden_layers": 48,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 5120,
        "vocab_size": 129280,
        "max_context_length": 131072,
        "default_context": 32768,
        "gqa_factor": 5.0,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html"
    },
    "deepseek-r1-32b": {
        "id": "deepseek-r1-32b",
        "name": "DeepSeek R1 Distill Qwen 32B",
        "family": "DeepSeek R1",
        "architecture": "dense",
        "total_params_b": 32.8,
        "active_params_b": 32.8,
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 5120,
        "vocab_size": 129280,
        "max_context_length": 131072,
        "default_context": 32768,
        "gqa_factor": 5.0,
        "reference_url": "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html"
    },
    "qwen-2.5-coder-32b": {
        "id": "qwen-2.5-coder-32b",
        "name": "Qwen 2.5 Coder 32B Instruct",
        "family": "Qwen 2.5",
        "architecture": "dense",
        "total_params_b": 32.5,
        "active_params_b": 32.5,
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 5120,
        "vocab_size": 152064,
        "max_context_length": 131072,
        "default_context": 32768,
        "gqa_factor": 5.0,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html"
    },
    "mixtral-8x7b-moe": {
        "id": "mixtral-8x7b-moe",
        "name": "Mistral AI Mixtral 8x7B MoE",
        "family": "Mixtral",
        "architecture": "moe",
        "total_params_b": 46.7,
        "active_params_b": 12.9,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 32000,
        "max_context_length": 32768,
        "default_context": 16384,
        "gqa_factor": 4.0,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html"
    },
    "gemma-2-9b": {
        "id": "gemma-2-9b",
        "name": "Google Gemma 2 9B Instruct",
        "family": "Gemma 2",
        "architecture": "dense",
        "total_params_b": 9.24,
        "active_params_b": 9.24,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "hidden_size": 3584,
        "vocab_size": 256000,
        "max_context_length": 8192,
        "default_context": 8192,
        "gqa_factor": 2.0,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html"
    },
    "phi-4-14b": {
        "id": "phi-4-14b",
        "name": "Microsoft Phi-4 14B",
        "family": "Phi-4",
        "architecture": "dense",
        "total_params_b": 14.7,
        "active_params_b": 14.7,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 10,
        "head_dim": 128,
        "hidden_size": 5120,
        "vocab_size": 100352,
        "max_context_length": 16384,
        "default_context": 16384,
        "gqa_factor": 4.0,
        "reference_url": "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html"
    }
}

HARDWARE_DB: Dict[str, Dict[str, Any]] = {
    "rtx-5090-32gb": {
        "id": "rtx-5090-32gb",
        "name": "NVIDIA GeForce RTX 5090",
        "category": "discrete_gpu",
        "vram_gb": 32.0,
        "usable_vram_gb": 31.2,
        "bandwidth_gb_s": 1792.0,
        "tflops_fp16": 209.0,
        "tops_int8": 418.0,
        "tdp_watts": 575,
        "efficiency_factor": 0.72,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html"
    },
    "rtx-4090-24gb": {
        "id": "rtx-4090-24gb",
        "name": "NVIDIA GeForce RTX 4090",
        "category": "discrete_gpu",
        "vram_gb": 24.0,
        "usable_vram_gb": 23.4,
        "bandwidth_gb_s": 1008.0,
        "tflops_fp16": 165.0,
        "tops_int8": 330.0,
        "tdp_watts": 450,
        "efficiency_factor": 0.72,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html"
    },
    "rtx-4070-super-12gb": {
        "id": "rtx-4070-super-12gb",
        "name": "NVIDIA GeForce RTX 4070 SUPER",
        "category": "discrete_gpu",
        "vram_gb": 12.0,
        "usable_vram_gb": 11.5,
        "bandwidth_gb_s": 504.0,
        "tflops_fp16": 71.0,
        "tops_int8": 142.0,
        "tdp_watts": 220,
        "efficiency_factor": 0.72,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html"
    },
    "rtx-4060-laptop-8gb": {
        "id": "rtx-4060-laptop-8gb",
        "name": "NVIDIA GeForce RTX 4060 Mobile",
        "category": "mobile_gpu",
        "vram_gb": 8.0,
        "usable_vram_gb": 7.6,
        "bandwidth_gb_s": 256.0,
        "tflops_fp16": 30.0,
        "tops_int8": 60.0,
        "tdp_watts": 115,
        "efficiency_factor": 0.72,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html"
    },
    "apple-m4-max-128gb": {
        "id": "apple-m4-max-128gb",
        "name": "Apple M4 Max (128GB Unified)",
        "category": "unified_memory",
        "vram_gb": 128.0,
        "usable_vram_gb": 120.0,
        "bandwidth_gb_s": 546.0,
        "tflops_fp16": 54.0,
        "tops_int8": 108.0,
        "tdp_watts": 90,
        "efficiency_factor": 0.72,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html"
    },
    "apple-m4-pro-48gb": {
        "id": "apple-m4-pro-48gb",
        "name": "Apple M4 Pro (48GB Unified)",
        "category": "unified_memory",
        "vram_gb": 48.0,
        "usable_vram_gb": 44.0,
        "bandwidth_gb_s": 273.0,
        "tflops_fp16": 32.0,
        "tops_int8": 64.0,
        "tdp_watts": 60,
        "efficiency_factor": 0.72,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html"
    },
    "apple-m4-base-16gb": {
        "id": "apple-m4-base-16gb",
        "name": "Apple M4 (16GB Unified)",
        "category": "unified_memory",
        "vram_gb": 16.0,
        "usable_vram_gb": 13.5,
        "bandwidth_gb_s": 120.0,
        "tflops_fp16": 18.0,
        "tops_int8": 38.0,
        "tdp_watts": 30,
        "efficiency_factor": 0.72,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html"
    },
    "amd-ryzen-ai-max-395-128gb": {
        "id": "amd-ryzen-ai-max-395-128gb",
        "name": "AMD Ryzen AI Max+ 395 (Strix Halo 128GB)",
        "category": "unified_apu_npu",
        "vram_gb": 96.0,
        "usable_vram_gb": 92.0,
        "bandwidth_gb_s": 256.0,
        "tflops_fp16": 48.0,
        "tops_int8": 126.0,
        "tdp_watts": 120,
        "efficiency_factor": 0.60,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html"
    },
    "intel-core-ultra-9-288v-32gb": {
        "id": "intel-core-ultra-9-288v-32gb",
        "name": "Intel Core Ultra 9 288V (Lunar Lake 32GB)",
        "category": "integrated_npu_arc",
        "vram_gb": 24.0,
        "usable_vram_gb": 22.0,
        "bandwidth_gb_s": 136.0,
        "tflops_fp16": 32.0,
        "tops_int8": 120.0,
        "tdp_watts": 30,
        "efficiency_factor": 0.60,
        "reference_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        "benchmark_guide": "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html"
    },
    "qualcomm-snapdragon-x-elite-64gb": {
        "id": "qualcomm-snapdragon-x-elite-64gb",
        "name": "Qualcomm Snapdragon X Elite (64GB)",
        "category": "unified_soc_npu",
        "vram_gb": 48.0,
        "usable_vram_gb": 44.0,
        "bandwidth_gb_s": 135.0,
        "tflops_fp16": 15.0,
        "tops_int8": 45.0,
        "tdp_watts": 45,
        "efficiency_factor": 0.60,
        "reference_url": "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html",
        "benchmark_guide": "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html"
    }
}

CANONICAL_URLS = {
    "homepage": "https://www.iaexplore.com/",
    "ai_hardware_guide_en": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
    "ai_hardware_guide_es": "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html",
    "ai_hardware_guide_pt": "https://www.iaexplore.com/html/pt/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html",
    "pc_benchmarks_2026_es": "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html",
    "pc_benchmarks_2026_en": "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
    "pc_benchmarks_2026_pt": "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html",
    "arm_tablets_2026_pt": "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html",
    "landauer_limit_es": "https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html",
    "neural_network_sim": "https://www.iaexplore.com/neural-network-sim/",
}


def get_model(model_id: str) -> Dict[str, Any]:
    """Retrieve model specification by ID."""
    key = model_id.lower().strip()
    if key not in MODELS_DB:
        raise KeyError(
            f"Unknown model '{model_id}'. Available models: {', '.join(MODELS_DB.keys())}"
        )
    return MODELS_DB[key]


def get_hardware(hardware_id: str) -> Dict[str, Any]:
    """Retrieve hardware profile by ID."""
    key = hardware_id.lower().strip()
    if key not in HARDWARE_DB:
        raise KeyError(
            f"Unknown hardware '{hardware_id}'. Available hardware: {', '.join(HARDWARE_DB.keys())}"
        )
    return HARDWARE_DB[key]


def get_quant_bits(quant_str: str) -> float:
    """Resolve bit-depth for quantization descriptor."""
    key = quant_str.upper().strip()
    if key in QUANTIZATION_BITS:
        return QUANTIZATION_BITS[key]
    try:
        return float(quant_str)
    except ValueError:
        raise ValueError(
            f"Unknown quantization '{quant_str}'. Supported options: {', '.join(QUANTIZATION_BITS.keys())} or custom float."
        )
