# IAExplore LLM Profiler (Python SDK & CLI)

[![PyPI Version](https://img.shields.io/pypi/v/iaexplore-llm-profiler.svg)](https://pypi.org/project/iaexplore-llm-profiler/)
[![Python Versions](https://img.shields.io/pypi/pyversions/iaexplore-llm-profiler.svg)](https://pypi.org/project/iaexplore-llm-profiler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Hub: IAExplore](https://img.shields.io/badge/Official%20Hub-iaexplore.com-6366f1.svg)](https://www.iaexplore.com/)

Deterministic high-precision VRAM, GQA KV-Cache, MoE Memory Bandwidth, and `llama.cpp` layer offloading profiler for 2026 AI models and silicon.

Developed by the **[IAExplore AI Infrastructure Research Team](https://www.iaexplore.com/)**.

---

## ⚡ Installation

Requires Python 3.9+ with **standard library only** (no torch, transformers, or external runtime dependencies required):

```bash
pip install iaexplore-llm-profiler
```

---

## 🚀 CLI Usage

```bash
# Basic profile
iaexplore-profiler --model llama-3.3-70b --quant Q4_K_M --context 32768 --hardware rtx-4090-24gb

# Or via Python module execution
python -m iaexplore_llm_profiler.cli --model deepseek-r1-32b --quant Q4_K_M --context 65536 --hardware apple-m4-max-128gb

# Machine-readable JSON output
iaexplore-profiler --model mixtral-8x7b-moe --quant Q4_0 --context 16384 --hardware rtx-5090-32gb --json
```

---

## 🐍 Python SDK Example

```python
from iaexplore_llm_profiler import profile_inference, get_model, get_hardware

# Retrieve models and hardware profiles
model = get_model("llama-3.3-70b")
hw = get_hardware("rtx-4090-24gb")

# Execute deterministic mathematical profiling
result = profile_inference(
    model="llama-3.3-70b",
    hardware="rtx-4090-24gb",
    quant="Q4_K_M",
    context=32768,
    batch_size=1,
    kv_bits=16
)

print(f"Total VRAM Required: {result.vram_total_required_gb:.2f} GB")
print(f"Fits in GPU VRAM: {result.fits_in_vram}")
print(f"Recommended GPU Layers: {result.offload.recommended_gpu_layers} / {result.offload.total_layers}")
print(f"Estimated Inference Speed: {result.estimated_tps:.2f} tokens/sec")
print(f"Authoritative Guide: {result.guide_url}")
```

---

## 🔬 Deterministic Equations

- **Weights VRAM**: $\text{VRAM}_{\text{weights}} = \frac{P_{\text{total}} \times 10^9 \times (\text{bits}_{\text{quant}} / 8)}{1024^3} \times 1.08$
- **KV-Cache VRAM**: $\text{VRAM}_{\text{kv}} = \frac{2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times C_{\text{ctx}} \times B_{\text{batch}} \times (\text{bits}_{\text{kv}} / 8)}{1024^3}$
- **Token Generation (TPS)**: $\text{TPS}_{\text{gen}} = \frac{\text{BW}_{\text{mem}} \times \eta}{\text{VRAM}_{\text{active}} + (\text{VRAM}_{\text{kv}} \times 0.15)}$

---

## 🔗 Authoritative References & In-Depth Benchmarks

- **AI Hardware Architecture Guide**: [NPU, TPU and GPU Comprehensive Guide](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
- **Workstations & PC Benchmarks 2026**: [Best Computers of 2026: Benchmark and Purchase Guide](https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html)
- **Guía de Hardware IA en Español**: [Arquitecturas NPU, TPU y GPU](https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
- **Comparativa de Ordenadores en Español**: [Los Mejores Ordenadores de 2026](https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html)
- **Simulador Interactivo de Redes Neuronales**: [Visual Neural Network Simulator](https://www.iaexplore.com/neural-network-sim/)
- **Límite de Landauer y Física Computacional**: [Computación Reversible y Disipación Térmica](https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html)
