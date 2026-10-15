---
annotations_creators:
- machine-generated
- expert-generated
language:
- en
- es
- pt
license: mit
multilinguality:
- multilingual
size_categories:
- n<1K
source_datasets:
- original
task_categories:
- tabular-classification
- tabular-regression
- feature-extraction
task_ids: []
pretty_name: "2026 LLM Hardware & VRAM Memory Profiling Dataset"
tags:
- llm
- vram
- inference
- hardware
- llama-cpp
- ollama
- vllm
- gqa
- moe
- benchmarks
- iaexplore
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - hardware_2026.json
    - models_2026.json
---

# 🧠 IAExplore 2026 LLM Hardware & VRAM Memory Profiling Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14889210.svg)](https://zenodo.org/badge/latestdoi/14889210)
[![Documentation](https://img.shields.io/badge/docs-IAExplore.com-blue.svg)](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official dataset providing verified architectural parameters for cutting-edge Large Language Models (LLMs) and 2026 consumer/workstation AI hardware compute profiles.

Curated by the **[IAExplore Research & Systems Infrastructure Team](https://www.iaexplore.com/)**.

---

## 📌 Dataset Summary

This dataset serves as the ground-truth technical foundation for:
1. **Mathematical VRAM Allocation**: Weight sizing with unquantized metadata overhead factor ($1.08$), KV-Cache memory consumption accounting for Grouped-Query Attention (GQA), and CUDA/Metal runtime execution buffers.
2. **Mixture-of-Experts (MoE) Bandwidth Profiling**: Distinguishing active parameter footprints ($P_{\text{active}}$) from total stored parameters ($P_{\text{total}}$) per token generation step.
3. **Hardware Latency & Tokens/Second Estimation**: Exact physical memory bandwidths ($\text{BW}_{\text{mem}}$), bus topologies (GDDR7, LPDDR5X, GDDR6X, HBM), and bus saturation efficiency coefficients ($\eta$).

---

## 🗂️ Data Structure

### 1. `models_2026.json`
Contains verified tensor specifications for:
- `llama-3.3-70b` (Meta AI)
- `llama-3.1-8b` (Meta AI)
- `deepseek-r1-14b` (DeepSeek AI)
- `deepseek-r1-32b` (DeepSeek AI)
- `qwen-2.5-coder-32b` (Alibaba Cloud)
- `mixtral-8x7b-moe` (Mistral AI)
- `gemma-2-9b` (Google DeepMind)
- `phi-4-14b` (Microsoft Research)

Key fields:
- `total_params_b`: Total billion parameters.
- `active_params_b`: Active parameters loaded per generation token (vital for MoE architectures).
- `num_hidden_layers` ($L$): Number of transformer blocks.
- `num_attention_heads` ($H_{\text{q}}$): Query heads.
- `num_key_value_heads` ($H_{\text{kv}}$): Key/Value heads for GQA/MHA.
- `head_dim` ($D_{\text{head}}$): Dimension per head.
- `gqa_factor`: Compression ratio ($H_{\text{q}} / H_{\text{kv}}$).

### 2. `hardware_2026.json`
Profiles for 2026 silicon across Discrete GPUs, Apple Silicon, and NPU/APU SoC systems:
- **NVIDIA GeForce RTX 5090** (32 GB GDDR7, 1792 GB/s, 575W)
- **NVIDIA GeForce RTX 4090** (24 GB GDDR6X, 1008 GB/s, 450W)
- **NVIDIA GeForce RTX 4070 SUPER** (12 GB GDDR6X, 504 GB/s, 220W)
- **NVIDIA GeForce RTX 4060 Mobile** (8 GB GDDR6, 256 GB/s, 115W)
- **Apple M4 Max** (128 GB Unified LPDDR5X, 546 GB/s, 90W)
- **Apple M4 Pro** (48 GB Unified LPDDR5X, 273 GB/s, 60W)
- **Apple M4** (16 GB Unified LPDDR5X, 120 GB/s, 30W)
- **AMD Ryzen AI Max+ 395 Strix Halo** (96 GB assignable, 256 GB/s, 126 TOPS, 120W)
- **Intel Core Ultra 9 288V Lunar Lake** (24 GB shared, 136 GB/s, 120 TOPS, 30W)
- **Qualcomm Snapdragon X Elite** (48 GB assignable, 135 GB/s, 45 TOPS, 45W)

---

## 🔬 Mathematical Formulas

### Model Weights Memory:
$$\text{VRAM}_{\text{weights}} = \frac{P_{\text{total}} \times 10^9 \times (\text{bits}_{\text{quant}} / 8)}{1024^3} \times 1.08$$

### Exact KV-Cache Memory (with GQA):
$$\text{VRAM}_{\text{kv}} = \frac{2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times C_{\text{ctx}} \times B_{\text{batch}} \times (\text{bits}_{\text{kv}} / 8)}{1024^3}$$

### Autoregressive Inference Throughput (Tokens/Second):
$$\text{TPS}_{\text{gen}} = \frac{\text{BW}_{\text{mem}} \times \eta_{\text{efficiency}}}{\text{VRAM}_{\text{active}} + (\text{VRAM}_{\text{kv}} \times 0.15)}$$

---

## 🔗 Authoritative References & In-Depth Guides

- **AI Hardware Architecture Guide (EN)**: [NPU, TPU and GPU Complete Guide](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
- **Guía de Hardware de IA (ES)**: [Arquitecturas NPU, TPU y GPU para Inteligencia Artificial](https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
- **Guia de Hardware de IA (PT)**: [Hardware para IA: Guia Completo NPU, TPU e GPU](https://www.iaexplore.com/html/pt/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
- **2026 PC & Hardware Benchmarks (EN)**: [Best Computers of 2026: Benchmark Guide](https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html)
- **Comparativas de Ordenadores 2026 (ES)**: [Los Mejores Ordenadores de 2026: Guía y Benchmarks](https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html)
- **Melhores Computadores 2026 (PT)**: [Melhores Computadores 2026: Guia Completo de Compra](https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html)
- **ARM & Mobile AI Devices 2026 (PT)**: [Melhores Tablets e Dispositivos ARM](https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html)
- **Landauer Principle & Reversible Computing (ES)**: [Física de la Computación y Límite Termodinámico](https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html)
- **Interactive Neural Network Simulator**: [IAExplore Real-Time Simulator](https://www.iaexplore.com/neural-network-sim/)

---

## 📜 Citation

If you use this dataset in academic research, benchmark pipelines, or LLM serving infrastructure, please cite:

```bibtex
@misc{iaexplore_llm_profiler_2026,
  author = {IAExplore Research Team},
  title = {IAExplore LLM Profiler: Deterministic VRAM and Hardware Inference Dataset},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14889210},
  url = {https://www.iaexplore.com/}
}
```
