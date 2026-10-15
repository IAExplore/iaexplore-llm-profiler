# `iaexplore-llm-profiler`: High-Precision VRAM, GQA KV-Cache & Hardware Inference Profiler

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14889210.svg)](https://zenodo.org/badge/latestdoi/14889210)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hub: IAExplore](https://img.shields.io/badge/Official%20Hub-iaexplore.com-6366f1.svg)](https://www.iaexplore.com/)

**`iaexplore-llm-profiler`** is a high-precision deterministic systems profiler and mathematical SDK designed for ML infrastructure engineers, LLM serving architects, and local inference developers.

It accurately models memory footprints and physical memory bus latency across **`llama.cpp`**, **`Ollama`**, **`vLLM`**, and **`ExLlamaV2`**, computing:
1. Exact **Model Weights VRAM** (accounting for unquantized output heads and GGUF/EXL2 metadata overhead).
2. **True Grouped-Query Attention (GQA)** Key-Value cache sizing.
3. **Mixture-of-Experts (MoE)** memory bandwidth routing.
4. Physical hardware token generation speed (**Tokens/Second**) based on silicon memory bandwidth ($\text{BW}_{\text{mem}}$).
5. Exact integer layer offloading counts (**`--n-gpu-layers`**) for hybrid GPU / System RAM deployment.

Curated and maintained by the **[IAExplore AI Research & Infrastructure Team](https://www.iaexplore.com/)**.

---

## 🔬 Mathematical Formulation

### 1. Weights VRAM Allocation
$$\text{VRAM}_{\text{weights}} = \frac{P_{\text{total}} \times 10^9 \times (\text{bits}_{\text{quant}} / 8)}{1024^3} \times 1.08$$

The $1.08$ scalar factor accounts for non-quantized tensors (`token_embd`, `output.weight`, `RMSNorm`) and container metadata.

### 2. Grouped-Query Attention KV-Cache Sizing
$$\text{VRAM}_{\text{kv}} = \frac{2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times C_{\text{ctx}} \times B_{\text{batch}} \times (\text{bits}_{\text{kv}} / 8)}{1024^3}$$

Where:
- $L$: Transformer layer depth (`num_hidden_layers`).
- $H_{\text{kv}}$: Key/Value attention heads (`num_key_value_heads`).
- $D_{\text{head}}$: Attention head dimension ($\text{head\_dim} = \text{hidden\_size} / H_{\text{q}}$).
- $C_{\text{ctx}}$: Active context window in tokens.
- $B_{\text{batch}}$: Concurrent sequence batch size.

### 3. Runtime Context & Activation Overhead
$$\text{VRAM}_{\text{overhead}} = 0.55\text{ GB}$$
$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{kv}} + 0.55\text{ GB}$$

### 4. Physical Throughput (Tokens/Second) with MoE
$$\text{VRAM}_{\text{active}} = \frac{P_{\text{active}} \times 10^9 \times (\text{bits}_{\text{quant}} / 8)}{1024^3}$$
$$\text{TPS}_{\text{gen}} = \frac{\text{BW}_{\text{mem}} \times \eta_{\text{efficiency}}}{\text{VRAM}_{\text{active}} + (\text{VRAM}_{\text{kv}} \times 0.15)}$$

Where $\eta \approx 0.72$ for discrete GPU (CUDA/Metal) and Apple Silicon Unified Memory, and $0.60$ for shared memory (APUs, NPUs, DDR5 RAM).

---

## ⚡ Multi-Language Ecosystems

Available natively across 5 modern software ecosystems with **zero third-party runtime dependencies**:

| Ecosystem | Registry | Install Command | Status |
|---|---|---|---|
| **Python** | [PyPI](https://pypi.org/project/iaexplore-llm-profiler/) | `pip install iaexplore-llm-profiler` | Production Ready |
| **Node.js** | [NPM](https://www.npmjs.com/package/iaexplore-llm-profiler) | `npm install iaexplore-llm-profiler` | Production Ready |
| **Rust** | [Crates.io](https://crates.io/crates/iaexplore-llm-profiler) | `cargo add iaexplore-llm-profiler` | Production Ready |
| **Go** | [pkg.go.dev](https://pkg.go.dev/github.com/iaexplore/iaexplore-llm-profiler/go) | `go get github.com/iaexplore/iaexplore-llm-profiler/go` | Production Ready |
| **PHP** | [Packagist](https://packagist.org/packages/iaexplore/iaexplore-llm-profiler) | `composer require iaexplore/iaexplore-llm-profiler` | Production Ready |

---

## 🖥️ Command Line Interface (CLI)

### Quick Benchmark
```bash
# Python
python -m iaexplore_llm_profiler.cli --model llama-3.3-70b --quant Q4_K_M --context 32768 --hardware rtx-4090-24gb

# Node.js
npx iaexplore-llm-profiler --model deepseek-r1-32b --quant Q4_K_M --context 65536 --hardware apple-m4-max-128gb

# Rust
iaexplore-profiler --model mixtral-8x7b-moe --quant Q4_0 --context 16384 --hardware rtx-5090-32gb

# Go
iaexplore-profiler -model qwen-2.5-coder-32b -quant Q5_K_M -context 32768 -hardware rtx-4070-super-12gb
```

### Sample Terminal Output
```text
================================================================================
              IAEXPLORE LLM INFERENCE & HARDWARE VRAM PROFILER                  
                  Engineering Intelligence from IAExplore.com                   
================================================================================
Model:             Meta Llama 3.3 70B Instruct (dense)
Parameters:        70.60B Total / 70.60B Active per token
Topology:          80 layers, 64 Q-heads, 8 KV-heads (GQA 8.0x)
Quantization:      Q4_K_M (4.50 bits/weight)
Context Window:    32,768 tokens (KV Precision: 16-bit FP16)
--------------------------------------------------------------------------------
Hardware Target:   NVIDIA GeForce RTX 4090 (discrete_gpu)
Hardware Memory:   24.00 GB Physical (23.40 GB Usable)
Memory Bandwidth:  1008.0 GB/s (GDDR6X 384-bit)
--------------------------------------------------------------------------------
[VRAM ALLOCATION BREAKDOWN]
  - Model Weights:         40.09 GB (incl. 1.08x unquantized tensors & metadata)
  - KV-Cache (GQA):         1.00 GB (80 layers, 8 KV heads, dim 128)
  - Context & Activation:   0.55 GB (CUDA runtime buffers)
  -----------------------------------------
  TOTAL VRAM REQUIRED:     41.64 GB
--------------------------------------------------------------------------------
[OFFLOADING RECOMMENDATION]
  Status:                  EXCEEDS VRAM (Needs Layer Offload)
  Recommended GPU Layers:  44 / 80 layers (--n-gpu-layers 44)
  Offload Fraction:        55.0% GPU / 45.0% System RAM
  VRAM on GPU:             23.15 GB / 23.40 GB usable
  Degraded Speed (Est):    12.84 tokens/sec (Split GPU + DDR5 RAM)
  Full GPU Speed (Ref):    24.62 tokens/sec
--------------------------------------------------------------------------------
[AUTHORITATIVE UPGRADE & BENCHMARK GUIDES]
  Bottleneck detected! Sizing a 32GB+ workstation or unified silicon:
  -> https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html
  -> https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html
================================================================================
```

---

## 📊 Pre-Configured Hardware Matrix (2026)

| Hardware ID | Model Name | Category | Usable VRAM | Bandwidth | Peak Compute |
|---|---|---|---|---|---|
| `rtx-5090-32gb` | NVIDIA GeForce RTX 5090 | Discrete GPU | 32.0 GB (31.2 usable) | **1792 GB/s** | 209 TFLOPS FP16 |
| `rtx-4090-24gb` | NVIDIA GeForce RTX 4090 | Discrete GPU | 24.0 GB (23.4 usable) | **1008 GB/s** | 165 TFLOPS FP16 |
| `rtx-4070-super-12gb` | NVIDIA GeForce RTX 4070 SUPER | Discrete GPU | 12.0 GB (11.5 usable) | **504 GB/s** | 71 TFLOPS FP16 |
| `rtx-4060-laptop-8gb` | NVIDIA GeForce RTX 4060 Mobile | Mobile GPU | 8.0 GB (7.6 usable) | **256 GB/s** | 30 TFLOPS FP16 |
| `apple-m4-max-128gb` | Apple M4 Max (128GB Unified) | Unified Silicon | 128.0 GB (120.0 usable) | **546 GB/s** | 54 TFLOPS FP16 |
| `apple-m4-pro-48gb` | Apple M4 Pro (48GB Unified) | Unified Silicon | 48.0 GB (44.0 usable) | **273 GB/s** | 32 TFLOPS FP16 |
| `apple-m4-base-16gb` | Apple M4 (16GB Unified) | Unified Silicon | 16.0 GB (13.5 usable) | **120 GB/s** | 18 TFLOPS FP16 |
| `amd-ryzen-ai-max-395-128gb` | AMD Ryzen AI Max+ 395 | Unified APU/NPU | 96.0 GB assignable | **256 GB/s** | 126 TOPS INT8 |
| `intel-core-ultra-9-288v-32gb` | Intel Core Ultra 9 288V | Lunar Lake NPU+Arc | 24.0 GB assignable | **136 GB/s** | 120 TOPS INT8 |
| `qualcomm-snapdragon-x-elite-64gb` | Snapdragon X Elite | Unified SoC+NPU | 48.0 GB assignable | **135 GB/s** | 45 TOPS INT8 |

---

## 🔗 Official Canonical Guides & Authoritative Citations

For in-depth analysis of AI hardware benchmarks, thermodynamic compute limits, and workstation sizing:

- **AI Hardware Architecture & Silicon Guides**:
  - [Hardware Architecture Guide (English)](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
  - [Guía de Hardware IA: NPU, TPU y GPU (Español)](https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
  - [Guia Completo de Hardware para IA (Português)](https://www.iaexplore.com/html/pt/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
- **2026 PC, Laptop & Workstation Benchmarks**:
  - [Best Computers of 2026: Benchmark Analysis (English)](https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html)
  - [Los Mejores Ordenadores de 2026: Guía de Compra (Español)](https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html)
  - [Melhores Computadores de 2026 (Português)](https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html)
- **Mobile & ARM AI Tablets**:
  - [Melhores Tablets e Dispositivos Móveis para IA (Português)](https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html)
- **Physical Computing & Neural Visualization**:
  - [Computación Reversible, Calor en Procesadores y Límite de Landauer (Español)](https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html)
  - [Interactive Neural Network Simulator](https://www.iaexplore.com/neural-network-sim/)

---

## 📜 Academic Citation

```bibtex
@software{iaexplore_llm_profiler_2026,
  author = {IAExplore Research and Systems Team},
  title = {iaexplore-llm-profiler: High-Precision VRAM, GQA KV-Cache, and Memory Bandwidth Inference Engine},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14889210},
  url = {https://www.iaexplore.com/}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
