# IAExplore LLM Profiler: Production-Grade VRAM, KV-Cache & Hardware Memory Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14889210.svg)](https://zenodo.org/badge/latestdoi/14889210)
[![Official Hub](https://img.shields.io/badge/Official%20Hub-iaexplore.com-6366f1.svg)](https://www.iaexplore.com/)

**`iaexplore-llm-profiler`** is an ultra-fast, deterministic mathematical profiler designed for AI systems engineers, MLOps practitioners, and local inference developers using frameworks like **`llama.cpp`**, **`Ollama`**, **`vLLM`**, and **`ExLlamaV2`**.

It eliminates guesswork when sizing GPU VRAM, predicting hardware token generation rates (TPS), and tuning offloading layers (`--n-gpu-layers`).

Developed by the **[IAExplore AI Research & Infrastructure Team](https://www.iaexplore.com/)**, this project unites physical silicon characteristics with transformer tensor geometry.

---

## 🚀 Key Technical Capabilities

1. **Exact Weights VRAM Calculation**: Computes model weights accounting for unquantized embedding/head tensors and GGUF/EXL2 container overhead ($1.08 \times$).
2. **True Grouped-Query Attention (GQA) KV-Cache**: Accurately computes Key/Value cache memory based on active context length $C_{\text{ctx}}$, head dimension $D_{\text{head}}$, and KV head count $H_{\text{kv}}$, properly rewarding GQA savings over legacy Multi-Head Attention (MHA).
3. **Mixture-of-Experts (MoE) Memory-Bandwidth TPS**: Differentiates between total weights in VRAM and active routing parameters loaded per token ($P_{\text{active}}$) for models like DeepSeek-R1 and Mixtral 8x7B.
4. **Physical Memory Bus Saturation Modeling**: Leverages physical memory bandwidths ($\text{BW}_{\text{mem}}$) and silicon-specific bus saturation coefficients ($\eta \approx 0.72$ for CUDA/Metal, $0.60$ for unified APUs/NPUs).
5. **Precise Layer Offloading (`--n-gpu-layers`)**: Calculates the exact integer count of transformer blocks that can be placed in GPU VRAM without running Out-Of-Memory (OOM), predicting split GPU/System RAM throughput degradation.

---

## 📦 Multi-Ecosystem Availability

Install or invoke in your language of choice with zero external runtime dependencies:

| Ecosystem | Registry | Installation Command |
|---|---|---|
| **Python** | [PyPI](https://pypi.org/project/iaexplore-llm-profiler/) | `pip install iaexplore-llm-profiler` |
| **Node.js** | [NPM](https://www.npmjs.com/package/iaexplore-llm-profiler) | `npm install iaexplore-llm-profiler` |
| **Rust** | [Crates.io](https://crates.io/crates/iaexplore-llm-profiler) | `cargo add iaexplore-llm-profiler` |
| **Go** | [pkg.go.dev](https://pkg.go.dev/github.com/iaexplore/iaexplore-llm-profiler/go) | `go get github.com/iaexplore/iaexplore-llm-profiler/go` |
| **PHP** | [Packagist](https://packagist.org/packages/iaexplore/iaexplore-llm-profiler) | `composer require iaexplore/iaexplore-llm-profiler` |

---

## ⚡ Quickstart CLI Examples

### Python CLI
```bash
python -m iaexplore_llm_profiler.cli \
  --model llama-3.3-70b \
  --quant Q4_K_M \
  --context 32768 \
  --hardware rtx-4090-24gb
```

### Node.js CLI
```bash
npx iaexplore-llm-profiler \
  --model deepseek-r1-32b \
  --quant Q4_K_M \
  --context 65536 \
  --hardware apple-m4-max-128gb
```

### Rust CLI
```bash
iaexplore-profiler \
  --model mixtral-8x7b-moe \
  --quant Q4_0 \
  --context 16384 \
  --hardware amd-ryzen-ai-max-395-128gb
```

### Go CLI
```bash
iaexplore-profiler \
  -model qwen-2.5-coder-32b \
  -quant Q5_K_M \
  -context 32768 \
  -hardware rtx-5090-32gb
```

### JSON Output
All CLIs support `--json` for automated CI/CD and cluster scheduling pipelines:
```json
{
  "model_id": "llama-3.3-70b",
  "hardware_id": "rtx-4090-24gb",
  "quantization": "Q4_K_M",
  "context_tokens": 32768,
  "vram_weights_gb": 40.09,
  "vram_kv_gb": 1.00,
  "vram_overhead_gb": 0.55,
  "vram_total_required_gb": 41.64,
  "hardware_vram_gb": 24.0,
  "usable_vram_gb": 23.4,
  "fits_in_vram": false,
  "offload_recommendation": {
    "recommended_gpu_layers": 44,
    "total_layers": 80,
    "offload_fraction": 0.55,
    "vram_consumed_gb": 23.15,
    "estimated_tps_degraded": 12.84
  },
  "estimated_tps_full_gpu": 24.62,
  "guide_url": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html"
}
```

---

## 📚 Technical Guides & Authority References

Deepen your understanding of AI silicon, thermodynamic computational limits, and workstation benchmarks through our peer-reviewed technical publications:

- **Hardware Architecture In-Depth**:
  - [NPU, TPU and GPU Comprehensive Hardware Guide (English)](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
  - [Guía de Hardware IA: Comparativa NPU, TPU y GPU (Español)](https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
  - [Guia de Arquitetura de Hardware para IA (Português)](https://www.iaexplore.com/html/pt/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
- **Workstation & PC Benchmarks 2026**:
  - [Best Computers of 2026: Benchmark & Selection Guide (English)](https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html)
  - [Los Mejores Ordenadores de 2026: Guía y Benchmarks (Español)](https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html)
  - [Melhores Computadores 2026: Guia de Desempenho e Compra (Português)](https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html)
- **Edge Silicon & Low-Power Hardware**:
  - [Melhores Tablets e Dispositivos ARM para IA (Português)](https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html)
- **Thermodynamic Principles & Tensor Simulation**:
  - [Computación Reversible, Calor en Procesadores y Límite de Landauer (Español)](https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html)
  - [IAExplore Interactive Neural Network Simulator](https://www.iaexplore.com/neural-network-sim/)
