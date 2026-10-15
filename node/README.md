# IAExplore LLM Profiler (Node.js & TypeScript SDK)

[![NPM Version](https://img.shields.io/npm/v/iaexplore-llm-profiler.svg)](https://www.npmjs.com/package/iaexplore-llm-profiler)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Hub: IAExplore](https://img.shields.io/badge/Official%20Hub-iaexplore.com-6366f1.svg)](https://www.iaexplore.com/)

Deterministic high-precision VRAM, GQA KV-Cache, MoE Memory Bandwidth, and `llama.cpp` layer offloading profiler for 2026 AI models and silicon.

Developed by the **[IAExplore AI Infrastructure Research Team](https://www.iaexplore.com/)**.

---

## ⚡ Installation

Zero runtime dependencies. Works with Node.js 16+ and all modern JavaScript / TypeScript runtimes:

```bash
npm install iaexplore-llm-profiler
```

---

## 🚀 CLI Usage

Run via `npx` or install globally:

```bash
# Basic execution
npx iaexplore-llm-profiler --model llama-3.3-70b --quant Q4_K_M --context 32768 --hardware rtx-4090-24gb

# Apple Silicon Unified Memory Profile
npx iaexplore-llm-profiler --model deepseek-r1-32b --quant Q4_K_M --context 65536 --hardware apple-m4-max-128gb

# Machine-readable JSON
npx iaexplore-llm-profiler --model mixtral-8x7b-moe --quant Q4_0 --hardware rtx-5090-32gb --json
```

---

## 💻 Programmatic JavaScript / TypeScript SDK

```javascript
const { profileInference, getModel, getHardware } = require('iaexplore-llm-profiler');

const result = profileInference({
  model: 'llama-3.3-70b',
  hardware: 'rtx-4090-24gb',
  quant: 'Q4_K_M',
  context: 32768
});

console.log(`Model: ${result.modelName}`);
console.log(`Total VRAM Required: ${result.vramTotalRequiredGb} GB`);
console.log(`Fits in GPU VRAM: ${result.fitsInVram}`);
console.log(`Recommended GPU Layers: ${result.offload.recommendedGpuLayers} / ${result.offload.totalLayers}`);
console.log(`Estimated Speed: ${result.estimatedTps} tokens/sec`);
console.log(`Authoritative Guide: ${result.guideUrl}`);
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
