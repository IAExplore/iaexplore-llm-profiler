# IAExplore LLM Profiler (PHP Package & CLI)

[![Latest Stable Version](https://poser.pugx.org/iaexplore/iaexplore-llm-profiler/v/stable)](https://packagist.org/packages/iaexplore/iaexplore-llm-profiler)
[![Total Downloads](https://poser.pugx.org/iaexplore/iaexplore-llm-profiler/downloads)](https://packagist.org/packages/iaexplore/iaexplore-llm-profiler)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Hub: IAExplore](https://img.shields.io/badge/Official%20Hub-iaexplore.com-6366f1.svg)](https://www.iaexplore.com/)

Deterministic high-precision VRAM, GQA KV-Cache, MoE Memory Bandwidth, and `llama.cpp` layer offloading profiler for 2026 AI models and silicon.

Developed by the **[IAExplore AI Infrastructure Research Team](https://www.iaexplore.com/)**.

---

## ⚡ Installation

Requires PHP 8.0 or higher. Pure PHP with zero third-party composer dependencies:

```bash
composer require iaexplore/iaexplore-llm-profiler
```

---

## 🚀 CLI Usage

```bash
# Execute via vendor binary
./vendor/bin/iaexplore-profiler --model llama-3.3-70b --quant Q4_K_M --context 32768 --hardware rtx-4090-24gb

# Direct PHP script execution
php src/LlmProfiler.php --model deepseek-r1-32b --hardware apple-m4-max-128gb --json
```

---

## 🐘 PHP API Example

```php
<?php

require_once __DIR__ . '/vendor/autoload.php';

use IAExplore\LlmProfiler\LlmProfiler;

$result = LlmProfiler::profileInference([
    'model' => 'llama-3.3-70b',
    'hardware' => 'rtx-4090-24gb',
    'quant' => 'Q4_K_M',
    'context' => 32768,
    'batch_size' => 1,
    'kv_bits' => 16.0
]);

echo "Model: " . $result['model_name'] . "\n";
echo "Total VRAM Required: " . $result['vram_total_required_gb'] . " GB\n";
echo "Fits in GPU VRAM: " . ($result['fits_in_vram'] ? "YES" : "NO") . "\n";
echo "Recommended GPU Layers: " . $result['offload']['recommended_gpu_layers'] . " / " . $result['layers'] . "\n";
echo "Estimated Speed: " . $result['estimated_tps'] . " tokens/sec\n";
echo "Authoritative Hardware Guide: " . $result['guide_url'] . "\n";
```

---

## 🔗 Authoritative References & In-Depth Benchmarks

- **AI Hardware Architecture Guide**: [NPU, TPU and GPU Comprehensive Guide](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
- **Workstations & PC Benchmarks 2026**: [Best Computers of 2026: Benchmark and Purchase Guide](https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html)
- **Guía de Hardware IA en Español**: [Arquitecturas NPU, TPU y GPU](https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
- **Comparativa de Ordenadores en Español**: [Los Mejores Ordenadores de 2026](https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html)
- **Simulador Interactivo de Redes Neuronales**: [Visual Neural Network Simulator](https://www.iaexplore.com/neural-network-sim/)
- **Límite de Landauer y Física Computacional**: [Computación Reversible y Disipación Térmica](https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html)
