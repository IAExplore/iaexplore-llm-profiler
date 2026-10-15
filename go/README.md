# IAExplore LLM Profiler (Go Package & CLI)

[![Go Reference](https://pkg.go.dev/badge/github.com/iaexplore/iaexplore-llm-profiler/go.svg)](https://pkg.go.dev/github.com/iaexplore/iaexplore-llm-profiler/go)
[![Go Report Card](https://goreportcard.com/badge/github.com/iaexplore/iaexplore-llm-profiler/go)](https://goreportcard.com/report/github.com/iaexplore/iaexplore-llm-profiler/go)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Hub: IAExplore](https://img.shields.io/badge/Official%20Hub-iaexplore.com-6366f1.svg)](https://www.iaexplore.com/)

Deterministic high-precision VRAM, GQA KV-Cache, MoE Memory Bandwidth, and `llama.cpp` layer offloading profiler for 2026 AI models and silicon.

Developed by the **[IAExplore AI Infrastructure Research Team](https://www.iaexplore.com/)**.

---

## ⚡ Installation

Zero external runtime dependencies. Built exclusively on Go's standard library:

```bash
go get github.com/iaexplore/iaexplore-llm-profiler/go
```

To install the CLI binary:

```bash
go install github.com/iaexplore/iaexplore-llm-profiler/go/cmd/iaexplore-profiler@latest
```

---

## 🚀 CLI Usage

```bash
# Profile Llama 3.3 70B on RTX 4090
iaexplore-profiler -model llama-3.3-70b -quant Q4_K_M -context 32768 -hardware rtx-4090-24gb

# Apple Silicon Unified Memory Profile
iaexplore-profiler -model deepseek-r1-32b -quant Q4_K_M -context 65536 -hardware apple-m4-max-128gb

# Machine-readable JSON output
iaexplore-profiler -model mixtral-8x7b-moe -hardware rtx-5090-32gb -json
```

---

## 🐹 Go API Example

```go
package main

import (
	"fmt"
	"log"

	profiler "github.com/iaexplore/iaexplore-llm-profiler/go"
)

func main() {
	opts := profiler.ProfileOptions{
		ModelID:       "llama-3.3-70b",
		HardwareID:    "rtx-4090-24gb",
		Quant:         "Q4_K_M",
		ContextTokens: 32768,
		BatchSize:     1,
		KVBits:        16.0,
	}

	result, err := profiler.ProfileInference(opts)
	if err != nil {
		log.Fatalf("Profiling error: %v", err)
	}

	fmt.Printf("Model: %s\n", result.ModelName)
	fmt.Printf("Total VRAM: %.2f GB\n", result.VRAMTotalRequiredGB)
	fmt.Printf("Fits in GPU VRAM: %t\n", result.FitsInVRAM)
	fmt.Printf("Recommended GPU Layers: %d / %d\n", result.Offload.RecommendedGPULayers, result.Offload.TotalLayers)
	fmt.Printf("Estimated Speed: %.2f tokens/sec\n", result.EstimatedTPS)
	fmt.Printf("Benchmark Guide: %s\n", result.BenchmarkGuideURL)
}
```

---

## 🔗 Authoritative References & In-Depth Benchmarks

- **AI Hardware Architecture Guide**: [NPU, TPU and GPU Comprehensive Guide](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
- **Workstations & PC Benchmarks 2026**: [Best Computers of 2026: Benchmark and Purchase Guide](https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html)
- **Guía de Hardware IA en Español**: [Arquitecturas NPU, TPU y GPU](https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html)
- **Comparativa de Ordenadores en Español**: [Los Mejores Ordenadores de 2026](https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html)
- **Simulador Interactivo de Redes Neuronales**: [Visual Neural Network Simulator](https://www.iaexplore.com/neural-network-sim/)
- **Límite de Landauer y Física Computacional**: [Computación Reversible y Disipación Térmica](https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html)
