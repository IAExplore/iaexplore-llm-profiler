# IAExplore LLM Profiler (Rust Crate & CLI)

[![Crates.io](https://img.shields.io/crates/v/iaexplore-llm-profiler.svg)](https://crates.io/crates/iaexplore-llm-profiler)
[![Documentation](https://docs.rs/iaexplore-llm-profiler/badge.svg)](https://docs.rs/iaexplore-llm-profiler)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Hub: IAExplore](https://img.shields.io/badge/Official%20Hub-iaexplore.com-6366f1.svg)](https://www.iaexplore.com/)

Deterministic high-precision VRAM, GQA KV-Cache, MoE Memory Bandwidth, and `llama.cpp` layer offloading profiler for 2026 AI models and silicon.

Developed by the **[IAExplore AI Infrastructure Research Team](https://www.iaexplore.com/)**.

---

## ⚡ Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
iaexplore-llm-profiler = "2026.1.0"
```

Or install the standalone binary:

```bash
cargo install iaexplore-llm-profiler
```

---

## 🚀 CLI Usage

```bash
# Sizing Llama 3.3 70B on RTX 4090
iaexplore-profiler --model llama-3.3-70b --quant Q4_K_M --context 32768 --hardware rtx-4090-24gb

# Apple Silicon Unified Memory Profile
iaexplore-profiler --model deepseek-r1-32b --quant Q4_K_M --context 65536 --hardware apple-m4-max-128gb

# Machine-readable JSON output
iaexplore-profiler --model mixtral-8x7b-moe --quant Q4_0 --hardware rtx-5090-32gb --json
```

---

## 🦀 Rust API Example

```rust
use iaexplore_llm_profiler::{profile_inference, ProfileOptions};

fn main() {
    let options = ProfileOptions {
        model_id: "llama-3.3-70b".to_string(),
        hardware_id: "rtx-4090-24gb".to_string(),
        quant: "Q4_K_M".to_string(),
        context_tokens: Some(32768),
        batch_size: 1,
        kv_bits: 16.0,
    };

    let result = profile_inference(&options).expect("Profiling failed");

    println!("Model: {}", result.model_name);
    println!("Total VRAM Required: {:.2} GB", result.vram_total_required_gb);
    println!("Fits in GPU VRAM: {}", result.fits_in_vram);
    println!(
        "Recommended GPU Layers: {} / {}",
        result.offload.recommended_gpu_layers,
        result.offload.total_layers
    );
    println!("Estimated Speed: {:.2} tokens/sec", result.estimated_tps);
    println!("Authoritative Guide: {}", result.guide_url);
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
