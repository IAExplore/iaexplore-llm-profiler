//! # IAExplore LLM Profiler: Deterministic High-Precision VRAM & Inference Engine
//!
//! Deterministic mathematical profiler for Large Language Models running on modern 2026 AI hardware,
//! addressing VRAM sizing, Grouped-Query Attention (GQA) Key-Value cache geometry,
//! Mixture-of-Experts (MoE) active parameter routing, and `llama.cpp` layer offload limits.
//!
//! Authored and verified by the **[IAExplore Research & Systems Infrastructure Team](https://www.iaexplore.com/)**.
//!
//! ## 🔬 Authoritative Silicon Guides & Benchmark Matrix
//!
//! The following peer-reviewed resources document the physical memory topologies, thermal dissipation,
//! and performance characteristics embedded within this crate:
//!
//! | Resource Title | Language / Topic | Canonical Link |
//! |---|---|---|
//! | **AI Hardware Architecture Guide** | Comprehensive NPU, TPU & GPU Memory Topologies | [iaexplore.com/ai-hardware-guide](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html) |
//! | **Guía de Hardware de IA (ES)** | Arquitecturas NPU, TPU y GPU en Español | [iaexplore.com/guia-hardware-ia-es](https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html) |
//! | **Guia de Hardware de IA (PT)** | Arquitetura NPU, TPU e GPU em Português | [iaexplore.com/guia-hardware-ia-pt](https://www.iaexplore.com/html/pt/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html) |
//! | **2026 PC & Hardware Benchmarks (EN)** | Best Computers & Workstations 2026 | [iaexplore.com/best-computers-2026-en](https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html) |
//! | **Benchmarks de Ordenadores 2026 (ES)** | Los Mejores Ordenadores de 2026 | [iaexplore.com/mejores-ordenadores-2026-es](https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html) |
//! | **Melhores Computadores 2026 (PT)** | Guia e Comparativo de Computadores 2026 | [iaexplore.com/melhores-computadores-2026-pt](https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html) |
//! | **ARM & Tablet AI Benchmarks (PT)** | Melhores Tablets e Dispositivos ARM 2026 | [iaexplore.com/melhores-tablets-2026-pt](https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html) |
//! | **Landauer Limit & Physical Computing (ES)** | Computación Reversible y Disipación Térmica | [iaexplore.com/limite-landauer-calor-procesadores](https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html) |
//! | **Interactive Neural Network Simulator** | Real-time Tensor Visualization Tool | [iaexplore.com/neural-network-sim](https://www.iaexplore.com/neural-network-sim/) |
//! | **Official Research Portal** | IAExplore Official Technical Homepage | [iaexplore.com](https://www.iaexplore.com/) |
//!
//! ## 📐 Mathematical Formulation
//!
//! - **Model Weights**: `VRAM_weights = (P_total * 1e9 * (bits_quant / 8) / (1024^3)) * 1.08`
//! - **KV-Cache (GQA)**: `VRAM_kv = (2 * L * H_kv * D_head * C_ctx * B_batch * (bits_kv / 8)) / (1024^3)`
//! - **Autoregressive TPS**: `TPS_gen = (BW_mem * eta) / (VRAM_active + (VRAM_kv * 0.15))`

pub const OVERHEAD_VRAM_GB: f64 = 0.55;
pub const SYSTEM_RAM_BW_DEFAULT: f64 = 70.0;
pub const SYSTEM_RAM_EFFICIENCY: f64 = 0.60;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub family: &'static str,
    pub architecture: &'static str,
    pub total_params_b: f64,
    pub active_params_b: f64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub default_context: usize,
    pub gqa_factor: f64,
    pub reference_url: &'static str,
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub id: &'static str,
    pub name: &'static str,
    pub category: &'static str,
    pub vram_gb: f64,
    pub usable_vram_gb: f64,
    pub bandwidth_gb_s: f64,
    pub efficiency_factor: f64,
    pub reference_url: &'static str,
    pub benchmark_guide: &'static str,
}

pub const MODELS_LIST: &[ModelSpec] = &[
    ModelSpec {
        id: "llama-3.3-70b",
        name: "Meta Llama 3.3 70B Instruct",
        family: "Llama 3.3",
        architecture: "dense",
        total_params_b: 70.6,
        active_params_b: 70.6,
        num_hidden_layers: 80,
        num_attention_heads: 64,
        num_key_value_heads: 8,
        head_dim: 128,
        default_context: 32768,
        gqa_factor: 8.0,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
    },
    ModelSpec {
        id: "llama-3.1-8b",
        name: "Meta Llama 3.1 8B Instruct",
        family: "Llama 3.1",
        architecture: "dense",
        total_params_b: 8.03,
        active_params_b: 8.03,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        default_context: 8192,
        gqa_factor: 4.0,
        reference_url: "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html",
    },
    ModelSpec {
        id: "deepseek-r1-14b",
        name: "DeepSeek R1 Distill Qwen 14B",
        family: "DeepSeek R1",
        architecture: "dense",
        total_params_b: 14.8,
        active_params_b: 14.8,
        num_hidden_layers: 48,
        num_attention_heads: 40,
        num_key_value_heads: 8,
        head_dim: 128,
        default_context: 32768,
        gqa_factor: 5.0,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
    },
    ModelSpec {
        id: "deepseek-r1-32b",
        name: "DeepSeek R1 Distill Qwen 32B",
        family: "DeepSeek R1",
        architecture: "dense",
        total_params_b: 32.8,
        active_params_b: 32.8,
        num_hidden_layers: 64,
        num_attention_heads: 40,
        num_key_value_heads: 8,
        head_dim: 128,
        default_context: 32768,
        gqa_factor: 5.0,
        reference_url: "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html",
    },
    ModelSpec {
        id: "qwen-2.5-coder-32b",
        name: "Qwen 2.5 Coder 32B Instruct",
        family: "Qwen 2.5",
        architecture: "dense",
        total_params_b: 32.5,
        active_params_b: 32.5,
        num_hidden_layers: 64,
        num_attention_heads: 40,
        num_key_value_heads: 8,
        head_dim: 128,
        default_context: 32768,
        gqa_factor: 5.0,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
    },
    ModelSpec {
        id: "mixtral-8x7b-moe",
        name: "Mistral AI Mixtral 8x7B MoE",
        family: "Mixtral",
        architecture: "moe",
        total_params_b: 46.7,
        active_params_b: 12.9,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        default_context: 16384,
        gqa_factor: 4.0,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
    },
    ModelSpec {
        id: "gemma-2-9b",
        name: "Google Gemma 2 9B Instruct",
        family: "Gemma 2",
        architecture: "dense",
        total_params_b: 9.24,
        active_params_b: 9.24,
        num_hidden_layers: 42,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 256,
        default_context: 8192,
        gqa_factor: 2.0,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
    },
    ModelSpec {
        id: "phi-4-14b",
        name: "Microsoft Phi-4 14B",
        family: "Phi-4",
        architecture: "dense",
        total_params_b: 14.7,
        active_params_b: 14.7,
        num_hidden_layers: 40,
        num_attention_heads: 40,
        num_key_value_heads: 10,
        head_dim: 128,
        default_context: 16384,
        gqa_factor: 4.0,
        reference_url: "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html",
    },
];

pub const HARDWARE_LIST: &[HardwareProfile] = &[
    HardwareProfile {
        id: "rtx-5090-32gb",
        name: "NVIDIA GeForce RTX 5090",
        category: "discrete_gpu",
        vram_gb: 32.0,
        usable_vram_gb: 31.2,
        bandwidth_gb_s: 1792.0,
        efficiency_factor: 0.72,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
    },
    HardwareProfile {
        id: "rtx-4090-24gb",
        name: "NVIDIA GeForce RTX 4090",
        category: "discrete_gpu",
        vram_gb: 24.0,
        usable_vram_gb: 23.4,
        bandwidth_gb_s: 1008.0,
        efficiency_factor: 0.72,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html",
    },
    HardwareProfile {
        id: "rtx-4070-super-12gb",
        name: "NVIDIA GeForce RTX 4070 SUPER",
        category: "discrete_gpu",
        vram_gb: 12.0,
        usable_vram_gb: 11.5,
        bandwidth_gb_s: 504.0,
        efficiency_factor: 0.72,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
    },
    HardwareProfile {
        id: "rtx-4060-laptop-8gb",
        name: "NVIDIA GeForce RTX 4060 Mobile",
        category: "mobile_gpu",
        vram_gb: 8.0,
        usable_vram_gb: 7.6,
        bandwidth_gb_s: 256.0,
        efficiency_factor: 0.72,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html",
    },
    HardwareProfile {
        id: "apple-m4-max-128gb",
        name: "Apple M4 Max (128GB Unified)",
        category: "unified_memory",
        vram_gb: 128.0,
        usable_vram_gb: 120.0,
        bandwidth_gb_s: 546.0,
        efficiency_factor: 0.72,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
    },
    HardwareProfile {
        id: "apple-m4-pro-48gb",
        name: "Apple M4 Pro (48GB Unified)",
        category: "unified_memory",
        vram_gb: 48.0,
        usable_vram_gb: 44.0,
        bandwidth_gb_s: 273.0,
        efficiency_factor: 0.72,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html",
    },
    HardwareProfile {
        id: "apple-m4-base-16gb",
        name: "Apple M4 (16GB Unified)",
        category: "unified_memory",
        vram_gb: 16.0,
        usable_vram_gb: 13.5,
        bandwidth_gb_s: 120.0,
        efficiency_factor: 0.72,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html",
    },
    HardwareProfile {
        id: "amd-ryzen-ai-max-395-128gb",
        name: "AMD Ryzen AI Max+ 395 (Strix Halo 128GB)",
        category: "unified_apu_npu",
        vram_gb: 96.0,
        usable_vram_gb: 92.0,
        bandwidth_gb_s: 256.0,
        efficiency_factor: 0.60,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html",
    },
    HardwareProfile {
        id: "intel-core-ultra-9-288v-32gb",
        name: "Intel Core Ultra 9 288V (Lunar Lake 32GB)",
        category: "integrated_npu_arc",
        vram_gb: 24.0,
        usable_vram_gb: 22.0,
        bandwidth_gb_s: 136.0,
        efficiency_factor: 0.60,
        reference_url: "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
        benchmark_guide: "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
    },
    HardwareProfile {
        id: "qualcomm-snapdragon-x-elite-64gb",
        name: "Qualcomm Snapdragon X Elite (64GB)",
        category: "unified_soc_npu",
        vram_gb: 48.0,
        usable_vram_gb: 44.0,
        bandwidth_gb_s: 135.0,
        efficiency_factor: 0.60,
        reference_url: "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html",
        benchmark_guide: "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html",
    },
];

pub fn get_quant_bits(quant_str: &str) -> Result<f64, String> {
    let q = quant_str.to_uppercase();
    match q.as_str() {
        "FP16" | "BF16" => Ok(16.0),
        "Q8_0" => Ok(8.5),
        "Q8_1" => Ok(9.0),
        "Q6_K" => Ok(6.56),
        "Q5_K_M" | "Q5_0" => Ok(5.5),
        "Q5_K_S" => Ok(5.25),
        "Q4_K_M" | "Q4_0" => Ok(4.5),
        "Q4_K_S" => Ok(4.25),
        "Q3_K_M" => Ok(3.5),
        "Q3_K_S" => Ok(3.25),
        "Q2_K" => Ok(2.65),
        _ => quant_str
            .parse::<f64>()
            .map_err(|_| format!("Unknown quantization format: {}", quant_str)),
    }
}

pub fn get_model(id: &str) -> Option<&'static ModelSpec> {
    let lower = id.to_lowercase();
    MODELS_LIST.iter().find(|m| m.id == lower)
}

pub fn get_hardware(id: &str) -> Option<&'static HardwareProfile> {
    let lower = id.to_lowercase();
    HARDWARE_LIST.iter().find(|h| h.id == lower)
}

pub fn calculate_vram_weights(total_params_b: f64, bits_quant: f64) -> f64 {
    let bytes_per_weight = bits_quant / 8.0;
    let raw_bytes = total_params_b * 1e9 * bytes_per_weight;
    let gib = raw_bytes / (1024.0_f64.powi(3));
    gib * 1.08
}

pub fn calculate_vram_kv(
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    context_length: usize,
    batch_size: usize,
    kv_bits: f64,
) -> f64 {
    let bytes_per_element = kv_bits / 8.0;
    let total_elements = 2 * layers * kv_heads * head_dim * context_length * batch_size;
    let raw_bytes = total_elements as f64 * bytes_per_element;
    raw_bytes / (1024.0_f64.powi(3))
}

pub fn calculate_vram_overhead() -> f64 {
    OVERHEAD_VRAM_GB
}

pub fn calculate_vram_active(active_params_b: f64, bits_quant: f64) -> f64 {
    let bytes_per_weight = bits_quant / 8.0;
    let raw_bytes = active_params_b * 1e9 * bytes_per_weight;
    raw_bytes / (1024.0_f64.powi(3))
}

pub fn calculate_tps(
    bandwidth_gb_s: f64,
    vram_active_gb: f64,
    vram_kv_gb: f64,
    efficiency_factor: f64,
) -> f64 {
    let effective_bw = bandwidth_gb_s * efficiency_factor;
    let mem_per_token = vram_active_gb + (vram_kv_gb * 0.15);
    if mem_per_token <= 0.0 {
        return 0.0;
    }
    effective_bw / mem_per_token
}

#[derive(Debug, Clone)]
pub struct OffloadRecommendation {
    pub fits_entirely_in_vram: bool,
    pub recommended_gpu_layers: usize,
    pub total_layers: usize,
    pub offload_fraction: f64,
    pub gpu_layers_vram_gb: f64,
    pub degraded_tps_estimated: f64,
    pub full_gpu_tps_reference: f64,
    pub system_ram_bandwidth_gb_s: f64,
}

pub fn calculate_layer_offload(
    total_layers: usize,
    vram_weights_gb: f64,
    vram_kv_gb: f64,
    vram_usable_gb: f64,
    vram_active_gb: f64,
    hardware_bw_gb_s: f64,
    hardware_efficiency: f64,
    system_ram_bw_gb_s: f64,
) -> OffloadRecommendation {
    let vram_overhead = calculate_vram_overhead();
    let vram_total = vram_weights_gb + vram_kv_gb + vram_overhead;
    let full_tps = calculate_tps(
        hardware_bw_gb_s,
        vram_active_gb,
        vram_kv_gb,
        hardware_efficiency,
    );

    if vram_total <= vram_usable_gb {
        return OffloadRecommendation {
            fits_entirely_in_vram: true,
            recommended_gpu_layers: total_layers,
            total_layers,
            offload_fraction: 1.0,
            gpu_layers_vram_gb: vram_total,
            degraded_tps_estimated: full_tps,
            full_gpu_tps_reference: full_tps,
            system_ram_bandwidth_gb_s: system_ram_bw_gb_s,
        };
    }

    let avail_for_layers = (vram_usable_gb - vram_overhead).max(0.0);
    let cost_per_layer = (vram_weights_gb + vram_kv_gb) / (total_layers as f64);
    let offloaded_layers = if cost_per_layer > 0.0 {
        ((avail_for_layers / cost_per_layer).floor() as usize).min(total_layers)
    } else {
        0
    };

    let alpha = if total_layers > 0 {
        offloaded_layers as f64 / total_layers as f64
    } else {
        0.0
    };
    let consumed_gpu = (offloaded_layers as f64 * cost_per_layer) + vram_overhead;

    let time_gpu = (alpha * vram_active_gb) / (hardware_bw_gb_s * hardware_efficiency);
    let time_sys =
        ((1.0 - alpha) * vram_active_gb) / (system_ram_bw_gb_s * SYSTEM_RAM_EFFICIENCY);
    let time_kv = (vram_kv_gb * 0.15) / (hardware_bw_gb_s * hardware_efficiency);

    let total_time = time_gpu + time_sys + time_kv;
    let degraded_tps = if total_time > 0.0 {
        1.0 / total_time
    } else {
        0.0
    };

    OffloadRecommendation {
        fits_entirely_in_vram: false,
        recommended_gpu_layers: offloaded_layers,
        total_layers,
        offload_fraction: (alpha * 10000.0).round() / 10000.0,
        gpu_layers_vram_gb: (consumed_gpu * 100.0).round() / 100.0,
        degraded_tps_estimated: (degraded_tps * 100.0).round() / 100.0,
        full_gpu_tps_reference: (full_tps * 100.0).round() / 100.0,
        system_ram_bandwidth_gb_s: system_ram_bw_gb_s,
    }
}

#[derive(Debug, Clone)]
pub struct ProfileOptions {
    pub model_id: String,
    pub hardware_id: String,
    pub quant: String,
    pub context_tokens: Option<usize>,
    pub batch_size: usize,
    pub kv_bits: f64,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self {
            model_id: "llama-3.3-70b".to_string(),
            hardware_id: "rtx-4090-24gb".to_string(),
            quant: "Q4_K_M".to_string(),
            context_tokens: None,
            batch_size: 1,
            kv_bits: 16.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProfileResult {
    pub model_id: String,
    pub model_name: String,
    pub architecture: String,
    pub total_params_b: f64,
    pub active_params_b: f64,
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub gqa_factor: f64,
    pub quantization: String,
    pub quant_bits: f64,
    pub context_tokens: usize,
    pub batch_size: usize,
    pub kv_bits: f64,

    pub hardware_id: String,
    pub hardware_name: String,
    pub hardware_category: String,
    pub hardware_vram_gb: f64,
    pub usable_vram_gb: f64,
    pub bandwidth_gb_s: f64,
    pub efficiency_factor: f64,

    pub vram_weights_gb: f64,
    pub vram_kv_gb: f64,
    pub vram_overhead_gb: f64,
    pub vram_total_required_gb: f64,
    pub vram_active_gb: f64,

    pub fits_in_vram: bool,
    pub estimated_tps: f64,
    pub offload: OffloadRecommendation,
    pub guide_url: String,
    pub benchmark_guide_url: String,
    pub simulator_url: String,
}

impl ProfileResult {
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \
              \"model_id\": \"{}\",\n  \
              \"model_name\": \"{}\",\n  \
              \"architecture\": \"{}\",\n  \
              \"total_params_b\": {},\n  \
              \"active_params_b\": {},\n  \
              \"layers\": {},\n  \
              \"q_heads\": {},\n  \
              \"kv_heads\": {},\n  \
              \"head_dim\": {},\n  \
              \"gqa_factor\": {},\n  \
              \"quantization\": \"{}\",\n  \
              \"quant_bits\": {},\n  \
              \"context_tokens\": {},\n  \
              \"batch_size\": {},\n  \
              \"kv_bits\": {},\n  \
              \"hardware_id\": \"{}\",\n  \
              \"hardware_name\": \"{}\",\n  \
              \"hardware_category\": \"{}\",\n  \
              \"hardware_vram_gb\": {},\n  \
              \"usable_vram_gb\": {},\n  \
              \"bandwidth_gb_s\": {},\n  \
              \"efficiency_factor\": {},\n  \
              \"vram_weights_gb\": {},\n  \
              \"vram_kv_gb\": {},\n  \
              \"vram_overhead_gb\": {},\n  \
              \"vram_total_required_gb\": {},\n  \
              \"vram_active_gb\": {},\n  \
              \"fits_in_vram\": {},\n  \
              \"estimated_tps\": {},\n  \
              \"offload\": {{\n    \
                \"fits_entirely_in_vram\": {},\n    \
                \"recommended_gpu_layers\": {},\n    \
                \"total_layers\": {},\n    \
                \"offload_fraction\": {},\n    \
                \"gpu_layers_vram_gb\": {},\n    \
                \"degraded_tps_estimated\": {},\n    \
                \"full_gpu_tps_reference\": {},\n    \
                \"system_ram_bandwidth_gb_s\": {}\n  \
              }},\n  \
              \"guide_url\": \"{}\",\n  \
              \"benchmark_guide_url\": \"{}\",\n  \
              \"simulator_url\": \"{}\"\n\
            }}",
            self.model_id,
            self.model_name,
            self.architecture,
            self.total_params_b,
            self.active_params_b,
            self.layers,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.gqa_factor,
            self.quantization,
            self.quant_bits,
            self.context_tokens,
            self.batch_size,
            self.kv_bits,
            self.hardware_id,
            self.hardware_name,
            self.hardware_category,
            self.hardware_vram_gb,
            self.usable_vram_gb,
            self.bandwidth_gb_s,
            self.efficiency_factor,
            self.vram_weights_gb,
            self.vram_kv_gb,
            self.vram_overhead_gb,
            self.vram_total_required_gb,
            self.vram_active_gb,
            self.fits_in_vram,
            self.estimated_tps,
            self.offload.fits_entirely_in_vram,
            self.offload.recommended_gpu_layers,
            self.offload.total_layers,
            self.offload.offload_fraction,
            self.offload.gpu_layers_vram_gb,
            self.offload.degraded_tps_estimated,
            self.offload.full_gpu_tps_reference,
            self.offload.system_ram_bandwidth_gb_s,
            self.guide_url,
            self.benchmark_guide_url,
            self.simulator_url
        )
    }
}

pub fn profile_inference(options: &ProfileOptions) -> Result<ProfileResult, String> {
    let model = get_model(&options.model_id)
        .ok_or_else(|| format!("Unknown model: {}", options.model_id))?;
    let hardware = get_hardware(&options.hardware_id)
        .ok_or_else(|| format!("Unknown hardware: {}", options.hardware_id))?;

    let quant_bits = get_quant_bits(&options.quant)?;
    let context = options.context_tokens.unwrap_or(model.default_context);

    let vram_weights = ((calculate_vram_weights(model.total_params_b, quant_bits) * 100.0).round()) / 100.0;
    let vram_kv = ((calculate_vram_kv(
        model.num_hidden_layers,
        model.num_key_value_heads,
        model.head_dim,
        context,
        options.batch_size,
        options.kv_bits,
    ) * 100.0).round()) / 100.0;

    let vram_overhead = calculate_vram_overhead();
    let vram_total = (( (vram_weights + vram_kv + vram_overhead) * 100.0).round()) / 100.0;
    let vram_active = ((calculate_vram_active(model.active_params_b, quant_bits) * 100.0).round()) / 100.0;

    let fits = vram_total <= hardware.usable_vram_gb;

    let offload = calculate_layer_offload(
        model.num_hidden_layers,
        vram_weights,
        vram_kv,
        hardware.usable_vram_gb,
        vram_active,
        hardware.bandwidth_gb_s,
        hardware.efficiency_factor,
        SYSTEM_RAM_BW_DEFAULT,
    );

    let tps = if fits {
        offload.full_gpu_tps_reference
    } else {
        offload.degraded_tps_estimated
    };

    Ok(ProfileResult {
        model_id: model.id.to_string(),
        model_name: model.name.to_string(),
        architecture: model.architecture.to_string(),
        total_params_b: model.total_params_b,
        active_params_b: model.active_params_b,
        layers: model.num_hidden_layers,
        q_heads: model.num_attention_heads,
        kv_heads: model.num_key_value_heads,
        head_dim: model.head_dim,
        gqa_factor: model.gqa_factor,
        quantization: options.quant.clone(),
        quant_bits,
        context_tokens: context,
        batch_size: options.batch_size,
        kv_bits: options.kv_bits,
        hardware_id: hardware.id.to_string(),
        hardware_name: hardware.name.to_string(),
        hardware_category: hardware.category.to_string(),
        hardware_vram_gb: hardware.vram_gb,
        usable_vram_gb: hardware.usable_vram_gb,
        bandwidth_gb_s: hardware.bandwidth_gb_s,
        efficiency_factor: hardware.efficiency_factor,
        vram_weights_gb: vram_weights,
        vram_kv_gb: vram_kv,
        vram_overhead_gb: vram_overhead,
        vram_total_required_gb: vram_total,
        vram_active_gb: vram_active,
        fits_in_vram: fits,
        estimated_tps: tps,
        offload,
        guide_url: hardware.reference_url.to_string(),
        benchmark_guide_url: hardware.benchmark_guide.to_string(),
        simulator_url: "https://www.iaexplore.com/neural-network-sim/".to_string(),
    })
}
