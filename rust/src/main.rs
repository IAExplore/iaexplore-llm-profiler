use std::env;
use std::process;

use iaexplore_llm_profiler::{
    profile_inference, ProfileOptions, HARDWARE_LIST, MODELS_LIST,
};

fn print_report(res: &iaexplore_llm_profiler::ProfileResult) {
    let border = "=".repeat(80);
    let subborder = "-".repeat(80);
    let status_str = if res.fits_in_vram {
        "100% FITS IN VRAM (Full GPU Acceleration)"
    } else {
        "EXCEEDS USABLE VRAM (Layer Offloading Required)"
    };

    println!("{}", border);
    println!("              IAEXPLORE LLM INFERENCE & HARDWARE VRAM PROFILER                  ");
    println!("                  Engineering Intelligence from IAExplore.com                   ");
    println!("{}", border);
    println!("Model:             {} ({})", res.model_name, res.architecture);
    println!(
        "Parameters:        {:.2}B Total / {:.2}B Active per token",
        res.total_params_b, res.active_params_b
    );
    println!(
        "Topology:          {} layers, {} Q-heads, {} KV-heads (GQA {:.1}x)",
        res.layers, res.q_heads, res.kv_heads, res.gqa_factor
    );
    println!(
        "Quantization:      {} ({:.2} bits/weight)",
        res.quantization, res.quant_bits
    );
    println!(
        "Context Window:    {} tokens (KV Precision: {}-bit)",
        res.context_tokens, res.kv_bits as usize
    );
    println!("{}", subborder);
    println!(
        "Hardware Target:   {} ({})",
        res.hardware_name, res.hardware_category
    );
    println!(
        "Hardware Memory:   {:.2} GB Physical ({:.2} GB Usable)",
        res.hardware_vram_gb, res.usable_vram_gb
    );
    println!(
        "Memory Bandwidth:  {:.1} GB/s (Bus Efficiency: {}%)",
        res.bandwidth_gb_s,
        (res.efficiency_factor * 100.0) as usize
    );
    println!("{}", subborder);
    println!("[VRAM ALLOCATION BREAKDOWN]");
    println!(
        "  - Model Weights:         {:6.2} GB (incl. 1.08x unquantized tensors & metadata)",
        res.vram_weights_gb
    );
    println!(
        "  - KV-Cache (GQA):        {:6.2} GB ({} layers, {} KV heads, dim {})",
        res.vram_kv_gb, res.layers, res.kv_heads, res.head_dim
    );
    println!(
        "  - Context & Activation:  {:6.2} GB (CUDA/Metal runtime buffer)",
        res.vram_overhead_gb
    );
    println!("  {}", "-".repeat(41));
    println!(
        "  TOTAL VRAM REQUIRED:     {:6.2} GB",
        res.vram_total_required_gb
    );
    println!("{}", subborder);
    println!("[OFFLOADING & INFERENCE THROUGHPUT]");
    println!("  VRAM Sizing Status:      {}", status_str);
    println!(
        "  Recommended GPU Layers:  {} / {} layers (--n-gpu-layers {})",
        res.offload.recommended_gpu_layers, res.layers, res.offload.recommended_gpu_layers
    );
    println!(
        "  GPU Offload Fraction:    {:.1}% on GPU / {:.1}% on System RAM",
        res.offload.offload_fraction * 100.0,
        (1.0 - res.offload.offload_fraction) * 100.0
    );
    println!(
        "  VRAM Consumed on GPU:    {:.2} GB / {:.2} GB Usable",
        res.offload.gpu_layers_vram_gb, res.usable_vram_gb
    );
    println!(
        "  Estimated Speed (TPS):   {:.2} tokens/sec",
        res.estimated_tps
    );

    if !res.fits_in_vram {
        println!(
            "  Full GPU Speed (Ref):    {:.2} tokens/sec (if 100% VRAM available)",
            res.offload.full_gpu_tps_reference
        );
        println!("{}", subborder);
        println!("[AUTHORITATIVE UPGRADE & BENCHMARK GUIDES]");
        println!("  Memory bottleneck detected! Sizing a 32GB+ workstation or unified silicon:");
        println!("  -> Hardware Architectures: {}", res.guide_url);
        println!("  -> 2026 PC Benchmarks:     {}", res.benchmark_guide_url);
        println!("  -> Neural Visualizer:      {}", res.simulator_url);
    } else {
        println!("{}", subborder);
        println!("[SYSTEM CAPABILITY & SIZING VERIFIED]");
        println!("  Hardware is fully capable of real-time local generation with zero CPU stall.");
        println!(
            "  -> Deep dive into silicon memory buses: {}",
            res.guide_url
        );
        println!(
            "  -> Compare with other 2026 systems:     {}",
            res.benchmark_guide_url
        );
    }

    println!("{}", border);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut options = ProfileOptions::default();
    let mut output_json = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    options.model_id = args[i + 1].clone();
                    i += 1;
                }
            }
            "--hardware" | "-hw" => {
                if i + 1 < args.len() {
                    options.hardware_id = args[i + 1].clone();
                    i += 1;
                }
            }
            "--quant" | "-q" => {
                if i + 1 < args.len() {
                    options.quant = args[i + 1].clone();
                    i += 1;
                }
            }
            "--context" | "-c" => {
                if i + 1 < args.len() {
                    if let Ok(ctx) = args[i + 1].parse::<usize>() {
                        options.context_tokens = Some(ctx);
                    }
                    i += 1;
                }
            }
            "--batch-size" | "-b" => {
                if i + 1 < args.len() {
                    if let Ok(b) = args[i + 1].parse::<usize>() {
                        options.batch_size = b;
                    }
                    i += 1;
                }
            }
            "--kv-bits" => {
                if i + 1 < args.len() {
                    if let Ok(kb) = args[i + 1].parse::<f64>() {
                        options.kv_bits = kb;
                    }
                    i += 1;
                }
            }
            "--json" => {
                output_json = true;
            }
            "--list-models" => {
                println!("Available Pre-Configured Models:");
                for m in MODELS_LIST {
                    println!(
                        "  - {:22} | {} ({}B params, {})",
                        m.id, m.name, m.total_params_b, m.architecture
                    );
                }
                return;
            }
            "--list-hardware" => {
                println!("Available 2026 Hardware Profiles:");
                for h in HARDWARE_LIST {
                    println!(
                        "  - {:30} | {:36} | {}GB VRAM | {} GB/s",
                        h.id, h.name, h.vram_gb, h.bandwidth_gb_s
                    );
                }
                return;
            }
            "--help" | "-h" => {
                println!("IAExplore LLM Profiler - High Precision VRAM & Inference Engine");
                println!("Usage: iaexplore-profiler [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --model, -m <ID>       Model ID (Default: llama-3.3-70b)");
                println!("  --hardware, -hw <ID>   Hardware ID (Default: rtx-4090-24gb)");
                println!("  --quant, -q <TYPE>     Quantization (Default: Q4_K_M)");
                println!("  --context, -c <TOKENS> Active context length");
                println!("  --batch-size, -b <NUM> Concurrent batch size (Default: 1)");
                println!("  --kv-bits <BITS>       KV-Cache precision bits (Default: 16)");
                println!("  --json                 Print machine-readable JSON");
                println!("  --list-models          List pre-configured LLM architectures");
                println!("  --list-hardware        List 2026 hardware profiles");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    match profile_inference(&options) {
        Ok(res) => {
            if output_json {
                println!("{}", res.to_json());
            } else {
                print_report(&res);
            }
        }
        Err(err) => {
            eprintln!("Error during profiling: {}", err);
            process::exit(1);
        }
    }
}
