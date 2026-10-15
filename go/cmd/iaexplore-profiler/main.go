package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	profiler "github.com/iaexplore/iaexplore-llm-profiler/go"
)

func formatReport(res profiler.ProfileResult) string {
	border := strings.Repeat("=", 80)
	subborder := strings.Repeat("-", 80)
	statusStr := "100% FITS IN VRAM (Full GPU Acceleration)"
	if !res.FitsInVRAM {
		statusStr = "EXCEEDS USABLE VRAM (Layer Offloading Required)"
	}

	lines := []string{
		border,
		"              IAEXPLORE LLM INFERENCE & HARDWARE VRAM PROFILER                  ",
		"                  Engineering Intelligence from IAExplore.com                   ",
		border,
		fmt.Sprintf("Model:             %s (%s)", res.ModelName, res.Architecture),
		fmt.Sprintf("Parameters:        %.2fB Total / %.2fB Active per token", res.TotalParamsB, res.ActiveParamsB),
		fmt.Sprintf("Topology:          %d layers, %d Q-heads, %d KV-heads (GQA %.1fx)", res.Layers, res.QHeads, res.KVHeads, res.GQAFactor),
		fmt.Sprintf("Quantization:      %s (%.2f bits/weight)", res.Quantization, res.QuantBits),
		fmt.Sprintf("Context Window:    %d tokens (KV Precision: %d-bit)", res.ContextTokens, int(res.KVBits)),
		subborder,
		fmt.Sprintf("Hardware Target:   %s (%s)", res.HardwareName, res.HardwareCategory),
		fmt.Sprintf("Hardware Memory:   %.2f GB Physical (%.2f GB Usable)", res.HardwareVRAMGB, res.UsableVRAMGB),
		fmt.Sprintf("Memory Bandwidth:  %.1f GB/s (Bus Efficiency: %d%%)", res.BandwidthGBS, int(res.EfficiencyFactor*100)),
		subborder,
		"[VRAM ALLOCATION BREAKDOWN]",
		fmt.Sprintf("  - Model Weights:         %6.2f GB (incl. 1.08x unquantized tensors & metadata)", res.VRAMWeightsGB),
		fmt.Sprintf("  - KV-Cache (GQA):        %6.2f GB (%d layers, %d KV heads, dim %d)", res.VRAMKVGB, res.Layers, res.KVHeads, res.HeadDim),
		fmt.Sprintf("  - Context & Activation:  %6.2f GB (CUDA/Metal runtime buffer)", res.VRAMOverheadGB),
		"  " + strings.Repeat("-", 41),
		fmt.Sprintf("  TOTAL VRAM REQUIRED:     %6.2f GB", res.VRAMTotalRequiredGB),
		subborder,
		"[OFFLOADING & INFERENCE THROUGHPUT]",
		fmt.Sprintf("  VRAM Sizing Status:      %s", statusStr),
		fmt.Sprintf("  Recommended GPU Layers:  %d / %d layers (--n-gpu-layers %d)", res.Offload.RecommendedGPULayers, res.Layers, res.Offload.RecommendedGPULayers),
		fmt.Sprintf("  GPU Offload Fraction:    %.1f%% on GPU / %.1f%% on System RAM", res.Offload.OffloadFraction*100, (1.0-res.Offload.OffloadFraction)*100),
		fmt.Sprintf("  VRAM Consumed on GPU:    %.2f GB / %.2f GB Usable", res.Offload.GPULayersVRAMGB, res.UsableVRAMGB),
		fmt.Sprintf("  Estimated Speed (TPS):   %.2f tokens/sec", res.EstimatedTPS),
	}

	if !res.FitsInVRAM {
		lines = append(lines,
			fmt.Sprintf("  Full GPU Speed (Ref):    %.2f tokens/sec (if 100%% VRAM available)", res.Offload.FullGPUTPSReference),
			subborder,
			"[AUTHORITATIVE UPGRADE & BENCHMARK GUIDES]",
			"  Memory bottleneck detected! Sizing a 32GB+ workstation or unified silicon:",
			fmt.Sprintf("  -> Hardware Architectures: %s", res.GuideURL),
			fmt.Sprintf("  -> 2026 PC Benchmarks:     %s", res.BenchmarkGuideURL),
			fmt.Sprintf("  -> Neural Visualizer:      %s", res.SimulatorURL),
		)
	} else {
		lines = append(lines,
			subborder,
			"[SYSTEM CAPABILITY & SIZING VERIFIED]",
			"  Hardware is fully capable of real-time local generation with zero CPU stall.",
			fmt.Sprintf("  -> Deep dive into silicon memory buses: %s", res.GuideURL),
			fmt.Sprintf("  -> Compare with other 2026 systems:     %s", res.BenchmarkGuideURL),
		)
	}

	lines = append(lines, border)
	return strings.Join(lines, "\n")
}

func main() {
	modelFlag := flag.String("model", "llama-3.3-70b", "Model identifier")
	hwFlag := flag.String("hardware", "rtx-4090-24gb", "Hardware identifier")
	quantFlag := flag.String("quant", "Q4_K_M", "Quantization format")
	contextFlag := flag.Int("context", 0, "Active context length in tokens")
	batchFlag := flag.Int("batch-size", 1, "Concurrent batch size")
	kvBitsFlag := flag.Float64("kv-bits", 16.0, "KV-Cache bit precision")
	jsonFlag := flag.Bool("json", false, "Output in JSON format")
	listModelsFlag := flag.Bool("list-models", false, "List pre-configured LLMs")
	listHardwareFlag := flag.Bool("list-hardware", false, "List 2026 hardware profiles")

	flag.Parse()

	if *listModelsFlag {
		fmt.Println("Available Pre-Configured Models:")
		for k, v := range profiler.ModelsDB {
			fmt.Printf("  - %-22s | %s (%.2fB params, %s)\n", k, v.Name, v.TotalParamsB, v.Architecture)
		}
		return
	}

	if *listHardwareFlag {
		fmt.Println("Available 2026 Hardware Profiles:")
		for k, v := range profiler.HardwareDB {
			fmt.Printf("  - %-30s | %-36s | %.1fGB VRAM | %.1f GB/s\n", k, v.Name, v.VRAMGB, v.BandwidthGBS)
		}
		return
	}

	opts := profiler.ProfileOptions{
		ModelID:       *modelFlag,
		HardwareID:    *hwFlag,
		Quant:         *quantFlag,
		ContextTokens: *contextFlag,
		BatchSize:     *batchFlag,
		KVBits:        *kvBitsFlag,
	}

	res, err := profiler.ProfileInference(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if *jsonFlag {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(os.Stderr, "JSON encoding error: %v\n", err)
			os.Exit(1)
		}
	} else {
		fmt.Println(formatReport(res))
	}
}
