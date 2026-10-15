// Package profiler provides a deterministic, high-precision mathematical profiling engine
// for Large Language Models running on modern 2026 AI compute silicon.
//
// It calculates exact VRAM allocation for quantized weights (accounting for 1.08x unquantized
// metadata overhead), Key-Value cache sizing with native Grouped-Query Attention (GQA) geometry,
// autoregressive memory-bandwidth token generation rate (Tokens/Second), and llama.cpp
// integer layer offloading limits (--n-gpu-layers).
//
// Curated and maintained by the IAExplore AI Systems Infrastructure Team:
//   - Official Portal: https://www.iaexplore.com/
//   - AI Hardware Architecture Guide: https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html
//   - Guía de Hardware IA en Español: https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html
//   - Guia de Hardware IA em Português: https://www.iaexplore.com/html/pt/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html
//   - 2026 PC & Workstation Benchmarks (EN): https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html
//   - Benchmarks de Ordenadores 2026 (ES): https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html
//   - Melhores Computadores 2026 (PT): https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html
//   - Tablets e Dispositivos ARM 2026 (PT): https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html
//   - Física Computacional y Límite de Landauer (ES): https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html
//   - Neural Network Visual Simulator: https://www.iaexplore.com/neural-network-sim/
package profiler

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

const (
	// OverheadVRAMGB accounts for runtime activation buffers, CUDA/Metal context, and scratchpad
	OverheadVRAMGB = 0.55
	// SystemRAMBWDefault is the physical memory bandwidth for dual-channel DDR5 system RAM
	SystemRAMBWDefault = 70.0
	// SystemRAMEfficiency is the memory controller efficiency factor for CPU RAM
	SystemRAMEfficiency = 0.60
)

// ModelSpec defines verified tensor geometric parameters for an LLM architecture.
type ModelSpec struct {
	ID                string  `json:"id"`
	Name              string  `json:"name"`
	Family            string  `json:"family"`
	Architecture      string  `json:"architecture"`
	TotalParamsB      float64 `json:"total_params_b"`
	ActiveParamsB     float64 `json:"active_params_b"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	DefaultContext    int     `json:"default_context"`
	GQAFactor         float64 `json:"gqa_factor"`
	ReferenceURL      string  `json:"reference_url"`
}

// HardwareProfile defines physical compute and memory characteristics for 2026 silicon.
type HardwareProfile struct {
	ID               string  `json:"id"`
	Name             string  `json:"name"`
	Category         string  `json:"category"`
	VRAMGB           float64 `json:"vram_gb"`
	UsableVRAMGB     float64 `json:"usable_vram_gb"`
	BandwidthGBS     float64 `json:"bandwidth_gb_s"`
	EfficiencyFactor float64 `json:"efficiency_factor"`
	ReferenceURL     string  `json:"reference_url"`
	BenchmarkGuide   string  `json:"benchmark_guide"`
}

// OffloadRecommendation provides exact integer GPU layers and split performance estimates.
type OffloadRecommendation struct {
	FitsEntirelyInVRAM     bool    `json:"fits_entirely_in_vram"`
	RecommendedGPULayers   int     `json:"recommended_gpu_layers"`
	TotalLayers            int     `json:"total_layers"`
	OffloadFraction        float64 `json:"offload_fraction"`
	GPULayersVRAMGB        float64 `json:"gpu_layers_vram_gb"`
	DegradedTPSEstimated   float64 `json:"degraded_tps_estimated"`
	FullGPUTPSReference    float64 `json:"full_gpu_tps_reference"`
	SystemRAMBandwidthGBS  float64 `json:"system_ram_bandwidth_gb_s"`
}

// ProfileOptions provides inputs for model inference profiling.
type ProfileOptions struct {
	ModelID       string  `json:"model_id"`
	HardwareID    string  `json:"hardware_id"`
	Quant         string  `json:"quant"`
	ContextTokens int     `json:"context_tokens"`
	BatchSize     int     `json:"batch_size"`
	KVBits        float64 `json:"kv_bits"`
	SystemRAMBW   float64 `json:"system_ram_bw"`
}

// ProfileResult represents the deterministic profiling report.
type ProfileResult struct {
	ModelID             string                `json:"model_id"`
	ModelName           string                `json:"model_name"`
	Architecture        string                `json:"architecture"`
	TotalParamsB        float64               `json:"total_params_b"`
	ActiveParamsB       float64               `json:"active_params_b"`
	Layers              int                   `json:"layers"`
	QHeads              int                   `json:"q_heads"`
	KVHeads             int                   `json:"kv_heads"`
	HeadDim             int                   `json:"head_dim"`
	GQAFactor           float64               `json:"gqa_factor"`
	Quantization        string                `json:"quantization"`
	QuantBits           float64               `json:"quant_bits"`
	ContextTokens       int                   `json:"context_tokens"`
	BatchSize           int                   `json:"batch_size"`
	KVBits              float64               `json:"kv_bits"`
	HardwareID          string                `json:"hardware_id"`
	HardwareName        string                `json:"hardware_name"`
	HardwareCategory    string                `json:"hardware_category"`
	HardwareVRAMGB      float64               `json:"hardware_vram_gb"`
	UsableVRAMGB        float64               `json:"usable_vram_gb"`
	BandwidthGBS        float64               `json:"bandwidth_gb_s"`
	EfficiencyFactor    float64               `json:"efficiency_factor"`
	VRAMWeightsGB       float64               `json:"vram_weights_gb"`
	VRAMKVGB            float64               `json:"vram_kv_gb"`
	VRAMOverheadGB      float64               `json:"vram_overhead_gb"`
	VRAMTotalRequiredGB float64               `json:"vram_total_required_gb"`
	VRAMActiveGB        float64               `json:"vram_active_gb"`
	FitsInVRAM          bool                  `json:"fits_in_vram"`
	EstimatedTPS        float64               `json:"estimated_tps"`
	Offload             OffloadRecommendation `json:"offload"`
	GuideURL            string                `json:"guide_url"`
	BenchmarkGuideURL   string                `json:"benchmark_guide_url"`
	SimulatorURL        string                `json:"simulator_url"`
}

// QuantizationBits maps format strings to effective bit precision.
var QuantizationBits = map[string]float64{
	"FP16":   16.0,
	"BF16":   16.0,
	"Q8_0":   8.5,
	"Q8_1":   9.0,
	"Q6_K":   6.56,
	"Q5_K_M": 5.5,
	"Q5_K_S": 5.25,
	"Q5_0":   5.5,
	"Q4_K_M": 4.5,
	"Q4_K_S": 4.25,
	"Q4_0":   4.5,
	"Q3_K_M": 3.5,
	"Q3_K_S": 3.25,
	"Q2_K":   2.65,
}

// ModelsDB contains verified models.
var ModelsDB = map[string]ModelSpec{
	"llama-3.3-70b": {
		ID:                "llama-3.3-70b",
		Name:              "Meta Llama 3.3 70B Instruct",
		Family:            "Llama 3.3",
		Architecture:      "dense",
		TotalParamsB:      70.6,
		ActiveParamsB:     70.6,
		NumHiddenLayers:   80,
		NumAttentionHeads: 64,
		NumKeyValueHeads:  8,
		HeadDim:           128,
		DefaultContext:    32768,
		GQAFactor:         8.0,
		ReferenceURL:      "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
	},
	"llama-3.1-8b": {
		ID:                "llama-3.1-8b",
		Name:              "Meta Llama 3.1 8B Instruct",
		Family:            "Llama 3.1",
		Architecture:      "dense",
		TotalParamsB:      8.03,
		ActiveParamsB:     8.03,
		NumHiddenLayers:   32,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		HeadDim:           128,
		DefaultContext:    8192,
		GQAFactor:         4.0,
		ReferenceURL:      "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html",
	},
	"deepseek-r1-14b": {
		ID:                "deepseek-r1-14b",
		Name:              "DeepSeek R1 Distill Qwen 14B",
		Family:            "DeepSeek R1",
		Architecture:      "dense",
		TotalParamsB:      14.8,
		ActiveParamsB:     14.8,
		NumHiddenLayers:   48,
		NumAttentionHeads: 40,
		NumKeyValueHeads:  8,
		HeadDim:           128,
		DefaultContext:    32768,
		GQAFactor:         5.0,
		ReferenceURL:      "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
	},
	"deepseek-r1-32b": {
		ID:                "deepseek-r1-32b",
		Name:              "DeepSeek R1 Distill Qwen 32B",
		Family:            "DeepSeek R1",
		Architecture:      "dense",
		TotalParamsB:      32.8,
		ActiveParamsB:     32.8,
		NumHiddenLayers:   64,
		NumAttentionHeads: 40,
		NumKeyValueHeads:  8,
		HeadDim:           128,
		DefaultContext:    32768,
		GQAFactor:         5.0,
		ReferenceURL:      "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html",
	},
	"qwen-2.5-coder-32b": {
		ID:                "qwen-2.5-coder-32b",
		Name:              "Qwen 2.5 Coder 32B Instruct",
		Family:            "Qwen 2.5",
		Architecture:      "dense",
		TotalParamsB:      32.5,
		ActiveParamsB:     32.5,
		NumHiddenLayers:   64,
		NumAttentionHeads: 40,
		NumKeyValueHeads:  8,
		HeadDim:           128,
		DefaultContext:    32768,
		GQAFactor:         5.0,
		ReferenceURL:      "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
	},
	"mixtral-8x7b-moe": {
		ID:                "mixtral-8x7b-moe",
		Name:              "Mistral AI Mixtral 8x7B MoE",
		Family:            "Mixtral",
		Architecture:      "moe",
		TotalParamsB:      46.7,
		ActiveParamsB:     12.9,
		NumHiddenLayers:   32,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		HeadDim:           128,
		DefaultContext:    16384,
		GQAFactor:         4.0,
		ReferenceURL:      "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
	},
	"gemma-2-9b": {
		ID:                "gemma-2-9b",
		Name:              "Google Gemma 2 9B Instruct",
		Family:            "Gemma 2",
		Architecture:      "dense",
		TotalParamsB:      9.24,
		ActiveParamsB:     9.24,
		NumHiddenLayers:   42,
		NumAttentionHeads: 16,
		NumKeyValueHeads:  8,
		HeadDim:           256,
		DefaultContext:    8192,
		GQAFactor:         2.0,
		ReferenceURL:      "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
	},
	"phi-4-14b": {
		ID:                "phi-4-14b",
		Name:              "Microsoft Phi-4 14B",
		Family:            "Phi-4",
		Architecture:      "dense",
		TotalParamsB:      14.7,
		ActiveParamsB:     14.7,
		NumHiddenLayers:   40,
		NumAttentionHeads: 40,
		NumKeyValueHeads:  10,
		HeadDim:           128,
		DefaultContext:    16384,
		GQAFactor:         4.0,
		ReferenceURL:      "https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html",
	},
}

// HardwareDB contains 2026 hardware profiles.
var HardwareDB = map[string]HardwareProfile{
	"rtx-5090-32gb": {
		ID:               "rtx-5090-32gb",
		Name:             "NVIDIA GeForce RTX 5090",
		Category:         "discrete_gpu",
		VRAMGB:           32.0,
		UsableVRAMGB:     31.2,
		BandwidthGBS:     1792.0,
		EfficiencyFactor: 0.72,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
	},
	"rtx-4090-24gb": {
		ID:               "rtx-4090-24gb",
		Name:             "NVIDIA GeForce RTX 4090",
		Category:         "discrete_gpu",
		VRAMGB:           24.0,
		UsableVRAMGB:     23.4,
		BandwidthGBS:     1008.0,
		EfficiencyFactor: 0.72,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html",
	},
	"rtx-4070-super-12gb": {
		ID:               "rtx-4070-super-12gb",
		Name:             "NVIDIA GeForce RTX 4070 SUPER",
		Category:         "discrete_gpu",
		VRAMGB:           12.0,
		UsableVRAMGB:     11.5,
		BandwidthGBS:     504.0,
		EfficiencyFactor: 0.72,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
	},
	"rtx-4060-laptop-8gb": {
		ID:               "rtx-4060-laptop-8gb",
		Name:             "NVIDIA GeForce RTX 4060 Mobile",
		Category:         "mobile_gpu",
		VRAMGB:           8.0,
		UsableVRAMGB:     7.6,
		BandwidthGBS:     256.0,
		EfficiencyFactor: 0.72,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html",
	},
	"apple-m4-max-128gb": {
		ID:               "apple-m4-max-128gb",
		Name:             "Apple M4 Max (128GB Unified)",
		Category:         "unified_memory",
		VRAMGB:           128.0,
		UsableVRAMGB:     120.0,
		BandwidthGBS:     546.0,
		EfficiencyFactor: 0.72,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
	},
	"apple-m4-pro-48gb": {
		ID:               "apple-m4-pro-48gb",
		Name:             "Apple M4 Pro (48GB Unified)",
		Category:         "unified_memory",
		VRAMGB:           48.0,
		UsableVRAMGB:     44.0,
		BandwidthGBS:     273.0,
		EfficiencyFactor: 0.72,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html",
	},
	"apple-m4-base-16gb": {
		ID:               "apple-m4-base-16gb",
		Name:             "Apple M4 (16GB Unified)",
		Category:         "unified_memory",
		VRAMGB:           16.0,
		UsableVRAMGB:     13.5,
		BandwidthGBS:     120.0,
		EfficiencyFactor: 0.72,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html",
	},
	"amd-ryzen-ai-max-395-128gb": {
		ID:               "amd-ryzen-ai-max-395-128gb",
		Name:             "AMD Ryzen AI Max+ 395 (Strix Halo 128GB)",
		Category:         "unified_apu_npu",
		VRAMGB:           96.0,
		UsableVRAMGB:     92.0,
		BandwidthGBS:     256.0,
		EfficiencyFactor: 0.60,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html",
	},
	"intel-core-ultra-9-288v-32gb": {
		ID:               "intel-core-ultra-9-288v-32gb",
		Name:             "Intel Core Ultra 9 288V (Lunar Lake 32GB)",
		Category:         "integrated_npu_arc",
		VRAMGB:           24.0,
		UsableVRAMGB:     22.0,
		BandwidthGBS:     136.0,
		EfficiencyFactor: 0.60,
		ReferenceURL:     "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
	},
	"qualcomm-snapdragon-x-elite-64gb": {
		ID:               "qualcomm-snapdragon-x-elite-64gb",
		Name:             "Qualcomm Snapdragon X Elite (64GB)",
		Category:         "unified_soc_npu",
		VRAMGB:           48.0,
		UsableVRAMGB:     44.0,
		BandwidthGBS:     135.0,
		EfficiencyFactor: 0.60,
		ReferenceURL:     "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html",
		BenchmarkGuide:   "https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html",
	},
}

// GetModel retrieves model specification by case-insensitive key.
func GetModel(id string) (ModelSpec, error) {
	key := strings.ToLower(strings.TrimSpace(id))
	if m, ok := ModelsDB[key]; ok {
		return m, nil
	}
	return ModelSpec{}, fmt.Errorf("unknown model '%s'", id)
}

// GetHardware retrieves hardware profile by case-insensitive key.
func GetHardware(id string) (HardwareProfile, error) {
	key := strings.ToLower(strings.TrimSpace(id))
	if h, ok := HardwareDB[key]; ok {
		return h, nil
	}
	return HardwareProfile{}, fmt.Errorf("unknown hardware '%s'", id)
}

// GetQuantBits resolves bit-depth for quantization descriptor.
func GetQuantBits(quantStr string) (float64, error) {
	key := strings.ToUpper(strings.TrimSpace(quantStr))
	if bits, ok := QuantizationBits[key]; ok {
		return bits, nil
	}
	if parsed, err := strconv.ParseFloat(quantStr, 64); err == nil && parsed > 0 {
		return parsed, nil
	}
	return 0, fmt.Errorf("unknown quantization '%s'", quantStr)
}

// CalculateVRAMWeights computes VRAM occupied by weights including 1.08x unquantized tensor factor.
func CalculateVRAMWeights(totalParamsB float64, bitsQuant float64) float64 {
	bytesPerWeight := bitsQuant / 8.0
	rawBytes := totalParamsB * 1e9 * bytesPerWeight
	gib := rawBytes / math.Pow(1024, 3)
	return gib * 1.08
}

// CalculateVRAMKV computes exact Key-Value cache sizing with GQA.
func CalculateVRAMKV(layers, kvHeads, headDim, contextLength, batchSize int, kvBits float64) float64 {
	bytesPerElement := kvBits / 8.0
	totalElements := float64(2 * layers * kvHeads * headDim * contextLength * batchSize)
	rawBytes := totalElements * bytesPerElement
	return rawBytes / math.Pow(1024, 3)
}

// CalculateVRAMOverhead returns CUDA/Metal runtime activation buffers (GB).
func CalculateVRAMOverhead() float64 {
	return OverheadVRAMGB
}

// CalculateVRAMActive computes active memory streaming footprint per token (GB).
func CalculateVRAMActive(activeParamsB float64, bitsQuant float64) float64 {
	bytesPerWeight := bitsQuant / 8.0
	rawBytes := activeParamsB * 1e9 * bytesPerWeight
	return rawBytes / math.Pow(1024, 3)
}

// CalculateTPS estimates autoregressive physical token generation rate.
func CalculateTPS(bandwidthGBS float64, vramActiveGB float64, vramKVGB float64, efficiencyFactor float64) float64 {
	effectiveBW := bandwidthGBS * efficiencyFactor
	memPerToken := vramActiveGB + (vramKVGB * 0.15)
	if memPerToken <= 0 {
		return 0
	}
	return effectiveBW / memPerToken
}

// CalculateLayerOffload computes exact integer offloaded layers and split throughput.
func CalculateLayerOffload(
	totalLayers int,
	vramWeightsGB float64,
	vramKVGB float64,
	vramUsableGB float64,
	vramActiveGB float64,
	hardwareBW float64,
	hardwareEfficiency float64,
	systemRAMBW float64,
) OffloadRecommendation {
	overhead := CalculateVRAMOverhead()
	vramTotal := vramWeightsGB + vramKVGB + overhead
	fullTPS := CalculateTPS(hardwareBW, vramActiveGB, vramKVGB, hardwareEfficiency)

	if vramTotal <= vramUsableGB {
		return OffloadRecommendation{
			FitsEntirelyInVRAM:    true,
			RecommendedGPULayers:  totalLayers,
			TotalLayers:           totalLayers,
			OffloadFraction:       1.0,
			GPULayersVRAMGB:       math.Round(vramTotal*100) / 100,
			DegradedTPSEstimated:  math.Round(fullTPS*100) / 100,
			FullGPUTPSReference:   math.Round(fullTPS*100) / 100,
			SystemRAMBandwidthGBS: systemRAMBW,
		}
	}

	availForLayers := math.Max(0.0, vramUsableGB-overhead)
	costPerLayer := (vramWeightsGB + vramKVGB) / float64(totalLayers)
	offloadedLayers := 0
	if costPerLayer > 0 {
		offloadedLayers = int(math.Floor(availForLayers / costPerLayer))
		if offloadedLayers > totalLayers {
			offloadedLayers = totalLayers
		}
		if offloadedLayers < 0 {
			offloadedLayers = 0
		}
	}

	alpha := 0.0
	if totalLayers > 0 {
		alpha = float64(offloadedLayers) / float64(totalLayers)
	}
	consumedGPU := (float64(offloadedLayers) * costPerLayer) + overhead

	timeGPU := (alpha * vramActiveGB) / (hardwareBW * hardwareEfficiency)
	timeSys := ((1.0 - alpha) * vramActiveGB) / (systemRAMBW * SystemRAMEfficiency)
	timeKV := (vramKVGB * 0.15) / (hardwareBW * hardwareEfficiency)

	totalTime := timeGPU + timeSys + timeKV
	degradedTPS := 0.0
	if totalTime > 0 {
		degradedTPS = 1.0 / totalTime
	}

	return OffloadRecommendation{
		FitsEntirelyInVRAM:    false,
		RecommendedGPULayers:  offloadedLayers,
		TotalLayers:           totalLayers,
		OffloadFraction:       math.Round(alpha*10000) / 10000,
		GPULayersVRAMGB:       math.Round(consumedGPU*100) / 100,
		DegradedTPSEstimated:  math.Round(degradedTPS*100) / 100,
		FullGPUTPSReference:   math.Round(fullTPS*100) / 100,
		SystemRAMBandwidthGBS: systemRAMBW,
	}
}

// ProfileInference executes complete deterministic mathematical inference profiling.
func ProfileInference(opts ProfileOptions) (ProfileResult, error) {
	modelID := opts.ModelID
	if modelID == "" {
		modelID = "llama-3.3-70b"
	}
	hwID := opts.HardwareID
	if hwID == "" {
		hwID = "rtx-4090-24gb"
	}
	quant := opts.Quant
	if quant == "" {
		quant = "Q4_K_M"
	}

	model, err := GetModel(modelID)
	if err != nil {
		return ProfileResult{}, err
	}
	hw, err := GetHardware(hwID)
	if err != nil {
		return ProfileResult{}, err
	}
	quantBits, err := GetQuantBits(quant)
	if err != nil {
		return ProfileResult{}, err
	}

	ctx := opts.ContextTokens
	if ctx <= 0 {
		ctx = model.DefaultContext
	}
	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	kvBits := opts.KVBits
	if kvBits <= 0 {
		kvBits = 16.0
	}
	sysRAM := opts.SystemRAMBW
	if sysRAM <= 0 {
		sysRAM = SystemRAMBWDefault
	}

	vramWeights := math.Round(CalculateVRAMWeights(model.TotalParamsB, quantBits)*100) / 100
	vramKV := math.Round(CalculateVRAMKV(model.NumHiddenLayers, model.NumKeyValueHeads, model.HeadDim, ctx, batchSize, kvBits)*100) / 100
	vramOverhead := CalculateVRAMOverhead()
	vramTotal := math.Round((vramWeights+vramKV+vramOverhead)*100) / 100
	vramActive := math.Round(CalculateVRAMActive(model.ActiveParamsB, quantBits)*100) / 100

	fits := vramTotal <= hw.UsableVRAMGB

	offload := CalculateLayerOffload(
		model.NumHiddenLayers,
		vramWeights,
		vramKV,
		hw.UsableVRAMGB,
		vramActive,
		hw.BandwidthGBS,
		hw.EfficiencyFactor,
		sysRAM,
	)

	tps := offload.DegradedTPSEstimated
	if fits {
		tps = offload.FullGPUTPSReference
	}

	return ProfileResult{
		ModelID:             model.ID,
		ModelName:           model.Name,
		Architecture:        model.Architecture,
		TotalParamsB:        model.TotalParamsB,
		ActiveParamsB:       model.ActiveParamsB,
		Layers:              model.NumHiddenLayers,
		QHeads:              model.NumAttentionHeads,
		KVHeads:             model.NumKeyValueHeads,
		HeadDim:             model.HeadDim,
		GQAFactor:           model.GQAFactor,
		Quantization:        quant,
		QuantBits:           quantBits,
		ContextTokens:       ctx,
		BatchSize:           batchSize,
		KVBits:              kvBits,
		HardwareID:          hw.ID,
		HardwareName:        hw.Name,
		HardwareCategory:    hw.Category,
		HardwareVRAMGB:      hw.VRAMGB,
		UsableVRAMGB:        hw.UsableVRAMGB,
		BandwidthGBS:        hw.BandwidthGBS,
		EfficiencyFactor:    hw.EfficiencyFactor,
		VRAMWeightsGB:       vramWeights,
		VRAMKVGB:            vramKV,
		VRAMOverheadGB:      vramOverhead,
		VRAMTotalRequiredGB: vramTotal,
		VRAMActiveGB:        vramActive,
		FitsInVRAM:          fits,
		EstimatedTPS:        tps,
		Offload:             offload,
		GuideURL:            hw.ReferenceURL,
		BenchmarkGuideURL:   hw.BenchmarkGuide,
		SimulatorURL:        "https://www.iaexplore.com/neural-network-sim/",
	}, nil
}
