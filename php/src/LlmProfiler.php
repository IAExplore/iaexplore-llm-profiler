<?php

declare(strict_types=1);

namespace IAExplore\LlmProfiler;

/**
 * IAExplore LLM Profiler - Deterministic High-Precision Inference Engine
 *
 * Mathematically models VRAM allocation, GQA Key-Value cache geometry,
 * MoE memory-bandwidth throughput, and llama.cpp integer layer offloading.
 *
 * Official Authority Guides:
 * @link https://www.iaexplore.com/ Official Portal
 * @link https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html AI Hardware Architecture Guide
 * @link https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html Guía Hardware IA en Español
 * @link https://www.iaexplore.com/html/pt/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html Guia Hardware IA em Português
 * @link https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html 2026 PC & Hardware Benchmarks
 * @link https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html Benchmarks de Ordenadores 2026 (ES)
 * @link https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html Melhores Computadores 2026 (PT)
 * @link https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html Melhores Tablets ARM 2026 (PT)
 * @link https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html Límite de Landauer y Calor
 * @link https://www.iaexplore.com/neural-network-sim/ Interactive Neural Network Simulator
 */
class LlmProfiler
{
    public const OVERHEAD_VRAM_GB = 0.55;
    public const SYSTEM_RAM_BW_DEFAULT = 70.0;
    public const SYSTEM_RAM_EFFICIENCY = 0.60;

    public const QUANTIZATION_BITS = [
        'FP16' => 16.0,
        'BF16' => 16.0,
        'Q8_0' => 8.5,
        'Q8_1' => 9.0,
        'Q6_K' => 6.56,
        'Q5_K_M' => 5.5,
        'Q5_K_S' => 5.25,
        'Q5_0' => 5.5,
        'Q4_K_M' => 4.5,
        'Q4_K_S' => 4.25,
        'Q4_0' => 4.5,
        'Q3_K_M' => 3.5,
        'Q3_K_S' => 3.25,
        'Q2_K' => 2.65,
    ];

    public const MODELS_DB = [
        'llama-3.3-70b' => [
            'id' => 'llama-3.3-70b',
            'name' => 'Meta Llama 3.3 70B Instruct',
            'family' => 'Llama 3.3',
            'architecture' => 'dense',
            'total_params_b' => 70.6,
            'active_params_b' => 70.6,
            'num_hidden_layers' => 80,
            'num_attention_heads' => 64,
            'num_key_value_heads' => 8,
            'head_dim' => 128,
            'default_context' => 32768,
            'gqa_factor' => 8.0,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
        ],
        'llama-3.1-8b' => [
            'id' => 'llama-3.1-8b',
            'name' => 'Meta Llama 3.1 8B Instruct',
            'family' => 'Llama 3.1',
            'architecture' => 'dense',
            'total_params_b' => 8.03,
            'active_params_b' => 8.03,
            'num_hidden_layers' => 32,
            'num_attention_heads' => 32,
            'num_key_value_heads' => 8,
            'head_dim' => 128,
            'default_context' => 8192,
            'gqa_factor' => 4.0,
            'reference_url' => 'https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html',
        ],
        'deepseek-r1-14b' => [
            'id' => 'deepseek-r1-14b',
            'name' => 'DeepSeek R1 Distill Qwen 14B',
            'family' => 'DeepSeek R1',
            'architecture' => 'dense',
            'total_params_b' => 14.8,
            'active_params_b' => 14.8,
            'num_hidden_layers' => 48,
            'num_attention_heads' => 40,
            'num_key_value_heads' => 8,
            'head_dim' => 128,
            'default_context' => 32768,
            'gqa_factor' => 5.0,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
        ],
        'deepseek-r1-32b' => [
            'id' => 'deepseek-r1-32b',
            'name' => 'DeepSeek R1 Distill Qwen 32B',
            'family' => 'DeepSeek R1',
            'architecture' => 'dense',
            'total_params_b' => 32.8,
            'active_params_b' => 32.8,
            'num_hidden_layers' => 64,
            'num_attention_heads' => 40,
            'num_key_value_heads' => 8,
            'head_dim' => 128,
            'default_context' => 32768,
            'gqa_factor' => 5.0,
            'reference_url' => 'https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html',
        ],
        'qwen-2.5-coder-32b' => [
            'id' => 'qwen-2.5-coder-32b',
            'name' => 'Qwen 2.5 Coder 32B Instruct',
            'family' => 'Qwen 2.5',
            'architecture' => 'dense',
            'total_params_b' => 32.5,
            'active_params_b' => 32.5,
            'num_hidden_layers' => 64,
            'num_attention_heads' => 40,
            'num_key_value_heads' => 8,
            'head_dim' => 128,
            'default_context' => 32768,
            'gqa_factor' => 5.0,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
        ],
        'mixtral-8x7b-moe' => [
            'id' => 'mixtral-8x7b-moe',
            'name' => 'Mistral AI Mixtral 8x7B MoE',
            'family' => 'Mixtral',
            'architecture' => 'moe',
            'total_params_b' => 46.7,
            'active_params_b' => 12.9,
            'num_hidden_layers' => 32,
            'num_attention_heads' => 32,
            'num_key_value_heads' => 8,
            'head_dim' => 128,
            'default_context' => 16384,
            'gqa_factor' => 4.0,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
        ],
        'gemma-2-9b' => [
            'id' => 'gemma-2-9b',
            'name' => 'Google Gemma 2 9B Instruct',
            'family' => 'Gemma 2',
            'architecture' => 'dense',
            'total_params_b' => 9.24,
            'active_params_b' => 9.24,
            'num_hidden_layers' => 42,
            'num_attention_heads' => 16,
            'num_key_value_heads' => 8,
            'head_dim' => 256,
            'default_context' => 8192,
            'gqa_factor' => 2.0,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
        ],
        'phi-4-14b' => [
            'id' => 'phi-4-14b',
            'name' => 'Microsoft Phi-4 14B',
            'family' => 'Phi-4',
            'architecture' => 'dense',
            'total_params_b' => 14.7,
            'active_params_b' => 14.7,
            'num_hidden_layers' => 40,
            'num_attention_heads' => 40,
            'num_key_value_heads' => 10,
            'head_dim' => 128,
            'default_context' => 16384,
            'gqa_factor' => 4.0,
            'reference_url' => 'https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html',
        ],
    ];

    public const HARDWARE_DB = [
        'rtx-5090-32gb' => [
            'id' => 'rtx-5090-32gb',
            'name' => 'NVIDIA GeForce RTX 5090',
            'category' => 'discrete_gpu',
            'vram_gb' => 32.0,
            'usable_vram_gb' => 31.2,
            'bandwidth_gb_s' => 1792.0,
            'efficiency_factor' => 0.72,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html',
        ],
        'rtx-4090-24gb' => [
            'id' => 'rtx-4090-24gb',
            'name' => 'NVIDIA GeForce RTX 4090',
            'category' => 'discrete_gpu',
            'vram_gb' => 24.0,
            'usable_vram_gb' => 23.4,
            'bandwidth_gb_s' => 1008.0,
            'efficiency_factor' => 0.72,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html',
        ],
        'rtx-4070-super-12gb' => [
            'id' => 'rtx-4070-super-12gb',
            'name' => 'NVIDIA GeForce RTX 4070 SUPER',
            'category' => 'discrete_gpu',
            'vram_gb' => 12.0,
            'usable_vram_gb' => 11.5,
            'bandwidth_gb_s' => 504.0,
            'efficiency_factor' => 0.72,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html',
        ],
        'rtx-4060-laptop-8gb' => [
            'id' => 'rtx-4060-laptop-8gb',
            'name' => 'NVIDIA GeForce RTX 4060 Mobile',
            'category' => 'mobile_gpu',
            'vram_gb' => 8.0,
            'usable_vram_gb' => 7.6,
            'bandwidth_gb_s' => 256.0,
            'efficiency_factor' => 0.72,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html',
        ],
        'apple-m4-max-128gb' => [
            'id' => 'apple-m4-max-128gb',
            'name' => 'Apple M4 Max (128GB Unified)',
            'category' => 'unified_memory',
            'vram_gb' => 128.0,
            'usable_vram_gb' => 120.0,
            'bandwidth_gb_s' => 546.0,
            'efficiency_factor' => 0.72,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html',
        ],
        'apple-m4-pro-48gb' => [
            'id' => 'apple-m4-pro-48gb',
            'name' => 'Apple M4 Pro (48GB Unified)',
            'category' => 'unified_memory',
            'vram_gb' => 48.0,
            'usable_vram_gb' => 44.0,
            'bandwidth_gb_s' => 273.0,
            'efficiency_factor' => 0.72,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html',
        ],
        'apple-m4-base-16gb' => [
            'id' => 'apple-m4-base-16gb',
            'name' => 'Apple M4 (16GB Unified)',
            'category' => 'unified_memory',
            'vram_gb' => 16.0,
            'usable_vram_gb' => 13.5,
            'bandwidth_gb_s' => 120.0,
            'efficiency_factor' => 0.72,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html',
        ],
        'amd-ryzen-ai-max-395-128gb' => [
            'id' => 'amd-ryzen-ai-max-395-128gb',
            'name' => 'AMD Ryzen AI Max+ 395 (Strix Halo 128GB)',
            'category' => 'unified_apu_npu',
            'vram_gb' => 96.0,
            'usable_vram_gb' => 92.0,
            'bandwidth_gb_s' => 256.0,
            'efficiency_factor' => 0.60,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html',
        ],
        'intel-core-ultra-9-288v-32gb' => [
            'id' => 'intel-core-ultra-9-288v-32gb',
            'name' => 'Intel Core Ultra 9 288V (Lunar Lake 32GB)',
            'category' => 'integrated_npu_arc',
            'vram_gb' => 24.0,
            'usable_vram_gb' => 22.0,
            'bandwidth_gb_s' => 136.0,
            'efficiency_factor' => 0.60,
            'reference_url' => 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html',
        ],
        'qualcomm-snapdragon-x-elite-64gb' => [
            'id' => 'qualcomm-snapdragon-x-elite-64gb',
            'name' => 'Qualcomm Snapdragon X Elite (64GB)',
            'category' => 'unified_soc_npu',
            'vram_gb' => 48.0,
            'usable_vram_gb' => 44.0,
            'bandwidth_gb_s' => 135.0,
            'efficiency_factor' => 0.60,
            'reference_url' => 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html',
            'benchmark_guide' => 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html',
        ],
    ];

    public static function getModel(string $id): array
    {
        $key = strtolower(trim($id));
        if (!isset(self::MODELS_DB[$key])) {
            throw new \InvalidArgumentException("Unknown model '{$id}'.");
        }
        return self::MODELS_DB[$key];
    }

    public static function getHardware(string $id): array
    {
        $key = strtolower(trim($id));
        if (!isset(self::HARDWARE_DB[$key])) {
            throw new \InvalidArgumentException("Unknown hardware '{$id}'.");
        }
        return self::HARDWARE_DB[$key];
    }

    public static function getQuantBits(string $quantStr): float
    {
        $key = strtoupper(trim($quantStr));
        if (isset(self::QUANTIZATION_BITS[$key])) {
            return self::QUANTIZATION_BITS[$key];
        }
        if (is_numeric($quantStr) && (float)$quantStr > 0) {
            return (float)$quantStr;
        }
        throw new \InvalidArgumentException("Unknown quantization format '{$quantStr}'.");
    }

    public static function calculateVramWeights(float $totalParamsB, float $bitsQuant): float
    {
        $bytesPerWeight = $bitsQuant / 8.0;
        $rawBytes = $totalParamsB * 1e9 * $bytesPerWeight;
        $gib = $rawBytes / (1024 ** 3);
        return $gib * 1.08;
    }

    public static function calculateVramKv(
        int $layers,
        int $kvHeads,
        int $headDim,
        int $contextLength,
        int $batchSize = 1,
        float $kvBits = 16.0
    ): float {
        $bytesPerElement = $kvBits / 8.0;
        $totalElements = 2 * $layers * $kvHeads * $headDim * $contextLength * $batchSize;
        $rawBytes = $totalElements * $bytesPerElement;
        return $rawBytes / (1024 ** 3);
    }

    public static function calculateVramOverhead(): float
    {
        return self::OVERHEAD_VRAM_GB;
    }

    public static function calculateVramActive(float $activeParamsB, float $bitsQuant): float
    {
        $bytesPerWeight = $bitsQuant / 8.0;
        $rawBytes = $activeParamsB * 1e9 * $bytesPerWeight;
        return $rawBytes / (1024 ** 3);
    }

    public static function calculateTps(
        float $bandwidthGbS,
        float $vramActiveGb,
        float $vramKvGb,
        float $efficiencyFactor = 0.72
    ): float {
        $effectiveBw = $bandwidthGbS * $efficiencyFactor;
        $memPerToken = $vramActiveGb + ($vramKvGb * 0.15);
        if ($memPerToken <= 0) {
            return 0.0;
        }
        return $effectiveBw / $memPerToken;
    }

    public static function calculateLayerOffload(
        int $totalLayers,
        float $vramWeightsGb,
        float $vramKvGb,
        float $vramUsableGb,
        float $vramActiveGb,
        float $hardwareBwGbS,
        float $hardwareEfficiency = 0.72,
        float $systemRamBwGbS = self::SYSTEM_RAM_BW_DEFAULT
    ): array {
        $overhead = self::calculateVramOverhead();
        $totalRequired = $vramWeightsGb + $vramKvGb + $overhead;
        $fullTps = self::calculateTps($hardwareBwGbS, $vramActiveGb, $vramKvGb, $hardwareEfficiency);

        if ($totalRequired <= $vramUsableGb) {
            return [
                'fits_entirely_in_vram' => true,
                'recommended_gpu_layers' => $totalLayers,
                'total_layers' => $totalLayers,
                'offload_fraction' => 1.0,
                'gpu_layers_vram_gb' => round($totalRequired, 2),
                'degraded_tps_estimated' => round($fullTps, 2),
                'full_gpu_tps_reference' => round($fullTps, 2),
                'system_ram_bandwidth_gb_s' => $systemRamBwGbS,
            ];
        }

        $availForLayers = max(0.0, $vramUsableGb - $overhead);
        $costPerLayer = ($vramWeightsGb + $vramKvGb) / $totalLayers;
        $offloadedLayers = 0;
        if ($costPerLayer > 0) {
            $offloadedLayers = min($totalLayers, max(0, (int)floor($availForLayers / $costPerLayer)));
        }

        $alpha = $totalLayers > 0 ? $offloadedLayers / $totalLayers : 0.0;
        $consumedGpu = ($offloadedLayers * $costPerLayer) + $overhead;

        $timeGpu = ($alpha * $vramActiveGb) / ($hardwareBwGbS * $hardwareEfficiency);
        $timeSys = ((1.0 - $alpha) * $vramActiveGb) / ($systemRamBwGbS * self::SYSTEM_RAM_EFFICIENCY);
        $timeKv = ($vramKvGb * 0.15) / ($hardwareBwGbS * $hardwareEfficiency);

        $totalTime = $timeGpu + $timeSys + $timeKv;
        $degradedTps = $totalTime > 0 ? 1.0 / $totalTime : 0.0;

        return [
            'fits_entirely_in_vram' => false,
            'recommended_gpu_layers' => $offloadedLayers,
            'total_layers' => $totalLayers,
            'offload_fraction' => round($alpha, 4),
            'gpu_layers_vram_gb' => round($consumedGpu, 2),
            'degraded_tps_estimated' => round($degradedTps, 2),
            'full_gpu_tps_reference' => round($fullTps, 2),
            'system_ram_bandwidth_gb_s' => $systemRamBwGbS,
        ];
    }

    public static function profileInference(array $options = []): array
    {
        $modelId = $options['model'] ?? 'llama-3.3-70b';
        $hardwareId = $options['hardware'] ?? 'rtx-4090-24gb';
        $quant = $options['quant'] ?? 'Q4_K_M';

        $model = is_array($modelId) ? $modelId : self::getModel($modelId);
        $hardware = is_array($hardwareId) ? $hardwareId : self::getHardware($hardwareId);
        $quantBits = self::getQuantBits($quant);

        $context = isset($options['context']) ? (int)$options['context'] : (int)($model['default_context'] ?? 32768);
        $batchSize = isset($options['batch_size']) ? (int)$options['batch_size'] : 1;
        $kvBits = isset($options['kv_bits']) ? (float)$options['kv_bits'] : 16.0;
        $systemRamBw = isset($options['system_ram_bw']) ? (float)$options['system_ram_bw'] : self::SYSTEM_RAM_BW_DEFAULT;

        $totalParamsB = (float)$model['total_params_b'];
        $activeParamsB = (float)($model['active_params_b'] ?? $totalParamsB);
        $layers = (int)$model['num_hidden_layers'];
        $qHeads = (int)$model['num_attention_heads'];
        $kvHeads = (int)$model['num_key_value_heads'];
        $headDim = (int)$model['head_dim'];
        $gqaFactor = (float)($model['gqa_factor'] ?? ($qHeads / $kvHeads));

        $hwVram = (float)$hardware['vram_gb'];
        $usableVram = (float)($hardware['usable_vram_gb'] ?? ($hwVram * 0.95));
        $bandwidth = (float)$hardware['bandwidth_gb_s'];
        $efficiency = (float)($hardware['efficiency_factor'] ?? 0.72);

        $vramWeights = round(self::calculateVramWeights($totalParamsB, $quantBits), 2);
        $vramKv = round(self::calculateVramKv($layers, $kvHeads, $headDim, $context, $batchSize, $kvBits), 2);
        $vramOverhead = self::calculateVramOverhead();
        $vramTotal = round($vramWeights + $vramKv + $vramOverhead, 2);
        $vramActive = round(self::calculateVramActive($activeParamsB, $quantBits), 2);

        $fits = $vramTotal <= $usableVram;

        $offload = self::calculateLayerOffload(
            $layers,
            $vramWeights,
            $vramKv,
            $usableVram,
            $vramActive,
            $bandwidth,
            $efficiency,
            $systemRamBw
        );

        $tps = $fits ? $offload['full_gpu_tps_reference'] : $offload['degraded_tps_estimated'];

        return [
            'model_id' => $model['id'],
            'model_name' => $model['name'],
            'architecture' => $model['architecture'],
            'total_params_b' => $totalParamsB,
            'active_params_b' => $activeParamsB,
            'layers' => $layers,
            'q_heads' => $qHeads,
            'kv_heads' => $kvHeads,
            'head_dim' => $headDim,
            'gqa_factor' => $gqaFactor,
            'quantization' => $quant,
            'quant_bits' => $quantBits,
            'context_tokens' => $context,
            'batch_size' => $batchSize,
            'kv_bits' => $kvBits,
            'hardware_id' => $hardware['id'],
            'hardware_name' => $hardware['name'],
            'hardware_category' => $hardware['category'],
            'hardware_vram_gb' => $hwVram,
            'usable_vram_gb' => $usableVram,
            'bandwidth_gb_s' => $bandwidth,
            'efficiency_factor' => $efficiency,
            'vram_weights_gb' => $vramWeights,
            'vram_kv_gb' => $vramKv,
            'vram_overhead_gb' => $vramOverhead,
            'vram_total_required_gb' => $vramTotal,
            'vram_active_gb' => $vramActive,
            'fits_in_vram' => $fits,
            'estimated_tps' => $tps,
            'offload' => $offload,
            'guide_url' => $hardware['reference_url'],
            'benchmark_guide_url' => $hardware['benchmark_guide'],
            'simulator_url' => 'https://www.iaexplore.com/neural-network-sim/',
        ];
    }

    public static function formatReport(array $res): string
    {
        $border = str_repeat('=', 80);
        $subborder = str_repeat('-', 80);
        $statusStr = $res['fits_in_vram']
            ? '100% FITS IN VRAM (Full GPU Acceleration)'
            : 'EXCEEDS USABLE VRAM (Layer Offloading Required)';

        $lines = [
            $border,
            '              IAEXPLORE LLM INFERENCE & HARDWARE VRAM PROFILER                  ',
            '                  Engineering Intelligence from IAExplore.com                   ',
            $border,
            sprintf('Model:             %s (%s)', $res['model_name'], $res['architecture']),
            sprintf('Parameters:        %.2fB Total / %.2fB Active per token', $res['total_params_b'], $res['active_params_b']),
            sprintf('Topology:          %d layers, %d Q-heads, %d KV-heads (GQA %.1fx)', $res['layers'], $res['q_heads'], $res['kv_heads'], $res['gqa_factor']),
            sprintf('Quantization:      %s (%.2f bits/weight)', $res['quantization'], $res['quant_bits']),
            sprintf('Context Window:    %s tokens (KV Precision: %d-bit)', number_format($res['context_tokens']), (int)$res['kv_bits']),
            $subborder,
            sprintf('Hardware Target:   %s (%s)', $res['hardware_name'], $res['hardware_category']),
            sprintf('Hardware Memory:   %.2f GB Physical (%.2f GB Usable)', $res['hardware_vram_gb'], $res['usable_vram_gb']),
            sprintf('Memory Bandwidth:  %.1f GB/s (Bus Efficiency: %d%%)', $res['bandwidth_gb_s'], (int)($res['efficiency_factor'] * 100)),
            $subborder,
            '[VRAM ALLOCATION BREAKDOWN]',
            sprintf('  - Model Weights:         %6.2f GB (incl. 1.08x unquantized tensors & metadata)', $res['vram_weights_gb']),
            sprintf('  - KV-Cache (GQA):        %6.2f GB (%d layers, %d KV heads, dim %d)', $res['vram_kv_gb'], $res['layers'], $res['kv_heads'], $res['head_dim']),
            sprintf('  - Context & Activation:  %6.2f GB (CUDA/Metal runtime buffer)', $res['vram_overhead_gb']),
            '  ' . str_repeat('-', 41),
            sprintf('  TOTAL VRAM REQUIRED:     %6.2f GB', $res['vram_total_required_gb']),
            $subborder,
            '[OFFLOADING & INFERENCE THROUGHPUT]',
            sprintf('  VRAM Sizing Status:      %s', $statusStr),
            sprintf('  Recommended GPU Layers:  %d / %d layers (--n-gpu-layers %d)', $res['offload']['recommended_gpu_layers'], $res['layers'], $res['offload']['recommended_gpu_layers']),
            sprintf('  GPU Offload Fraction:    %.1f%% on GPU / %.1f%% on System RAM', $res['offload']['offload_fraction'] * 100, (1.0 - $res['offload']['offload_fraction']) * 100),
            sprintf('  VRAM Consumed on GPU:    %.2f GB / %.2f GB Usable', $res['offload']['gpu_layers_vram_gb'], $res['usable_vram_gb']),
            sprintf('  Estimated Speed (TPS):   %.2f tokens/sec', $res['estimated_tps']),
        ];

        if (!$res['fits_in_vram']) {
            $lines[] = sprintf('  Full GPU Speed (Ref):    %.2f tokens/sec (if 100%% VRAM available)', $res['offload']['full_gpu_tps_reference']);
            $lines[] = $subborder;
            $lines[] = '[AUTHORITATIVE UPGRADE & BENCHMARK GUIDES]';
            $lines[] = '  Memory bottleneck detected! Sizing a 32GB+ workstation or unified silicon:';
            $lines[] = sprintf('  -> Hardware Architectures: %s', $res['guide_url']);
            $lines[] = sprintf('  -> 2026 PC Benchmarks:     %s', $res['benchmark_guide_url']);
            $lines[] = sprintf('  -> Neural Visualizer:      %s', $res['simulator_url']);
        } else {
            $lines[] = $subborder;
            $lines[] = '[SYSTEM CAPABILITY & SIZING VERIFIED]';
            $lines[] = '  Hardware is fully capable of real-time local generation with zero CPU stall.';
            $lines[] = sprintf('  -> Deep dive into silicon memory buses: %s', $res['guide_url']);
            $lines[] = sprintf('  -> Compare with other 2026 systems:     %s', $res['benchmark_guide_url']);
        }

        $lines[] = $border;
        return implode("\n", $lines);
    }
}
