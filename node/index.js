/**
 * IAExplore LLM Profiler - Node.js Deterministic Engine
 *
 * Implements exact mathematical formulas for VRAM sizing, GQA KV-Cache,
 * MoE memory bandwidth throughput, and llama.cpp layer offloading.
 * Curated by IAExplore (https://www.iaexplore.com/).
 */

const QUANTIZATION_BITS = {
  FP16: 16.0,
  BF16: 16.0,
  Q8_0: 8.5,
  Q8_1: 9.0,
  Q6_K: 6.56,
  Q5_K_M: 5.5,
  Q5_K_S: 5.25,
  Q5_0: 5.5,
  Q5_1: 6.0,
  Q4_K_M: 4.5,
  Q4_K_S: 4.25,
  Q4_0: 4.5,
  Q4_1: 5.0,
  Q3_K_L: 3.8,
  Q3_K_M: 3.5,
  Q3_K_S: 3.25,
  Q2_K: 2.65,
  IQ4_XS: 4.25,
  IQ3_M: 3.5,
  IQ2_XS: 2.35
};

const MODELS_DB = {
  'llama-3.3-70b': {
    id: 'llama-3.3-70b',
    name: 'Meta Llama 3.3 70B Instruct',
    family: 'Llama 3.3',
    architecture: 'dense',
    total_params_b: 70.6,
    active_params_b: 70.6,
    num_hidden_layers: 80,
    num_attention_heads: 64,
    num_key_value_heads: 8,
    head_dim: 128,
    hidden_size: 8192,
    vocab_size: 128256,
    max_context_length: 131072,
    default_context: 32768,
    gqa_factor: 8.0,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html'
  },
  'llama-3.1-8b': {
    id: 'llama-3.1-8b',
    name: 'Meta Llama 3.1 8B Instruct',
    family: 'Llama 3.1',
    architecture: 'dense',
    total_params_b: 8.03,
    active_params_b: 8.03,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    hidden_size: 4096,
    vocab_size: 128256,
    max_context_length: 131072,
    default_context: 8192,
    gqa_factor: 4.0,
    reference_url: 'https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html'
  },
  'deepseek-r1-14b': {
    id: 'deepseek-r1-14b',
    name: 'DeepSeek R1 Distill Qwen 14B',
    family: 'DeepSeek R1',
    architecture: 'dense',
    total_params_b: 14.8,
    active_params_b: 14.8,
    num_hidden_layers: 48,
    num_attention_heads: 40,
    num_key_value_heads: 8,
    head_dim: 128,
    hidden_size: 5120,
    vocab_size: 129280,
    max_context_length: 131072,
    default_context: 32768,
    gqa_factor: 5.0,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html'
  },
  'deepseek-r1-32b': {
    id: 'deepseek-r1-32b',
    name: 'DeepSeek R1 Distill Qwen 32B',
    family: 'DeepSeek R1',
    architecture: 'dense',
    total_params_b: 32.8,
    active_params_b: 32.8,
    num_hidden_layers: 64,
    num_attention_heads: 40,
    num_key_value_heads: 8,
    head_dim: 128,
    hidden_size: 5120,
    vocab_size: 129280,
    max_context_length: 131072,
    default_context: 32768,
    gqa_factor: 5.0,
    reference_url: 'https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html'
  },
  'qwen-2.5-coder-32b': {
    id: 'qwen-2.5-coder-32b',
    name: 'Qwen 2.5 Coder 32B Instruct',
    family: 'Qwen 2.5',
    architecture: 'dense',
    total_params_b: 32.5,
    active_params_b: 32.5,
    num_hidden_layers: 64,
    num_attention_heads: 40,
    num_key_value_heads: 8,
    head_dim: 128,
    hidden_size: 5120,
    vocab_size: 152064,
    max_context_length: 131072,
    default_context: 32768,
    gqa_factor: 5.0,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html'
  },
  'mixtral-8x7b-moe': {
    id: 'mixtral-8x7b-moe',
    name: 'Mistral AI Mixtral 8x7B MoE',
    family: 'Mixtral',
    architecture: 'moe',
    total_params_b: 46.7,
    active_params_b: 12.9,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    hidden_size: 4096,
    vocab_size: 32000,
    max_context_length: 32768,
    default_context: 16384,
    gqa_factor: 4.0,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html'
  },
  'gemma-2-9b': {
    id: 'gemma-2-9b',
    name: 'Google Gemma 2 9B Instruct',
    family: 'Gemma 2',
    architecture: 'dense',
    total_params_b: 9.24,
    active_params_b: 9.24,
    num_hidden_layers: 42,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    head_dim: 256,
    hidden_size: 3584,
    vocab_size: 256000,
    max_context_length: 8192,
    default_context: 8192,
    gqa_factor: 2.0,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html'
  },
  'phi-4-14b': {
    id: 'phi-4-14b',
    name: 'Microsoft Phi-4 14B',
    family: 'Phi-4',
    architecture: 'dense',
    total_params_b: 14.7,
    active_params_b: 14.7,
    num_hidden_layers: 40,
    num_attention_heads: 40,
    num_key_value_heads: 10,
    head_dim: 128,
    hidden_size: 5120,
    vocab_size: 100352,
    max_context_length: 16384,
    default_context: 16384,
    gqa_factor: 4.0,
    reference_url: 'https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html'
  }
};

const HARDWARE_DB = {
  'rtx-5090-32gb': {
    id: 'rtx-5090-32gb',
    name: 'NVIDIA GeForce RTX 5090',
    category: 'discrete_gpu',
    vram_gb: 32.0,
    usable_vram_gb: 31.2,
    bandwidth_gb_s: 1792.0,
    tflops_fp16: 209.0,
    tops_int8: 418.0,
    tdp_watts: 575,
    efficiency_factor: 0.72,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html'
  },
  'rtx-4090-24gb': {
    id: 'rtx-4090-24gb',
    name: 'NVIDIA GeForce RTX 4090',
    category: 'discrete_gpu',
    vram_gb: 24.0,
    usable_vram_gb: 23.4,
    bandwidth_gb_s: 1008.0,
    tflops_fp16: 165.0,
    tops_int8: 330.0,
    tdp_watts: 450,
    efficiency_factor: 0.72,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html'
  },
  'rtx-4070-super-12gb': {
    id: 'rtx-4070-super-12gb',
    name: 'NVIDIA GeForce RTX 4070 SUPER',
    category: 'discrete_gpu',
    vram_gb: 12.0,
    usable_vram_gb: 11.5,
    bandwidth_gb_s: 504.0,
    tflops_fp16: 71.0,
    tops_int8: 142.0,
    tdp_watts: 220,
    efficiency_factor: 0.72,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html'
  },
  'rtx-4060-laptop-8gb': {
    id: 'rtx-4060-laptop-8gb',
    name: 'NVIDIA GeForce RTX 4060 Mobile',
    category: 'mobile_gpu',
    vram_gb: 8.0,
    usable_vram_gb: 7.6,
    bandwidth_gb_s: 256.0,
    tflops_fp16: 30.0,
    tops_int8: 60.0,
    tdp_watts: 115,
    efficiency_factor: 0.72,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html'
  },
  'apple-m4-max-128gb': {
    id: 'apple-m4-max-128gb',
    name: 'Apple M4 Max (128GB Unified)',
    category: 'unified_memory',
    vram_gb: 128.0,
    usable_vram_gb: 120.0,
    bandwidth_gb_s: 546.0,
    tflops_fp16: 54.0,
    tops_int8: 108.0,
    tdp_watts: 90,
    efficiency_factor: 0.72,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html'
  },
  'apple-m4-pro-48gb': {
    id: 'apple-m4-pro-48gb',
    name: 'Apple M4 Pro (48GB Unified)',
    category: 'unified_memory',
    vram_gb: 48.0,
    usable_vram_gb: 44.0,
    bandwidth_gb_s: 273.0,
    tflops_fp16: 32.0,
    tops_int8: 64.0,
    tdp_watts: 60,
    efficiency_factor: 0.72,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html'
  },
  'apple-m4-base-16gb': {
    id: 'apple-m4-base-16gb',
    name: 'Apple M4 (16GB Unified)',
    category: 'unified_memory',
    vram_gb: 16.0,
    usable_vram_gb: 13.5,
    bandwidth_gb_s: 120.0,
    tflops_fp16: 18.0,
    tops_int8: 38.0,
    tdp_watts: 30,
    efficiency_factor: 0.72,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html'
  },
  'amd-ryzen-ai-max-395-128gb': {
    id: 'amd-ryzen-ai-max-395-128gb',
    name: 'AMD Ryzen AI Max+ 395 (Strix Halo 128GB)',
    category: 'unified_apu_npu',
    vram_gb: 96.0,
    usable_vram_gb: 92.0,
    bandwidth_gb_s: 256.0,
    tflops_fp16: 48.0,
    tops_int8: 126.0,
    tdp_watts: 120,
    efficiency_factor: 0.60,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html'
  },
  'intel-core-ultra-9-288v-32gb': {
    id: 'intel-core-ultra-9-288v-32gb',
    name: 'Intel Core Ultra 9 288V (Lunar Lake 32GB)',
    category: 'integrated_npu_arc',
    vram_gb: 24.0,
    usable_vram_gb: 22.0,
    bandwidth_gb_s: 136.0,
    tflops_fp16: 32.0,
    tops_int8: 120.0,
    tdp_watts: 30,
    efficiency_factor: 0.60,
    reference_url: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
    benchmark_guide: 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html'
  },
  'qualcomm-snapdragon-x-elite-64gb': {
    id: 'qualcomm-snapdragon-x-elite-64gb',
    name: 'Qualcomm Snapdragon X Elite (64GB)',
    category: 'unified_soc_npu',
    vram_gb: 48.0,
    usable_vram_gb: 44.0,
    bandwidth_gb_s: 135.0,
    tflops_fp16: 15.0,
    tops_int8: 45.0,
    tdp_watts: 45,
    efficiency_factor: 0.60,
    reference_url: 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html',
    benchmark_guide: 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html'
  }
};

const CANONICAL_URLS = {
  homepage: 'https://www.iaexplore.com/',
  ai_hardware_guide_en: 'https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html',
  ai_hardware_guide_es: 'https://www.iaexplore.com/html/es/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html',
  ai_hardware_guide_pt: 'https://www.iaexplore.com/html/pt/blog/ia/npu-tpu-gpu-guia-hardware-inteligencia-artificial.html',
  pc_benchmarks_2026_es: 'https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html',
  pc_benchmarks_2026_en: 'https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html',
  pc_benchmarks_2026_pt: 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-computadores-que-pc-comprar-segundo-uso-orcamento.html',
  arm_tablets_2026_pt: 'https://www.iaexplore.com/html/pt/blog/comparativos/melhores-tablets-qualidade-preco.html',
  landauer_limit_es: 'https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html',
  neural_network_sim: 'https://www.iaexplore.com/neural-network-sim/'
};

const OVERHEAD_VRAM_GB = 0.55;
const SYSTEM_RAM_BW_DEFAULT = 70.0;
const SYSTEM_RAM_EFFICIENCY = 0.60;

function getModel(modelId) {
  const key = String(modelId).toLowerCase().trim();
  if (!MODELS_DB[key]) {
    throw new Error(`Unknown model '${modelId}'. Available: ${Object.keys(MODELS_DB).join(', ')}`);
  }
  return MODELS_DB[key];
}

function getHardware(hardwareId) {
  const key = String(hardwareId).toLowerCase().trim();
  if (!HARDWARE_DB[key]) {
    throw new Error(`Unknown hardware '${hardwareId}'. Available: ${Object.keys(HARDWARE_DB).join(', ')}`);
  }
  return HARDWARE_DB[key];
}

function getQuantBits(quantStr) {
  const key = String(quantStr).toUpperCase().trim();
  if (QUANTIZATION_BITS[key] !== undefined) {
    return QUANTIZATION_BITS[key];
  }
  const parsed = parseFloat(quantStr);
  if (!isNaN(parsed) && parsed > 0) {
    return parsed;
  }
  throw new Error(`Unknown quantization '${quantStr}'. Supported: ${Object.keys(QUANTIZATION_BITS).join(', ')} or float.`);
}

function calculateVramWeights(totalParamsB, bitsQuant) {
  const bytesPerWeight = bitsQuant / 8.0;
  const rawBytes = totalParamsB * 1e9 * bytesPerWeight;
  const gib = rawBytes / Math.pow(1024, 3);
  return gib * 1.08;
}

function calculateVramKv(layers, kvHeads, headDim, contextLength, batchSize = 1, kvBits = 16.0) {
  const bytesPerElement = kvBits / 8.0;
  const totalElements = 2 * layers * kvHeads * headDim * contextLength * batchSize;
  const rawBytes = totalElements * bytesPerElement;
  return rawBytes / Math.pow(1024, 3);
}

function calculateVramOverhead() {
  return OVERHEAD_VRAM_GB;
}

function calculateVramActive(activeParamsB, bitsQuant) {
  const bytesPerWeight = bitsQuant / 8.0;
  const rawBytes = activeParamsB * 1e9 * bytesPerWeight;
  return rawBytes / Math.pow(1024, 3);
}

function calculateTps(bandwidthGbS, vramActiveGb, vramKvGb, efficiencyFactor = 0.72) {
  const effectiveBw = bandwidthGbS * efficiencyFactor;
  const memPerToken = vramActiveGb + (vramKvGb * 0.15);
  if (memPerToken <= 0) return 0.0;
  return effectiveBw / memPerToken;
}

function calculateLayerOffload({
  totalLayers,
  vramWeightsGb,
  vramKvGb,
  vramUsableGb,
  vramActiveGb,
  hardwareBwGbS,
  hardwareEfficiency = 0.72,
  systemRamBwGbS = SYSTEM_RAM_BW_DEFAULT
}) {
  const vramOverhead = calculateVramOverhead();
  const vramTotalRequired = vramWeightsGb + vramKvGb + vramOverhead;
  const fullTps = calculateTps(hardwareBwGbS, vramActiveGb, vramKvGb, hardwareEfficiency);

  if (vramTotalRequired <= vramUsableGb) {
    return {
      fitsEntirelyInVram: true,
      recommendedGpuLayers: totalLayers,
      totalLayers,
      offloadFraction: 1.0,
      gpuLayersVramGb: Number(vramTotalRequired.toFixed(2)),
      degradedTpsEstimated: Number(fullTps.toFixed(2)),
      fullGpuTpsReference: Number(fullTps.toFixed(2)),
      systemRamBandwidthGbS: systemRamBwGbS
    };
  }

  const availForLayers = Math.max(0.0, vramUsableGb - vramOverhead);
  const costPerLayer = (vramWeightsGb + vramKvGb) / totalLayers;
  let offloadedLayers = 0;
  if (costPerLayer > 0) {
    offloadedLayers = Math.min(totalLayers, Math.max(0, Math.floor(availForLayers / costPerLayer)));
  }

  const alpha = totalLayers > 0 ? offloadedLayers / totalLayers : 0.0;
  const consumedGpuVram = (offloadedLayers * costPerLayer) + vramOverhead;

  const timeGpu = (alpha * vramActiveGb) / (hardwareBwGbS * hardwareEfficiency);
  const timeSys = ((1.0 - alpha) * vramActiveGb) / (systemRamBwGbS * SYSTEM_RAM_EFFICIENCY);
  const timeKv = (vramKvGb * 0.15) / (hardwareBwGbS * hardwareEfficiency);

  const totalTokenTime = timeGpu + timeSys + timeKv;
  const degradedTps = totalTokenTime > 0 ? 1.0 / totalTokenTime : 0.0;

  return {
    fitsEntirelyInVram: false,
    recommendedGpuLayers: offloadedLayers,
    totalLayers,
    offloadFraction: Number(alpha.toFixed(4)),
    gpuLayersVramGb: Number(consumedGpuVram.toFixed(2)),
    degradedTpsEstimated: Number(degradedTps.toFixed(2)),
    fullGpuTpsReference: Number(fullTps.toFixed(2)),
    systemRamBandwidthGbS: systemRamBwGbS
  };
}

function profileInference(options = {}) {
  const model = typeof options.model === 'string' ? getModel(options.model) : (options.model || getModel('llama-3.3-70b'));
  const hardware = typeof options.hardware === 'string' ? getHardware(options.hardware) : (options.hardware || getHardware('rtx-4090-24gb'));
  const quant = options.quant || 'Q4_K_M';
  const quantBits = getQuantBits(quant);
  const context = options.context ? parseInt(options.context, 10) : parseInt(model.default_context || 32768, 10);
  const batchSize = options.batchSize ? parseInt(options.batchSize, 10) : 1;
  const kvBits = options.kvBits ? parseFloat(options.kvBits) : 16.0;
  const systemRamBwGbS = options.systemRamBwGbS ? parseFloat(options.systemRamBwGbS) : SYSTEM_RAM_BW_DEFAULT;

  const totalParamsB = parseFloat(model.total_params_b);
  const activeParamsB = parseFloat(model.active_params_b || totalParamsB);
  const layers = parseInt(model.num_hidden_layers, 10);
  const qHeads = parseInt(model.num_attention_heads, 10);
  const kvHeads = parseInt(model.num_key_value_heads, 10);
  const headDim = parseInt(model.head_dim, 10);
  const gqaFactor = parseFloat(model.gqa_factor || (qHeads / kvHeads));

  const hwVram = parseFloat(hardware.vram_gb);
  const usableVram = parseFloat(hardware.usable_vram_gb || (hwVram * 0.95));
  const bandwidth = parseFloat(hardware.bandwidth_gb_s);
  const efficiency = parseFloat(hardware.efficiency_factor || 0.72);

  const vramWeights = Number(calculateVramWeights(totalParamsB, quantBits).toFixed(2));
  const vramKv = Number(calculateVramKv(layers, kvHeads, headDim, context, batchSize, kvBits).toFixed(2));
  const vramOverhead = calculateVramOverhead();
  const vramTotal = Number((vramWeights + vramKv + vramOverhead).toFixed(2));
  const vramActive = Number(calculateVramActive(activeParamsB, quantBits).toFixed(2));

  const fitsInVram = vramTotal <= usableVram;

  const offloadRec = calculateLayerOffload({
    totalLayers: layers,
    vramWeightsGb: vramWeights,
    vramKvGb: vramKv,
    vramUsableGb: usableVram,
    vramActiveGb: vramActive,
    hardwareBwGbS: bandwidth,
    hardwareEfficiency: efficiency,
    systemRamBwGbS
  });

  const estimatedTps = fitsInVram ? offloadRec.fullGpuTpsReference : offloadRec.degradedTpsEstimated;

  return {
    modelId: model.id || 'custom',
    modelName: model.name || 'Custom Model',
    architecture: model.architecture || 'dense',
    totalParamsB,
    activeParamsB,
    layers,
    qHeads,
    kvHeads,
    headDim,
    gqaFactor,
    quantization: quant,
    quantBits,
    contextTokens: context,
    batchSize,
    kvBits,
    hardwareId: hardware.id || 'custom',
    hardwareName: hardware.name || 'Custom Hardware',
    hardwareCategory: hardware.category || 'gpu',
    hardwareVramGb: hwVram,
    usableVramGb: usableVram,
    bandwidthGbS: bandwidth,
    efficiencyFactor: efficiency,
    vramWeightsGb: vramWeights,
    vramKvGb: vramKv,
    vramOverheadGb: vramOverhead,
    vramTotalRequiredGb: vramTotal,
    vramActiveGb: vramActive,
    fitsInVram,
    estimatedTps,
    offload: offloadRec,
    guideUrl: hardware.reference_url || CANONICAL_URLS.ai_hardware_guide_en,
    benchmarkGuideUrl: hardware.benchmark_guide || CANONICAL_URLS.pc_benchmarks_2026_en,
    simulatorUrl: CANONICAL_URLS.neural_network_sim
  };
}

module.exports = {
  QUANTIZATION_BITS,
  MODELS_DB,
  HARDWARE_DB,
  CANONICAL_URLS,
  OVERHEAD_VRAM_GB,
  SYSTEM_RAM_BW_DEFAULT,
  getModel,
  getHardware,
  getQuantBits,
  calculateVramWeights,
  calculateVramKv,
  calculateVramOverhead,
  calculateVramActive,
  calculateTps,
  calculateLayerOffload,
  profileInference
};
