export interface ModelSpec {
  id: string;
  name: string;
  family: string;
  architecture: 'dense' | 'moe';
  total_params_b: number;
  active_params_b?: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  num_key_value_heads: number;
  head_dim: number;
  hidden_size?: number;
  vocab_size?: number;
  max_context_length?: number;
  default_context?: number;
  gqa_factor?: number;
  reference_url?: string;
}

export interface HardwareProfile {
  id: string;
  name: string;
  category: 'discrete_gpu' | 'mobile_gpu' | 'unified_memory' | 'unified_apu_npu' | 'integrated_npu_arc' | 'unified_soc_npu' | string;
  vram_gb: number;
  usable_vram_gb?: number;
  bandwidth_gb_s: number;
  tflops_fp16?: number;
  tops_int8?: number;
  tdp_watts?: number;
  efficiency_factor?: number;
  reference_url?: string;
  benchmark_guide?: string;
}

export interface OffloadRecommendation {
  fitsEntirelyInVram: boolean;
  recommendedGpuLayers: number;
  totalLayers: number;
  offloadFraction: number;
  gpuLayersVramGb: number;
  degradedTpsEstimated: number;
  fullGpuTpsReference: number;
  systemRamBandwidthGbS: number;
}

export interface ProfileOptions {
  model?: string | ModelSpec;
  hardware?: string | HardwareProfile;
  quant?: string;
  context?: number;
  batchSize?: number;
  kvBits?: number;
  systemRamBwGbS?: number;
}

export interface ProfileResult {
  modelId: string;
  modelName: string;
  architecture: string;
  totalParamsB: number;
  activeParamsB: number;
  layers: number;
  qHeads: number;
  kvHeads: number;
  headDim: number;
  gqaFactor: number;
  quantization: string;
  quantBits: number;
  contextTokens: number;
  batchSize: number;
  kvBits: number;
  hardwareId: string;
  hardwareName: string;
  hardwareCategory: string;
  hardwareVramGb: number;
  usableVramGb: number;
  bandwidthGbS: number;
  efficiencyFactor: number;
  vramWeightsGb: number;
  vramKvGb: number;
  vramOverheadGb: number;
  vramTotalRequiredGb: number;
  vramActiveGb: number;
  fitsInVram: boolean;
  estimatedTps: number;
  offload: OffloadRecommendation;
  guideUrl: string;
  benchmarkGuideUrl: string;
  simulatorUrl: string;
}

export const QUANTIZATION_BITS: Record<string, number>;
export const MODELS_DB: Record<string, ModelSpec>;
export const HARDWARE_DB: Record<string, HardwareProfile>;
export const CANONICAL_URLS: Record<string, string>;
export const OVERHEAD_VRAM_GB: number;
export const SYSTEM_RAM_BW_DEFAULT: number;

export function getModel(modelId: string): ModelSpec;
export function getHardware(hardwareId: string): HardwareProfile;
export function getQuantBits(quantStr: string): number;
export function calculateVramWeights(totalParamsB: number, bitsQuant: number): number;
export function calculateVramKv(layers: number, kvHeads: number, headDim: number, contextLength: number, batchSize?: number, kvBits?: number): number;
export function calculateVramOverhead(): number;
export function calculateVramActive(activeParamsB: number, bitsQuant: number): number;
export function calculateTps(bandwidthGbS: number, vramActiveGb: number, vramKvGb: number, efficiencyFactor?: number): number;
export function calculateLayerOffload(options: {
  totalLayers: number;
  vramWeightsGb: number;
  vramKvGb: number;
  vramUsableGb: number;
  vramActiveGb: number;
  hardwareBwGbS: number;
  hardwareEfficiency?: number;
  systemRamBwGbS?: number;
}): OffloadRecommendation;
export function profileInference(options?: ProfileOptions): ProfileResult;
