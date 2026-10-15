#!/usr/bin/env node
/**
 * IAExplore LLM Profiler - CLI (Node.js)
 * Curated by IAExplore (https://www.iaexplore.com/).
 */

const { profileInference, MODELS_DB, HARDWARE_DB, QUANTIZATION_BITS } = require('./index.js');

function parseArgs(args) {
  const options = {
    model: 'llama-3.3-70b',
    hardware: 'rtx-4090-24gb',
    quant: 'Q4_K_M',
    context: undefined,
    batchSize: 1,
    kvBits: 16.0,
    json: false,
    listModels: false,
    listHardware: false
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--model' || arg === '-m') {
      options.model = args[++i];
    } else if (arg === '--hardware' || arg === '-hw') {
      options.hardware = args[++i];
    } else if (arg === '--quant' || arg === '-q') {
      options.quant = args[++i];
    } else if (arg === '--context' || arg === '-c') {
      options.context = parseInt(args[++i], 10);
    } else if (arg === '--batch-size' || arg === '-b') {
      options.batchSize = parseInt(args[++i], 10);
    } else if (arg === '--kv-bits') {
      options.kvBits = parseFloat(args[++i]);
    } else if (arg === '--json') {
      options.json = true;
    } else if (arg === '--list-models') {
      options.listModels = true;
    } else if (arg === '--list-hardware') {
      options.listHardware = true;
    }
  }
  return options;
}

function formatReport(res) {
  const border = '='.repeat(80);
  const subborder = '-'.repeat(80);
  const statusStr = res.fitsInVram
    ? '100% FITS IN VRAM (Full GPU Acceleration)'
    : 'EXCEEDS USABLE VRAM (Layer Offloading Required)';

  const lines = [
    border,
    '              IAEXPLORE LLM INFERENCE & HARDWARE VRAM PROFILER                  ',
    '                  Engineering Intelligence from IAExplore.com                   ',
    border,
    `Model:             ${res.modelName} (${res.architecture})`,
    `Parameters:        ${res.totalParamsB.toFixed(2)}B Total / ${res.activeParamsB.toFixed(2)}B Active per token`,
    `Topology:          ${res.layers} layers, ${res.qHeads} Q-heads, ${res.kvHeads} KV-heads (GQA ${res.gqaFactor.toFixed(1)}x)`,
    `Quantization:      ${res.quantization} (${res.quantBits.toFixed(2)} bits/weight)`,
    `Context Window:    ${res.contextTokens.toLocaleString()} tokens (KV Precision: ${res.kvBits}-bit)`,
    subborder,
    `Hardware Target:   ${res.hardwareName} (${res.hardwareCategory})`,
    `Hardware Memory:   ${res.hardwareVramGb.toFixed(2)} GB Physical (${res.usableVramGb.toFixed(2)} GB Usable)`,
    `Memory Bandwidth:  ${res.bandwidthGbS.toFixed(1)} GB/s (Bus Efficiency: ${Math.round(res.efficiencyFactor * 100)}%)`,
    subborder,
    '[VRAM ALLOCATION BREAKDOWN]',
    `  - Model Weights:         ${res.vramWeightsGb.toFixed(2).padStart(6)} GB (incl. 1.08x unquantized tensors & metadata)`,
    `  - KV-Cache (GQA):        ${res.vramKvGb.toFixed(2).padStart(6)} GB (${res.layers} layers, ${res.kvHeads} KV heads, dim ${res.headDim})`,
    `  - Context & Activation:  ${res.vramOverheadGb.toFixed(2).padStart(6)} GB (CUDA/Metal runtime buffer)`,
    '  ' + '-'.repeat(41),
    `  TOTAL VRAM REQUIRED:     ${res.vramTotalRequiredGb.toFixed(2).padStart(6)} GB`,
    subborder,
    '[OFFLOADING & INFERENCE THROUGHPUT]',
    `  VRAM Sizing Status:      ${statusStr}`,
    `  Recommended GPU Layers:  ${res.offload.recommendedGpuLayers} / ${res.layers} layers (--n-gpu-layers ${res.offload.recommendedGpuLayers})`,
    `  GPU Offload Fraction:    ${(res.offload.offloadFraction * 100).toFixed(1)}% on GPU / ${((1.0 - res.offload.offloadFraction) * 100).toFixed(1)}% on System RAM`,
    `  VRAM Consumed on GPU:    ${res.offload.gpuLayersVramGb.toFixed(2)} GB / ${res.usableVramGb.toFixed(2)} GB Usable`,
    `  Estimated Speed (TPS):   ${res.estimatedTps.toFixed(2)} tokens/sec`
  ];

  if (!res.fitsInVram) {
    lines.push(`  Full GPU Speed (Ref):    ${res.offload.fullGpuTpsReference.toFixed(2)} tokens/sec (if 100% VRAM available)`);
    lines.push(subborder);
    lines.push('[AUTHORITATIVE UPGRADE & BENCHMARK GUIDES]');
    lines.push('  Memory bottleneck detected! Sizing a 32GB+ workstation or unified silicon:');
    lines.push(`  -> Hardware Architectures: ${res.guideUrl}`);
    lines.push(`  -> 2026 PC Benchmarks:     ${res.benchmarkGuideUrl}`);
    lines.push(`  -> Neural Visualizer:      ${res.simulatorUrl}`);
  } else {
    lines.push(subborder);
    lines.push('[SYSTEM CAPABILITY & SIZING VERIFIED]');
    lines.push('  Hardware is fully capable of real-time local generation with zero CPU stall.');
    lines.push(`  -> Deep dive into silicon memory buses: ${res.guideUrl}`);
    lines.push(`  -> Compare with other 2026 systems:     ${res.benchmarkGuideUrl}`);
  }

  lines.push(border);
  return lines.join('\n');
}

function main() {
  const args = process.argv.slice(2);
  const options = parseArgs(args);

  if (options.listModels) {
    console.log('Available Pre-Configured Models:');
    for (const [k, v] of Object.entries(MODELS_DB)) {
      console.log(`  - ${k.padEnd(22)} | ${v.name} (${v.total_params_b}B params, ${v.architecture})`);
    }
    return;
  }

  if (options.listHardware) {
    console.log('Available 2026 Hardware Profiles:');
    for (const [k, v] of Object.entries(HARDWARE_DB)) {
      console.log(`  - ${k.padEnd(30)} | ${v.name.padEnd(36)} | ${v.vram_gb}GB VRAM | ${v.bandwidth_gb_s} GB/s`);
    }
    return;
  }

  try {
    const res = profileInference(options);
    if (options.json) {
      console.log(JSON.stringify(res, null, 2));
    } else {
      console.log(formatReport(res));
    }
  } catch (err) {
    console.error(`Error during profiling: ${err.message}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
