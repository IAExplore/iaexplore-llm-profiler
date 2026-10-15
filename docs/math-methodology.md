# Mathematical Methodology: Deterministic LLM Inference Profiling

This document outlines the rigorous mathematical equations implemented across all runtime packages of **`iaexplore-llm-profiler`**.

Unlike naive heuristics that approximate memory by multiplying parameter count by bytes, our engine implements exact tensor topology equations verified across modern inference engines (`llama.cpp`, `vLLM`, `Ollama`, and `TGI`).

---

## 1. Model Weights Memory ($\text{VRAM}_{\text{weights}}$)

During inference, transformer model weights occupy static VRAM. While core attention and feed-forward projection matrices are quantized, standard quantized formats (e.g., GGUF, EXL2, AWQ) leave token embeddings (`token_embd`), output heads (`output.weight`), normalization layers (`RMSNorm`), and container metadata unquantized or quantized at higher precisions (FP16/FP32).

$$\text{VRAM}_{\text{weights}} = \frac{P_{\text{total}} \times 10^9 \times (\text{bits}_{\text{quant}} / 8)}{1024^3} \times 1.08$$

Where:
- $P_{\text{total}}$: Total parameter count of the model in billions (e.g., $70.6$ for Llama 3.3 70B, $32.8$ for DeepSeek R1 32B).
- $\text{bits}_{\text{quant}}$: Effective quantization bit depth per weight.
- $1024^3$: Conversion denominator from bytes to gigabytes (GiB).
- $1.08$: Verified empirical coefficient representing tensor overhead, unquantized normalization weights, output projection heads, and tensor page alignment.

### Quantization Bit-Depth Matrix
| Format | Effective Bits / Weight ($\text{bits}_{\text{quant}}$) | Description |
|---|---|---|
| **FP16 / BF16** | 16.00 | Unquantized floating point baseline |
| **Q8_0** | 8.50 | 8-bit quantization with block scaling factors |
| **Q6_K** | 6.56 | 6-bit k-quant with super-block scaling |
| **Q5_K_M** | 5.50 | 5-bit medium mixed k-quant |
| **Q5_K_S** | 5.25 | 5-bit small k-quant |
| **Q4_K_M** | 4.50 | 4-bit medium mixed k-quant (industry standard for local inference) |
| **Q4_K_S** | 4.25 | 4-bit small k-quant |
| **Q4_0** | 4.50 | Traditional 4-bit linear quantization |
| **Q3_K_M** | 3.50 | 3-bit medium k-quant |
| **Q2_K** | 2.65 | Extreme 2-bit quantization |

---

## 2. Grouped-Query Attention (GQA) KV-Cache Memory ($\text{VRAM}_{\text{kv}}$)

In autoregressive generation, past tokens require storing Key ($K$) and Value ($V$) activation states for each layer:

$$\text{VRAM}_{\text{kv}} = \frac{2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times C_{\text{ctx}} \times B_{\text{batch}} \times (\text{bits}_{\text{kv}} / 8)}{1024^3}$$

### Geometric Breakdown:
- **$2$**: Multiplier for the distinct Key and Value state matrices.
- **$L$**: Number of transformer hidden layers (`num_hidden_layers`), e.g., 80 in Llama 3.3 70B.
- **$H_{\text{kv}}$**: Number of Key/Value attention heads (`num_key_value_heads`). In classic Multi-Head Attention (MHA), $H_{\text{kv}} = H_{\text{q}}$. In Grouped-Query Attention (GQA), multiple Query heads share a single Key/Value head:
  $$\text{GQA Compression Factor} = \frac{H_{\text{q}}}{H_{\text{kv}}}$$
  For example, Llama 3.3 70B features $H_{\text{q}} = 64$ and $H_{\text{kv}} = 8$, yielding an **$8\times$ reduction in KV-cache size** compared to legacy MHA architectures!
- **$D_{\text{head}}$**: Attention head dimension ($\text{head\_dim} = \text{hidden\_size} / H_{\text{q}}$), e.g., 128 for Llama and DeepSeek, 256 for Gemma 2.
- **$C_{\text{ctx}}$**: Active context window in tokens (e.g., 4096, 32768, 131072).
- **$B_{\text{batch}}$**: Concurrent batch size (defaults to 1 for interactive local inference).
- **$\text{bits}_{\text{kv}}$**: Bit precision of cached activations (16 for FP16 cache, 8 for quantized Q8_0 cache).

---

## 3. Runtime Activation Buffers & Framework Overhead ($\text{VRAM}_{\text{overhead}}$)

Running GPU compute kernels requires scratchpad workspaces, CUDA / Metal runtime context, and activation memory:

$$\text{VRAM}_{\text{overhead}} = 0.55\text{ GB}$$

$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{kv}} + \text{VRAM}_{\text{overhead}}$$

---

## 4. Physical Token Generation Throughput ($\text{TPS}_{\text{gen}}$) with MoE Support

During autoregressive token generation with $B_{\text{batch}} = 1$, inference is strictly **memory bandwidth bound** (Memory Wall). Compute units spend the majority of cycles stalled waiting for weights to be streamed from VRAM/DRAM into register files.

For Mixture-of-Experts (MoE) architectures (e.g., Mixtral 8x7B, DeepSeek R1), only active routing experts ($P_{\text{active}}$) are fetched from memory per token, while non-activated expert weights remain idle in VRAM.

$$\text{VRAM}_{\text{active}} = \frac{P_{\text{active}} \times 10^9 \times (\text{bits}_{\text{quant}} / 8)}{1024^3}$$

The physical token generation rate is given by:

$$\text{TPS}_{\text{gen}} = \frac{\text{BW}_{\text{mem}} \times \eta_{\text{efficiency}}}{\text{VRAM}_{\text{active}} + (\text{VRAM}_{\text{kv}} \times 0.15)}$$

Where:
- $\text{BW}_{\text{mem}}$: Physical bus bandwidth in gigabytes per second (GB/s).
- $\eta_{\text{efficiency}}$: Bus saturation efficiency coefficient:
  - $\eta \approx 0.72$ for discrete GPU (CUDA/Metal) and Apple Silicon Unified Memory.
  - $\eta \approx 0.60$ for shared memory architectures (integrated APUs, NPUs, DDR5 RAM).
- $0.15$: KV-cache attention scanning bandwidth factor per generated token.

---

## 5. Layer Offloading & Split Degradation (`--n-gpu-layers`)

When $\text{VRAM}_{\text{total}} > \text{VRAM}_{\text{usable}}$, inference engines must partition transformer blocks between GPU VRAM and System RAM.

Let available GPU space for layers be:
$$\text{VRAM}_{\text{avail\_layers}} = \max(0.0, \text{VRAM}_{\text{usable}} - \text{VRAM}_{\text{overhead}})$$

The memory footprint per transformer layer is:
$$\text{VRAM}_{\text{layer}} = \frac{\text{VRAM}_{\text{weights}}}{L} + \frac{\text{VRAM}_{\text{kv}}}{L}$$

The exact count of layers offloaded to GPU is:
$$N_{\text{gpu\_layers}} = \min\left(L, \max\left(0, \left\lfloor \frac{\text{VRAM}_{\text{avail\_layers}}}{\text{VRAM}_{\text{layer}}} \right\rfloor \right)\right)$$

The offload fraction is $\alpha = N_{\text{gpu\_layers}} / L$.

### Degraded Throughput Modeling
When layers are split across high-speed GPU VRAM ($\text{BW}_{\text{gpu}}$) and system memory ($\text{BW}_{\text{sys}} \approx 70\text{ GB/s}$ for dual-channel DDR5):

$$\tau_{\text{gpu}} = \frac{\alpha \times \text{VRAM}_{\text{active}}}{\text{BW}_{\text{gpu}} \times \eta_{\text{gpu}}}$$

$$\tau_{\text{sys}} = \frac{(1 - \alpha) \times \text{VRAM}_{\text{active}}}{\text{BW}_{\text{sys}} \times \eta_{\text{sys}}}$$

$$\text{TPS}_{\text{degraded}} = \frac{1}{\tau_{\text{gpu}} + \tau_{\text{sys}} + (\text{VRAM}_{\text{kv}} \times 0.15 / \text{BW}_{\text{gpu}})}$$

---

## 🔬 Further Reading on Physical Silicon Limits

To understand why memory bandwidth represents the fundamental physical bottleneck in deep learning hardware, consult our comprehensive analyses:

- [NPU, TPU and GPU Complete Hardware Guide](https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html)
- [Computación Reversible y Límite de Landauer en Procesadores](https://www.iaexplore.com/html/es/blog/tecnologia/computacion-reversible-principio-landauer-calor-procesadores.html)
- [Comparativa de Ordenadores y Estaciones de Trabajo de 2026](https://www.iaexplore.com/html/es/blog/comparativas/los-mejores-ordenadores-que-pc-comprar-segun-tu-uso-y-presupuesto.html)
- [Interactive Neural Network Simulator](https://www.iaexplore.com/neural-network-sim/)
