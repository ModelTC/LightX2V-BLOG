---
layout: post
title: "Wan2.2-NVFP4-Sparse: Extremely Fast Wan 2.2 14B Inference"
author: "Chengtao Lv, LightX2V Team"
date: 2026-06-09
tags: [Wan2.2, NVFP4, Sparse Attention, Video Generation]
---

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Wan2.2--NVFP4--Sparse-yellow)](https://huggingface.co/lightx2v/Wan2.2-NVFP4-Sparse)

Video generation has received broad attention in recent years, driven by the impressive visual quality and motion consistency of models such as Wan, Sora, Seedance, and other large-scale diffusion Transformers. These models have made it possible to generate high-resolution, temporally coherent videos from text or image prompts, opening up new workflows for creative production, simulation, advertising, and interactive content.

However, the same capabilities also make modern video generation extremely resource intensive. For 14B video DiT models, inference can easily become impractical on consumer GPUs because both latency and memory usage scale aggressively with model size, video resolution, frame count, and sequence length. In practice, running a large model such as Wan2.2-A14B on a single consumer GPU is challenging without a carefully optimized inference stack.

The bottleneck comes from several sources:

1. **Multiple denoising steps.**  
   Diffusion-based video generation repeatedly evaluates the DiT backbone during denoising. For example, Wan2.2 commonly uses a 40-step schedule, which means the full Transformer must be executed again and again for a single video.

2. **Expensive per-step computation.**  
   Inside each denoising step, self-attention becomes one of the dominant costs because its complexity grows quadratically with sequence length. At the same time, the linear layers in the Transformer blocks require large matrix multiplications, which also contribute heavily to total latency.

3. **Non-trivial operator overhead.**  
   Some operations around the main attention and linear layers, such as 3D RoPE and RMSNorm, can still introduce noticeable overhead when implemented with naive Python or unfused tensor operations. These costs become more visible when the main model path is optimized.

4. **High peak memory usage.**  
   Large model weights, long video token sequences, attention buffers, and intermediate activations together push peak VRAM usage very high. This is often the immediate blocker for running 14B-class models on GPUs with around 30 GB of memory.

To address these challenges, the LightX2V team performs joint optimization across both the model and inference framework. The goal is not only to make Wan2.2-A14B faster, but also to make it runnable on a single RTX 5090-class GPU with roughly 30 GB of VRAM. This work combines step distillation, low-precision kernels, sparse attention, efficient operators, and multi-granularity offload into one practical inference path.

The main techniques include:

- **Step Distillation**: Two high-noise expert steps followed by two low-noise expert steps, enabling extremely fast Wan2.2 MoE generation on a single Blackwell GPU.
- **NVFP4 Quantization**: Quantization-aware step distillation reduces memory traffic and compute cost while targeting Blackwell architecture.
- **Sparse Attention**: Accelerates the costly O(n^2) self-attention workload with sparse attention, reducing end-to-end latency for high-resolution video generation.
- **Efficient Operators**: Replaces naive implementations of operations such as 3D RoPE and RMSNorm with efficient parallel kernels, reducing framework overhead around the main Transformer computation.
- **Offload Optimization**: In addition to model-level offload, LightX2V provides block-level offload, allowing finer-grained weight movement and lower peak GPU memory usage during inference.

## Step Distillation

The first and most direct way to reduce video generation latency is to reduce the number of denoising steps. Few-step distillation provides a practical path for this: instead of running the original teacher model for dozens of denoising steps, the student model learns to approximate the same generation trajectory with only a small number of steps.

The challenge is how to make few-step distillation scalable without sacrificing video quality, diversity, and motion dynamics. The comparison below summarizes the main design choices.

![Phased DMD overview]({{ site.baseurl }}/assets/wan22-nvfp4-sparse/phaseddmd.png)

*Phased DMD compared with direct multi-step distillation and stochastic gradient truncation.*

In **(a)**, direct multi-step distillation keeps gradients through every generator step. This gives supervision to the full denoising trajectory, but it also creates a deep computational graph. For large video models, this leads to high memory overhead and makes training difficult to scale to 14B-class or larger DiT backbones.

In **(b)**, stochastic gradient truncation reduces this cost by keeping gradients only on the final denoising step while detaching the other steps. This improves memory efficiency and training stability, and it still allows intermediate states to be sampled during training. However, when the backward simulation terminates after only one step, that iteration effectively becomes one-step distillation. For video generation, this can reduce output diversity and weaken motion dynamics, making the few-step generator behave too much like a one-step model.

In **(c)**, Phased DMD avoids this one-step degeneration by splitting the signal-to-noise ratio (SNR) range into multiple phases. Each phase distills one expert for a specific SNR subinterval, and the backward simulation terminates at that phase's target SNR level rather than always going to the clean sample. This lets the model learn complex denoising behavior progressively, while each phase still records gradients for only one sampling step.

In **(d)**, Phased DMD is combined with stochastic gradient truncation. This makes it possible to train a 4-step generator with only two phases, reducing framework complexity while preserving the key benefit of phased training. For Wan2.2, this naturally leads to a compact MoE-style few-step generator: two high-noise expert steps followed by two low-noise expert steps.

This design is especially suitable for Wan2.2-style video generation. Diffusion models perform different types of work across the denoising trajectory: low-SNR stages focus more on global visual structure and motion dynamics, while high-SNR stages refine visual details. By assigning different experts to different SNR ranges, Phased DMD increases model capacity where it matters without adding inference-time cost.

In this work, the distilled Wan2.2 model uses a 4-step inference schedule: two high-noise expert steps followed by two low-noise expert steps. Compared with the original 40-step denoising process, this directly cuts the number of DiT forward passes by an order of magnitude. At the same time, the phased training objective helps preserve motion dynamics, visual fidelity, and output diversity, which are often the first qualities to degrade in aggressive few-step video distillation.

## Low-Precision QAT

Step distillation reduces the number of DiT forward passes. The next goal is to reduce the latency of each remaining denoising step. We first apply model quantization to the large matrix multiplications in the Transformer blocks, and later further reduce attention cost with sparse attention. Quantization converts high-precision tensors, such as FP16 or BF16 weights and activations, into lower-precision representations. Lower precision reduces memory footprint, memory bandwidth, and Tensor Core compute cost, which is critical for running 14B-class video models on consumer Blackwell GPUs.

In a typical quantization flow, a high-precision value $x$ is mapped to a low-precision code and then dequantized back to an approximate value for computation:

$$
Q_U(x)=\mathrm{round}\left(\frac{x}{\Delta}\right)\Delta,
\quad
\Delta=\frac{u-l}{2^b-1}.
$$

Here, $b$ is the bit width, $x$ is a floating-point activation or weight in the clipping range $(l,u)$, and $\Delta$ is the interval length after dividing the original range into $2^b-1$ intervals. This quantization-dequantization process maps continuous high-precision values onto a finite low-precision grid. The smaller the data type, the more important the scaling strategy becomes: 4-bit formats have very limited representable values, so a single coarse scale can introduce large quantization error.

Blackwell GPUs introduce hardware support for FP4 computation, and NVFP4 is NVIDIA's block-scaled 4-bit floating-point format designed for this generation. Each NVFP4 tensor element uses an E2M1 4-bit value, with 1 sign bit, 2 exponent bits, and 1 mantissa bit. To preserve dynamic range, NVFP4 uses hierarchical scaling:

$$
\hat{x}=x_{\mathrm{e2m1}}\cdot s_{\mathrm{block}}\cdot s_{\mathrm{global}}.
$$

where $x_{\mathrm{e2m1}}$ is the 4-bit floating-point value, $s_{\mathrm{block}}$ is an FP8 E4M3 scale shared by a block of 16 consecutive elements, and $s_{\mathrm{global}}$ is an FP32 scale applied to the whole tensor. The scaling factors can be computed from the tensor and block maximum absolute values:

$$
s_{\mathrm{global}}=
\frac{\mathrm{global\_amax}}{\mathrm{fp8}_{\max}\cdot \mathrm{fp4}_{\max}},
\quad
s_{\mathrm{block}}=
\frac{\mathrm{block\_amax}/\mathrm{fp4}_{\max}}{s_{\mathrm{global}}}.
$$

With NVFP4 E2M1, $\mathrm{fp4}_{\max}$ is 6.0, and the FP8 E4M3 block scale provides finer-grained, non-power-of-two scaling. This makes NVFP4 more expressive than a plain 4-bit format with only one tensor-level scale, while still keeping the storage and bandwidth close to FP4. In practice, the expensive linear layers can be executed with NVFP4 weights and activations, reducing both memory traffic and GEMM latency on Blackwell Tensor Cores.

For aggressive 4-bit quantization, post-training quantization alone is often not enough for video generation quality. We therefore use quantization-aware training during distillation. During the forward pass, fake-quantized tensors are used so that the student model learns under the same numerical constraints it will see at inference time:

$$
x_{\mathrm{qat}}=Q_U(x),
\quad
y=f(x_{\mathrm{qat}}).
$$

The difficulty is that quantization contains rounding and clipping, which are not directly differentiable. To train through this path, we use the straight-through estimator (STE). In the forward pass, the model still sees the quantized value. In the backward pass, the gradient is approximated as if the quantization operation were the identity function within the valid range:

$$
\text{Forward:}\quad x_{\mathrm{qat}}=Q_U(x),
$$

$$
\frac{\partial Q_U(x)}{\partial x}\approx
\begin{cases}
1, & x_{\min}\le x\le x_{\max},\\
0, & \text{otherwise},
\end{cases}
\quad
\frac{\partial \mathcal{L}}{\partial x}
\approx
\frac{\partial \mathcal{L}}{\partial x_{\mathrm{qat}}}
\frac{\partial Q_U(x)}{\partial x}.
$$

Equivalently, STE lets gradients update the high-precision master weights while the forward computation remains aware of NVFP4 quantization error. This is especially important for step-distilled video models, where the student already has fewer denoising opportunities to correct numerical errors.

At implementation level, the NVFP4 path relies on CUTLASS kernels for the core matrix multiplications. CUTLASS provides optimized support for block-scaled data types, including NVIDIA NVFP4, and maps these low-precision GEMMs to Blackwell Tensor Core execution. This allows LightX2V to combine quantization-aware distillation at the model level with efficient production kernels at the inference framework level.

## Dynamic Sparse Attention

After step distillation and low-precision QAT, attention becomes the next major target for reducing per-step latency. In video DiT models, the sequence length grows quickly with resolution and frame count. Since full self-attention has quadratic complexity with respect to sequence length, the attention module can dominate end-to-end inference time. For example, with a sequence length of around 120K, self-attention in Wan2.2-A14B can take more than 80% of the total DiT latency.

The key observation is that full token-to-token attention is often redundant. Many query regions only need to attend to a subset of key/value regions, especially in high-resolution video latents where local spatial-temporal structure is strong. LightX2V therefore uses dynamic blockwise sparse attention: it predicts important attention blocks at runtime, then computes attention only on the selected blocks.

This design has two stages.

**First, estimate block importance with block-level mean vectors.**  
Given query, key, and value tensors $Q$, $K$, $V \in \mathbb{R}^{n \times d}$, we partition the sequence dimension into query blocks and key/value blocks:

$$
Q = [Q_1;\ldots;Q_{n_q}],\quad
K = [K_1;\ldots;K_{n_k}],\quad
V = [V_1;\ldots;V_{n_k}],
$$

where $Q_i \in \mathbb{R}^{b_q \times d}$ and $K_j,V_j \in \mathbb{R}^{b_{kv} \times d}$. The number of blocks is $n_q = \lceil n / b_q \rceil$ and $n_k = \lceil n / b_{kv} \rceil$.

For each block, we compute a mean vector along the token dimension:

$$
\bar{Q}_i = \mathrm{mean}(Q_i),\quad
\bar{K}_j = \mathrm{mean}(K_j).
$$

The compressed attention score matrix is then computed as:

$$
P_c = \mathrm{Softmax}\left(\frac{\bar{Q}\bar{K}^{\top}}{\sqrt{d_k}}\right),
\quad P_c \in \mathbb{R}^{n_q \times n_k}.
$$

Each element in $P_c$ estimates the importance of a query block attending to a key/value block. Based on these scores, LightX2V builds a dynamic block mask $B \in \{0,1\}^{n_q \times n_k}$, where $B_{ij}=1$ means the $(i,j)$ attention tile is selected and $B_{ij}=0$ means it can be skipped. In practice, we use an 80%-90% sparsity ratio, which means keeping the top 10%-20% most important blocks for sparse attention.

**Second, compute sparse attention only on selected blocks.**  
With the block mask $B$, block-sparse attention can be written as:

$$
\mathrm{SparseAttn}(Q,K,V;B)
= \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} \odot M(B)\right)V,
$$

where $M(B)$ expands the block mask into an element-wise mask that is constant inside each $b_q \times b_{kv}$ tile, and $\odot$ denotes element-wise multiplication. In the actual kernel implementation, LightX2V does not materialize the full $n \times n$ mask. Instead, it computes only the tile products $Q_iK_j^\top$ and the corresponding value aggregation for active block pairs with $B_{ij}=1$, skipping entire tiles when $B_{ij}=0$.

As a result, the computational and memory costs scale with the number of active blocks rather than the full $n^2$ attention matrix. At the same time, computation inside each selected tile remains dense and GPU-friendly, which makes the method compatible with high-throughput attention kernels and modern accelerator memory access patterns.

## 🚀 Quick Start

We strongly recommend using the official LightX2V Docker image for the cleanest environment and best reproducibility.

### Option A: Docker Recommended

```bash
# 1. Pull LightX2V Docker image
docker pull lightx2v/lightx2v:26052801-cu130-5090

# 2. Run text-to-video inference
bash scripts/wan22/distill/run_wan22_moe_t2v_extreme.sh

# 3. Run image-to-video inference
bash scripts/wan22/distill/run_wan22_moe_i2v_extreme.sh
```

### Option B: Manual Installation

If Docker is not available, install the environment manually:

```bash
# 1. Install LightX2V
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
uv pip install -v .

# 2. Install NVFP4 Kernel
pip install scikit_build_core uv
git clone https://github.com/NVIDIA/cutlass.git
cd lightx2v_kernel

MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
uv build --wheel \
  -Cbuild-dir=build . \
  -Ccmake.define.CUTLASS_PATH=/path/to/cutlass \
  --verbose --color=always --no-build-isolation

pip install dist/*whl --force-reinstall --no-deps

# 3. Run text-to-video inference
bash scripts/wan22/distill/run_wan22_moe_t2v_extreme.sh

# 4. Run image-to-video inference
bash scripts/wan22/distill/run_wan22_moe_i2v_extreme.sh
```

Scripts:
- [run_wan22_moe_t2v_extreme.sh](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/distill/run_wan22_moe_t2v_extreme.sh)
- [run_wan22_moe_i2v_extreme.sh](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/distill/run_wan22_moe_i2v_extreme.sh)

**Test Environment**: RTX 5090 Single GPU | LightX2V Framework | End-to-End Latency

| Resolution | Wan2.2-T2V-14B | Wan2.2-NVFP4-Sparse | Speedup |
| --- | ---: | ---: | ---: |
| 480p | 734s | 14.15s | 51.9x |
| 720p | 2668s | 45s | 59.3x |
