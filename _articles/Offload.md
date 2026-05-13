---
layout: post
title: "Running Large Video Models on Consumer GPUs: LightX2V Offload Explained"
author: "LightX2V Team"
date: 2026-05-13
tags: [Offload, Video Generation, Consumer GPU, Inference Optimization]
---

Video generation models are growing rapidly. A 14B, 28B, or even larger DiT / Transformer backbone can easily exceed the memory capacity of a single RTX 4090 or RTX 5090 in BF16. Once the text encoder, image encoder, VAE, attention buffers, and intermediate activations are added, the full pipeline becomes even more memory hungry.

LightX2V Offload addresses a practical problem: **during inference, the GPU only needs the weights for the part of the model currently being computed. The remaining weights can stay in CPU memory, or even on NVMe storage, and be moved to the GPU right before they are needed.**

This post explains LightX2V's multi-level Offload design, including:

- three offload granularities: `model`, `block`, and `phase`;
- CPU ↔ GPU weight transfer, plus Disk / NVMe as an additional source when `lazy_load=true`;
- asynchronous prefetching and double buffering;
- how LightX2V turns offload from a single-model optimization into a framework-level capability;
- practical recommendations for consumer GPUs such as RTX 3060, RTX 4090, and RTX 5090.

**Table of contents:**

- [Why Video Generation Needs Offload](#why-video-generation-needs-offload)
- [LightX2V Offload Architecture](#lightx2v-offload-architecture)
- [Three Granularities: Model, Block, and Phase](#three-granularities-model-block-and-phase)
- [CPU-GPU Offload and Lazy Load](#cpu-gpu-offload-and-lazy-load)
- [Wan2.2-A14B as an Example](#wan22-a14b-as-an-example)
- [Practical Recommendations for Consumer GPUs](#practical-recommendations-for-consumer-gpus)
- [How to Enable Offload in Config Files](#how-to-enable-offload-in-config-files)
- [Performance Example](#performance-example)
- [Conclusion](#conclusion)

---

## Why Video Generation Needs Offload

Video generation hits the memory wall more easily than image generation, and not only because model parameter counts are increasing. A typical X-to-Video pipeline contains several large components:

| Component | Examples | Memory Pressure |
|---|---|---|
| Text / Image Encoder | T5, Qwen2.5-VL, CLIP, SigLIP | Prompt / image condition preprocessing |
| Transformer / DiT | Wan, HunyuanVideo, LTX, Qwen-Image, SeedVR2 | Dominant model weight and activation footprint |
| VAE Encoder / Decoder | Video VAE | High-resolution latent / pixel conversion |

LightX2V provides offload capabilities for different modules. To keep the discussion focused, this post mainly explains offload for the DiT / Transformer backbone, which is usually the largest part of the pipeline. Offload for other modules, such as text encoders, image encoders, and VAEs, is supported but not covered in detail here.

In data centers, insufficient GPU memory can often be handled with larger GPUs or multi-GPU deployment. In local creation and development environments, however, consumer GPUs with 12 GB, 16 GB, 24 GB, or 32 GB of VRAM are much more common. In these setups, system memory and NVMe storage are often more abundant than GPU memory, making offload a practical engineering solution.

LightX2V uses a three-level storage hierarchy:

```text
GPU memory: current compute weights, activations, workspace
CPU memory: warm weight pool and pinned transfer buffers
Disk / NVMe: optional weight source when lazy_load=true
```

Offload is not a free speedup. It trades extra data movement, asynchronous scheduling, and buffer management for lower peak GPU memory. This allows models that would otherwise OOM to run on consumer hardware. When implemented carefully, part of the transfer overhead can also be hidden behind GPU computation.

---

## LightX2V Offload Architecture

LightX2V treats Offload as a framework-level capability rather than a one-off patch for a specific model. The same design can be applied across several model families:

- video generation: Wan2.1 / Wan2.2, HunyuanVideo, LTX;
- image generation: Qwen-Image;
- video restoration / super-resolution: SeedVR2;
- world models: Matrix Game, HY-WorldMirror;
- autoregressive video models: Self-Forcing / Lingbot-style pipelines.

The core execution model can be summarized as follows:

```text
                ┌────────────────┐
                │  CPU / Disk    │
                │ weight storage │
                └───────┬────────┘
                        │ prefetch
                        ▼
┌──────────────┐  H2D  ┌──────────────┐
│ CPU buffer   │ ────→ │ GPU buffer   │
│ pinned / hot │       │ current unit │
└──────────────┘       └──────┬───────┘
                              │
                              ▼
                         Transformer
                          compute
```

At the weight-container level, LightX2V packages blocks and phases as movable units. At the inference level, an offload manager handles prefetching, copying, stream synchronization, and buffer swapping. When `lazy_load=true`, the weight source is further extended to Disk / NVMe.

The key idea is not simply "put weights on CPU." More importantly, different models can share the same scheduling abstraction while keeping the model-specific execution details they need. For models with regular Transformer structures, LightX2V can use two GPU buffers for ping-pong prefetching. For models whose block structures are not fully identical, a more conservative per-block transfer strategy can be used instead.

![LightX2V Offload Overview]({{ site.baseurl }}/assets/offload-blog/offload_fig1.png)
*Figure 1: LightX2V Offload overview, including the motivation, three granularities, the CPU ↔ GPU path, and Disk / NVMe as an additional source when `lazy_load=true`.*

In the CPU-to-GPU transfer path, LightX2V relies on two important kinds of buffers.

**Pinned memory** is a stable staging area on the CPU side. Regular CPU memory may be moved or paged by the operating system, making it hard for the GPU to read efficiently during high-speed transfers. Pinned memory is fixed in place, so it is suitable as a source address for GPU DMA reads. Intuitively, regular CPU memory is like a temporary storage area, while pinned memory is like a dedicated loading dock for the GPU.

**GPU buffer** is a fixed workspace in GPU memory. After weights are copied from CPU to GPU, they need to land in stable GPU memory so attention, MLP, and other kernels can read them directly. For block offload, a common strategy is to allocate two GPU buffers: one for computing the current block and another for receiving the next block's weights. With fixed GPU buffers, each iteration only needs to overwrite the buffer with the next set of weights through H2D. There is no need to copy the current weights back to CPU (D2H), nor to repeatedly free and reallocate GPU memory. This reduces copies, memory allocation/free overhead, and potential synchronization stalls.

---

## Three Granularities: Model, Block, and Phase

LightX2V Offload can be understood through three granularities: `model`, `block`, and `phase`. Finer granularity reduces peak GPU memory but increases scheduling complexity and the number of transfers.

### Model-Level Offload

`model` granularity treats the whole module as one unit. For example, the Transformer can be moved to the GPU before inference and moved back to CPU afterward, or non-critical modules can be placed on demand.

This is suitable when:

- the model is close to the GPU memory limit but does not exceed it by too much;
- only coarse movement between pipeline stages is needed;
- implementation simplicity and low scheduling overhead are preferred.

The limitation is clear: the entire DiT or Transformer still needs to reside on GPU during execution, so peak-memory reduction is limited.

### Block-Level Offload

`block` granularity is the most common balance point in LightX2V. A Transformer is composed of multiple blocks, and inference only moves the current block, or the next block, to GPU:

```text
Compute block i on GPU buffer A
Prefetch block i+1 into GPU buffer B
Swap A/B
Compute block i+1
```

This strategy works well for most consumer GPUs: peak memory is much lower than keeping the whole model resident, while scheduling remains more manageable than phase-level offload.

For models with regular structures, such as Wan, HunyuanVideo, and Qwen-Image, block offload can typically use two GPU buffers for ping-pong prefetching.

### Phase-Level Offload

`phase` granularity further splits a Transformer block into smaller computation stages, such as:

```text
Self-Attention → Cross-Attention → FFN → Post-Adapter
```

This reduces peak memory further and is useful for very memory-constrained devices such as RTX 3060 / RTX 4070-class GPUs. The cost is higher scheduling complexity: intermediate results must be preserved between phases, and weight transfer, compute streams, and buffer lifetimes must be carefully aligned.

### Granularity Trade-off

| Granularity | Peak GPU Memory | Scheduling Complexity | Typical Use Case |
|---|---|---|---|
| `model` | Highest | Low | Coarse module placement |
| `block` | Medium | Medium | Consumer GPUs with enough CPU memory |
| `phase` | Lowest | High | Very tight VRAM budget |

---

## CPU-GPU Offload and Lazy Load

Offload is not only about granularity. It also depends on where weights come from and how they are moved to the GPU. LightX2V's default path is CPU ↔ GPU. When `lazy_load=true`, Disk / NVMe is added as an additional weight source.

### CPU ↔ GPU Offload

This is the most common mode: weights are prepared in CPU memory and copied to GPU by block or by phase during inference.

```text
CPU pinned buffer ──H2D──> GPU buffer ──compute──> next unit
```

The key ingredients are pinned memory, asynchronous copy, and double buffering. Ideally, while the GPU is computing the current block, the next block's weights are already being copied into another GPU buffer on a separate stream. In this case, part or most of the H2D transfer can be hidden behind computation.

In other words, CPU ↔ GPU offload does not mean "the GPU computes directly using CPU-resident weights." The actual process is: weights are prepared in a pinned CPU buffer, copied into a GPU buffer through H2D, and then GPU kernels read from the GPU buffer.

Because the GPU buffer is fixed and reused, the current block's weights typically do not need to be copied back to CPU through D2H after execution. The buffer also does not need to be freed. The next iteration simply overwrites it with a new H2D copy.

![CPU CUDA Offload Inference]({{ site.baseurl }}/assets/offload-blog/offload_fig2.png)
*Figure 2: CPU ↔ GPU block offload. Ideally, compute for the current block can overlap with H2D copy for the next block. Fixed GPU buffers also avoid D2H transfer and repeated GPU memory allocation/free.*

Advantages:

- no dependency on real-time disk reads;
- more stable latency;
- suitable for local workstations with enough system memory.

Main bottlenecks:

- CPU memory usage can still be high;
- PCIe bandwidth can limit transfer speed;
- if blocks are small or computation is fast, transfers are harder to hide.

### Lazy Load

`lazy_load=true` simply means weights do not all need to be resident in CPU memory in advance. Instead, they can be loaded from Disk / NVMe on demand and then enter the normal offload path.

```text
Disk / NVMe → CPU buffer → GPU buffer
```

Therefore, regular CPU offload mainly addresses GPU memory pressure, while `lazy_load=true` further introduces disk storage to reduce the need for all weights to stay resident in CPU memory.

![Disk CPU CUDA Offload Inference]({{ site.baseurl }}/assets/offload-blog/offload_fig3.png)
*Figure 3: When `lazy_load=true`, Disk / NVMe becomes an additional weight source. The figure shows the relationship between Disk, CPU buffers, and GPU buffers.*

---

## Wan2.2-A14B as an Example

Wan2.2-A14B contains two DiT backbones: a high-noise model and a low-noise model. With traditional whole-model offload, switching between the two models often requires large CPU/GPU weight transfers. As a result, some denoising steps can show obvious latency spikes.

Block offload moves the transfer granularity from "the whole model" down to "a single block." While the current block is computing on GPU, the next block's weights can be prefetched into another GPU buffer. This reduces peak GPU memory and also reduces the whole-model transfer overhead when switching between the high-noise and low-noise models.

More concretely, near the end of the high-noise model, the GPU may be computing the last high-noise block while the offload stream starts copying the first low-noise block into an idle GPU buffer. When high-noise computation finishes, the first low-noise block is already ready, or nearly ready. The transition no longer has to wait for the entire low-noise model to be transferred, reducing the stall at the model boundary.

---

## Practical Recommendations for Consumer GPUs

The offload strategy should be chosen based on GPU memory, CPU memory, disk bandwidth, and model structure. A simple rule is: **start with the coarsest strategy that runs, and only move to a finer granularity when memory is still insufficient.**

### RTX 5090 / RTX 4090

These high-end consumer GPUs usually work best with `block` offload:

- use `block` granularity for large DiT / Transformer backbones;
- keep small or frequently used modules resident on GPU when possible;
- combine with FP8 / INT8 / NVFP4 quantization;
- if system memory is sufficient, `lazy_load` is usually unnecessary.

Recommended starting point:

```json
{
  "cpu_offload": true,
  "offload_granularity": "block",
  "lazy_load": false
}
```

### RTX 3060 / RTX 4070-Class GPUs

For 8 GB to 16 GB GPUs, the first goal is often simply to run the model. Consider:

- switching to `phase` if block offload still OOMs;
- enabling `lazy_load` if CPU memory is also limited, introducing Disk / NVMe;
- reducing resolution, frame count, or batch size;
- combining offload with quantization;
- avoiding long-term GPU residency for non-critical modules.

Example configuration:

```json
{
  "cpu_offload": true,
  "offload_granularity": "phase",
  "lazy_load": true,
  "num_disk_workers": 4
}
```

---

## How to Enable Offload in Config Files

LightX2V Offload is mainly controlled through configuration files. The most important fields are:

- `cpu_offload`: whether to enable weight offload.
- `offload_granularity`: the offload granularity, commonly `model`, `block`, or `phase`.
- `lazy_load`: whether to introduce Disk / NVMe as a weight source.

The simplest model-level offload:

```json
{
  "cpu_offload": true,
  "offload_granularity": "model"
}
```

More commonly used block-level offload:

```json
{
  "cpu_offload": true,
  "offload_granularity": "block",
  "lazy_load": false
}
```

When GPU memory is tighter, try phase-level offload:

```json
{
  "cpu_offload": true,
  "offload_granularity": "phase",
  "lazy_load": false
}
```

If you do not want all weights to stay resident in CPU memory, enable lazy load:

```json
{
  "cpu_offload": true,
  "offload_granularity": "block",
  "lazy_load": true,
  "num_disk_workers": 4
}
```

In practice, try the following order:

1. If memory is almost enough, start with `offload_granularity="model"`.
2. For large models on consumer GPUs, prefer `offload_granularity="block"`.
3. If block offload still OOMs, try `offload_granularity="phase"`.
4. If CPU memory is also limited, enable `lazy_load=true`.

Offload is often combined with quantization. For example, large models can use `cpu_offload=true` together with FP8 / INT8 / NVFP4 quantization to further reduce GPU memory usage.

---

## Performance Example

The following figure shows a performance example for Wan2.2-A14B on a 5-second 480P generation task. The left chart shows per-iteration latency, where lower is better. The right chart compares peak VRAM usage against the GPU's maximum VRAM capacity.

![Wan2.2-A14B Offload Performance]({{ site.baseurl }}/assets/offload-blog/image.png)
*Figure 4: Speed and peak VRAM example for Wan2.2-A14B 5s 480P generation.*

The figure shows that Offload allows the same 14B-level video model to run across different VRAM tiers, from RTX 5090 and RTX 4090D to RTX 4060. High-end GPUs have lower per-iteration latency, while the 8 GB RTX 4060 can still keep peak VRAM within capacity, at the cost of significantly slower inference.

This reflects the core trade-off of Offload: it first solves the "can it run?" problem, and then uses block / phase granularity, quantization, and prefetching strategies to improve speed.

---

## Conclusion

LightX2V Offload can be summarized in one sentence:

> Instead of requiring all weights to stay resident on GPU, LightX2V moves the weights that will be needed next to the GPU during inference.

This is built on three design choices:

1. split weights by `model`, `block`, or `phase`;
2. move weights through CPU ↔ GPU transfer, and introduce Disk / NVMe when `lazy_load=true`;
3. overlap prefetching, transfer, and computation as much as possible.

For high-end consumer GPUs, block offload usually provides a good balance between memory and speed. For low-memory GPUs, phase offload and lazy load further lower the runtime requirement. In production deployment, the same mechanism can also be combined with quantization, feature caching, attention optimization, and disaggregated inference to improve overall resource efficiency.

As video generation models continue to grow, Offload will become less like a fallback for insufficient VRAM and more like a core capability of large-model inference runtimes.
---
layout: post
title: "让大视频模型跑上消费级显卡：LightX2V Offload 技术解析"
author: "LightX2V Team"
date: 2026-05-13
tags: [Offload, Video Generation, Consumer GPU, Inference Optimization]
---

视频生成模型正在快速变大。14B、28B 甚至更大规模的 DiT / Transformer 主干，在 BF16 精度下往往已经超过单张 RTX 4090 或 RTX 5090 的显存上限；如果再加上文本编码器、图像编码器、VAE、attention buffer 和中间激活，完整 pipeline 的显存压力会更明显。

LightX2V 的 Offload 机制要解决的不是“如何让所有权重同时放进 GPU”，而是一个更实际的问题：**推理过程中，GPU 只需要当前正在计算的那部分权重；其余权重可以放在 CPU 内存甚至 NVMe 磁盘中，并在即将使用前提前搬运到 GPU。**

这篇文章介绍 LightX2V 的多层 Offload 设计，包括：

- `model` / `block` / `phase` 三种 offload 粒度；
- CPU ↔ GPU 权重搬运，以及 `lazy_load=true` 时引入 Disk / NVMe；
- 异步预取与双缓冲的执行方式；
- 与只针对单个模型做 layerwise offload 的方案相比，LightX2V 如何把 Offload 做成跨模型的统一能力；
- 在 RTX 3060、RTX 4090、RTX 5090 等消费级显卡上的推荐策略。

**目录：**

- [为什么视频生成特别需要 Offload](#为什么视频生成特别需要-offload)
- [LightX2V Offload 的整体设计](#lightx2v-offload-的整体设计)
- [三种粒度：Model、Block、Phase](#三种粒度modelblockphase)
- [CPU-GPU Offload 与 Lazy Load](#cpu-gpu-offload-与-lazy-load)
- [从 Wan2.2 单点优化到框架级 Offload](#从-wan22-单点优化到框架级-offload)
- [消费级显卡上的实践建议](#消费级显卡上的实践建议)
- [如何在配置文件中开启 Offload](#如何在配置文件中开启-offload)
- [性能示例](#性能示例)
- [结论](#结论)

---

## 为什么视频生成特别需要 Offload

视频生成比图像生成更容易撞上显存墙，原因不只是模型参数变多。一个典型的 X-to-Video pipeline 通常包含多类模块：

| Component | Examples | Memory Pressure |
|---|---|---|
| Text / Image Encoder | T5, Qwen2.5-VL, CLIP, SigLIP | Prompt / image condition preprocessing |
| Transformer / DiT | Wan, HunyuanVideo, LTX, Qwen-Image, SeedVR2 | Dominant model weight and activation footprint |
| VAE Encoder / Decoder | Video VAE | High-resolution latent / pixel conversion |

LightX2V 对不同模块都提供了相应的 offload 能力。为了聚焦核心机制，本文主要讲解显存占用最大的 DiT / Transformer 主干部分的 offload；文本编码器、图像编码器、VAE 等其他模块的 offload 不在本文展开。

在数据中心 GPU 上，显存不足可以通过更大的单卡或多卡部署缓解；但在本地创作和开发场景中，更常见的是 12 GB、16 GB、24 GB 或 32 GB 显存的消费级显卡。此时，系统内存和 NVMe 磁盘通常比 GPU 显存更宽裕，Offload 就成为一种非常直接的工程手段。

LightX2V 将存储层级拆成三层：

```text
GPU memory: current compute weights, activations, workspace
CPU memory: warm weight pool and pinned transfer buffers
Disk / NVMe: optional weight source when lazy_load=true
```

Offload 的收益不是免费提速。它用额外的数据搬运、异步调度和 buffer 管理，换取更低的峰值 GPU 显存，让原本 OOM 的模型可以在消费级设备上跑起来。做得好的 Offload，还可以把部分搬运开销隐藏在 GPU 计算之后，让可运行性和速度之间取得更好的平衡。

---

## LightX2V Offload 的整体设计

LightX2V 的设计目标是把 Offload 做成框架能力，而不是某个模型里的临时补丁。当前同一套思路可以覆盖多种模型形态：

- video generation: Wan2.1 / Wan2.2, HunyuanVideo, LTX;
- image generation: Qwen-Image;
- video restoration / super-resolution: SeedVR2;
- world models: Matrix Game, HY-WorldMirror;
- autoregressive video models: Self-Forcing / Lingbot-style pipelines.

核心执行模型可以概括为：

```text
                ┌────────────────┐
                │  CPU / Disk    │
                │ weight storage │
                └───────┬────────┘
                        │ prefetch
                        ▼
┌──────────────┐  H2D  ┌──────────────┐
│ CPU buffer   │ ────→ │ GPU buffer   │
│ pinned / hot │       │ current unit │
└──────────────┘       └──────┬───────┘
                              │
                              ▼
                         Transformer
                          compute
```

LightX2V 在权重容器层面把 block / phase 封装成可搬运单元；在推理层面通过 offload manager 负责预取、拷贝、stream 同步和 buffer 交换。当 `lazy_load=true` 时，权重来源会进一步扩展到 Disk / NVMe。

这套设计的关键并不只是“把权重放到 CPU”。更重要的是：不同模型可以共享同一个资源调度抽象，同时保留必要的模型特化实现。例如结构规整的 Transformer 可以使用双 GPU buffer 做 ping-pong 预取；而 block 结构不完全一致的模型，则可以采用更保守的逐 block 搬运策略。

![LightX2V Offload Overview]({{ site.baseurl }}/assets/offload-blog/offload_fig1.png)
*Figure 1: LightX2V Offload 总览，包括 offload 动机、三种粒度、CPU ↔ GPU 路径以及 `lazy_load=true` 时引入 Disk / NVMe。*

在 CPU 到 GPU 的搬运链路里，LightX2V 重点依赖两类缓冲区。

**Pinned memory** 是 CPU 侧的稳定中转区。普通 CPU 内存可能被操作系统移动或换页，GPU 做高速拷贝时很难直接稳定读取；pinned memory 则会被固定住，适合作为 GPU 通过 DMA 读取的源地址。简单理解，普通 CPU 内存像临时堆放区，pinned memory 更像专门给 GPU 取货的装卸台。

**GPU buffer** 是 GPU 显存里的固定工作区。权重从 CPU 搬到 GPU 后，需要落在一块稳定的显存中，后续 attention、MLP 等 kernel 才能直接读取。对于 block offload，常见做法是准备两个 GPU buffer：一个用于当前 block 计算，另一个提前接收下一 block 的权重。固定 GPU buffer 后，每一轮只需要把下一组权重从 CPU 覆盖写入 GPU buffer（H2D），不需要把当前权重再从 GPU 搬回 CPU（D2H），也不需要频繁释放再重新申请显存，从而减少拷贝、显存分配/释放以及潜在的同步等待。

---

## 三种粒度：Model、Block、Phase

LightX2V Offload 可以按粒度分成 `model`、`block`、`phase` 三类。粒度越细，峰值显存越低，但调度复杂度和搬运次数也越高。

### Model-Level Offload

`model` 粒度把整个模块视作一个整体。例如在推理前把 Transformer 移到 GPU，推理后再移回 CPU；或者让非核心模块按需驻留。

适合场景：

- 模型本身接近显存上限，但还没有大幅超出；
- 只需要在 pipeline 不同阶段之间做粗粒度迁移；
- 希望实现简单、调度开销低。

局限也很明显：整个 DiT 或 Transformer 仍然需要在执行阶段常驻 GPU，因此显存下降空间有限。

### Block-Level Offload

`block` 粒度是 LightX2V 中最常用的平衡点。Transformer 由多个 block 组成，推理时只把当前 block 或即将执行的 block 搬到 GPU：

```text
Compute block i on GPU buffer A
Prefetch block i+1 into GPU buffer B
Swap A/B
Compute block i+1
```

这种方式适合大多数消费级显卡：显存峰值比整模型常驻低很多，调度复杂度又比 phase 粒度更可控。

对于 Wan、HunyuanVideo、Qwen-Image 这类结构比较规整的模型，block offload 通常可以使用双 GPU buffer 做 ping-pong 预取。

### Phase-Level Offload

`phase` 粒度继续把一个 Transformer block 拆成更小的计算阶段，例如：

```text
Self-Attention → Cross-Attention → FFN → Post-Adapter
```

它进一步降低了峰值显存，适合 RTX 3060 / 4070 这类显存非常紧张的设备。但代价是调度更复杂：phase 之间必须保存中间结果，并保证权重搬运、compute stream 和 buffer 生命周期完全对齐。

### Granularity Trade-off

| Granularity | Peak GPU Memory | Scheduling Complexity | Typical Use Case |
|---|---|---|---|
| `model` | Highest | Low | Coarse module placement |
| `block` | Medium | Medium | Consumer GPUs with enough CPU memory |
| `phase` | Lowest | High | Very tight VRAM budget |

---

## CPU-GPU Offload 与 Lazy Load

Offload 不只是一种粒度选择，也涉及权重从哪里来、如何搬到 GPU。LightX2V 的默认路径是 CPU ↔ GPU；当 `lazy_load=true` 时，会额外引入 Disk / NVMe 作为权重来源。

### CPU ↔ GPU Offload

这是最常见的模式：权重预先保存在 CPU 内存中，推理时按 block 或 phase 拷贝到 GPU。

```text
CPU pinned buffer ──H2D──> GPU buffer ──compute──> next unit
```

它的关键在于 pinned memory、异步拷贝和双缓冲。理想情况下，当 GPU 正在计算当前 block 时，下一 block 的权重已经在另一个 stream 中拷贝到 GPU。这样 H2D 搬运可以被部分或大部分隐藏在计算后面。

换句话说，CPU ↔ GPU offload 并不是“GPU 直接用 CPU 上的权重算”。真实过程是：权重先在 CPU 的 pinned buffer 中准备好，再通过 H2D 拷贝进入 GPU buffer，最后 GPU kernel 从 GPU buffer 读取权重并执行计算。

因为 GPU buffer 是固定复用的，执行完当前 block 后通常不需要把权重 D2H 传回 CPU，也不需要释放这块显存；下一轮直接用新的 H2D 拷贝覆盖 buffer 内容即可。

![CPU CUDA Offload Inference]({{ site.baseurl }}/assets/offload-blog/offload_fig2.png)
*Figure 2: CPU ↔ GPU block offload 的执行流程。理想情况下，当前 block 的 GPU 计算可以和下一 block 的 H2D 拷贝重叠；固定 GPU buffer 也避免了 D2H 回传和频繁释放显存。*

优点：

- 不依赖实时磁盘读取；
- 延迟更稳定；
- 适合系统内存充足的本地工作站。

主要瓶颈：

- CPU 内存占用仍然较高；
- PCIe 带宽会限制搬运速度；
- 如果模型 block 很小或计算很快，拷贝更难被隐藏。

### Lazy Load

`lazy_load=true` 的含义很简单：权重不再要求全部提前放在 CPU 内存中，而是可以从 Disk / NVMe 按需加载，再进入后续 offload 路径。

```text
Disk / NVMe → CPU buffer → GPU buffer
```

因此，普通 CPU offload 主要解决 GPU 显存压力；`lazy_load=true` 则是在此基础上进一步引入磁盘，降低权重对 CPU 内存常驻的要求。

![Disk CPU CUDA Offload Inference]({{ site.baseurl }}/assets/offload-blog/offload_fig3.png)
*Figure 3: `lazy_load=true` 时，Disk / NVMe 会作为额外权重来源。图中展示了 Disk、CPU buffer 和 GPU buffer 的关系。*

---

## 以 Wan2.2-A14B为例

以 Wan2.2-A14B 为例，它包含 high-noise model 和 low-noise model 两个 DiT 主干。传统整模型 offload 在两个模型切换时，往往需要进行大块权重的 CPU/GPU 搬运，某些 denoise step 因此会出现明显的延迟尖刺。

Block offload 的思路是把搬运粒度从“整个模型”下沉到“单个 block”：当前 block 在 GPU 上计算时，下一 block 的权重可以提前预取到另一个 GPU buffer 中。这样既能降低峰值显存，也能减少 high-noise model 与 low-noise model 切换时的整模型搬运开销。

更具体地说，在 high-noise model 即将结束时，GPU 正在计算 high-noise 的最后一个 block；与此同时，offload stream 可以开始把 low-noise model 的第一个 block 搬到空闲的 GPU buffer 中。等 high-noise 计算结束后，low-noise 的首个 block 已经准备好或接近准备好，模型切换就不再需要等待整套 low-noise 权重完成搬运，从而减少切换处的停顿。

---

## 消费级显卡上的实践建议

Offload 策略应该根据 GPU 显存、CPU 内存、磁盘带宽和模型结构来选择。一个简单原则是：**优先使用能跑通的最粗粒度策略；只有显存仍然不足时，再切到更细粒度。**

### RTX 5090 / RTX 4090

这类高端消费级 GPU 通常最适合 `block` offload：

- 对大 DiT / Transformer 使用 `block` 粒度；
- 小模块或频繁使用模块尽量常驻 GPU；
- 结合 FP8 / INT8 / NVFP4 等量化；
- 如果系统内存足够，通常不需要开启 `lazy_load`。

推荐起点：

```json
{
  "cpu_offload": true,
  "offload_granularity": "block",
  "lazy_load": false
}
```

### RTX 3060 / RTX 4070-Class GPUs

对于 8 GB 到 16 GB 显存的设备，第一目标通常是“能跑”。可以考虑：

- block offload 不够时切到 `phase`；
- CPU 内存也有限时开启 `lazy_load`，引入 Disk / NVMe；
- 降低分辨率、帧数或 batch；
- 与量化组合使用；
- 避免让非关键模块长期驻留 GPU。

可能配置：

```json
{
  "cpu_offload": true,
  "offload_granularity": "phase",
  "lazy_load": true,
  "num_disk_workers": 4
}
```
---

## 如何在配置文件中开启 Offload

LightX2V 的 Offload 主要通过配置文件控制。最核心的字段有三个：

- `cpu_offload`: 是否启用权重 offload。
- `offload_granularity`: 使用哪种粒度，常见取值是 `model`、`block`、`phase`。
- `lazy_load`: 是否引入 Disk / NVMe 作为权重来源。

最简单的 model 粒度 offload：

```json
{
  "cpu_offload": true,
  "offload_granularity": "model"
}
```

更常用的 block 粒度 offload：

```json
{
  "cpu_offload": true,
  "offload_granularity": "block",
  "lazy_load": false
}
```

显存更紧张时，可以尝试 phase 粒度：

```json
{
  "cpu_offload": true,
  "offload_granularity": "phase",
  "lazy_load": false
}
```

如果 CPU 内存也不希望常驻完整权重，可以开启 lazy load：

```json
{
  "cpu_offload": true,
  "offload_granularity": "block",
  "lazy_load": true,
  "num_disk_workers": 4
}
```

实践中可以按以下顺序尝试：

1. 显存接近够用时，先尝试 `offload_granularity="model"`。
2. 大模型在消费级显卡上运行时，优先尝试 `offload_granularity="block"`。
3. block 仍然 OOM 时，再尝试 `offload_granularity="phase"`。
4. CPU 内存也有限时，再打开 `lazy_load=true`。

Offload 通常也会和量化一起使用。例如大模型可以同时配置 `cpu_offload=true` 与 FP8 / INT8 / NVFP4 等量化方案，以进一步降低显存占用。

---

## 性能示例

下面是 Wan2.2-A14B 在 5s 480P 生成任务上的一个性能示例。左图是单 iter 延迟，越低越好；右图是峰值显存占用与 GPU 显存容量的对比。

![Wan2.2-A14B Offload Performance]({{ site.baseurl }}/assets/offload-blog/image.png)
*Figure 4: Wan2.2-A14B 5s 480P 生成任务上的速度与峰值显存示例。*

从图中可以看到，Offload 让同一个 14B 级视频模型可以覆盖从 RTX 5090、RTX 4090D 到 RTX 4060 的不同显存档位。高端显卡上单 iter 延迟更低，而 8 GB 显存的 RTX 4060 也能把峰值显存控制在容量内，代价是推理速度明显下降。

这也体现了 Offload 的基本取舍：它首先解决“能不能跑”的问题，然后再通过 block / phase 粒度、量化和预取策略去优化速度。

---

## 结论

LightX2V Offload 可以概括为一句话：

> 不再要求所有权重同时驻留 GPU，而是在模型推理过程中，把即将用到的权重提前搬到 GPU。

这背后对应三层设计：

1. 按 `model`、`block` 或 `phase` 拆分权重；
2. 通过 CPU ↔ GPU 搬运权重，并在 `lazy_load=true` 时引入 Disk / NVMe；
3. 尽可能重叠预取、传输和计算。

对于高端消费级显卡，block offload 通常能在显存与速度之间取得较好平衡；对于低显存显卡，phase offload 和 lazy load 能进一步降低运行门槛；在服务化场景中，同一套机制还可以与量化、特征缓存、attention 优化和 disaggregated inference 结合，提升整体资源效率。

随着视频生成模型继续变大，Offload 会越来越不像一个“显存不够时的 fallback”，而更像大模型推理 runtime 的核心能力之一。
