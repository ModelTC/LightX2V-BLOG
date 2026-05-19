---
layout: post
title: "Parallel Mechanism of LightX2V"
author: "LightX2V Team"
date: 2026-05-19
tags: [Parallel]
---

## I. Overview of LightX2V's Parallel Mechanism

LightX2V is a distributed inference engine specifically designed for video generation tasks. It achieves multi-GPU acceleration through DiT parallelism, CFG parallelism, and hybrid parallelism.

#### DiT Parallelism

The video generation workload has the unique feature of "long sequences + small batches," which fundamentally distinguishes its best parallel strategies from the traditional methods used for large language models (LLMs) with "large models + high concurrency." The DiT parallelism of LightX2V mainly adopts sequence parallelism (SP) rather than the traditional tensor parallelism (TP) or data parallelism (DP). The specific reasons are as follows:

1. **Long sequence: SP is superior to TP in many aspects**

   TP mainly addresses the pressure of the model dimension (d_model), while SP focuses on relieving pressure of the sequence dimension (L). Currently, the primary challenge of video generation tasks lies in "ultra-long sequences" rather than "super-large models," which makes the advantages of TP less prominent, while SP is a more direct and efficient solution.

   At present, the parameter sizes of mainstream video generation models range from several billion to hundreds of billions, usually smaller than trillion-parameter LLMs, which leads to relatively limited computing and memory pressure on a single gpu. At the same time, video generation needs to take into account both temporal and spatial dimensions, and the resulting sequence is much longer than the typical LLM input sequence. The sequence length is calculated as frames × spatial_tokens_per_frame. Even for short videos, it can reach tens of thousands of tokens. Therefore, solving the problem of "ultra-long sequences" is a more urgent requirement for video generation tasks.

   SP distributes the computing load and memory pressure of long sequences across multiple devices. Compared with TP, it has advantages in terms of communication volume, scalability and computing efficiency, which will be detailed later.

2. **Small Batch: A Staged Limitation of DP**

   At this period, the core goal of video generation inference is to achieve high-quality real-time generation for a single task, and thus batch_size=1 is widely used. Under this setting, DP becomes invalid due to the fact that "there is no data to be divided," and thus cannot exert its advantages.

   This is not an inherent defect of DP, but a stage state determined by the current task objective. If future workloads shift towards batch generation or need to serve multiple independent requests simultaneously, DP will re-emerge as a valuable parallel strategy, complementing SP/TP.

#### CFG Parallelism

Unlike traditional LLMs, video diffusion models use Classifier-Free Guidance (CFG), which requires computing both conditional predictions and unconditional predictions. LightX2V implements CFG parallelization, where different GPUs concurrently compute the conditional branches and the unconditional branches, and then aggregate the results.

This CFG parallelism mechanism is limited to 2 GPUs, one for conditional judgment and the other for unconditional judgment.

#### Hybrid Parallelism

LightX2V supports enabling both DiT parallelism and CFG parallelism simultaneously to jointly utilize the distributed GPU resources.

LightX2V employs a two-dimensional grid called device_mesh for parallel management, organizing the GPUs into a (cfg_p_size, seq_p_size) dimension, where cfg_p_size × seq_p_size equals the total number of GPUs:

- The **cfg_p** dimension: used for Classifier-Free Guidance parallelism.
- The **seq_p** dimension: used for DiT sequence parallelism.

---

## II. Main Methods of DiT Sequence Parallelism

DiT computation is the bottleneck of video generation inference, and DiT sequence parallelism is the core of LightX2V's parallel strategy. Currently, LightX2V supports two representative sequence parallelism methods: Ulysses and Ring-Attention.

### Ulysses

The Ulysses method proposed by the DeepSpeed team works on the principle of performing data redistribution through two efficient all-to-all communications. This enables parallel processing of the attention calculation stage in the attention head dimension and parallel processing of other calculation stages in the sequence dimension.

#### Principle

The Ulysses method proposed by the DeepSpeed team redistributes data through two all-to-all communications. This enables the attention calculation stage to be processed across the head dimension, and the other calculation stages to be processed across the sequence dimension.

![Ulysses workflow diagram]({{ site.baseurl }}/assets/Parallel-blog/img2.PNG)

#### Workflow

1. **Sequence splitting:** Divide the input tokens along the sequence dimension and distribute them to each GPU.
2. **Sequence parallel computation:** Complete the calculation phases before the attention phase.
3. **Attention phase:**
   1. **Sequence-to-head exchange:** Before the attention calculation, through all-to-all communication, each GPU obtains the complete sequence but only handles a subset of the attention heads.
   2. **Local computation:** Each GPU performs Attention computation on the complete sequence for its assigned subset of attention heads.
   3. **Head-to-sequence exchange:** After the attention calculation, through all-to-all communication, each GPU recovers to the split along the sequence dimension and has all the attention heads.
4. **Sequence parallel computation:** Complete the calculation phases after the attention phase.

### Ulysses-4090 Variant

LightX2V has specially designed the Ulysses-4090 variant for the RTX 4090 GPU. It replaces all-to-all collective communication with manual P2P communication to achieve better performance on consumer-grade hardware.

### Ring-Attention

#### Principle

Ring Attention, proposed by the University of California, Berkeley, draws on the block-wise computation principle of FlashAttention. It treats the long sequence as a series of "building blocks" (K/V blocks) that need to be assembled sequentially, and distributes them to each GPU. Then, it exchanges the blocks along a ring-shaped communication structure (Ring) like a pipeline until the attention computation for the entire sequence is completed.

![Ring-Attention workflow diagram]({{ site.baseurl }}/assets/parallel-blog/ring-attention-workflow.png)

#### Workflow

1. **Sequence splitting:** Divide the input tokens along the sequence dimension and distribute them to each GPU.
2. **Sequence parallel computation:** Complete the calculation phases before the attention phase.
3. **Attention phase:**
   1. **Ring communication:** Gradually send and receive the respective current K/V blocks between adjacent GPUs through ring-shaped point-to-point communication.
   2. **Block-wise computation:** After receiving a new set of K/V blocks, each GPU performs the calculation with the local Q block, and updates the attention output and normalization factors.
   3. **Iterative traversal:** Iterate through steps 2 and 3 sequentially until the entire ring is traversed.
4. **Sequence parallel computation:** Complete the calculation phases after the attention phase.

### Technology Selection

We compare the core characteristics of Ulysses, Ring-Attention, and Tensor Parallelism (TP) across metrics such as communication volume, scalability, and computational efficiency, using the following definitions:

- **L:** Total sequence length.
- **N:** Number of parallel devices.
- **d:** Model hidden dimension.
- **H:** Total number of attention heads.
- **Hardware friendliness:**
  - For linear layers, computational efficiency is highest if the matrix is regular and wide, i.e., the hidden dimension d are multiples of specific values (e.g., 128, 256) and the multiplier is large enough to fully utilize the number of SMs. SP maintains the integrity of d, making it easier to meet the requirements of hardware. But TP splits d into d/N, which may degrade the computation shape and reduce computational efficiency.
  - For attention, extremely long sequences impose cache pressure. After the sequence is divided into blocks, the shorter sequence length brings better data locality and higher computational efficiency.

#### Comparative Analysis

The following is a breakdown and comparison of the three parallelization strategies:

1. **Ulysses**
   - **Parallel Dimension:** Attention Heads.
   - **Communication Volume:**
     - Global: ≈ 4 × L × d.
     - Per Device: ≈ (4 × L × d) / N.
     - Note: (4: Number of qkvo tensors)
     - Feature: The global communication volume is constant, independent of N. Increasing N directly reduces the communication burden on each device, resulting in minimal per-device overhead.
   - **Scalability:**
     - Upper Bound: ≤ H.
     - Feature: Limited by the inherent structure of the model (the number of heads), only valid for counting devices that can be divided evenly by H. The limitations of the GQA/MQA architecture are even more prominent.
   - **Computational Efficiency:**
     - Linear Layers: hardware friendliness = High
     - Attention Layers: hardware friendliness = Medium
     - Feature: Preserves full d for optimal linear shape. Full sequence L brings cache pressure for attention computation.

2. **Ring-Attention**
   - **Parallel Dimension:** Sequence Length.
   - **Communication Volume:**
     - Global: ≈ 2 × L × d × N.
     - Per Device: ≈ 2 × L × d.
     - Note: (2: Number of kv tensors)
     - Feature: Global volume increases with N, and per-device volume is constant.
   - **Scalability:**
     - Upper Bound: ≤ L.
     - Feature: Excellent in theory, limited only by the total sequence length. Enables processing of infinitely long sequences with sufficient devices.
   - **Computational Efficiency:**
     - Linear Layers: hardware friendliness = High
     - Attention Layers: hardware friendliness = High
     - Feature: Preserves full d for optimal linear shape. Splits sequence L and can achieve more hardware-efficient attention computation.

3. **Tensor Parallelism (TP)**
   - **Parallel Dimension:** Attention Heads.
   - **Communication Volume:**
     - Global: ≈ 2 × L × d × N.
     - Per Device: ≈ 2 × L × d.
     - Note: (2: Allreduce two-stage communication ratio)
     - Feature: Global volume increases with N, and per-device volume is constant.
   - **Scalability:**
     - Upper Bound: ≤ H.
     - Feature: Limited by the model's number of heads, similar to Ulysses.
   - **Computational Efficiency:**
     - Linear Layers: hardware friendliness = Low
     - Attention Layers: hardware friendliness = Medium
     - Feature: Splits d, degrading linear shape. Full sequence L brings cache pressure for attention computation.

#### Synthesis and Selection Guidelines

Based on the analysis above:

- **Communication Advantage:** Ulysses > Ring-Attention > TP
- **Scalability Advantage:** Ring-Attention > Ulysses ≈ TP
- **Computational Efficiency Advantage:** Ring-Attention > Ulysses > TP

TP exhibits comprehensive disadvantages compared to SP for long-sequence video generation tasks. Ulysses excels in minimizing communication burden but is limited by the head count (H). Ring-Attention imposes higher global communication pressure, but offers superior scalability and computational efficiency.

Therefore, for sequence parallelism technology selection:

- Prefer Ulysses when the model is relatively lightweight with abundant heads, and inter-device bandwidth is limited.
- For other long-sequence scenarios, Ring-Attention might be more advantageous.

---

## III. Performance Evaluation

### Parallel Configuration Example

A typical parallel configuration of LightX2V is as follows. This configuration uses 4-way sequence parallelism and 2-way CFG parallelism, requiring a total of 8 GPUs. The DiT parallelism method is specified as Ulysses, with fp8 communication optimization enabled.

```json
"parallel": {
    "cfg_p_size": 2,
    "seq_p_size": 4,
    "seq_p_fp8_comm": true,
    "seq_p_attn_type": "ulysses"
}
```

### Parallel Performance

On the RTX 5090 platform, for a video generation task using the Hunyuan 1.5-8B model for Image-to-Video (I2V) at 480p resolution with CFG enabled, all evaluated cases utilized int8 linear and sage attention optimizations. The per-step Diffusion Transformer (DiT) latency comparison is shown in the figure below, demonstrating good parallel efficiency even on consumer-grade GPUs with lower-speed interconnection.

![Hunyuan 1.5-8B parallel performance on RTX 5090]({{ site.baseurl }}/assets/Parallel-blog/img3.png)

Compared to leading open-source frameworks such as SGLang Diffusion, xDiT, and FastVideo, LightX2V demonstrates significant performance advantages. The comparison of per-step DiT latency for the Wan 2.1-14B model on an I2V 480p video generation task across multiple platforms is shown in the figure below. Specifically, LightX2V delivers a 1.59x speedup over the fastest open-source baseline on H100, a 2.73x speedup on RTX 5090, and a 3.26x speedup on RTX 4090.

![Wan 2.1-14B parallel performance comparison]({{ site.baseurl }}/assets/Parallel-blog/img4.png)
