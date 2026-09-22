---
layout: post
title: "SwiftVR Inference Optimization: From One GPU to Causal Chunk Parallelism"
subtitle: "Reducing high-resolution intermediate computation and running ReAE and DiT across multiple GPUs for a single video"
author: "LightX2V Team"
date: 2026-09-18
tags: [SwiftVR, Video Restoration, Inference Optimization]
---

[SwiftVR](https://github.com/H-oliday/SwiftVR/tree/5ca168cef6ca7200f135fdfea85e5e13d12c5b53) uses Wan2.2-TI2V-5B to restore low-resolution video to higher resolutions in one generative step. It is the first generative video restoration model to achieve **real-time 1080p streaming on a consumer GPU**. Mask-free shifted-window attention (MFSWA) reduces attention cost, and a Restoration-aware Autoencoder (ReAE) maps between pixels and latent features. SwiftVR processes long videos in temporal chunks. Each chunk requires one diffusion Transformer (DiT) forward pass, while ReAE preserves causal state across chunks.

LightX2V integrates SwiftVR natively and optimizes the complete restoration pipeline. We first reduce computation and memory use on one GPU by reordering ReAE operations and controlling the number of frames processed together. Causal boundary exchange then lets multiple GPUs restore different chunks of the same video in parallel. Finally, we batch output from the decoder’s final layers and encode video within each process so restored frames reach the output file sooner.

The H100 workload is **4× video super-resolution**: **640×360** input and **2560×1440** output, with both width and height scaled by a factor of 4. The video contains **361 frames at 24 FPS**, and the output retains its frame count and frame rate. All resolutions use width × height. We also use this workload to measure each optimization independently.

On **H100 80GB**, the official implementation takes **17.781 s** to process the full video. LightX2V takes **9.297 s** on one GPU, a **47.72%** reduction. Four GPUs reduce the time to **2.870 s**, a **6.20×** speedup over the official single-GPU implementation. LightX2V also reduces peak memory per request on one GPU from the official implementation’s **51.059 GiB to 17.507 GiB**, a reduction of about **65.71%**.

| Implementation | GPU | Full video time (s) | Throughput (FPS) | Peak Mem. (GiB) |
|---|---:|---:|---:|---:|
| Official SwiftVR | 1 | 17.781 | 20.30 | 51.059 |
| LightX2V | 1 | 9.297 | 38.83 | **17.507** |
| LightX2V | 2 | 5.191 | 69.55 | 20.784 |
| LightX2V | 4 | **2.870** | **125.78** | 20.919 |

In both overview tables, **full video time** covers frame reading through completion of the output file, including model restoration, data transfer, and video encoding. Multi-GPU runs also include communication and final segment concatenation. **Throughput (FPS) = 361 / mean full video time** measures how many frames the complete pipeline processes per second. The output video's playback frame rate remains **24 FPS**.

**Peak Mem.** is the mean request peak measured by NVML. Multi-GPU runs use the largest per-GPU peak for each request. See <a href="#validation">Experimental Setup and Validation</a> for sampling and aggregation details.

On **RTX 5090**, LightX2V and the official implementation use the same input video and produce **1920×1080 output with 361 frames at 24 FPS**. The official implementation takes **29.233 s** on one GPU, while LightX2V takes **13.678 s**, a reduction of about **53.21%**. Four GPUs reduce the time to **3.982 s**, a speedup of about **7.34×** over the official single-GPU implementation. Peak memory per request on one GPU falls from **31.781 GiB to 16.861 GiB**, a reduction of about **46.95%**.

| Implementation | GPU | Full video time (s) | Throughput (FPS) | Peak Mem. (GiB) |
|---|---:|---:|---:|---:|
| Official SwiftVR | 1 | 29.233 | 12.35 | 31.781 |
| LightX2V | 1 | 13.678 | 26.39 | **16.861** |
| LightX2V | 2 | 7.417 | 48.67 | 18.730 |
| LightX2V | 4 | **3.982** | **90.66** | 18.732 |

<a id="pipeline"></a>

## From Pixels to Latents and Back

Restoration starts with low-resolution pixels: upsample the video to the target size, encode it with ReAE, restore the latents with one DiT step, and decode them into high-resolution frames.

```text
Low-resolution chunk → Upsample to target size → ReAE encode → One-step DiT restoration → ReAE decode → Encode and write video
```

DiT runs at a fixed timestep, with self-attention alternating between regular and shifted windows. LightX2V arranges these windows into dense batches for its existing attention backends and reuses Wan Transformer execution components. Converting the weight names offline lets LightX2V load the official weights directly.

Long videos are divided into temporal chunks. With `clip_len=24`, the reader takes 28 frames for the first chunk, 24 for each middle chunk, and the remaining frames for the final chunk. ReAE preserves boundary features at each layer across chunks, while DiT tracks global temporal positions. After decoding, the first chunk drops the 3 frames used for temporal alignment. The final chunk writes only valid frames, keeping the total output frame count equal to the input.

DiT computes in the smaller latent space, while ReAE progressively expands features into high-resolution pixels. High-resolution features are costly to compute, and processing more frames together increases the memory occupied by intermediate features. We therefore change the decoding order first, then control the number of frames processed together.

## Reducing ReAE Computation and Memory Use

### TemporalGrow: Compute Before Spatial Upsampling

The ReAE decoder uses TemporalGrow to increase the frame count and spatial upsampling to increase each frame’s resolution. The original order upsamples first, so TemporalGrow works on a feature map with twice the width and height.

**TemporalGrow does not mix neighboring spatial positions.** Its `1×1` spatial projection and temporal convolution with a `3×1×1` kernel compute independently at each position. The adjacent nearest-neighbor upsampling copies each position’s features to four output positions.

We can therefore replace “create four copies, then compute each copy” with “compute once, then create four copies”:

```text
Original:  H×W features → Nearest-neighbor upsample to 2H×2W → TemporalGrow → Spatial convolution
Reordered: H×W features → TemporalGrow → Nearest-neighbor upsample to 2H×2W → Spatial convolution
```

LightX2V moves all three TemporalGrow operations before their adjacent nearest-neighbor upsampling operations. Each now processes one quarter as many spatial positions; the following spatial convolution receives the same feature shape. This swap leaves spatial convolutions in place and preserves the original weight keys. The [decoder definition](https://github.com/ModelTC/LightX2V/blob/40744764aacc141166d9aa9c9596ba47ab1eab0e/lightx2v/models/networks/swiftvr/reae.py#L140) declares the layers in the new order.

We compare the two orders on one H100, with all other settings held constant:

| TemporalGrow order | Full video time (s) | Chunk compute time (s/24 frames) | Peak Mem. (GiB) |
|---|---:|---:|---:|
| After upsampling | 10.049 | 0.6305 | 21.531 |
| Before upsampling | **9.297** | **0.5811** | **17.507** |

Moving TemporalGrow earlier reduces full video time by **7.48%** and peak memory per request by **4.023 GiB (18.69%)**.

We also decoded the same DiT output latents with the same weights in both execution orders. Across all 361 frames, the BF16 decoded tensors match, and RGB values before video encoding are bitwise identical.

<a id="frame-batching"></a>

### ReAE Frame Batching: Reducing Intermediate Memory Use

Frame batching keeps fewer high-resolution intermediate features in memory at once. LightX2V uses `reae_frame_batch_size` to process small batches through consecutive ReAE layers that operate independently on each frame. A value of `1` processes one frame at a time. With `2`, two frames pass through the entire group of operators before the next two begin. With `0`, the group processes the current video chunk as a whole.

The key is to **pass each small batch through the entire group of operators**. Batching only one convolution still requires concatenating the full chunk for the next layer, leaving many complete high-resolution intermediates in memory. Running each batch through the group keeps only that batch’s intermediate features and writes the final results into one output buffer. MemoryBlocks with causal dependencies retain their existing state rules.

The following table compares three frame batch sizes on one H100:

| `reae_frame_batch_size` | Full video time (s) | Chunk compute time (s/24 frames) | Peak Mem. (GiB) |
|---|---:|---:|---:|
| 0, full chunk | **9.065** | **0.5669** | 29.329 |
| 1 | 9.297 | 0.5811 | **17.507** |
| 2 | 9.228 | 0.5764 | 17.893 |

**batch=1 uses the least memory of these three configurations**. Its request peak is **17.507 GiB**, **40.31%** below full-chunk execution, with a full video time of **9.297 s**.

On consumer GPUs with less memory, we can reduce the frame batch size to limit intermediate memory use. The H100 overview table, the TemporalGrow comparison, and the two-GPU output comparison below all use **batch=1**.

ReAE’s internal frame batch size is independent of <a href="#output">how many frames are transferred and delivered to the video encoder at a time</a>. In a supplementary test on two H100 GPUs with `batch=0`, switching from full-chunk output to 4 frames at a time reduces peak memory per request from **32.725 GiB to 30.526 GiB**, a decrease of **2.199 GiB**. This memory benefit applies when ReAE processes the full chunk internally.

<a id="parallel"></a>

## Scaling Across GPUs Along the Time Axis

With these optimizations, LightX2V processes the full video in **9.297 s** on one H100. Temporal chunk parallelism reduces the wait further by computing several chunks concurrently. Each GPU holds the complete model and performs ReAE encoding, DiT restoration, and ReAE decoding for its assigned chunks. Its worker process encodes the resulting video segments on the CPU.

Four GPUs process this video’s 15 chunks in four waves, assigned to GPUs 0–3 as follows:

| GPU | Wave 1 | Wave 2 | Wave 3 | Wave 4 |
|---|---|---|---|---|
| 0 | chunk 0 | chunk 4 | chunk 8 | chunk 12 |
| 1 | chunk 1 | chunk 5 | chunk 9 | chunk 13 |
| 2 | chunk 2 | chunk 6 | chunk 10 | chunk 14 |
| 3 | chunk 3 | chunk 7 | chunk 11 | Boundary communication, no output |

Restoring chunk 1 requires history from chunk 0. If each chunk had to wait for its predecessor to finish, execution would remain serial. SwiftVR’s dependency structure lets GPUs exchange the required boundary features between layers, then compute in parallel.

### ReAE: Causal Halo Exchange

A ReAE MemoryBlock concatenates the current frame’s input features with those of the previous frame, then applies a convolution. Consider a three-frame example. At this layer, A, B, and C denote each frame’s input features; P denotes those of the previous chunk’s last frame. The layer processes these three input pairs:

```text
Current frame:    A       B       C
Previous frame:   P       A       B
Conv input:     (A, P)  (B, A)  (C, B)
```

B uses A’s features **at the input to this layer**. The inputs for A, B, and C are already available, so all three convolutions can run together. Across chunks, the current chunk needs only the input features of the previous chunk’s last frame at this layer; it can proceed before the previous chunk finishes ReAE.

At each MemoryBlock, every GPU sends the input features of its chunk’s last frame to the next GPU while receiving boundary features from the preceding GPU. The GPUs then compute the layer in parallel. This exchange of boundary features is called **Halo Exchange**. LightX2V applies it along the time axis to transfer causal state between adjacent video chunks.

```text
                         GPU 0                     GPU 1
Layer input            [A, B, C]                  [D, E, F]
Boundary exchange            C ────────────────────→ C
Parallel layer compute (A,P) (B,A) (C,B)          (D,C) (E,D) (F,E)
                             ↓                         ↓
                       Next layer input          Next layer input
```

Both the ReAE encoder and decoder follow this rule. The video’s first chunk uses zeros as its historical input. GPU 0 saves the boundary features from the last GPU in each wave for the first chunk of the next wave. Each GPU resumes computation when its layer’s input boundary arrives.

### DiT: Exchange Input Context, Then Predict Independently

DiT’s context across chunks comes from **input latents produced by the ReAE encoder**. After encoding, the GPUs exchange context and run their DiT forward passes independently. Parallel execution preserves global temporal positions for rotary position embeddings (RoPE) and pads a short final chunk with input history.

Causal Halo Exchange in ReAE and chunk parallelism in DiT let GPUs process different chunks concurrently from encoding through decoding. ReAE history boundaries and final-chunk handling remain active even with `dit_overlap=0`. The [streaming implementation](https://github.com/ModelTC/LightX2V/blob/40744764aacc141166d9aa9c9596ba47ab1eab0e/lightx2v/models/networks/swiftvr/streaming.py#L81) prepares input history before model prediction.

The official version used in this comparison runs on one GPU. On H100, LightX2V processes the same video in **5.191 s on two GPUs** and **2.870 s on four**, giving **1.79×** and **3.24×** speedups over LightX2V on one GPU. At 1080p on RTX 5090, two and four GPUs provide approximately **1.84×** and **3.43×** speedups over LightX2V on one RTX 5090. These times include video segment encoding and final concatenation.

<a id="output"></a>

## Other Engineering Optimizations

LightX2V also overlaps frame reading, data movement, and video encoding with GPU computation:

- **Overlapping I/O and data movement.** Bounded queues connect the reader thread, GPU computation, and writer thread. They let the pipeline prefetch later chunks, compute the current chunk, and encode restored frames concurrently. LightX2V reuses input decoding buffers through DLPack, converts output to `uint8` on the GPU, and transfers it asynchronously to the CPU. For the H100 test video, this halves the pixel payload transferred to the CPU, from **7.436 GiB** in BF16 to **3.718 GiB** in `uint8`.
- **Batching output from the final decoder layers.** Layers after the last TemporalGrow have no state dependencies across frames. After causal decoding and boundary exchange, the multi-GPU path produces 4 frames at a time, allowing transfer and encoding of one batch to overlap with computation of the next. The single-GPU path still outputs a whole chunk at a time.
- **Encoding in process with PyAV.** Each GPU’s process encodes its own segments directly, avoiding the pipe transfer of RGB pixels to an FFmpeg subprocess. It releases the encoder after writing each segment. The main process then concatenates the segments in chunk order.

On two H100 GPUs, we separately revert to full-chunk output and FFmpeg subprocess encoding, keeping all other settings unchanged:

| Two-GPU output path | Full video time (s) | Peak Mem. (GiB) |
|---|---:|---:|
| 4 frames at a time + PyAV | **5.191** | 20.784 |
| Full-chunk output + PyAV | 5.254 | 20.666 |
| 4 frames at a time + FFmpeg subprocess | 5.360 | 20.784 |

Compared with full-chunk output, decoder output batching reduces full video time by **1.21%**. Compared with FFmpeg subprocess encoding, PyAV reduces it by **3.16%**. Both comparisons measure the complete pipeline, including pixel transfer, color format conversion, and codec execution.

<a id="validation"></a>

## Experimental Setup and Validation

- **H100 environment and configuration:** H100 80GB HBM3, PyTorch 2.11.0+cu130, CUDA 13.0, and cuDNN 9.19.0. Both implementations use BF16 and FlashAttention 3, with no quantization or CPU offload.
- **Request statistics:** We launch each configuration once, process five consecutive requests, and report the mean of the last four. Timing excludes model loading, startup warmup, and client polling.
- **Chunk timing:** We average compute time across middle chunks **2–13** from the last four requests: **48 chunks** in steady state, each producing 24 frames. Timing covers only GPU preprocessing, model restoration, and necessary communication.
- **Memory sampling:** We sample NVML approximately every **10 ms** throughout each request, including the first chunk but excluding model loading and startup warmup. For one GPU, we record the request peak; for multiple GPUs, we take the largest per-GPU peak within each request. We average the peaks from the last four requests and divide bytes by **1024³** to report GiB.
- **Output validation:** LightX2V outputs on both platforms passed full decoding checks and retain **361 frames at 24 FPS**. H100 outputs are **2560×1440**. All **15 outputs** from the 5090 tests are **1920×1080**, with no audio track.

<details markdown="block">
<summary>Steady-state performance per chunk (click to expand)</summary>

The table below reports compute performance for individual chunks in steady state. **Chunk processing rate (FPS) = 24 / mean chunk compute time**. Timing excludes video I/O and encoding. For multiple GPUs, this still measures the processing rate of one chunk, not overall service throughput. Full video throughput appears in the overview tables.

| GPU model | Implementation | GPU | Chunk compute time (s/24 frames) | Chunk processing rate (FPS) |
|---|---|---:|---:|---:|
| H100 | Official SwiftVR | 1 | 0.7509 | 31.96 |
| H100 | LightX2V | 1 | 0.5811 | 41.30 |
| H100 | LightX2V | 2 | 0.5879 | 40.82 |
| H100 | LightX2V | 4 | 0.6112 | 39.26 |
| RTX 5090 | Official SwiftVR | 1 | 1.0726 | 22.38 |
| RTX 5090 | LightX2V | 1 | 0.8649 | 27.75 |
| RTX 5090 | LightX2V | 2 | 0.8658 | 27.72 |
| RTX 5090 | LightX2V | 4 | 0.9055 | 26.51 |

</details>

<a id="usage"></a>

## Usage

See the [SwiftVR usage guide](https://github.com/ModelTC/LightX2V/blob/main/scripts/swiftvr/README.md) for model preparation, service deployment, and request examples.
