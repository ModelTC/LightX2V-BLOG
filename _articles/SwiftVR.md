---
layout: post
title: "SwiftVR Inference Optimization: From One GPU to Causal Chunk Parallelism"
subtitle: "Reducing high-resolution intermediate computation and running ReAE and DiT across multiple GPUs for a single video"
author: "LightX2V Team"
date: 2026-09-18
tags: [SwiftVR, Video Restoration, Inference Optimization]
---

[SwiftVR](https://github.com/H-oliday/SwiftVR) is a one-step video restoration model built on Wan2.2-TI2V-5B. It restores low-resolution video at a higher resolution, using mask-free shifted-window attention (MFSWA) to reduce attention cost and a Restoration-aware Autoencoder (ReAE) to map between pixels and latent features. Long videos are processed in temporal chunks. Each chunk requires one diffusion Transformer (DiT) forward pass, while ReAE preserves causal state across chunks.

LightX2V integrates SwiftVR natively and optimizes the complete restoration pipeline. We first improved operator execution on a single GPU and reduced high-resolution ReAE computation and data movement. We then introduced causal boundary exchange so that multiple GPUs can restore different chunks of the same video concurrently. With GPU computation accelerated, we batched the decoder's final layers and moved video encoding into each process to write restored frames sooner.

The workload is **4× video super-resolution**: **640×360** input and **2560×1440** output, with both width and height increased by a factor of 4. The video contains **361 frames at 24 FPS**, and the output retains its frame count and frame rate. All resolutions below use width × height. The tables report results for the full video on H100 80GB and NVIDIA GeForce RTX 5090 GPUs; the effects of individual optimizations are measured through controlled comparisons on H100 in BF16.

On **H100 80GB**, both the official implementation and LightX2V disable cuDNN `benchmark` and CUDA matmul TF32. Official SwiftVR takes **15.597 s** to process the full video. LightX2V takes **8.943 s** on one GPU, a **42.66%** reduction, and **2.826 s** on four GPUs, a **5.52×** speedup over the official single-GPU implementation. The single-GPU low-memory configuration reduces peak NVML usage from the official implementation's **45.100 GiB** to **21.940 GiB**, a **51.35%** reduction.

| Implementation | GPU | Avg. Time (s) | FPS | Allocated (GiB) | NVML (GiB) |
|---|---:|---:|---:|---:|---:|
| Official SwiftVR | 1 | 15.597 | 23.15 | 38.042 | 45.100 |
| LightX2V (speed configuration) | 1 | **8.943** | **40.37** | 25.560 | 39.690 |
| LightX2V (low-memory configuration) | 1 | 9.176 | 39.34 | **15.431** | **21.940** |
| LightX2V | 2 | **4.977** | **72.53** | 17.567 | 34.948 |
| LightX2V | 4 | **2.826** | **127.74** | 17.567 | 34.999 |

The H100 single-GPU low-memory configuration uses `reae_frame_batch_size=1`, which has the lowest NVML peak among the configurations tested here. The speed and multi-GPU configurations use `0`, so ReAE processes the current video chunk as a whole internally, without splitting it into smaller frame batches. The multi-GPU decoder tail independently [outputs 4 frames at a time](#output); this parameter does not control that output batching.

On **NVIDIA GeForce RTX 5090**, the official implementation takes **34.399 s** on one GPU. LightX2V's single-GPU speed configuration takes **24.218 s**, a reduction of about **29.60%**. Four GPUs reduce the time to **7.073 s**, a speedup of about **4.86×** over the official single-GPU implementation on the same machine. The single-GPU low-memory configuration peaks at **17.367 GiB** in NVML, about **45.46%** below the official implementation's **31.841 GiB**.

| Implementation | GPU | Avg. Time (s) | FPS | Allocated (GiB) | NVML (GiB) |
|---|---:|---:|---:|---:|---:|
| Official SwiftVR | 1 | 34.399 | 10.495 | 30.104 | 31.841 |
| LightX2V (speed configuration) | 1 | **24.218** | **14.907** | 25.530 | 29.124 |
| LightX2V (low-memory configuration) | 1 | 24.378 | 14.809 | **15.623** | **17.367** |
| LightX2V | 2 | **13.247** | **27.251** | 17.537 | 22.613 |
| LightX2V | 4 | **7.073** | **51.040** | 17.537 | 31.764 |

Both implementations use BF16 model precision on the 5090. Official SwiftVR uses SDPA; LightX2V uses SageAttention, with QK INT8 and PV FP8 internally, and FlashInfer RoPE. Both disable cuDNN `benchmark` and CUDA matmul TF32.

The 5090 speed configuration uses compilation with `reae_frame_batch_size=0`; the low-memory configuration uses compilation with `reae_frame_batch_size=2`. The latter adds only about **0.66%** to processing time while reducing peak NVML usage by about **40.37%**.

In both tables, **Avg. Time** is the mean processing time for the complete video, from reading frames to finishing the output file. We use the Python API with the model kept loaded across calls and average **8 warm calls across two process launches**, excluding model loading and startup warmup. **FPS = 361 / Avg. Time**; the output video remains at 24 FPS. **Allocated** is peak PyTorch tensor allocation, while **NVML** is peak total device memory usage, including initialization and warmup. For multiple GPUs, both columns report the largest per-GPU peak. Use NVML to assess actual device capacity requirements. See the [experimental setup](#validation) for sampling details and the separate comparison with the paper's timing per 24 frames.

<a id="pipeline"></a>

## Integrating the Complete Restoration Pipeline

Restoration starts with low-resolution pixels. The video is upsampled to the target dimensions, encoded by ReAE, restored by a single DiT step, and decoded into high-resolution frames.

```text
Low-resolution chunk → Upsample to target size → ReAE encode → One-step DiT restoration → ReAE decode → Encode and write video
```

DiT runs at a fixed timestep, with self-attention alternating between regular and shifted windows. LightX2V packs these windows into dense batches for its existing attention backends and reuses Wan execution components for conditioning, Transformer blocks, and output projection. Official weight names are converted offline, so runtime execution directly loads the LightX2V checkpoint.

Long videos advance in temporal chunks. With `clip_len=24`, the first chunk reads 28 frames, middle chunks read 24, and the final chunk reads the remainder. ReAE preserves boundary features at each layer between chunks, while DiT tracks global temporal positions. After decoding, the first chunk drops the 3 frames used for temporal alignment, and writing stops at the original frame count.

These two parts of the model call for different optimizations. DiT repeatedly executes Transformer blocks in the smaller latent space, making efficient operators and compilation useful. ReAE progressively expands features into high-resolution pixels, where execution order, batch size, and intermediate tensor lifetimes matter more.

<a id="single-gpu"></a>

## Improving Execution on One GPU

### Reusing Attention Kernels, RoPE, and Compilation

LightX2V reuses FlashAttention 3, FlashInfer rotary position embeddings (RoPE), and `torch.compile` in SwiftVR's DiT. Window layouts are prepared and reused for each input shape. The fixed timestep and text conditioning are reused across chunks. Compilation focuses on the repeated Transformer blocks.

On a single H100, with ReAE and I/O held constant, adopting FlashAttention 3, FlashInfer RoPE, and compilation in sequence reduces full video processing time from **11.202 s to 8.943 s**:

| DiT execution configuration | Full video time | Time reduction vs. previous row |
|---|---:|---:|
| SDPA + Complex RoPE, eager | 11.202 s | — |
| FlashAttention 3 + Complex RoPE, eager | 10.790 s | 3.68% |
| FlashAttention 3 + FlashInfer RoPE, eager | 10.440 s | 3.25% |
| FlashAttention 3 + FlashInfer RoPE, compile | 8.943 s | **14.34%** |

All four configurations peak at approximately 39.690 GiB in NVML. Startup warmup covers the computation shapes of the first, middle, and short final chunks. No new recompilations occurred during the measured calls. See [Warmup and Compile]({{ site.baseurl }}/posts/Warmup-and-Compile/) for the roles of warmup and compilation.

### TemporalGrow: Compute Before Duplicating Spatial Positions

The ReAE decoder increases both frame count and spatial resolution. In the original execution order, spatial upsampling comes before TemporalGrow: the feature map doubles in width and height before TemporalGrow processes it.

One property makes this order worth changing: **TemporalGrow does not mix neighboring spatial positions.** Its spatial projection uses a `1×1` convolution, and its temporal convolution has a `3×1×1` kernel. Computation along the time axis is independent at each spatial position. The adjacent nearest-neighbor upsampling operation simply creates four copies of each spatial position.

We can therefore replace “create four copies, then compute each copy” with “compute once, then copy the result to four positions”:

```text
Original: H×W features → Nearest-neighbor upsample to 2H×2W → TemporalGrow → Spatial convolution
Reordered: H×W features → TemporalGrow → Nearest-neighbor upsample to 2H×2W → Spatial convolution
```

LightX2V moves TemporalGrow before nearest-neighbor spatial upsampling, reducing the number of spatial positions it processes to one quarter. The following spatial convolution receives the same feature shape. This equivalence relies on operations that are pointwise in space and on nearest-neighbor duplication. The implementation swaps only these adjacent operations, preserving the position of spatial convolutions and the original weight keys. The [decoder definition](https://github.com/ModelTC/LightX2V/blob/12c3f00eccc7af2618728d05b39089ba3dc515df/lightx2v/models/networks/swiftvr/reae.py) declares the layers directly in this order.

On one H100, with all other LightX2V settings unchanged, this optimization reduces full video processing time from **9.351 s to 8.943 s**, a **4.36%** reduction. Peak NVML usage falls from **55.381 GiB to 39.690 GiB**, and peak allocated memory from **33.936 GiB to 25.560 GiB**. The comparison covers all 361 frames; RGB values before video encoding are bitwise identical before and after reordering.

### ReAE Frame Batching: Trading a Little Time for Memory

LightX2V uses `reae_frame_batch_size` to process small batches of frames through consecutive ReAE layers that can operate independently on each frame. With a value of `2`, two frames pass through the entire group of operators before the next two frames begin. With `0`, the group processes the current video chunk as a whole.

The key is to carry each small batch **through a group of layers**. Batching just one convolution and then concatenating the full chunk for the next layer still leaves many high-resolution intermediates in memory. Processing the group together keeps only the current batch's intermediate features, while final results are written into a single output buffer. MemoryBlocks with causal dependencies retain their existing state rules.

| `reae_frame_batch_size` | Full video time | Allocated (GiB) | NVML (GiB) |
|---|---:|---:|---:|
| 0, full chunk | 8.943 s | 25.560 | 39.690 |
| 1 | 9.176 s | **15.431** | **21.940** |
| 2 | 9.102 s | 15.652 | 22.317 |

**Batch=1 has the lowest memory peak among the H100 configurations tested here**, corresponding to the low-memory row in the opening table. Relative to batch=0, it reduces peak NVML usage by **44.72%** at a **2.61%** increase in time. Batch=2 uses about 0.377 GiB more than batch=1 but is slightly faster; relative to batch=0, it reduces peak NVML usage by **43.77%** at a **1.78%** increase in time.

Frame batching provides a deployment option for devices with less available memory. The H100 speed and multi-GPU experiments use batch=0 throughout.

### Overlapping Frame Reading, GPU Computation, and Encoding

After model computation, restored frames must return to the CPU and be compressed into a video. LightX2V connects the reader thread, GPU execution path, and writer thread with bounded queues so that different chunks can occupy different processing stages:

```text
Reader thread    Prefetch later chunks and prepare pinned memory
GPU path         Preprocess, ReAE encode, DiT, ReAE decode
Copy stream      Asynchronously transfer restored uint8 pixels to the CPU
Writer thread    Wait for the copy to finish, then convert colors and encode
```

On input, DLPack reuses decoding buffers. Pixels move to the GPU as `uint8` before conversion to the inference dtype and layout. On output, a separate copy stream and completion events let the CPU encode frames that have arrived while the GPU continues with later data. Queue limits keep buffer usage from growing with video length.

LightX2V combines DLPack, input conversion on the GPU, pinned inputs, and asynchronous transfers back to the CPU to reduce data movement overhead. On one H100, with the same model and reader/writer threads, these changes reduce full video processing time from **9.939 s to 8.943 s**, a **10.02%** reduction, while peak NVML usage increases by about 0.070 GiB. Conditioning caches and timely release of temporary tensors are enabled in both runs; their individual speed benefits were not measured.

<a id="parallel"></a>

## Scaling Across GPUs Along the Time Axis

The preceding optimizations bring full video processing time on one H100 down to **8.943 s**. Reducing the wait further requires computing multiple chunks concurrently. LightX2V introduces temporal chunk parallelism: each GPU holds the complete model and handles its assigned chunks through ReAE encoding, DiT restoration, ReAE decoding, and video segment encoding.

For this video, four GPUs process 15 chunks in four waves. GPU 0–3 below are logical ranks; the physical devices used in the experiment are listed in the experimental setup.

| GPU | Wave 1 | Wave 2 | Wave 3 | Wave 4 |
|---|---|---|---|---|
| 0 | chunk 0 | chunk 4 | chunk 8 | chunk 12 |
| 1 | chunk 1 | chunk 5 | chunk 9 | chunk 13 |
| 2 | chunk 2 | chunk 6 | chunk 10 | chunk 14 |
| 3 | chunk 3 | chunk 7 | chunk 11 | Boundary communication, no output |

This assignment must preserve causal dependencies: restoring chunk 1 uses history from chunk 0. If each chunk had to wait for its predecessor to finish the entire model, execution would remain serial. SwiftVR's dependency structure lets us reduce that wait to individual layers.

### ReAE: Causal Halo Exchange

A ReAE MemoryBlock concatenates the current frame's input features with those of the previous frame, then applies a convolution. Let A, B, and C be the current chunk's features, and P the final frame from the previous chunk. The layer processes these three input pairs:

```text
Current frame:    A       B       C
Previous frame:   P       A       B
Conv input:     (A, P)  (B, A)  (C, B)
```

B uses A's features at the input to this layer. Since the inputs for A, B, and C are already available, all three convolutions can run together. Across chunks, the same principle applies: the current chunk only needs the previous chunk's last input frame, without waiting for that chunk to finish the entire ReAE.

At each MemoryBlock, every GPU sends its last input frame to the next GPU and receives the boundary from the preceding GPU. The GPUs then compute the layer in parallel. Exchanging only boundary features is known as **Halo Exchange**; in SwiftVR, these are causal boundaries along the time axis.

```text
                         GPU 0                     GPU 1
Layer input            [A, B, C]                  [D, E, F]
Boundary exchange            C ────────────────────→ C
Parallel layer compute (A,P) (B,A) (C,B)          (D,C) (E,D) (F,E)
                             ↓                         ↓
                       Next layer input          Next layer input
```

Both the encoder and decoder follow this rule. The first chunk of the video uses zero history. GPU 0 saves the boundary from the last GPU in each wave for the first chunk of the next wave. Each GPU begins computing the layer once its boundary exchange is complete, reducing the dependency from “the entire preceding chunk has finished” to “this layer's input boundary has arrived.”

### DiT: Exchange Input Context, Then Predict Independently

DiT's cross-chunk context comes from **input latents produced by the ReAE encoder**. Once encoding finishes on each GPU, that context is already available. The GPUs can exchange it and then run their DiT forward passes independently. The parallel path preserves global temporal positions so that RoPE uses the original video's positions; a short final chunk is padded with input history.

Complete chunk parallelism therefore combines two mechanisms: causal Halo Exchange between ReAE layers, and parallel DiT execution after input preparation. ReAE history boundaries and tail handling remain active even with `dit_overlap=0`. The [streaming implementation](https://github.com/ModelTC/LightX2V/blob/12c3f00eccc7af2618728d05b39089ba3dc515df/lightx2v/models/networks/swiftvr/streaming.py) separates input-history preparation from model prediction, making this parallelization boundary explicit.

The [official version used for comparison](https://github.com/H-oliday/SwiftVR/tree/5ca168cef6ca7200f135fdfea85e5e13d12c5b53) runs on one device and has no multi-GPU chunk path for a single video. LightX2V processes the complete video in **4.977 s on two GPUs** and **2.826 s on four**, giving **1.80×** and **3.16×** speedups over LightX2V on one GPU. RGB values before video encoding are bitwise identical across all frames of this video on one, two, and four GPUs.

<a id="output"></a>

## Getting Multi-GPU Results to the Video Encoder Earlier

With multiple GPUs working in parallel, high-resolution outputs also arrive concurrently. If every GPU decodes a complete chunk, converts it to RGB, and only then sends it to the CPU, the full decoder-tail features and pixel buffers occupy GPU memory at the same time.

After ReAE's last TemporalGrow, only spatial upsampling, two-dimensional convolutions, activations, and pixel rearrangement remain. These layers have no cross-frame state dependencies. LightX2V batches their output: it first completes causal decoding and boundary exchange, then generates 4 frames at a time for the copy stream and video encoder.

```text
Causal decoding complete → Tail layers for 4 frames → Tail layers for next 4 frames → …
                                      ↓                          ↓
                               Transfer, encode           Transfer, encode
```

This reduces the high-resolution decoder-tail features held in memory at once and lets transfer and encoding of one batch overlap with computation of the next. The first chunk's 3 frames that would eventually be discarded can also be removed early, saving their tail-layer computation. Moving TemporalGrow earlier places the final spatial upsampling operation inside this batchable region.

Output batching and `reae_frame_batch_size` serve different purposes: the former controls how many frames are handed to the copy stream and encoder at a time; the latter controls the batch size within a group of ReAE layers. Even when the internal value is `0`, the multi-GPU decoder tail still outputs 4 frames at a time. The single-GPU path delivers the entire chunk at once. Required causal communication happens before output generation, so a rank with no valid output in the final wave still completes communication before skipping pixel generation.

LightX2V also uses PyAV so that each GPU process encodes its own video segment in process and releases the encoder when finished. The main rank concatenates the segments in chunk order. For this video, RGB data before video encoding totals approximately **3.718 GiB**. In-process encoding avoids transferring those pixels from Python through a pipe into an FFmpeg subprocess.

The following comparisons measure the two output optimizations on four H100 GPUs. Each row independently compares the complete pipeline before and after the corresponding change, with other settings held constant:

| Four-GPU optimization | Full video time, before → after | Time reduction | Peak NVML per GPU, before → after |
|---|---:|---:|---:|
| Full-chunk output → 4 frames at a time | 2.876 → 2.826 s | **1.73%** | 40.934 → 34.999 GiB |
| FFmpeg subprocess → PyAV | 3.020 → 2.826 s | **6.41%** | 35.001 → 34.999 GiB |

Batching the decoder tail reduces peak allocated memory from **25.560 GiB to 17.567 GiB**; its main benefit is lower memory usage. PyAV encoding in process reduces full video processing time by **6.41%**. GPU-to-CPU transfer is still required, and the two encoding paths link against different codec libraries, so this result measures the change in the complete encoding path.

<a id="validation"></a>

## Experimental Setup and Output Validation

### H100 Experimental Setup

The controlled comparisons cover 14 configurations, 28 independent process launches, and 140 measured calls. Each comparison uses the same workload and changes only the corresponding optimization. Gains are calculated within each comparison and are not additive.

- **Environment:** H100 80GB HBM3 with NVLink, using physical GPUs 4–7; PyTorch 2.10.0+cu128, cuDNN 9.10.2, BF16, with no quantization or CPU offload.
- **Workload:** The video and output specifications from the opening section, with no input audio; `clip_len=24`, `dit_overlap=0`, and `queue_size=3`. Output uses libx265, CRF 20, ultrafast, and yuv420p.
- **Timing:** With the model kept loaded across calls, we time the Python API from frame reading through completion of the output file, excluding model loading and startup warmup. Each configuration is launched twice, with five calls per launch. The last four from each launch provide **8 warm samples**.
- **Memory:** Allocated memory is the maximum request peak across all ten calls, including the first call of each launch. NVML is sampled approximately every 10 ms and takes the maximum across initialization, warmup, and measured calls. Multi-GPU results report the largest per-GPU peak. All values use GiB.
- **Flags and versions:** cuDNN `benchmark=False` and `deterministic=False`, CUDA matmul `allow_tf32=False`, and cuDNN `allow_tf32=True`. The official version is `5ca168ce`; LightX2V uses an experimental snapshot with `12c3f00e` as HEAD.

In a separate reproduction of the official default implementation, with both cuDNN `benchmark` and CUDA matmul TF32 enabled, stable chunk timings on H100 in BF16 are close to those reported in the [SwiftVR paper](https://arxiv.org/pdf/2606.09516):

| Source | Time per 24 frames | FPS | Peak memory |
|---|---:|---:|---|
| Paper | 0.766 s | 31.32 | 38.01 GB |
| Local reproduction of official defaults | 0.7743 s | 30.99 | allocated 38.042 GiB; NVML 45.678 GiB |

This timing covers GPU preprocessing and model computation, excluding frame reading, pixel transfers back to the CPU, and video encoding. It is measured separately from the full video processing times in this article.

### H100 Output Validation

We checked **956 tensors** before and after weight conversion; shapes, dtypes, and contents all matched. The **166 videos** generated by measured calls and additional validation all passed full decoding, resolution, frame-count, and frame-rate checks.

Pixel comparisons use **RGB from all 361 frames before video encoding**. Outputs are bitwise identical before and after TemporalGrow reordering, decoder-tail batching, and data movement optimizations, as well as between single- and multi-GPU execution and between the two encoding paths.

The LightX2V single-GPU speed configuration reaches a PSNR of **55.62 dB** against the official control. ReAE batch=1 and batch=2 reach **56.55 dB and 56.56 dB**, respectively, against batch=0.

<a id="usage"></a>

## Usage

The following examples use the CLI. From the LightX2V repository root, install the required operator backends, then download and convert the official weights:

```bash
hf download H-oliday/SwiftVR --local-dir /path/to/SwiftVR
python tools/convert/examples/convert_swiftvr.py \
  --source /path/to/SwiftVR \
  --output /path/to/SwiftVR_lightx2v
```

For one H100, use the compiled configuration. The output size is specified as height, then width: `1440 2560`.

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 -m lightx2v.infer \
  --model_cls swiftvr --task sr \
  --model_path /path/to/SwiftVR_lightx2v \
  --config_json configs/swiftvr/h100/swiftvr_compile.json \
  --video_path /path/to/input.mp4 --size 1440 2560 \
  --save_result_path /path/to/output.mp4
```

For four H100 GPUs, use the parallel configuration and ensure that `parallel.chunk_p_size=4` matches the process count:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 -m lightx2v.infer \
  --model_cls swiftvr --task sr \
  --model_path /path/to/SwiftVR_lightx2v \
  --config_json configs/swiftvr/h100/swiftvr_parallel.json \
  --video_path /path/to/input.mp4 --size 1440 2560 \
  --save_result_path /path/to/output.mp4
```

Related implementation: [native integration #1400](https://github.com/ModelTC/LightX2V/pull/1400), [compilation and pipelining #1406](https://github.com/ModelTC/LightX2V/pull/1406), and [multi-GPU and decoder optimizations #1525](https://github.com/ModelTC/LightX2V/pull/1525).
