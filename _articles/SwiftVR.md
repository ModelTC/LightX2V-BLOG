---
layout: post
title: "SwiftVR in LightX2V: Native Integration and 44.27 FPS Steady-State Serving"
subtitle: "On a single H100, RUN pipeline time for 362-frame 2× super-resolution fell from 12.992 seconds to 8.236 seconds"
author: "LightX2V Team"
date: 2026-08-26
tags: [SwiftVR, Video Restoration, Inference Optimization]
---

Upscaling generated video from 768p to 2K is a common post-processing task. Restoration quality matters, but so does speed: if super-resolution cannot keep up with generation, it becomes the bottleneck for the entire service.

[SwiftVR](https://github.com/H-oliday/SwiftVR) is a one-step video restoration model built on the Wan2.2-TI2V-5B backbone. It reduces the cost of high-resolution restoration through mask-free shifted-window self-attention (MFSWA), a Restoration-aware Autoencoder (ReAE), and causal chunking. The [paper](https://arxiv.org/abs/2606.09516) reports 31 FPS at 2560×1440 on a single H100 and 26 FPS at 1920×1080 on a single RTX 5090.

LightX2V now supports native SwiftVR image and video super-resolution without importing the upstream `SwiftVRPipeline`. On a single H100 80 GB, the steady-state `RUN pipeline` for a 362-frame, 2× super-resolution request completed in 8.236 seconds, with an internal pipeline throughput of 44.27 FPS.

**Contents**

- [Final Results](#results)
- [Native Integration and Streaming Restoration](#native-integration)
- [From 12.992 Seconds to 8.236 Seconds](#optimization)
- [Quick Start](#quick-start)
- [Measurement Boundaries and Current Limitations](#validation)

---

## Final Results {:#results}

The cross-GPU benchmark used the same 768×1024 input video at 24 FPS, with 362 frames (about 15 seconds). With `sr_ratio=2`, the output resolution was 1536×2048. Each GPU ran a separate single-GPU service with warmup and compilation enabled. We sent three requests and recorded the third. Inference used BF16. Video output used `libx265`, `quality=60` (CRF 20), the `ultrafast` preset, and `yuv420p`.

| GPU | RUN pipeline | Processing speed | Peak whole-device memory (NVML) |
|---|---:|---:|---:|
| H100 80 GB | **8.236 s** | **44.27 FPS** | 35.32 GiB |
| A800 | 19.330 s | 18.81 FPS | 35.147 GiB |
| RTX 5090 | 22.625 s | 16.05 FPS | **18.699 GiB** |

`RUN pipeline` includes reading, preprocessing, ReAE/DiT restoration, D2H transfer, video writing, and audio muxing, but excludes HTTP queuing and network transfer. Processing speed is calculated over the internal pipeline interval before audio muxing, so it cannot be reproduced as `362 / RUN pipeline`. Resolutions in prose and tables use width × height; the API field `target_shape` uses `[height, width]`.

The three rows use different attention backends and ReAE batch sizes, so this is not a controlled comparison in which only the GPU changes. The H100 configuration trades more memory for throughput, while the 5090 configuration retains small-batch ReAE execution for memory-constrained devices. The workload, output resolution, and timing boundaries also differ from the paper, so these numbers are not a reproduction of the official benchmark.

### Comparison with SeedVR2-7B

A separate service benchmark used a 960×544, 24 FPS, 124-frame video and again recorded the third request:

| Target resolution | SeedVR2-7B | SwiftVR | Speedup with SwiftVR | SeedVR2-7B peak NVML | SwiftVR peak NVML |
|---|---:|---:|---:|---:|---:|
| 1920×1080 | 66.502 s | 3.818 s | 17.4× | 24.23 GiB | 26.61 GiB |
| 2560×1440 | 108.892 s | 5.776 s | 18.9× | 42.03 GiB | 35.34 GiB |

This is a comparison of complete deployment configurations rather than model operators alone. SeedVR2-7B used block-level CPU offload and produced H.264, while SwiftVR ran without offload and produced HEVC.

The current implementation supports image and video inference through both CLI and service paths, accepts either `sr_ratio` or `target_shape`, and follows the source video's FPS by default. The image pipeline can return a CPU tensor. The synchronous HTTP image endpoint (`/v1/tasks/image/sync`) encodes that in-memory result as PNG bytes, while the asynchronous endpoint used by `post_image.py` writes the image to `save_result_path` and returns task metadata. CPU offload and multi-GPU execution are not yet supported.

---

## Native Integration and Streaming Restoration {:#native-integration}

### Reusing Wan Components While Preserving SwiftVR Semantics

SwiftVR's patch embedding, conditioning, Transformer blocks, and unpatchify stage map onto LightX2V's Wan components, but the model retains its own restoration semantics:

- Inference is fixed at the degradation endpoint `t=1000` and performs one DiT forward pass.
- Self-attention uses MFSWA, alternating between regular and shifted windows.
- ReAE maps pixels to and from latents instead of using the general Wan VAE.
- ReAE carries causal feature state across chunks; the DiT tracks temporal offsets and retains latent context for overlap and tail handling.

LightX2V therefore reuses `WanPreInfer`, `WanTransformerInfer`, and `WanPostInfer`, while the SwiftVR layer implements fixed conditioning, temporally offset RoPE, MFSWA layouts, ReAE, and streaming. The runtime does not import the upstream Python inference library.

The official DiT uses Diffusers-style weight names, while LightX2V's Wan execution layer expects a different layout. We resolve that difference offline:

```text
official Diffusers checkpoint
  → LightX2V wan_dit backward conversion
  → validate tensor count / dtype / shape / required keys
  → SwiftVR_lightx2v checkpoint
```

At runtime, LightX2V loads only the converted checkpoint; no regular-expression key rewriting occurs during startup or requests.

### Causal Chunking Without Changing the Output Frame Count

ReAE requires the frame count to have the form `4k+1`. For an input with `T` frames, the internal processing length is:

```text
T_padded = 4 × ceil((T - 1) / 4) + 1
```

For example, 362 frames are internally padded to 365. Reads beyond the source repeat its final frame, but the writer accepts only the original `T` frames, so the output still contains 362 frames.

With `clip_len=24`, the first chunk contains 28 frames, middle chunks contain 24, and the last chunk receives the remainder. ReAE MemoryBlocks preserve boundary state across chunks, while DiT maintains a global temporal offset so that RoPE positions remain continuous. Even when `dit_overlap=0`, causal state still flows through ReAE.

MFSWA gathers Q/K/V into dense batches for regular and shifted windows, calls one of LightX2V's existing attention backends—SDPA, FlashAttention, or SageAttention—and scatters the result back to the original positions. Window layouts are precomputed by shape and device, so compiled blocks do not evaluate Python window branches.

An image is treated as a single LAST chunk and shares the same `SwiftVRRestorer` as video. A long-running service creates fresh lightweight request state for each call. Images and videos can therefore reuse loaded weights, compiled graphs, and warmup results without inheriting path fields from a previous request.

### Reducing ReAE Peak Memory

The original memory peak occurred near the end of the ReAE decoder, during high-resolution temporal expansion:

```text
14 frames × 128 channels × 720 × 1280
  → TemporalGrow
28 frames × 128 channels × 720 × 1280
  → Conv2D chain and CUDA workspace
```

LightX2V preallocates the final output and runs each small frame batch through the complete `Upsample → TemporalGrow → Conv2D → ReLU → Conv2D` sequence before moving to the next batch. This avoids rematerializing a full-chunk high-resolution intermediate between adjacent layers. The encoder uses the same strategy.

`reae_frame_batch_size=0` processes the full chunk for higher speed and higher memory usage; values of `1` or `2` target memory-constrained devices. In one 124-frame comparison, ReAE-stage peak memory fell from 39.70 GiB to 23.44 GiB, a 41.0% reduction. The per-frame MD5 hashes matched exactly; PSNR was infinite and SSIM was 1.0.

---

## From 12.992 Seconds to 8.236 Seconds {:#optimization}

Frame reading, D2H transfer, and encoding all contribute to service latency alongside model operators. LightX2V replaces its initial serial path with a bounded pipeline:

```text
reader thread: prefetch and pin chunk N+1
main GPU path: preprocess and restore chunk N
copy stream: async D2H for restored chunk N-1
writer thread: wait for D2H, then encode older output frames
```

`queue_size` bounds prefetched and pending-write chunks, keeping pinned memory and CPU output tensors under control. With `reae_frame_batch_size=2` fixed, adding frame prefetching, pinned memory, and asynchronous D2H reduced `RUN pipeline` from 10.822 seconds to 9.383 seconds. The `restore` time remained nearly unchanged; the gain came from hiding work outside the model graph.

Warmup for the compiled service profile covers two spatial sizes, FIRST, MIDDLE, and one-frame LAST chunks, as well as both DiT latent lengths. Live requests then use the compiled graphs, while one-shot CLI inference remains eager to avoid compilation startup cost.

The final hardware profiles use LightX2V's standard backend names:

| Profile | Self-attention | Cross-attention | RoPE | ReAE batch |
|---|---|---|---|---:|
| H100 | `flash_attn3` | `flash_attn3` | `flashinfer_rope` | 0 |
| A800 | `torch_sdpa` | `torch_sdpa` | `flashinfer_rope` | 0 |
| RTX 5090 | `sage_attn2` | `sage_attn2` | `flashinfer_rope` | 2 |

The table below summarizes historical H100 A/B measurements under the same workload. Stage 0 already includes the first ReAE frame-batching memory optimization. Each stage records only the third request, not a multi-run average, so differences below roughly 1% should be treated as close to run-to-run noise.

| Stage | Incremental change | RUN pipeline | Reduction vs. Stage 0 | restore | Processing speed | Peak PyTorch | Peak whole-device memory (NVML) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Eager, SDPA/SDPA, complex RoPE, ReAE batch=1, serial read/D2H, x265 medium | 12.992 s | — | 10.404 s | 28.00 FPS | 18.604 GiB | 20.89 GiB |
| 1 | + warmup and compilation | 11.520 s | 11.3% | 8.883 s | 31.72 FPS | 18.604 GiB | 20.75 GiB |
| 2 | + FlashAttention 3 for self-attention | 11.328 s | 12.8% | 8.662 s | 32.20 FPS | 18.604 GiB | 20.75 GiB |
| 3 | + `torch_real_rope` | 11.262 s | 13.3% | 8.536 s | 32.44 FPS | 18.604 GiB | 20.24 GiB |
| 4 | + FlashAttention 3 for cross-attention | 11.043 s | 15.0% | 8.438 s | 33.03 FPS | 18.604 GiB | 20.52 GiB |
| 5 | + ReAE batch=2 | 10.822 s | 16.7% | 8.181 s | 33.74 FPS | 20.292 GiB | 22.69 GiB |
| 6 | + frame prefetch, pinned memory, and asynchronous D2H | 9.383 s | 27.8% | 8.263 s | 38.86 FPS | 20.292 GiB | 22.99 GiB |
| 7 | + batching across consecutive ReAE operators | 9.238 s | 28.9% | 8.243 s | 39.48 FPS | 16.747 GiB | 18.89 GiB |
| 8 | + x265 `ultrafast` | 8.512 s | 34.5% | 8.238 s | 42.80 FPS | 16.747 GiB | **18.89 GiB** |
| 9 | H100 speed path: full-chunk ReAE | 8.277 s | 36.3% | 7.983 s | 44.10 FPS | 32.292 GiB | 35.45 GiB |
| 10 | + `flashinfer_rope` | **8.236 s** | **36.6%** | **7.949 s** | **44.27 FPS** | 32.292 GiB | 35.32 GiB |

Three conclusions matter most:

1. Warmup moves one-time work ahead of live traffic; most of the steady-state gain in Stage 1 comes from compilation.
2. Prefetching and asynchronous D2H produced the largest incremental percentage reduction in the table (13.3%), while batching across the consecutive ReAE operator chain primarily reduced memory.
3. Stage 8 is the balanced 18.89 GiB profile. Stages 9–10 improve on it by only about 3.2% while using roughly 16.4 GiB more whole-device memory.

FlashInfer RoPE was only 0.024 seconds (0.29%) faster than `torch_real_rope`. The outputs were not pixel-identical: with the `torch_real_rope` output as the reference, the FlashInfer output reached a VMAF of 94.063, a PSNR of 45.932 dB, and an SSIM of 0.987521. These measurements describe similarity between the two outputs; they do not establish which restoration is better relative to ground truth. Likewise, x265 `ultrafast` does not alter the raw frames produced by the model, but it does change the final video's compression quality and file size.

---

## Quick Start {:#quick-start}

From the LightX2V repository root, download and convert the checkpoint:

```bash
cd /path/to/LightX2V
hf download H-oliday/SwiftVR --local-dir /path/to/SwiftVR

python tools/convert/examples/convert_swiftvr.py \
  --source /path/to/SwiftVR \
  --output /path/to/SwiftVR_lightx2v
```

After setting the LightX2V root, checkpoint, input, and output paths at the top of each script, run one-shot video or image super-resolution:

```bash
bash scripts/swiftvr/inference/run_swiftvr_video_sr.sh
bash scripts/swiftvr/inference/run_swiftvr_image_sr.sh
```

Use the compiled profile for a long-running service:

```bash
bash scripts/swiftvr/server/start_server.sh

# In another shell
python scripts/swiftvr/server/post_video.py
python scripts/swiftvr/server/post_image.py
```

Example video request:

```json
{
  "video_path": "/path/to/input.mp4",
  "sr_ratio": 2,
  "save_result_path": "/path/to/output.mp4"
}
```

Replace `sr_ratio` with `"target_shape": [1440, 2520]` to request an exact 2520×1440 output. Use only one output-size option.

The repository provides three explicit compiled profiles:

- [H100 compile config](https://github.com/ModelTC/LightX2V/blob/aa1b7b5921d73fb42a605a3f4f3519b0554bb7e6/configs/swiftvr/h100/swiftvr_compile.json)
- [A800 compile config](https://github.com/ModelTC/LightX2V/blob/aa1b7b5921d73fb42a605a3f4f3519b0554bb7e6/configs/swiftvr/a800/swiftvr_compile.json)
- [RTX 5090 compile config](https://github.com/ModelTC/LightX2V/blob/aa1b7b5921d73fb42a605a3f4f3519b0554bb7e6/configs/swiftvr/5090/swiftvr_compile.json)

`start_server.sh` uses the H100 profile by default. Select the matching profile explicitly on other GPUs; the script does not infer hardware at runtime.

---

## Measurement Boundaries and Current Limitations {:#validation}

This article describes the final implementation at LightX2V commit [`aa1b7b59`](https://github.com/ModelTC/LightX2V/commit/aa1b7b5921d73fb42a605a3f4f3519b0554bb7e6). The performance tables use hardware measurements retained during development; we did not replay every historical stage when preparing this article. The commit and configurations are pinned, but we did not retain a complete PyTorch, CUDA, and attention-kernel version matrix for all three machines, so exact reproduction may vary with the software stack.

- Peak PyTorch memory covers only the allocator. To judge whether a workload fits on a GPU, use the peak whole-device memory measured through high-frequency NVML sampling.
- ReAE batching was verified with frame-by-frame MD5 equality. FlashInfer RoPE was verified only as highly similar, and the x265 preset changes the final compressed video.
- We did not establish tensor-by-tensor equivalence between LightX2V and the upstream `SwiftVRPipeline`, so we do not claim bitwise-identical outputs.
- The current implementation is single-GPU only and does not support CPU offload. Video length has no explicit limit, but total processing time and output size still grow with frame count.
- Before restoration, inputs are cropped on the bottom and right to dimensions divisible by 8. With `sr_ratio`, scaling therefore uses the aligned dimensions; video output dimensions are rounded to the nearest even values for encoding.
- Image super-resolution is available through both CLI and service paths, but all performance results in this article come from video.

**Resources**

- [SwiftVR paper](https://arxiv.org/abs/2606.09516)
- [Official SwiftVR repository](https://github.com/H-oliday/SwiftVR)
- [SwiftVR checkpoint](https://huggingface.co/H-oliday/SwiftVR)
- LightX2V implementation: [#1400](https://github.com/ModelTC/LightX2V/pull/1400), [#1406](https://github.com/ModelTC/LightX2V/pull/1406), [#1407](https://github.com/ModelTC/LightX2V/pull/1407), [#1409](https://github.com/ModelTC/LightX2V/pull/1409), [#1421](https://github.com/ModelTC/LightX2V/pull/1421), [#1427](https://github.com/ModelTC/LightX2V/pull/1427), [#1429](https://github.com/ModelTC/LightX2V/pull/1429), [#1438](https://github.com/ModelTC/LightX2V/pull/1438)

The final integration has a clear separation of responsibilities: offline conversion resolves checkpoint differences; Wan components provide the Transformer execution path; the SwiftVR layer preserves MFSWA, ReAE, and causal chunk semantics; and the Runner owns reading, restoration, D2H transfer, encoding, and the request lifecycle. Explicit hardware profiles then choose between the speed and memory paths.
