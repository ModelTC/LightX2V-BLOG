---
layout: post
title: "LightX2V Reuse: Cross-Worker Encoder Caching and Segment Reuse for Long-Form InfiniteTalk Videos"
author: "LightX2V Team"
date: 2026-08-26
tags: [Reuse, Encoder Cache, InfiniteTalk, Video Generation]
---

Changing the seed is one of the most common ways to iterate on image and video generation. Once the prompt and any reference image or audio conditioning are fixed, we usually want a new sample—not another identical pass through the Text Encoder, Image Encoder, VAE Encoder, or Audio Encoder. Long-form video adds another requirement: if the first few segments are already satisfactory, regenerating from a later segment under a new seed should not require recomputing the entire video.

Both workflows involve reuse, but they reuse different artifacts. **Encoder Reuse** restores encoded model inputs while rerunning the full DiT path. **Segment Reuse** additionally preserves the first N segments of the previous successful video and resumes generation from the corresponding motion boundary.

The original process-local cache could only be consumed by the same Runner instance. In a multi-worker service, the next request may be dispatched to another process and miss the cache entirely. We moved encoder outputs, request compatibility metadata, the previous successful result path, and InfiniteTalk motion boundaries into a shared-disk cache. Reuse no longer requires worker affinity, while a staging-and-commit lifecycle prevents failed requests from replacing the last usable cache.

The implementation was merged in [LightX2V #1428](https://github.com/ModelTC/LightX2V/pull/1428). This article focuses on the behavior, mechanism, and operating boundaries of both reuse layers. The validation section covers static checks, cache lifecycle semantics, the prefix-frame formula, and FFmpeg merging; it does not report latency measurements from production checkpoints because a complete benchmark record is not yet available.

**Table of Contents**

- [Motivation: Why Reuse Has Two Layers](#motivation)
- [Support Matrix and Capability Boundaries](#support-matrix)
- [Cross-Worker Reuse: From Process-Local State to Shared Disk](#cross-worker-cache)
- [Encoder Reuse: Skipping Unchanged Input Encoding](#encoder-reuse)
- [Segment Reuse: Resuming InfiniteTalk from a Motion Boundary](#segment-reuse)
- [Publishing Only Successful Requests: The Cache Transaction Lifecycle](#cache-transaction)
- [Validation Scope: What We Verified and What Remains Unquantified](#validation)
- [Cost Model: What Each Reuse Layer Saves](#cost-model)
- [Configuration and Request Examples](#usage)
- [Usage Contract and Limitations](#constraints)
- [Conclusion](#conclusion)

---

## Motivation: Why Reuse Has Two Layers {:#motivation}

A typical diffusion video generation request can be simplified as follows:

```text
Text / Image / VAE / Audio Encoder
  → scheduler.prepare
  → Segment 0: N × DiT steps → VAE Decoder
  → Segment 1: N × DiT steps → VAE Decoder
  → ...
  → save video and audio
```

When the inputs remain unchanged and only the seed changes, the encoder outputs remain unchanged as well. Encoding them again adds latency and may repeatedly load or move large encoders. Encoder Reuse caches these outputs so that a new request can begin directly from the scheduler and DiT.

For a long video composed of multiple segments, however, skipping the encoders is not enough. Suppose a video contains eight segments and the first five already meet the requirements. What we actually want is:

```text
previous successful video: [S0][S1][S2][S3][S4][S5][S6][S7]
                                      keep ─────┘  └──── regenerate

new request:              [previous S0...S4][new seed: S5...S7]
```

Saving only the previous video cannot support this workflow. Adjacent InfiniteTalk segments are connected through motion frames, so the new S5 needs the latent conditioning at the S4→S5 boundary of the previous video. Segment Reuse therefore stores both the previous video and the `latent_motion_frames` at every segment boundary.

The two layers can be summarized as follows:

| Mode | `reuse` | `reuse_prefix_segments` | Reused artifacts | Recomputed work |
|---|---:|---:|---|---|
| Normal request | `false` | `0` | None. If the service sets `enable_reuse=true`, writes to a local file, and uses `return_result_tensor=false`, a successful request creates the cache | Encoder, plus DiT and Decoder work for every segment |
| Encoder Reuse | `true` | `0` | Encoder outputs | DiT and Decoder work for every segment |
| Segment Reuse | `true` | `N > 0` | Encoder outputs, the first N segments of the previous video, and the corresponding motion boundary | DiT and Decoder work starting from segment N |

Segment Reuse includes Encoder Reuse; it is not an independent side path.

The mechanism targets workflows in which generation conditions remain stable while sampling continues:

| Scenario | What remains unchanged | Recommended mode | Primary benefit |
|---|---|---|---|
| Trying multiple seeds in sequence | Prompt, negative prompt, dimensions, and inference configuration | Encoder Reuse | Skip repeated Text Encoder work while resampling the complete output |
| Generating multiple I2V or I2I versions from the same reference image | Prompt, input image, and preprocessing mode | Encoder Reuse | Reuse Image/VAE Encoder or multimodal encoding results |
| Iterating on a long InfiniteTalk video with the same subjects, audio, and reference video | All generation conditions except the seed | Segment Reuse | Preserve the accepted prefix and compute only the requested suffix |
| A serial creative session on a multi-worker service | Shared cache path and compatible worker configuration | Either mode | The next request does not need to return to the worker that produced the cache |

When the prompt, reference image, or audio has changed, send a normal request. When the inputs remain unchanged but the prefix should also change with the new seed, set `reuse_prefix_segments` to 0. Concurrent creative branches should use separate cache paths. Reuse is not a general-purpose result cache for arbitrary requests.

---

## Support Matrix and Capability Boundaries {:#support-matrix}

Reuse is an explicit capability, not default behavior inherited by every Runner. `BaseRunner` rejects both Encoder Reuse and Segment Reuse by default; each concrete Runner enables only the modes it actually supports.

| Runner / model family | Task | Encoder Reuse | Segment Reuse |
|---|---|---:|---:|
| Wan2.1 / Wan2.2 | T2V, I2V | ✓ | — |
| Qwen-Image | T2I, I2I | ✓ | — |
| InfiniteTalk | S2V | ✓ | ✓ |

This table reflects the capabilities exposed by the merged code and the configurations for which they are enabled. It does not imply that every model, checkpoint, and hardware combination has completed formal end-to-end testing; the validation section below states the actual coverage separately.

Runner capability and deployment intent are independent conditions. The service must also set `enable_reuse=true` and provide a dedicated `reuse_cache_path`. The complete configuration and operating boundaries appear later in this article.

---

## Cross-Worker Reuse: From Process-Local State to Shared Disk {:#cross-worker-cache}

### 1. Why a Process-Local Cache Is Not Enough

The most direct implementation stores encoder outputs in a Runner member:

```text
Worker A: request 1 → self.reuse_cache
Worker B: request 2 → cannot see Worker A's Python object
```

This can work with one worker, but it ties correctness to scheduling. To hit the cache, the service must route the next request back to Worker A. The cache also disappears as soon as that worker restarts.

Shared storage separates reuse state from the Runner lifecycle:

```text
                    ┌────────────────────────┐
Worker A ── write ─→│ shared reuse cache     │←─ load ── Worker B
Worker C ── load  ─→│ manifest + tensors     │←─ write ── Worker D
                    └────────────────────────┘
```

As long as every worker can access the same `reuse_cache_path`, any compatible worker can execute the next request. A multi-process service on one host can use a shared local NVMe directory. A multi-host deployment needs a shared filesystem visible to every worker.

### 2. Cache Directory Layout

A typical cache has the following structure:

```text
reuse_cache_path/
├── manifest.json
├── inputs_rank_00000.pt
├── inputs_rank_00001.pt
├── ...
├── boundary_00000.pt       # InfiniteTalk only
├── boundary_00001.pt
└── ...
```

`manifest.json` stores two pieces of metadata:

```json
{
  "reuse_key": {"...": "..."},
  "result_path": "/shared/results/previous-success.mp4"
}
```

- `reuse_key` determines whether the current request can consume this cache.
- `result_path` points to the previous successful result, from which Segment Reuse reads the video prefix.
- `inputs_rank_XXXXX.pt` stores encoder outputs for each distributed rank, preventing rank 0 tensors from being loaded on another rank.
- `boundary_XXXXX.pt` stores the motion latent between adjacent InfiniteTalk segments.

In a distributed request, each rank reads and writes its own tensor files and synchronizes at key stages with barriers. Only the main process publishes the directory and manifest.

### 3. `reuse_key` Should Describe the Cached Artifact, Not Copy the Entire Request

The purpose of `reuse_key` is to determine whether the artifacts on disk remain compatible with the current request. It should not mirror every API field. A parameter that affects only the current DiT sample, but not the encoder outputs, should not prevent Encoder Reuse.

| Runner | Main contents of `reuse_key` |
|---|---|
| Wan | The prompt and negative prompt actually encoded, plus video length; I2V also includes the image path and resize mode, while T2V includes the effective target shape |
| Qwen-Image | Prompt and negative prompt; I2I also includes the image path |
| InfiniteTalk | Prompt, negative prompt, audio, reference video, mask/bbox, duration, segment length, inference steps, shape, fps, motion frame, and other fields required for complete prefix compatibility |

The InfiniteTalk key contains more fields because it protects not only encoder tensors but also generated video content and segment boundaries. `seed`, reuse-control fields, and the output path are deliberately excluded: these are the values the next request is allowed to change.

---

## Encoder Reuse: Skipping Unchanged Input Encoding {:#encoder-reuse}

### 1. The First Successful Request Builds the Cache

When the service enables `enable_reuse`, writes results to a local path, and sets `return_result_tensor=false`, a normal request prepares the cache for its successor even if that request does not set `reuse=true`:

```text
run_input_encoder()
  → write inputs and required InputInfo state to .tmp
  → run DiT / Decoder / save as usual
  → publish the cache after the request succeeds
```

The cached content differs by Runner:

| Runner | Main cached content |
|---|---|
| Wan T2V | Text Encoder outputs and latent/target-shape state |
| Wan I2V | Text Encoder, Image Encoder, and VAE Encoder outputs, plus shape state |
| Qwen-Image T2I | Text embeddings and sequence length |
| Qwen-Image I2I | Multimodal text results, input-image VAE latents, sequence length, and original size |
| InfiniteTalk | Text embeddings, full audio embeddings, speaker count, and the audio array used for the video track |

The cache stores tensors already computed by the encoders, not the encoder models themselves. The service still keeps the encoder models resident, offloads them, or loads them on demand according to its configuration. Reuse changes only the request-level input computation.

### 2. The Next Request Restores the Cache from Disk

When a request sets `reuse=true`, the Runner first reads the manifest and compares `reuse_key`. Once the key matches, each rank loads its own `inputs_rank_XXXXX.pt`, restores the required `InputInfo` state, and continues along the original pipeline:

```python
inputs = load_reused_inputs() if reuse else run_input_encoder()
stage_reuse_cache()
result = run_dit_and_decode(inputs)
commit_reuse_result()
```

Wan and Qwen-Image still initialize new random latents and execute the complete scheduler, DiT, and Decoder path under the new seed. Encoder Reuse therefore produces a new result rather than copying the previous one.

Encoder Reuse in InfiniteTalk follows the same pattern. With `reuse_prefix_segments=0`, it restores only the encoded inputs and not the previous video prefix. Every video segment is regenerated under the new seed, producing a new set of motion boundaries.

---

## Segment Reuse: Resuming InfiniteTalk from a Motion Boundary {:#segment-reuse}

### 1. Long-Video Segments Are Not Simply Appended End to End

Let each InfiniteTalk segment contain `F` frames, and let adjacent segments share `M` motion frames. After the first segment, each new segment contributes only the following number of frames to the final video:

```text
segment_stride = F - M
```

The starting frame of segment `i` is therefore:

```text
start(i) = i × (F - M)
```

If the request reuses the first `N` segments, the video prefix contains:

```text
P(0) = 0
P(N) = N × (F - M) + M,  N > 0
```

With the defaults `F=81` and `M=9`:

| Reused segments N | Preserved prefix frames P(N) |
|---:|---:|
| 0 | 0 |
| 1 | 81 |
| 2 | 153 |
| 3 | 225 |

Using `N × F` would double-count the overlap between segments and shift the video join point.

### 2. Saving the Actual Continuation Boundary

When InfiniteTalk generates a non-initial segment, it takes the last `motion_frame` frames from the decoded previous segment, passes them through the VAE Encoder to obtain `latent_motion_frames`, and uses that latent as the scheduler's motion condition for the next segment.

```text
decoded tail of segment i
  → last M motion frames
  → VAE Encoder
  → latent_motion_frames
  → scheduler.prepare(segment i + 1)
```

The new implementation saves this latent at every adjacent segment boundary:

```text
boundary_00000.pt  = motion latent from S0 → S1
boundary_00001.pt  = motion latent from S1 → S2
...
```

Caching the latent directly has two benefits. First, it is the exact boundary state consumed by the scheduler. Second, it avoids decoding tail frames from a lossy video and running the VAE Encoder again, eliminating both extra computation and codec-induced distortion.

### 3. Skipping the First N Segments and Resuming at Segment N

After a request specifies `reuse_prefix_segments=N`, the generation loop starts at segment N:

```text
load encoder outputs
  → load boundary_(N-1)
  → scheduler.prepare(segment N)
  → generate segment N ... final segment
  → save boundaries for the new suffix
```

Segment N is not the first segment of the complete video, but it is the first segment actually executed by the current request. `InfiniteTalkScheduler.prepare()` updates its request ID only when `is_first_clip=true`. Because segment 0 is skipped, the Runner explicitly calls `begin_request()` so that request-scoped state such as the RoPE cache enters a new lifecycle.

During generation, boundaries required by the preserved prefix are copied from the previous cache into the staging directory. Boundaries produced by the new suffix then populate the remaining positions. After a successful request, the new cache contains the complete boundary chain for both the preserved prefix and regenerated suffix, so it can serve as the baseline for another seed change.

### 4. Merging the Previous Prefix with the New Suffix

Segment Reuse does not rerun the VAE Decoder to generate the previous prefix. The Runner first saves the newly generated suffix to a work video. FFmpeg then decodes the previous successful video, trims its prefix, and concatenates that prefix with the new suffix:

```text
previous successful video -- trim first P(N) frames --┐
                                                       ├─ concat → new result
new-seed suffix ---------------------------------------┘

audio = audio track from the previous successful video
```

The prefix and suffix timestamps are reset before the video streams are concatenated, and the result is encoded as H.264. The complete audio track is copied from the previous successful result. Segment Reuse requires audio and the other generation inputs to remain unchanged, so this preserves one continuous track without separately trimming and recombining suffix audio.

What remains unchanged is the visual content of the previous prefix. The merge re-encodes the output and does not promise byte-for-byte identity of the video file.

---

## Publishing Only Successful Requests: The Cache Transaction Lifecycle {:#cache-transaction}

The phrase “previous successful request” cannot be merely a usage convention; it must be enforced by the cache lifecycle. If the active cache were replaced immediately after the Encoder completed, a subsequent failure in DiT, VAE decoding, video saving, or FFmpeg merging would leave the next request with an incomplete cache and no valid result behind it.

LightX2V therefore stages cache updates, commits them only after generation succeeds, and discards them on failure:

```text
prepare
  → resolve final result, previous result, and cache paths

stage
  → create reuse_cache_path.tmp
  → write or copy encoder outputs
  → write the manifest
  → let InfiniteTalk write motion boundaries as generation proceeds

generate
  → DiT / Decoder
  → let InfiniteTalk write a work video and merge the prefix when needed

commit on success
  → current cache → .old
  → .tmp → current cache
  → work video → final result
  → remove .old

discard on failure
  → remove .tmp and the work video
  → preserve the previous successful cache and result
```

The main process performs directory replacement. If an exception occurs during commit, `.old` is restored as the active cache. Cache visibility therefore matches request success: cache state from unfinished requests remains only in staging, and other requests cannot treat it as the new baseline.

This also defines latest-success semantics. B consumes the successful result of A; if B succeeds, it becomes the new baseline, while if B fails, A remains the baseline for the next request. Across repeated seed changes, each request reuses the immediately preceding successful result rather than the original video.

This design does not support concurrent writers on the same cache path. The workflow is deliberately serial: the caller must wait for the previous request to succeed before issuing a reuse request.

---

## Validation Scope: What We Verified and What Remains Unquantified {:#validation}

The implementation described here is based on merged commit `a34806c0` (PR #1428). We reran Ruff and `py_compile` on the relevant Python files and used temporary directories and synthetic videos to validate the critical state transitions.

| Check | Method | Behavior established by the check |
|---|---|---|
| Static checks | Ruff and `py_compile` on the Base, Default, Wan, Qwen-Image, and InfiniteTalk Runners, plus the scheduler, schema, and worker | The relevant code passes the current lint and syntax checks |
| Cache semantics | An A→B handoff across two independent Runner instances, followed by separate key-mismatch and commit-failure cases | Cache consumption does not require the same Runner; incompatible inputs are rejected; a failed request does not replace latest-success |
| Segment semantics | The actual calculation for `F=81`, `M=9`, and `N=2`, followed by a merge of a four-frame prefix from a ten-frame old video with audio and a five-frame new suffix | The prefix formula produces 153 frames; the merged synthetic result contains nine frames and retains the old video's audio track |

These checks cover the cache, failure, and merge semantics discussed in this article, but should not be interpreted more broadly:

- This article does not yet document a formal end-to-end run with a production checkpoint under multi-process scheduling or across hosts using a shared filesystem.
- No Encoder Reuse or Segment Reuse latency benchmark has yet been collected under a fixed hardware setup, world size, video length, and sample count.
- Synthetic-video validation establishes frame count and audio-track mapping; it cannot establish visual continuity at segment boundaries in generated video.

The timing expressions below are therefore an analytical model for understanding where savings come from, not a table of measured performance. A formal benchmark should separately record encoder, segment, disk I/O, and video-merge time in the target deployment environment.

---

## Cost Model: What Each Reuse Layer Saves {:#cost-model}

Let `T_encoder` be the Encoder time, `T_segment` the DiT and Decoder time for one segment, `T_load` the disk-loading time, `T_merge` the video-merging time, `S` the total number of segments, and `N` the number of prefix segments reused.

The three paths can be approximated as:

```text
T_normal        ≈ T_encoder + S × T_segment
T_encoder_reuse ≈ T_load + S × T_segment
T_segment_reuse ≈ T_load + (S - N) × T_segment + T_merge
```

To highlight the primary difference, these expressions omit result-saving work shared by every path and do not separately model request-dependent staging and cache-write overhead.

The two reuse layers eliminate repeated work at different scales:

- The benefit of Encoder Reuse depends on how much time the complete pipeline spends in the Text, Image, VAE, or Audio Encoder.
- Segment Reuse skips multiple complete diffusion-and-decoding segments. Long videos and larger values of N are generally more likely to amortize disk reads and FFmpeg merging.
- When `N=0`, there is no segment-level saving, but encoder-level savings still apply.

Actual gains also depend on disk bandwidth, world size, video-encoding speed, segment length, and diffusion-step count. This article therefore does not claim a fixed speedup independent of deployment conditions.

---

## Configuration and Request Examples {:#usage}

### 1. Configure a Dedicated Cache Path for the Service

Add the following fields to an existing model configuration. Other fields such as model paths and parallel settings are omitted. Each compatible combination of model, task, and configuration should use its own `reuse_cache_path`:

```json
{
  "enable_reuse": true,
  "reuse_cache_path": "/shared/lightx2v/reuse/infinitetalk-480p-multi",
  "target_video_length": 81,
  "motion_frame": 9,
  "target_fps": 25
}
```

For production deployment, use an absolute path that resolves to the same location for every worker. Segment Reuse also requires the previous result path stored in the manifest to be visible to the current worker.

All requests below are sent to `POST /v1/tasks/video/`. The first JSON object is a complete request. The next two show only fields that change from the preceding request and cannot be sent as standalone request bodies. The API does not inherit fields from an earlier request; an actual request must still include every unchanged generation input.

### 2. First Request: Generate the Video and Build the Cache

```json
{
  "prompt": "A man and a woman are talking in a park.",
  "negative_prompt": "blur, artifacts, subtitles",
  "image_path": "/shared/inputs/reference.png",
  "audio_path": "/shared/inputs/person1.wav,/shared/inputs/person2.wav",
  "target_video_length": 81,
  "video_duration": 20,
  "infer_steps": 40,
  "seed": 42,
  "reuse": false,
  "save_result_path": "/shared/results/seed-42.mp4"
}
```

The cache becomes available for reuse only after this request succeeds.

### 3. Reuse Only the Encoders and Change the Seed for the Entire Video

Keep the generation inputs unchanged, change the seed, and set:

```json
{
  "seed": 2026,
  "reuse": true,
  "reuse_prefix_segments": 0,
  "save_result_path": "/shared/results/seed-2026.mp4"
}
```

This object shows only the changed fields. The complete request must keep all other generation fields identical to the previous successful request. `reuse_prefix_segments=0` selects ordinary Encoder Reuse.

### 4. Keep the First Two Segments and Change the Seed Starting from the Third

```json
{
  "seed": 2027,
  "reuse": true,
  "reuse_prefix_segments": 2,
  "save_result_path": "/shared/results/seed-2027-prefix-2.mp4"
}
```

The complete request must likewise carry the same prompt, reference image/video, audio, mask/bbox, duration, segment length, and inference configuration as the previous successful request. After validation, the service skips segments 0 and 1 and begins generation at segment 2.

The task API is asynchronous. Before sending the next reuse request, the client should poll `GET /v1/tasks/{task_id}/status` until the returned `status` is `completed`; receiving a task ID from POST is not sufficient.

---

## Usage Contract and Limitations {:#constraints}

To keep the encoder tensors, motion boundaries, and previous video prefix on disk semantically aligned, the current implementation follows these rules:

1. **Wait for the previous request to succeed.** A `reuse_cache_path` represents one serial latest-success chain; concurrent requests must not write to it together.
2. **Keep generation inputs unchanged.** Except for the seed, reuse-control fields, and output path, Segment Reuse requires the prompt, reference video/image, audio, mask/bbox, dimensions, duration, and inference configuration to match the previous successful request.
3. **Use stable local input paths.** `reuse_key` compares paths and associated parameters rather than hashing file contents. Do not modify an input file in place at the same path between requests.
4. **Keep service configuration and distributed topology compatible.** Rank caches correspond to the current world topology. After changing the model, task, world size, or a critical configuration value, use a new cache path or first run a new normal request.
5. **Use a separate cache path for each service.** Wan, Qwen-Image, InfiniteTalk, and configurations at different resolutions must not share one directory.
6. **Request a valid number of segments.** `reuse_prefix_segments` must be smaller than the total segment count because every request must generate at least one suffix segment.
7. **Use local-file results only.** `return_result_tensor=true`, HTTP(S)/RTMP output, and disaggregated inference do not currently enter the reuse path.

These constraints keep the implementation focused: consistency comes from explicit configuration and a serial request contract, without worker binding, a central state service, hardware autodetection, or input-file hashing.

---

## Conclusion {:#conclusion}

LightX2V divides reuse into two layers. Encoder Reuse restores seed-independent encoding results from shared disk across workers. InfiniteTalk Segment Reuse additionally saves `latent_motion_frames` and the previous video prefix, regenerating only the content after a requested segment boundary. Together, `reuse_key`, per-rank files, and the staging/commit lifecycle preserve cache compatibility and latest-success semantics, allowing stable-input iteration workflows to spend computation only on the parts that actually need another attempt.
