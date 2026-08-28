---
layout: post
title: "LightX2V Warmup and Compile: Optimizing Cold Starts and Steady-State Inference"
author: "LightX2V Team"
date: 2026-07-30
tags: [Warmup, Dynamic Compilation]
---

The first request to an image or video generation service is often much slower than the requests that follow. It may absorb kernel loading or generation, GPU memory allocation, workspace initialization, and compiled-graph specialization for the input shape. Meanwhile, other inference frameworks have been narrowing their performance gap with LightX2V through `torch.compile`. Preserving LightX2V's performance lead therefore required native compilation support.

To solve both problems, we added warmup and `use_compile` support for Wan, Qwen-Image, LTX2/LTX2.3, and LingBot-Video. We then distilled the lessons that held across these models into two reusable skills. This article examines the design tradeoffs, implementation challenges, and performance results, and offers a practical guide for future model integrations.

The scope is limited to LightX2V's native `--warmup` and `use_compile` features on the `normal`, CPU offload, and lazy-load paths that each model already supports. We do not cover `compiled_method` or `use_magi_compile`. For details on Magi Compiler, see [Graph Fusion for DiT Inference: Magi Compiler in LightX2V]({{ site.baseurl }}/posts/MagiCompiler/).

In representative tests on H100 GPUs, steady-state latency fell by approximately 3%–40% across the models. Profiler traces and cross-platform results suggest that the exact gains depend on how extensively a model's operators are already optimized and on which hardware resource is the bottleneck. Later sections describe the methodology and results in detail.

**Contents**

- [Motivation: Why Warmup and Compile Matter](#motivation)
- [Support Matrix](#support-matrix)
- [Warmup: Simulating Real Requests](#warmup)
- [Compile: Stable Boundaries, Dynamic Shapes, and Execution Reuse](#compile)
- [Benchmark Methodology and Steady-State Gains](#benchmark)
- [Evolution: From Model-Specific Cases to Shared Patterns](#history)
- [Skills: Turning Experience into Playbooks](#skills)
- [Conclusion](#conclusion)

---

## Motivation: Why Warmup and Compile Matter {:#motivation}

LightX2V is built for long-running inference services: load a model once, then serve a continuous stream of requests that vary in resolution, aspect ratio, and content. Extra startup time is acceptable; charging live requests for cold starts or recompilation is not.

In LightX2V, Step 1 of the first request has consistently taken much longer than subsequent steps. Warmup therefore exercises the production request path with representative inputs during startup, moving one-time work—such as kernel loading or generation, GPU memory allocation, and workspace initialization—ahead of live traffic.

A single diffusion request invokes the DiT/Transformer forward pass dozens of times:

```text
Text / Image / VAE Encoder
  → scheduler.prepare
  → N × (step_pre → DiT → step_post)
  → VAE Decoder
```

Compilation, by contrast, targets steady-state computation. `torch.compile` can optimize blocks that execute repeatedly, reducing Python dispatch, intermediate tensor reads and writes, and overhead from many small operators. Those savings compound across diffusion steps and requests. Compilation should therefore be judged by latency and throughput after the service is ready, not by startup time.

Specializing only for the first shape is not enough. If the compiled graph is tied to the first resolution, a later request may trigger graph recompilation or kernel generation simply because its resolution differs, putting seconds of latency—or more—back on the production request path. For long-running services, effective compilation requires dynamic-shape support: as long as the computation semantics remain unchanged, requests at different resolutions should reuse compiled artifacts whenever possible.

During startup, LightX2V therefore warms up two representative resolutions through the same path used by live requests:

1. The first resolution specializes the graph and initializes the kernels, allocator, and workspace needed along that path.
2. In the current tests, an input at a second resolution fails a guard under `dynamic=None`; Dynamo then recompiles and generalizes the dimensions it has observed to vary.

The division of labor is clear: warmup moves one-time costs out of the first request, while compilation accelerates steady-state computation. Warming up two representative resolutions also moves dynamic-shape adaptation into service initialization.

---

## Support Matrix {:#support-matrix}

Warmup and compilation respect each model's existing inference modes; they do not add CPU offload or lazy-load capabilities that the model did not already support.

| Model / task | `normal` | CPU model | CPU block | CPU phase | lazy block | lazy phase |
|---|---:|---:|---:|---:|---:|---:|
| Wan T2V/I2V/FLF2V | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Qwen-Image T2I/I2I, excluding layered mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LTX2/LTX2.3 T2AV/I2AV | ✓ | ✓ | ✓ | — | — | — |
| LingBot-Video T2I/T2V/I2V | ✓ | — | — | — | — | — |

---

## Warmup: Simulating Real Requests {:#warmup}

### 1. Where `warmup()` Should Run

`warmup()` should have one well-defined call site: after all modules have been initialized but before the first live request arrives. At that point, the model, scheduler, and compilation settings are ready. This placement lets warmup follow the production path without adding work to every `run_pipeline()` call.

```text
init_modules()
  → load model / encoder / VAE
  → attach scheduler
  → warmup()            # once only

first run_pipeline()
```

**Implementation**

When a Runner subclass is created, `BaseRunner.__init_subclass__()` automatically wraps its `init_modules()` method. The wrapper waits for the outermost initialization call to return before invoking `warmup()`, while `_warmup_done` guarantees that it runs only once. New Runners require no manual wiring, and nested calls introduced by `super().init_modules()` do not trigger duplicate warmups.

### 2. Scope of a Simulated Request

Warmup must exercise the critical path of a live request without paying the cost of a complete denoising run. The flow has three parts: input construction, required computation, and state cleanup.

```text
Build representative prompt / image / InputInfo
  → run the required text / image / VAE encoders
  → scheduler.prepare
  → run representative steps and required branches
      (step_pre → model.infer → step_post)
  → run stage transitions and the VAE decoder
  → synchronize the device
  → clear state that must not persist across requests
```

Representative inputs should replace only user-provided content; they must not bypass any stage that needs to be warmed up. The denoising loop need not run in full, but the selected steps must cover every computation branch that a live request may take.

### 3. What to Retain and What to Clear

| Keep | Release |
|---|---|
| Compiled graphs, kernels, resident weights, allocator cache, and safe immutable caches | Request state, temporary conditioning state, lazy-loaded models, offload managers, staging buffers, and related strong references |

What survives warmup matters as much as what runs during it. The rule is simple: keep performance-related state that can be reused across requests, and clear request-specific state that belongs only to the warmup run.

Both encoders and `scheduler.prepare()` may advance random-number-generator state. Each warmup path should set `generator` to `None` before either component first uses it and restore it to `None` on exit. Each live request can then create a fresh generator from its own seed. Latents, timesteps, solver history, and request-level caches must likewise be cleared.

The easiest way to undo much of warmup's benefit is to call the following unconditionally:

```python
torch.cuda.empty_cache()
gc.collect()
```

`torch.cuda.empty_cache()` releases unused reserved blocks from PyTorch's caching allocator, which may force Step 1 of the first live request to request memory from CUDA again. These calls do not necessarily clear every workspace held by third-party operators, but they weaken warmup's effect on the memory-allocation path. The final implementation uses a pressure-aware `maybe_empty_cache()`: it releases the device cache only when free GPU memory is low and the gap between reserved and allocated memory contains enough reclaimable space to justify doing so.

After warming up a resident model, LightX2V can run `gc.collect()` and `gc.freeze()` once to move the stable object graph out of subsequent GC scans. Lazy-loaded models and paths that use `unload_modules` rebuild objects for every request, so freezing is skipped there.

### 4. Model Differences and Implementation Challenges

A shared flow can unify the lifecycle, but it cannot eliminate model-specific semantics. Input construction, representative steps, and offload policies still need to be verified model by model.

**Inputs and stages**

- Resolution-independent text-encoder outputs can be reused across T2I/T2V warmup shapes.
- FLF2V requires both the first and last frames.
- Some decoders do not begin computation until their outputs are consumed, so the results must be fully materialized.
- A multi-stage model must feed the actual Stage 1 output into the upsampler and Stage 2.

**Representative steps**

For a model with a single denoising path, warmup usually runs `step_index=0`, which appears as Step 1 in a live-request log. Some models require more than this one pass:

- Wan2.2 MoE must cover the high-noise and low-noise branches separately.
- If warmup jumps between representative steps, the scheduler's solver history determines whether a reset is required.
- The final LTX2 step performs unpatchify and must run before its output can enter Stage 2 or the VAE.
- Multi-stage LTX2 must also follow the actual upsampler preparation path.

**Offload and lazy-load**

- CPU model offload must restore the intended device placement after warmup.
- Warmup in a lazy-load mode must exercise the actual loading path and then break strong references to the model, manager, and staging buffers.
- Each shape's warmup is wrapped in `try/finally` so that an exception cannot leak temporary state into a live request.

---

## Compile: Stable Boundaries, Dynamic Shapes, and Execution Reuse {:#compile}

### 1. Choose a Stable Compilation Boundary First

Transformer/DiT blocks run repeatedly across dozens of diffusion steps, making block execution a stable, high-value compilation boundary. Scheduler state, stage transitions, offload transfers, and stream synchronization remain outside the graph and continue to use the existing eager path.

```text
block loop
  → prefetch / swap (if any)
  → run_block()
      ├─ eager: infer_block()
      └─ compiled: reuse compiled block
  → stream synchronize (if any)
```

This boundary applies to normal execution, CPU model offload, and CPU block offload, as well as the phase-offload and lazy-load paths already supported by each model. It keeps the entire state-heavy pipeline out of Dynamo.

### 2. Keep the Eager Path Unchanged with a Shared Dispatcher

The shared layer keeps compilation concerns out of model-specific code. Its only jobs are dispatching calls and caching compiled entry points:

```python
class BaseTransformerInfer:
    def init_compile(self, config):
        self.use_compile = config.get("use_compile", False)
        self.compiled_blocks = {}

    def get_compiled_block(self, block_idx, block):
        key = self.get_compile_block_key(block_idx, block)
        cached = self.compiled_blocks.get(key)
        if cached is not None and cached[0] is block:
            return cached[1]

        def block_runner(*args):
            return self.infer_block(block, *args)

        compiled = torch.compile(block_runner, dynamic=None)
        self.compiled_blocks[key] = (block, compiled)
        return compiled

    def get_compile_block_key(self, block_idx, block):
        return block_idx

    def run_block(self, block_idx, block, *args):
        if self.use_compile:
            return self.get_compiled_block(block_idx, block)(*args)
        return self.infer_block(block, *args)
```

Every block call flows through `run_block()`. With compilation disabled, `run_block()` calls the existing `infer_block()` directly; with compilation enabled, it looks up or creates a compiled entry point. The shared layer neither modifies `self.block_idx` nor takes ownership of offload logic or model-specific branching, so the original eager semantics remain unchanged.

### 3. Dynamic Shapes: From `dynamic=True` to Automatic Generalization

Our first implementation used `dynamic=True` in the hope that one graph would cover every resolution. In practice, the steady-state gains were minimal. Further analysis showed that the problem was not dynamic shapes themselves, but LightX2V's weight representation.

Standard PyTorch models usually register weights as `nn.Parameter`, allowing Dynamo to recognize them as parameter sources and generally treat their shapes as static. LightX2V uses custom weight objects whose underlying weights are plain tensors. In the current traces, `dynamic=True` appears to make additional weight and structural dimensions dynamic. Fixed matrix-multiplication dimensions then lose their static guarantees, limiting Inductor's ability to specialize effectively.

After identifying the cause, we initially kept `dynamic=True` and progressively fixed weight and structural dimensions with `mark_static`, using performance tests to determine which dimensions needed to remain static. We then switched to `dynamic=False` with `mark_dynamic`, manually marking only resolution-dependent dimensions as dynamic. Steady-state performance improved, but input structures vary across models, making it easy to miss a dimension that should be marked.

The final design is simpler: `dynamic=None`. The first input typically produces a specialized graph. When the second shape fails a shape guard, Dynamo recompiles the graph and generalizes the dimensions it has observed to vary. For the models and inputs tested, warming up two shapes in sequence completes the required dynamic-shape adaptation before the service becomes ready. These shapes are internal to warmup; they need not be supplied through the application configuration or match every production resolution.

This result shaped the final warmup design: warmup should both move cold-start costs out of the first live request and drive dynamic-shape generalization with two representative resolutions.

### 4. Compile Once and Reuse Thereafter

The same blocks run at every diffusion step. Calling `torch.compile` again on each pass would create a new Python wrapper and closure every time. Even if the generated code hits a cache, those wrappers would still introduce unnecessary entry points and state management. We therefore create one compiled entry point for each block object or logical layer that must be distinguished, then reuse it across subsequent steps and requests.

The appropriate cache key depends on how block objects are reused and on whether computation depends on the logical layer index:

- **Normal execution and CPU model offload:** each layer has a fixed block, so the entry point is keyed by `block_idx`.
- **CPU block offload:** the GPU holds only a small number of staging blocks into which weights are copied in turn, so the entry point is keyed by `id(block)`.
- **CPU phase offload:** each phase type has a fixed buffer, so the entry point is keyed by `phase_idx`.

If computation also depends on the logical layer index, a block key must include `block_idx`, while a phase key must include `(block_idx, phase_idx)`. This creates more Python entry points, but it does not imply one-to-one CUDA kernel generation; Dynamo/Inductor may still reuse underlying graphs and code.

Wan's block computation reads `self.block_idx`, so its cache key must distinguish layer indices. Qwen-Image's offload path does not read the layer index, so block offload uses `id(block)`, while its four phase types require only four entry points. Lazy-load follows the same rules and recreates the corresponding entry points when it rebuilds the objects.

### 5. Expose Only Stable State to the Compiled Graph

A stable compiled graph depends on more than predictable tensor shapes: any Python state read by the graph must also remain stable. `torch.compile` guards those values, so a change in a field's value or type may trigger recompilation. Three main cases arose during adaptation.

**Turn branches into explicit inputs**

LTX2.3's multi-branch guider originally queried Python sets and instance state inside each block. Different layers and guiders produced different answers, quickly hitting the recompilation limit. The implementation now resolves a small set of boolean flags outside the graph and passes them into the block:

```python
skip_video_self = block_idx in self._skip_video_self_blocks
self.run_block(block_idx, block, ..., skip_video_self)
```

**Move one-time initialization earlier**

In the LTX2 self-attention path, `cu_seqlens` starts as `None` and is lazily replaced by a tensor. This is harmless in eager mode, but under compilation it exposes blocks to two different states. The compiled path now prepares the video and audio `cu_seqlens` once before entering the block loop, ensuring that every compiled block sees a tensor. The original eager path retains lazy initialization.

**Resolve stable configuration in advance**

LingBot-Video originally read its compute dtype inside the graph through an environment helper. Because the value remains constant throughout the service lifetime, it is now resolved once during initialization:

```python
self.compute_dtype = GET_DTYPE()
```

All three treatments follow the same principle: pass branch decisions as explicit inputs, prepare one-time state in advance, and resolve fixed configuration before entering the compiled graph.

### 6. Understand the Performance Boundary for Offload

Block offload roughly follows this sequence:

```text
CPU→GPU weight copy
  → swap staging buffers
  → synchronize the load stream
  → run the compiled block
  → synchronize the compute stream
```

Within the current compilation boundary, `torch.compile` can optimize only block computation. It cannot automatically optimize H2D transfers, prefetch/swap, or stream synchronization, nor can it cover lazy-load disk reads and object reconstruction.

When transfers and synchronization outside the graph dominate runtime, the saved compute time may not offset dispatch overhead, guard checks, or additional workspace costs, making compilation slower overall. In such cases, the better optimization targets are transfer overlap and offload scheduling, not a larger compiled graph.

### 7. Third-Party Operators: Runtime Routing and Tracing Boundaries {:#operators}

#### 7.1 Operator Routing Under `torch.compile`

A stable compilation boundary does not guarantee that third-party operators will follow the same runtime path. Some implementations detect that they are running under `torch.compile` and automatically switch or fall back to another path, causing subtle performance regressions. The Qwen-Image RMSNorm investigation revealed the following:

- **Eager:** `sgl_kernel.rmsnorm()` calls FlashInfer's `_flashinfer_norm.rmsnorm`.
- **Under `torch.compile`:** execution switches to `_rmsnorm_internal`, which calls SGL Kernel's own CUDA custom op, `torch.ops.sgl_kernel.rmsnorm.default`.

Both configurations report `sgl-kernel`, yet the two paths execute different RMSNorm kernels. We ultimately registered FlashInfer RMSNorm as a custom op: Dynamo sees only an opaque leaf node, while runtime continues to call the original FlashInfer kernel.

Because the configuration hides this distinction, the resulting speedup or regression can easily be misattributed to compilation itself. A profiler is therefore essential for verifying which kernels actually run.

#### 7.2 Keeping Dynamo Out of FlashInfer RoPE's JIT Wrapper

FlashInfer RoPE's Python entry point retrieves and loads a JIT module through `lru_cache`. Dynamo did not recognize the entry point as a single external operator; instead, it traced through the cache wrapper and continued into module lookup and loading. Python operations unrelated to model computation—including `datetime.now`, thread locks, and `posix.stat`—therefore became visible to Dynamo, expanding the tracing boundary and increasing compilation overhead.

The RoPE CUDA kernel was not the problem; tracing into its JIT-management logic was. The solution was to register the original call as a leaf operator:

```text
FlashInfer RoPE Python/JIT wrapper
  → torch.library.custom_op (declares in-place mutation of query and key)
  → Dynamo records only the rope_flashinfer_ node
  → runtime calls the original FlashInfer CUDA implementation
```

`register_fake` provides output metadata for FakeTensor/meta inference, so Dynamo does not need to enter the real CUDA implementation. The custom op establishes a tracing boundary; it does not eliminate RoPE's own kernel launch. The eager path remains unchanged and continues to call `apply_rope_with_cos_sin_cache_inplace()` directly.

---

## Benchmark Methodology and Steady-State Gains {:#benchmark}

For each comparison between eager and compiled execution, we run at least three trials with warmup enabled in both configurations. In each trial, we record only `infer_main cost` and calculate two steady-state means: one from Step 3 onward and another from Step 6 onward. We aggregate each window across trials. Comparing the two windows helps reveal lingering startup effects, while multiple trials reduce run-to-run noise. The table below reports relative improvements within each model; because the model-specific configurations differ, the rows are not direct cross-model comparisons.

```text
Steady-state latency reduction = (eager steady-state latency - compiled steady-state latency) / eager steady-state latency × 100%
```

The representative H100 results are:

| Model | Approx. steady-state latency reduction |
|---|---:|
| Wan | 3% |
| Qwen-Image | 3% |
| LTX2/LTX2.3 | 25% |
| LingBot-Video | 40% |

Profiler traces and observed operator paths suggest one plausible explanation: Wan and Qwen-Image already rely heavily on hand-optimized kernels. These kernels often appear as leaf nodes in the compiled graph, leaving relatively little room for Inductor to optimize. By contrast, LTX2/LTX2.3 and LingBot-Video contain more small, fusible operators and give Inductor more optimization opportunities. That interpretation is consistent with the larger gains, though it does not rule out other factors such as model architecture, input size, and runtime overhead.

Hardware matters as well. In cross-platform tests, Wan2.2 showed only a limited improvement on H100. On a platform with lower GPU memory bandwidth, where small operators accounted for a larger share of runtime, total time fell from 28 seconds to 22 seconds—a reduction of approximately 21%. This result suggests that hardware bottlenecks affect compilation gains. Separating the contributions of memory bandwidth and small-operator overhead will require finer-grained controlled experiments.

---

## Evolution: From Model-Specific Cases to Shared Patterns {:#history}

The shared abstraction was not predetermined; it emerged incrementally as we adapted more models.

| Date | Main work | Key challenge or outcome |
|---|---|---|
| 2026-07-15 | [Wan2.1 warmup #1251](https://github.com/ModelTC/LightX2V/pull/1251), [Wan2.2 MoE warmup #1254](https://github.com/ModelTC/LightX2V/pull/1254), [Wan compile #1255](https://github.com/ModelTC/LightX2V/pull/1255) | A single `step_index=0` pass could not cover both the high-noise and low-noise MoE branches |
| 2026-07-20–22 | [Wan lazy/offload warmup #1271](https://github.com/ModelTC/LightX2V/pull/1271), [Wan offload compile #1276](https://github.com/ModelTC/LightX2V/pull/1276) | Extended warmup and compilation to CPU offload and lazy-load paths |
| 2026-07-27 | [Qwen-Image and LTX2 warmup #1297](https://github.com/ModelTC/LightX2V/pull/1297) | Effective warmup had to cover the live request path: Qwen-Image required image conditioning tailored to each resolution, while LTX also had to perform final-step unpatchify and connect the upsampler, Stage 2, and Decoder end to end |
| 2026-07-28 | [Qwen-Image compile #1303](https://github.com/ModelTC/LightX2V/pull/1303) | Qwen phases are independent of layer indices, allowing four fixed staging buffers to reuse four compiled entry points; this work also revealed that the third-party RMSNorm implementation selected different kernels inside and outside Dynamo |
| 2026-07-29 | [LTX2/LTX2.3 compile #1307](https://github.com/ModelTC/LightX2V/pull/1307) | LTX2.3's multi-branch guider read mutable Python state and quickly hit Dynamo's recompilation limit; the graph became stable after the finite branches were converted into explicit boolean inputs |
| 2026-07-29 | [LingBot-Video warmup/compile #1308](https://github.com/ModelTC/LightX2V/pull/1308) | Adapted a new model using the existing skills and filled in missing steps discovered in practice |
| 2026-07-29 | [Warmup/compile skills #1309](https://github.com/ModelTC/LightX2V/pull/1309) | Refined and finalized both skills based on the LingBot-Video validation results |

LightX2V models differ substantially in their pipelines, schedulers, computation branches, and offload lifecycles. Designing the shared layer too early would have risked turning assumptions from one model into general rules.

We therefore began with Wan, then adapted Qwen-Image and LTX, extracting stable patterns from the differences we observed. As a final validation, we approached LingBot-Video as a fresh integration, applied the existing method from scratch, and fed the issues it exposed back into both the implementation and the skills. The shared abstraction grew out of an iterative cycle of implementation, extraction, validation, and refinement.

---

## Skills: Turning Experience into Playbooks {:#skills}

After implementing and validating the approach across multiple models, we distilled it into two skills: `support_model_warmup` and `support_model_compile`. They are not simple code templates; they are layered playbooks that guide an LLM through the process of adapting a new model.

We then used both skills to add warmup and compilation support to LingBot-Video. This provided an end-to-end test of whether an LLM could follow the playbooks and complete the task independently in a fresh conversation. Because LingBot-Video originally supported only `normal` inference—not CPU offload or lazy-load—the validation covered only the `normal` path.

| File | Purpose |
|---|---|
| `SKILL.md` | The primary playbook for the LLM, defining the order of auditing, implementation, and validation |
| `references/implementation-patterns.md` | Code patterns and lessons from existing models, consulted as needed during implementation |
| `references/casebook.md` | A troubleshooting guide for the `support_model_compile` skill, covering recompilation, third-party operators, and performance anomalies |

The primary playbook contains only the core steps needed to perform the task; implementation details and troubleshooting cases live in the references. Once new model-specific lessons have been validated, they are folded back into the corresponding files.

---

## Conclusion {:#conclusion}

Warmup and compilation solve two distinct problems in long-running services. Warmup moves one-time initialization out of the first live request, while compilation reduces the steady-state cost of repeated block computation. Together, warming up at two resolutions and using `dynamic=None` also move dynamic-shape generalization out of the production request path.

After module initialization, the final approach exercises the live request path at two resolutions. With `dynamic=None` enabled, those warmup runs absorb both cold-start work and dynamic-shape generalization. Only repeated block/phase computation is compiled; the existing eager control flow, offload lifecycle, and lazy-load lifecycle remain unchanged. Mutable Python state, one-time initialization, and third-party JIT wrappers are each handled at the narrowest appropriate boundary.

In representative H100 tests, steady-state latency fell by approximately 3%–40%, with the exact gain shaped by both existing operator optimizations and hardware bottlenecks. The approach has also been distilled into warmup/compile skills and independently validated on LingBot-Video's normal-only path, providing a reusable starting point for future model integrations.
