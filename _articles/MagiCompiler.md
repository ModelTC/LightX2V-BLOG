---
layout: post
title: "Graph Fusion for DiT Inference: Magi Compiler in LightX2V"
author: "LightX2V and SandAI"
date: 2026-05-21
tags: [MagiCompiler, torch.compile, QwenImage, Performance]
---

In DiT inference for video and image generation, compute-heavy core operators often already have dedicated optimized kernels, yet many small operator chains still lack mature, efficient optimizations—memory traffic and kernel launch overhead remain non-trivial. Hand-written Triton can optimize hot spots in those chains, but each pattern requires separate development and maintenance; using `torch.compile` directly often yields unstable gains or even regressions when subgraph boundaries and dynamic shapes are not handled carefully.

This post introduces [Magi Compiler](https://github.com/SandAI-org/MagiCompiler) integration in [LightX2V](https://github.com/ModelTC/LightX2V)—a `torch.compile`-based compilation stack tailored for Transformer-style inference. This integration lowers adoption and maintenance cost while delivering:

- **Low engineering cost**: packages subgraph boundaries, dynamic shapes, piecewise compilation, and other `torch.compile` plumbing into a reusable integration path, reducing development and maintenance cost for graph-level compilation in LightX2V.
- **Steady-state speedups on pre-Triton paths**: ~15–20% per-step improvement on [NeoPP](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) at steady state; ~20% on [Qwen Image](https://huggingface.co/Qwen/Qwen-Image-2512) with Triton OFF; closes much of the gap to hand-tuned Triton (~93–94%).
- **Multi-resolution serving on the compile path**: within a single process, switching resolutions can reuse compiled subgraphs, avoiding repeated per-shape recompilation that inflates first-step latency.

We thank the **SandAI** team for developing and open-sourcing [Magi Compiler](https://github.com/SandAI-org/MagiCompiler), and for their pioneering integration work in [LightX2V-MagiCompiler](https://github.com/SandAI-org/LightX2V-MagiCompiler)—their compiler design and reference implementation made this LightX2V integration possible.

**Table of contents:**

- [Why DiT Inference Needs Magi Compiler](#why-dit-inference-needs-magi-compiler)
- [Magi Compiler Integration in LightX2V](#magi-compiler-integration-in-lightx2v)
- [Benchmarks and Recommendations](#benchmarks-and-recommendations)
- [Porting Magi Compiler to a New Model](#porting-magi-compiler-to-a-new-model)
- [Limitations](#limitations)

## Why DiT Inference Needs Magi Compiler

DiT inference tends to split into two worlds: FlashAttention, GEMM, MoE, and other large operators are already heavily optimized, while small operator chains—norm, RoPE, modulate, residual handling—still leave room for improvement.

Two common approaches target those small chains:

- **Hand-fused kernels (e.g., Triton)**
  - Hot operator chains are rewritten as dedicated fused kernels; when patterns are fixed and tuning is thorough, performance is excellent.
  - The cost is per-pattern development and maintenance; model structure changes often require new kernels or re-tuning.

- **Compiler-driven fusion (e.g., `torch.compile`)**
  - The compiler traces the computation graph and fuses operator chains automatically, without manual kernel work.
  - On complex DiT graphs, however, gains are unstable or negligible without careful subgraph boundaries and dynamic-shape design; robust support for custom ops and dynamic shapes is itself a substantial engineering effort.

**Magi Compiler** follows the compiler-driven path. Built on `torch.compile`, it adds Transformer-oriented enhancements—subgraph boundary conventions, piecewise compilation, and declarative dynamic-shape configuration—packaged for easier, maintainable integration and lower engineering cost when enabling compilation in LightX2V.

## Magi Compiler Integration in LightX2V

### Integration overview

![Figure 1: LightX2V + Magi Compiler integration overview]({{ site.baseurl }}/assets/magi-compiler-blog/magi_fig1.png)

*Figure 1: LightX2V + Magi Compiler integration overview.*

As shown in Figure 1, Magi is enabled via `use_magi_compile` in config. Each model performs one-time setup in `TransformerInfer.__init__`; the inference loop runs paths decorated with `@magi_compile`. Green **Compiled regions** cover **fusible small operator chains**; orange **Eager** blocks are modules that do not participate in compile fusion. How many compile regions to use varies by model.

### Core mechanisms

#### 1. Declaring compile regions with `@magi_compile`

Decorate selected functions with `@magi_compile`, declare variable dimensions via `dynamic_arg_dims`, and tune compile behavior with `config_patch`. Example:

```python
def _magi_config_patch(c):
    return c.model_copy(update={"disable_cache": True})

@magi_compile(
    dynamic_arg_dims={"hidden_states": 0, "freqs": 0},
    config_patch=_magi_config_patch,
)
def infer_region_magi(self, block, hidden_states, freqs, ...):
    return self.infer_region(block, hidden_states, freqs, ...)

for block in blocks:
    hidden_states = self.infer_region_magi(block, hidden_states, freqs, ...)
```

The main loop calls `infer_region_magi` instead of `infer_region` directly, matching the `@magi_compile decorated regions` in Figure 1’s inference loop.

#### 2. Subgraph boundaries: isolating non-fusible ops inside compile regions

When Attention, state writes, or similar ops must live **inside** a compile region (e.g., an entire decoder layer wrapped in one decorator), the FX graph needs **explicit cuts**: mark ops Inductor must not cross using `@magi_register_custom_op(..., is_subgraph_boundary=True)` or `splitting_ops`, so those segments run eagerly or as opaque calls—the orange Eager boxes in Figure 1. Example:

```python
@magi_register_custom_op(
    "magi_compiler::flash_attn",
    infer_output_meta_fn=["q"],
    is_subgraph_boundary=True,
)
def flash_attn_func(q, k, v):
    from flash_attn import flash_attn_func as fa2
    return fa2(q, k, v)
```

#### 3. Shared configuration

When Magi is enabled, models also call during `__init__`:

- `configure_dynamo_for_magi_compile()` for unified Dynamo settings;
- `set_magi_custom_op_mode()` where needed, so relevant ops use custom-op subgraph boundaries.

Users do not configure these separately—they take effect together with `use_magi_compile: true`.

### Design principles

1. **Shared framework infrastructure, per-model boundaries**  
   Dynamo setup and custom-op mode live in `lightx2v/common/magi_custom_op_mode.py`; `@magi_compile` scope and `dynamic_arg_dims` are designed per model. See Chapter 4 for porting new models.

2. **Transparent to users**  
   Enable with `"use_magi_compile": true` in config (default `false`); if `magi_compiler` is missing, log a warning and fall back to eager.

3. **Orthogonal and composable with other optimizations**  
   Magi is an optional compile layer—it does not replace Attention/GEMM kernels, Triton, quantization, parallelism, or Offload, and can be combined with them. Whether to enable it and whether it helps should be decided by benchmark.

## Benchmarks and Recommendations

### How to enable

LightX2V’s official Docker image and `requirements` include `magi_compiler`. Set `"use_magi_compile": true` in the model config JSON and run inference as usual (e.g., `python -m lightx2v.infer ...`). Magi-related log lines confirm it is active; if the dependency is missing, a warning is printed and execution falls back to eager.

### Performance data

**Benchmark conditions**: NVIDIA H100; metric is average `infer_main` latency per DiT step.

#### [NeoPP](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) (2K FP8, three-turn dialogue)

[NeoPP](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) is a recently integrated model whose small operator chains are **not yet deeply Triton-optimized**. Tests use 2048×2048 resolution, FP8 dense, three consecutive dialogue turns in one script; KV cache grows by ~4k tokens per turn. All statistics **skip step 0**.

**Table 1** [NeoPP](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) three-turn dialogue `infer_main` latency (skip step 0)

| Turn | Magi ON | Magi OFF | Speedup |
|------|--------:|---------:|--------:|
| Turn 0 | 0.204 s | 0.250 s | 20% |
| Turn 1 | 0.240 s | 0.293 s | 17% |
| Turn 2 | 0.276 s | 0.335 s | 15% |

As Table 1 shows, Magi ON vs OFF yields a stable **~15%–20%** gain across all three turns. As dialogue progresses, Attention share rises and small-op share falls, so relative speedup eases from 20% toward 15%.

#### [Qwen Image](https://huggingface.co/Qwen/Qwen-Image-2512) (I2I, multi-resolution)

[Qwen Image](https://huggingface.co/Qwen/Qwen-Image-2512) is a heavily optimized model in LightX2V with arbitrary-resolution image generation. Tests use I2I across representative H×W sizes. All statistics **skip step 0**.

Table 2 covers four config combinations, labeled `triton_{on|off}_magi_{on|off}`. The last three columns are three comparison metrics:

- **Speedup ①**: `triton_off_magi_on / triton_off_magi_off` — Magi benefit on the Triton-OFF path
- **Speedup ②**: `triton_on_magi_off / triton_off_magi_on` — how close PyTorch + Magi gets to hand-tuned Triton (closer to 100% means closer to `triton_on_magi_off`)
- **Speedup ③**: `triton_on_magi_on / triton_on_magi_off` — marginal Magi benefit on the Triton-ON path

**Table 2** [Qwen Image](https://huggingface.co/Qwen/Qwen-Image-2512) I2I multi-resolution `infer_main` latency (skip step 0)

| H×W | triton_off_magi_off | triton_off_magi_on | triton_on_magi_off | triton_on_magi_on | Speedup ① | Speedup ② | Speedup ③ |
|-----|--------------------:|-------------------:|-------------------:|------------------:|--:|--:|--:|
| 512×512 | 467 ms | 377 ms | 350 ms | 352 ms | 1.24× | 93% | 1.00× |
| 1024×1024 | 773 ms | 636 ms | 594 ms | 596 ms | 1.22× | 93% | 1.00× |
| 1328×1328 | 1083 ms | 907 ms | 850 ms | 854 ms | 1.19× | 94% | 0.99× |
| 928×1664 | 986 ms | 824 ms | 772 ms | 776 ms | 1.20× | 94% | 0.99× |

Speedup ① stays around **~1.2×** across resolutions; ② is **~93%–94%**—Magi alone clearly narrows the gap to hand-tuned Triton but remains slightly slower; ③ is near **~1.0×**, so enabling Magi on top of Triton-ON does not yield stable extra benefit.

#### [Qwen Image](https://huggingface.co/Qwen/Qwen-Image-2512) (single-process variable resolution)

[Qwen Image](https://huggingface.co/Qwen/Qwen-Image-2512) supports arbitrary resolution. On compile-based serving paths, one worker often handles multiple H×W values; without proper dynamic-shape handling, a resolution change can trigger recompilation and sharply raise first-step latency. This section switches resolutions in one process with Magi ON, comparing first-step vs steady-state latency to verify that dynamic shapes can reuse compiled subgraphs in-process.

**Table 3** [Qwen Image](https://huggingface.co/Qwen/Qwen-Image-2512) single-process variable resolution `infer_main` latency (Magi ON)

| Phase | H×W | step 0 | steady avg |
|-------|-----|-------:|-----------:|
| A (first) | 1024×1024 | 8751 ms | 603 ms |
| B (switch) | 768×1024 | 527 ms | 522 ms |
| C (switch) | 1328×1328 | 870 ms | 867 ms |
| A2 (switch back) | 1024×1024 | 607 ms | 609 ms |

The first 1024×1024 step 0 takes **8s+**, mostly compile cost. After that, step 0 and steady avg stay close when switching resolutions; switching back to 1024×1024 does not reproduce the first-run compile spike.

### Recommendations

1. **Pre-Triton** is Magi’s main value zone—prioritize evaluation for new models or paths not yet hand-Triton-optimized.
2. Magi **significantly closes the gap** to hand-tuned Triton, but a gap remains.
3. After **post-Triton** deep manual optimization, **do not enable Magi by default**.
4. For **single-process variable resolution on compile paths**, moving from raw `torch.compile` to Magi has clear engineering value.

## Porting Magi Compiler to a New Model

This chapter is for developers integrating Magi on new models: reusable assessment items and steps. Exact partitioning varies by model—use the reference implementations below as guides, not templates.

### Pre-integration assessment

- **Does Magi match your optimization stage?**
  - If the model is not yet hand-Triton-optimized and small ops are still mostly PyTorch (**pre-Triton**), evaluate Magi first.
  - If the path is fully Triton-ized, expect limited benefit.

- **What share do small operator chains have?**
  - With very long sequences and Attention-dominated steps, end-to-end Magi speedup may be limited.

- **Which chains belong in compile vs eager islands?**
  - Dedicated hot kernels and ops Dynamo cannot stably trace or fuse are better kept as eager islands.

- **Dynamic shapes (sequence length, resolution, etc.)?**
  - If dynamic shapes are involved, declare them via `dynamic_arg_dims` and consider variable-resolution tests.

### Integration steps

#### Define compile regions

In the model’s `TransformerInfer`, pick paths with dense, fusible operator chains and decorate them with `@magi_compile`.

#### Set subgraph boundaries

For eager islands inside compile regions, cut the graph at those ops with `@magi_register_custom_op(..., is_subgraph_boundary=True)` or `splitting_ops` so Inductor does not fuse across them incorrectly.

#### Declare dynamic shapes

Configure `dynamic_arg_dims` on `@magi_compile` for tensor dimensions that vary with dynamic shapes. Avoid hiding variable dimensions in `tuple` arguments, `self.xxx`, or Python scalars, which can cause Dynamo to specialize.

#### Config and fallback

- Read `use_magi_compile` from config; call `configure_dynamo_for_magi_compile()` when true.
- Call `set_magi_custom_op_mode(True)` if the model needs custom-op boundaries.
- On init, warn and fall back to eager if `magi_compiler` is not installed.

#### Benchmark sign-off

After integration, benchmark with **production-equivalent settings** (same Triton, quantization, parallelism, etc.), comparing `use_magi_compile: true` vs `false`. Time single-step `infer_main`; for steady-state stats, **skip step 0**.

If the model has dynamic shapes (multi-resolution, variable sequence length, etc.), add a variable-resolution test: in one process, switch through representative shapes and confirm that—except for the first shape—step 0 latency matches steady state, with no abnormal recompilation.

#### Iterating compile boundaries with profiling

Optional fine-tuning: after an initial boundary layout, profile Magi ON vs OFF. If a segment regresses inside compile, a common cause is Inductor fusing across a boundary incorrectly or emitting a kernel worse than the dedicated or eager path—move that segment out of the compile region or add a subgraph boundary, then re-run sign-off.

### Reference implementations

Examples already landed in LightX2V for `@magi_compile`, `@magi_register_custom_op`, `dynamic_arg_dims`, etc.; compile granularity varies by model.

- [**SandAI LightX2V-MagiCompiler @3ed48d6**](https://github.com/SandAI-org/LightX2V-MagiCompiler/commit/3ed48d6835098643cb946f75f353710a61daf506) — Smallest change surface: `@magi_compile` directly on `infer_without_offload`, FlashAttention isolated via `@magi_register_custom_op(..., is_subgraph_boundary=True)`; suited to pre-Triton models with regular block structure.

- [**LightX2V PR #1070**](https://github.com/ModelTC/LightX2V/pull/1070) — Per-block `@magi_compile`, `dynamic_arg_dims` for sequence length and cos/sin; KV cache writes isolated with `splitting_ops` to prevent Inductor from fusing across updates.

- [**LightX2V PR #1089**](https://github.com/ModelTC/LightX2V/pull/1089) — Multiple compile segments per block (before and after Attention), RoPE via custom-op boundaries; `dynamic_arg_dims` for dual image/text streams and multi-resolution.

## Limitations

- **First-step compile latency**: enabling Magi triggers graph compilation and codegen on the first step, which is much slower than steady-state steps.
- **Mainline support**: [NeoPP](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) and [Qwen Image](https://huggingface.co/Qwen/Qwen-Image-2512) are validated and maintained in LightX2V mainline; other models require porting per Chapter 4.
- **Other unverified scenarios**: complex parallel combinations have not been systematically tested; complete numerical correctness and steady-state sign-off before production use.
