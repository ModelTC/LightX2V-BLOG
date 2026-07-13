---
layout: post
title: "LightX2V x FlagOS: A Plugin Design for Cross-Chip Video Generation Inference"
author: "Huiting Yu, LightX2V Team"
date: 2026-07-10
tags: [FlagOS, Plugin, Multi-Platform Deployment, FlagGems, FlagCX]
---

Video generation inference is increasingly a cross-chip deployment problem.
Modern DiT pipelines depend on optimized Attention, quantized MatMul,
normalization, RoPE, and distributed communication kernels; when the target
accelerator changes, those hot paths often need to be adapted again.

This post introduces the `lightx2v-plugin-FL` integration: an out-of-tree
LightX2V backend powered by FlagOS. It complements LightX2V's existing
per-chip in-tree platform mechanism by registering a single meta-platform:

```bash
PLATFORM=flagos
```

Under this platform, LightX2V continues to use its existing model runners,
schedulers, offload logic, and config-driven operator registries. FlagOS handles
the heterogeneous substrate through FlagGems for compute operators and FlagCX
for collective communication where available.

The integration is designed around four goals:

- **One backend, multiple chips**: expose `flagos` as a LightX2V platform while
  delegating physical chip coverage to the FlagOS stack.
- **No LightX2V fork**: keep the integration in a pip-installable plugin rather
  than editing LightX2V model code.
- **Config-driven operator selection**: reuse LightX2V's existing registry keys
  for Attention, MatMul, Norm, and RoPE.
- **Graceful fallback**: prefer FlagGems kernels, but keep torch reference paths
  for development, smoke tests, and missing operator coverage.

**Table of contents:**

- [Why a FlagOS Plugin](#why-a-flagos-plugin)
- [Integration Overview](#integration-overview)
- [Core Mechanisms](#core-mechanisms)
- [Operator Coverage](#operator-coverage)
- [Quick Start](#quick-start)
- [Practical Recommendations](#practical-recommendations)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Resources](#resources)

---

## Why a FlagOS Plugin

LightX2V already separates hardware-specific logic into `lightx2v_platform`.
That layer initializes devices, registers chip-specific operators, and lets the
upper inference stack stay mostly hardware-agnostic. This is LightX2V's original
platform mechanism and remains its main in-tree integration path: each chip is
typically supported by a dedicated backend implementation.

![Existing LightX2V in-tree platform design]({{ site.baseurl }}/assets/flagos-plugin/existing-in-tree-platform-design.png)

*Figure 1: Existing LightX2V in-tree platform design. Chip-specific device and
operator modules are maintained under `lightx2v_platform`, and each additional
chip follows the same in-tree extension pattern.*

In practice, supporting another accelerator may require a new `base/<chip>`
device adapter, corresponding `ops/*/<chip>` implementations, configuration
files, launch scripts, and numerical validation. Even when two chips run the
same video generation model, their operator APIs and communication backends may
differ, so this per-chip work can become repetitive as the hardware matrix
grows.

FlagOS is built to address this fragmentation. In the 2.1 release, it provides
a cross-chip heterogeneous AI system software stack, including:

- **FlagGems**: a large-scale operator library with broad chip coverage.
- **FlagTree**: a cross-chip compiler layer.
- **KernelGen**: an automated operator generation platform.
- **FlagScale Agent**: tooling for automated model migration, accuracy
  alignment, and performance optimization.
- **FlagCX**: a collective communication layer that can plug into PyTorch
  distributed execution.

For LightX2V inference, the key idea is simple: make FlagOS the platform
boundary. LightX2V sees `PLATFORM=flagos`; FlagOS absorbs the chip matrix below
that boundary.

## Integration Overview

`lightx2v-plugin-FL` is intentionally small. It does not replace the LightX2V
pipeline or introduce a separate inference engine. It contributes a Python
package, `lightx2v_fl`, whose `register()` function wires the FlagOS backend
into LightX2V's normal extension points.

Figure 2 shows how that boundary appears at runtime. The user selects
`PLATFORM=flagos` and the `flagos_*` operator keys through configuration, while
LightX2V continues to own the runner, scheduler, and offload workflow.

![LightX2V x FlagOS runtime path]({{ site.baseurl }}/assets/flagos-plugin/lightx2v-flagos-runtime-path.png)

*Figure 2: LightX2V x FlagOS runtime path. Registry lookup selects the FlagOS
operator adapters; the plugin then routes compute to FlagGems with torch
fallbacks, and distributed communication to FlagCX with vendor-native backend
fallbacks.*

The two lower branches isolate the hardware-dependent work. Both ultimately
target the concrete accelerator detected by the FlagOS environment, while the
upper LightX2V execution path remains unchanged.

The plugin repository is organized around the same functional boundaries as
LightX2V's platform layer:

| Path | Role |
|---|---|
| `lightx2v_fl/__init__.py` | Idempotent `register()` entry point and optional auto-registration. |
| `lightx2v_fl/device/` | `FlagOSDevice`, registered as `flagos` and `fl`. |
| `lightx2v_fl/ops/attn.py` | `flagos_flash_attn`. |
| `lightx2v_fl/ops/mm.py` | `flagos`, `flagos-fp8`, and `flagos-int8` MatMul schemes. |
| `lightx2v_fl/ops/norm.py` | `flagos_rms_norm` and `flagos_layer_norm`. |
| `lightx2v_fl/ops/rope.py` | `flagos_rope`. |
| `lightx2v_fl/configs/` | Ready-to-use Wan T2V configs for FlagOS. |
| `Dockerfile_ppu_flagos` | Example PPU-ZW810E image build with LightX2V, FlagGems, FlagCX, and the plugin. |

This structure keeps the plugin easy to audit: device, operator, config, and
packaging concerns are separate, and each operator adapter is thin.

## Core Mechanisms

### 1. A `flagos` Meta-Platform <!-- omit from toc -->

The plugin registers one LightX2V platform key:

```text
PLATFORM=flagos
PLATFORM=fl       # alias
```

`FlagOSDevice` detects the concrete torch device by asking FlagGems first, then
probing common torch backends such as `cuda`, `npu`, `mlu`, `musa`, and `xpu`.
This lets the LightX2V side stay stable even when the physical device changes.

### 2. Entry-Point Activation with an Import Fallback <!-- omit from toc -->

The package exposes a standard Python entry point:

```toml
[project.entry-points."lightx2v.platform_plugins"]
flagos = "lightx2v_fl:register"
```

Once LightX2V loads this entry-point group during platform initialization,
installing the plugin is enough. Until that hook is available in a given
checkout, users can explicitly import the plugin before entering LightX2V:

```bash
python -c "import lightx2v_fl; from lightx2v.infer import main; main()" \
  <lightx2v-infer-arguments>
```

The `register()` function is idempotent, so it is safe to call from either path.

### 3. Registry Timing: Device vs Operators <!-- omit from toc -->

LightX2V merges platform operator registries into final runtime registries at
import time. Because that merge is a snapshot, a late plugin registration into
the staging `PLATFORM_*` operator tables may be invisible to model code.

The FlagOS plugin therefore uses a split strategy:

- The **device** registers into `PLATFORM_DEVICE_REGISTER`, because
  `set_ai_device()` must find `flagos` during platform initialization.
- The **operators** register directly into the final LightX2V registries:
  `ATTN_WEIGHT_REGISTER`, `MM_WEIGHT_REGISTER`, `RMS_WEIGHT_REGISTER`,
  `LN_WEIGHT_REGISTER`, and `ROPE_REGISTER`.

Figure 3 separates registration from lookup. Importing `lightx2v_fl` first
registers `FlagOSDevice` and the `flagos_*` operators; it may also enable the
optional process-wide FlagGems patch. LightX2V then initializes the selected
platform and resolves the config-selected operators from those populated
registries.

![Registration timing in the LightX2V x FlagOS plugin]({{ site.baseurl }}/assets/flagos-plugin/registration-timing-lightx2v-flagos.png)

*Figure 3: Registration timing in the LightX2V x FlagOS plugin. Phase 1 writes
the device and operator entries; Phase 2 reads `FlagOSDevice` during platform
initialization and looks up operators in the final runtime registries.*

The `FlagOSDevice` box in Phase 2 is therefore the lookup target established in
Phase 1, not a second registration step. This split registration strategy avoids
platform-registry snapshot-order issues and lets the plugin remain an
out-of-tree package without forking LightX2V.

### 4. FlagCX First, Vendor Backend Fallback <!-- omit from toc -->

For distributed inference, the plugin prefers FlagCX when it can be imported:

```text
cpu:gloo,<device>:flagcx
```

If FlagCX is not installed, or if users set
`LIGHTX2V_FL_DISABLE_FLAGCX=1`, the plugin falls back to the vendor-native
backend:

| Torch device | Fallback backend |
|---|---|
| `cuda` | `nccl` |
| `npu` | `hccl` |
| `mlu` | `cncl` |
| `musa` | `mccl` |
| `xpu` | `ccl` |

This keeps the single-card path lightweight while giving multi-card deployments
a clean route to FlagCX.

## Operator Coverage

The current plugin focuses on the core DiT operator path used by Wan-style
video generation:

| LightX2V config field | FlagOS value |
|---|---|
| `self_attn_1_type` | `flagos_flash_attn` |
| `cross_attn_1_type` | `flagos_flash_attn` |
| `cross_attn_2_type` | `flagos_flash_attn` |
| `rms_norm_type` | `flagos_rms_norm` |
| `layer_norm_type` | `flagos_layer_norm` |
| `rope_type` | `flagos_rope` |
| `dit_quant_scheme` | `flagos`, `flagos-fp8`, or `flagos-int8` |

The implementation policy is consistent: try FlagGems first, then fall back to
a torch reference path when needed.

### Attention <!-- omit from toc -->

`flagos_flash_attn` adapts LightX2V's WAN-style attention layout and probes
FlagGems attention entry points. If the FlagGems call is unavailable or fails,
it falls back to `torch.nn.functional.scaled_dot_product_attention`.

### MatMul and Quantized MatMul <!-- omit from toc -->

The plugin registers:

- `flagos` for BF16/FP16 linear layers.
- `flagos-fp8` for FP8 per-channel symmetric stored weights.
- `flagos-int8` for INT8 per-channel symmetric stored weights.

The quantized paths reuse LightX2V's existing `MMWeightQuantTemplate` loaders,
so model weight formats remain aligned with the rest of the framework.

### Normalization and RoPE <!-- omit from toc -->

`flagos_rms_norm` and `flagos_layer_norm` call FlagGems normalization operators
when present and otherwise use torch references. `flagos_rope` adapts
LightX2V's `[S, H, D]` query/key layout to the layout expected by FlagGems,
then restores the original layout after rotary embedding.

## Quick Start

As one concrete hardware example, the plugin provides a Docker image for the
PPU-ZW810E platform:

```bash
docker pull lightx2v/lightx2v-plugin-fl:26070101-platform-ppu
```

Start a container with your model mounted. Add the device flags required by the
PPU-ZW810E runtime or your cluster scheduler.

```bash
mkdir -p outputs

docker run --rm -it \
  --ipc=host \
  --network=host \
  -v /path/to/Wan2.1-T2V-14B:/models/wan2.1-t2v:ro \
  -v "$PWD/outputs:/workspace/LightX2V/save_results" \
  lightx2v/lightx2v-plugin-fl:26070101-platform-ppu \
  bash
```

Run Wan2.1 T2V with the FlagOS config:

```bash
cd /workspace/LightX2V

export PLATFORM=flagos

python -c "import lightx2v_fl; from lightx2v.infer import main; main()" \
  --model_cls wan2.1 \
  --task t2v \
  --model_path /models/wan2.1-t2v \
  --config_json /workspace/lightx2v-plugin-FL/lightx2v_fl/configs/wan_t2v_flagos.json \
  --prompt "A cinematic shot of a futuristic research lab where a robot arm assembles a tiny glowing aircraft." \
  --negative_prompt "low quality, blurry, distorted, watermark, text" \
  --save_result_path /workspace/LightX2V/save_results/wan_t2v_flagos.mp4
```

For source installation, install LightX2V, the plugin, and the FlagOS runtime
packages in the same environment:

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
pip install -e .

cd /path/to/lightx2v-plugin-FL
pip install -e .

# Install the FlagGems build matching your chip; NVIDIA shown as an example.
pip install "flag_gems[nvidia]"
```

For FP8 inference, switch the config to:

```text
lightx2v_fl/configs/wan_t2v_flagos_fp8.json
```

For multi-card inference, install or enable FlagCX and launch with `torchrun`.
Once the LightX2V entry-point hook is available, the command can use the normal
module form:

```bash
export PLATFORM=flagos

torchrun --nproc_per_node=2 -m lightx2v.infer \
  --model_cls wan2.1 \
  --task t2v \
  --model_path /path/to/Wan2.1-T2V-14B \
  --config_json /path/to/lightx2v-plugin-FL/lightx2v_fl/configs/wan_t2v_flagos.json \
  --prompt "A high-speed train crossing a mountain bridge at sunset." \
  --save_result_path save_results/wan_t2v_flagos_dist.mp4
```

To verify plugin wiring without real accelerator kernels:

```bash
cd /path/to/lightx2v-plugin-FL
pytest -q
```

These smoke tests cover registration and torch fallback correctness. Real
FlagGems kernels, FlagCX collectives, and FP8/INT8 numerical behavior should
still be validated on the target chip.

## Practical Recommendations

### Start with the Registry Path <!-- omit from toc -->

Use the explicit FlagOS config keys first:

```json
{
  "self_attn_1_type": "flagos_flash_attn",
  "cross_attn_1_type": "flagos_flash_attn",
  "cross_attn_2_type": "flagos_flash_attn",
  "rms_norm_type": "flagos_rms_norm",
  "layer_norm_type": "flagos_layer_norm",
  "rope_type": "flagos_rope"
}
```

This path is easy to debug because operator selection is visible in the config.

### Enable Global FlagGems Patching Carefully <!-- omit from toc -->

The plugin can optionally call `flag_gems.enable()` for generic torch ops that
do not pass through LightX2V's weight-template registries:

```bash
export LIGHTX2V_FL_GLOBAL_GEMS=1
```

Because this changes torch behavior process-wide, enable it after the registry
path is already correct and benchmark it per model and chip. If a patched op
regresses, exclude it:

```bash
export LIGHTX2V_FL_GEMS_UNUSED=softmax,gelu
```

### Debug with the Smallest Checks First <!-- omit from toc -->

If `PLATFORM=flagos` is not recognized, check device registration:

```bash
python - <<'PY'
import lightx2v_fl
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

print("flagos" in PLATFORM_DEVICE_REGISTER)
print("fl" in PLATFORM_DEVICE_REGISTER)
PY
```

If an operator key is missing, check the final runtime registries:

```bash
python - <<'PY'
import lightx2v_fl
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER

print("flagos_flash_attn" in ATTN_WEIGHT_REGISTER)
print("flagos-fp8" in MM_WEIGHT_REGISTER)
PY
```

If FlagCX is not available, the plugin falls back to the vendor-native backend.
To disable FlagCX intentionally:

```bash
export LIGHTX2V_FL_DISABLE_FLAGCX=1
```

## Limitations and Next Steps

The current plugin should be treated as an MVP integration path rather than a
fully tuned backend for every FlagOS-supported chip. The important remaining
work is hardware validation:

- confirm numerical alignment for Attention, RoPE, Norm, and quantized MatMul;
- benchmark FlagGems kernels against chip-native tuned kernels;
- validate FlagCX behavior under LightX2V parallel workloads;
- expand model coverage beyond the initial Wan T2V configs;
- upstream the entry-point hook so `pip install lightx2v-plugin-fl` becomes the
  normal activation path.

Even with those caveats, the design direction is useful: LightX2V keeps owning
the video inference pipeline, while FlagOS owns the cross-chip operator and
communication layer. As FlagOS coverage expands, the same `PLATFORM=flagos`
backend can reduce repeated per-vendor integration work and make cross-chip
video generation deployment more maintainable.

## Resources

- [LightX2V x FlagOS plugin repository](https://github.com/ModelTC/lightx2v-plugin-FL/)
- [LightX2V integration pull request #1126](https://github.com/ModelTC/LightX2V/pull/1126)
