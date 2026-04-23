---
layout: post
title: "LightX2V Expands World Model Capabilities with Matrix Game 3.0 and HY-WorldMirror 2.0"
author: "LightX2V Team, Longzan Luo, Shunzi Yang"
date: 2026-04-23
tags: [World Models, Matrix Game, HY-WorldMirror, Real-time Generation, 3D Reconstruction]
---

We're excited to announce that LightX2V is expanding from video generation model inference to world modeling capabilities. LightX2V now supports two cutting-edge world models: **Matrix Game 3.0** for real-time interactive video generation and **HY-WorldMirror 2.0** for 3D world reconstruction. These integrations bring significant performance improvements and new capabilities to our unified inference platform.

## Matrix Game 3.0: Real-Time Interactive World Modeling

[Matrix Game 3.0](https://matrix-game-v3.github.io/) by Skywork AI is a memory-augmented interactive world model that achieves **up to 40 FPS real-time generation at 720p resolution**. It's the first industrial-scale deployable world model for interactive applications, supporting mouse and keyboard inputs with long-term memory consistency over minute-long sequences.

### Performance Improvements in LightX2V

Our integration ([PR #989](https://github.com/ModelTC/LightX2V/pull/989)) delivers significant performance improvements over the official implementation:

![Matrix Game 3.0 Performance]({{ site.baseurl }}/assets/world-models-blog/matrix-game-3.0.png)
*Performance comparison between LightX2V implementation and Matrix-Game official*

<video controls width="100%">
  <source src="{{ site.baseurl }}/assets/world-models-blog/matrix-game.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
*Matrix Game 3.0 real-time interactive generation demo*

Key integration benefits:
- **Enhanced memory efficiency**: Optimized attention mechanisms for long sequences
- **Improved hardware utilization**: Better performance on A800 GPUs with FlashAttention2
- **Streamlined deployment**: Unified inference APIs with existing LightX2V workflows

## HY-WorldMirror 2.0: 3D World Reconstruction

[HY-WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) by Tencent's Hunyuan team generates actual 3D assets instead of temporary video files. The ~1.2B parameter model supports both world generation from text/images and world reconstruction from multi-view inputs.

The model creates navigable 3D worlds through a 4-stage pipeline from panorama generation to final 3D scene composition for world generation, while its reconstruction capability predicts depth, surface normals, camera parameters, 3D point clouds, and 3DGS attributes in a single forward pass. It achieves state-of-the-art results with 0.012 accuracy and 0.016 completeness on the 7-Scenes dataset, along with 35% improvement in rotation accuracy compared to version 1.0.

![HY-WorldMirror Performance]({{ site.baseurl }}/assets/world-models-blog/HY-World-Mirror.png)
*Performance acceleration comparison between LightX2V implementation and baseline*

<video controls width="100%">
  <source src="{{ site.baseurl }}/assets/world-models-blog/HY-World-Mirror-2.0.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
*HY-WorldMirror 2.0 3D world reconstruction demo*

Our integration [PR #1022](https://github.com/ModelTC/LightX2V/pull/1022)) delivers significant performance acceleration with multi-GPU scaling through sequence-parallel inference for large scenes, advanced FP8 quantization support with real-time calibration, Intel XPU acceleration through SYCL/ESIMD kernels, and memory optimization via CPU offloading and lazy loading for efficient inference.

## Getting Started

Both models are available in the latest LightX2V release. Check our [documentation](https://github.com/ModelTC/LightX2V) for installation and usage examples.

**Matrix Game 3.0**: Perfect for real-time interactive applications requiring responsive world modeling.

**HY-WorldMirror 2.0**: Ideal for 3D content creation and applications requiring persistent 3D assets.

## Looking Forward

These integrations demonstrate LightX2V's commitment to bringing cutting-edge world modeling technologies to developers through optimized, production-ready implementations. We continue to push the boundaries of what's possible in AI-powered world creation.

---

*LightX2V is an open-source unified framework for X-to-Video generation. Follow us on [GitHub](https://github.com/ModelTC/LightX2V) for the latest updates.*