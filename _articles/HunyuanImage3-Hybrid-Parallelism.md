---
layout: post
title: "权重不动，拓扑切换：Hunyuan Image 3.0 四卡混合并行实践"
subtitle: "Weights Stay Put, Topology Changes: Phase-Aware Hybrid Parallelism for Hunyuan Image 3.0"
author: "LightX2V Team"
date: 2026-08-17
tags: [HunyuanImage3, Hybrid Parallelism, Tensor Parallelism, Sequence Parallelism, MoE]
---

## TL;DR

Hunyuan Image 3.0 的一次 T2I 或 TI2I 推理包含两个形态明显不同的计算阶段：前半段是 autoregressive generation（AR），后半段是 diffusion denoising（denoise）。AR 逐 token 生成更适合进一步切分模型权重的 Tensor Parallel（TP）；denoise 面对更长的图像 token 序列，更适合结合 Sequence Parallel （SP）分摊序列计算。如果用同一种并行方式覆盖整个推理流程，会让其中一个阶段为另一个阶段妥协。

LightX2V 在四张 GPU 上采用阶段自适应混合并行（phase-aware hybrid parallelism）：**AR prefill/decode 使用 TP4，随后切换到 TP2+SP2 完成 diffusion denoise。**

本文提出的并行切换方案能够做到不搬运权重，其依赖一个关键观察： SP 并不切分权重，denoise 的每个物理 rank 持有的是一份没有再按 SP 细分的 TP2 本地分片（local shard）；而同一物理 rank 在 AR TP4 中需要的局部权重，恰好是这份 denoise resident TP2 分片的一个 micro 子集。于是 AR 阶段只需激活子集，denoise 则激活其父 TP2 本地分片，不需要在阶段边界补齐或交换权重。

基于以上观察，本方案使模型从官方 checkpoint 直接初始化，不需要离线权重转换。设完整模型权重集合为 M：其中需要 TP 的参数按 storage TP2 形成两个基于rank的局部分片（rank-local shard）A 和 B，其他不参与 TP 的参数分别复制到 A、B。Denoise 的两个 SP rank 都需要同一套 TP2 权重，因此四张卡上的常驻权重结果为 A、B、A、B。在 AR 推理侧，驻留权重 A 和 B 分别包含两个互补的 micro-shard，即 `A = [a1, a2]`、`B = [b1, b2]`，四个 rank 依次选取 `a1、b1、a2、b2`，从而在不搬移权重的情况下构造出 TP4 计算视图；denoise 推理侧则直接使用完整 A/B shard，并启用 SP2。两个推理阶段共享同一套常驻权重，阶段边界没有模型重载、 模型权重再次打包或跨 GPU 的权重移动。

进一步地，在denoise阶段的 MoE 部分，TP2 常驻权重的 micro-major 布局不能直接由标准 FlashInfer fused-MoE 一次表达。LightX2V 因此增加了 multi-micro-shard MoE 路径：同一rank 的两个 micro 共享一次路由重排（route permutation）和专家分段偏移（expert offsets），各自完成必要的分组矩阵乘法（grouped GEMM），最后只执行一次路由权重缩放（routing-scale）和 top-k加权归并（top-k finalize）。

当前四卡测量中，混合方案同时取得两类任务中最低的 AR 每 token 时延和最低的 DiT 单步时延：

| Task | AR time/token | DiT one step |
|---|---:|---:|
| T2I hybrid | **46.333 ms** | **285.096 ms** |
| TI2I hybrid | **45.608 ms** | **336.162 ms** |

以上数据说明不同阶段采用不同并行拓扑的方向是有效的。

![Hunyuan Image 3.0 四卡混合并行总览]({{ site.baseurl }}/assets/hunyuan-image3-hybrid-parallel/figure-01-overview.png)

*图 1：官方权重 M 形成 TP2 的 A/B 分片，并以 ABAB 常驻四卡；AR 与 denoise 只切换权重视图和通信组。*

## 1. 为什么一个模型需要两套并行拓扑

Hunyuan Image 3.0 的图像生成不是单一阶段的工作流。模型先执行 AR prefill，再以逐 token 的方式进行 decode，生成后续图像合成使用的文本或思维链；之后 pipeline 构造 diffusion 输入，在图像 token 序列上重复执行多个 denoise step。两部分共享模型权重，却有不同的序列长度、调用粒度和通信特征。

### 1.1 AR：逐 token 路径需要更细的权重切分

AR prefill 一次处理完整上下文，decode 的每一步则只增加少量 token。对本文测量的四卡 Hunyuan Image 3.0 workload，继续沿序列维切分（SP）很难提供足够大的局部计算规模，通信和调度开销反而更容易占据较高比例。

TP4 将线性层、attention projection 和 MoE 权重进一步切到四个 rank，每张卡只完成对应 shard 的局部计算，再通过 collective 合并结果。对于重复执行的 decode 路径，这种方式可以降低单卡计算量，并保持完整 token 序列参与每一层计算。

### 1.2 Denoise：长图像序列给 SP 提供了空间

Denoise 阶段的图像 token 序列更长，并且相同的 Transformer block 会在多个 denoise step 中重复执行。SP 可以沿序列维度分摊 attention 和 activation 的计算与显存，同时保留 TP2 来容纳大模型权重。

在当前四卡拓扑中，TP2+SP2 让两组 TP2 分别处理一个 sequence shard。Ulysses attention 在 SP group 内通过 all-to-all 在 sequence shard 与 head shard 之间转换，完成 attention 后再变回 sequence shard。此外也支持 KV All-Gather SP：各 rank 保留本地的 Query sequence shard，通过 All-Gather 汇集完整的 K/V，在本地完成 Attention 计算后，输出仍保持 sequence shard 布局。所有 Transformer block 结束后，沿 SP group 聚合完整序列。

固定 TP4 会错过 denoise 的长序列并行机会；固定 TP2+SP2 又不能充分利用 AR TP4 的逐 token 优势。由此得到整套设计最基本的判断：最合适的并行拓扑会随推理阶段变化，优化对象应当是完整的随阶段适应的并行计划，而不是孤立地为整个模型选择一个 TP 或 SP 并行方案。

## 2. 设计目标

明确约束：

1. 资源限制为四张 H200 GPU 。

2. 直接读取官方 checkpoint，不依赖离线权重转换工具。

3. 权重只初始化为一套常驻布局。

4. AR 使用 TP4，denoise 使用 TP2+SP2。

5. 阶段切换时不重新加载、跨卡搬运或重新打包权重。

它并不是一个任意 TP/SP 层面的通用张量重分片框架，也不是根据在线请求负载实时选择并行策略的服务调度器。当前实现解决的是一个更具体的问题：在固定四卡上，让 Hunyuan Image 3.0 的两个业务阶段使用各自更合适的拓扑，同时保持一套稳定常驻权重。

## 3. 主线：利用权重子集关系完成 TP4 ↔ TP2+SP2 切换

关键问题：两种拓扑需要的权重不同，为什么切换时可以不搬权重？

原因是团队在四卡布局中发现并利用了一个恰好成立的包含关系：每张卡在 AR TP4 中需要的本地权重，恰好是它在 denoise TP2+SP2 中所持有权重的一个子集。

### 3.1 SP 权重

SP 切分的是输入序列和 activation，TP 沿参数维度切分权重。本文所提的“SP 侧权重”，指的是 denoise TP2+SP2 拓扑下，每个物理 rank 为了执行本地 sequence shard 而持有的、没有再被 SP 切分的 TP2 本地常驻权重分片。

设 denoise 的本地 TP2 分片为 A、B（它们如何从完整权重 M 生成将在第 4 节展开），并进一步写成：

```text
A = [a1, a2]
B = [b1, b2]
```

在两个 SP rank 上，A/B 各复制一份，因此四张卡常驻：

```text
rank: 0 1 2 3
denoise resident: A B A B
[a1,a2] [b1,b2] [a1,a2] [b1,b2]
```

AR TP4 需要的四个局部 shard 分别是 a1、b1、a2、b2。将每个 shard 放回它所在的 resident tensor，可以看到逐 rank 的包含关系：

| Physical rank | Denoise/SP-side resident weight | AR TP4 active weight | Containment |
|---:|---|---|---|
| 0 | A = [a1, a2] | a1 | a1 ⊂ A |
| 1 | B = [b1, b2] | b1 | b1 ⊂ B |
| 2 | A = [a1, a2] | a2 | a2 ⊂ A |
| 3 | B = [b1, b2] | b2 | b2 ⊂ B |

记物理 rank r 上的 denoise 阶段常驻权重为 `W_res(r)`，AR active weight 为 `W_AR(r)`，则：

```text
W_AR(r) = micro_view(W_res(r), local_micro_id(r))
W_AR(r) ⊆ W_res(r)
```

对参与 TP 的 projection，AR 使用的是严格更小的 micro 子集；对 norm、router 等 replicated 参数，两阶段使用的是同一份参数。因此从完整 rank-local 参数集合看，用 `W_AR(r) ⊆ W_res(r)` 表示更准确。

这就是混合并行可以零权重迁移切换的基础。AR 阶段只暴露父 tensor 中的一个 micro view；进入 denoise 后，不需要从其他 rank 取回缺失部分，而是直接把当前 active view 扩展回本卡早已常驻的完整 A 或 B。

### 3.2 为什么这个子集关系恰好能覆盖四个 AR shard

当前三个并行度满足：

```text
storage TP size = 2
AR TP size = 4
denoise SP size = 2

micro_shard_count
= AR TP size / storage TP size
= 4 / 2
= 2
= denoise SP size
```

一个 storage TP2 shard 恰好能再分为两个 TP4 micro-shard，而它又恰好在两个 SP rank 上各有一个 replica。因此，可以让：

```text
SP rank 0 replica → 选择 micro0
SP rank 1 replica → 选择 micro1
```

这样，同一个 storage shard 的两个 replica 在 denoise 中内容相同，在 AR 中却分别激活不同 micro，合起来刚好覆盖 TP4 所需的全部 shard。代码会显式校验：

```text
AR_TP / storage_TP == denoise_SP
```

这不是一个偶然的 rank 编号技巧，而是这套无搬运布局必须满足的结构约束。对于其他并行度，只有仍能建立相同的 replica-to-micro 一一映射时，才能沿用这套方法。

### 3.3 Phase switch 实际做了什么

完整执行顺序如下。上方是实际计算阶段，底部绿色泳道表示 ABAB resident weights 在整个 pipeline 中始终驻留于原地址。

![从权重加载到 AR、denoise 和 VAE decode 的阶段时间线]({{ site.baseurl }}/assets/hunyuan-image3-hybrid-parallel/figure-02-phase-timeline.png)

*图 2：Phase switch 激活新的本地权重视图和通信拓扑，但不改变常驻权重地址。*

这一过程不会创建新的 process group，也不会创建、复制或重排权重 tensor。权重侧的切换只是从 `W_AR(r)` 这个 micro 子集回到已经常驻的 `W_res(r)` 父 tensor。

![逐 rank 权重子集关系与两阶段通信组]({{ site.baseurl }}/assets/hunyuan-image3-hybrid-parallel/figure-03-local-subset-topology.png)

*图 3：每个 AR micro-shard 已经包含在本卡 denoise TP2 shard 中；四张卡不变，process group 随阶段切换。*

## 4. 从完整权重 M 到 TP2 常驻权重分片，再到 TP4 视图

只解决通信组切换还不够。Hunyuan Image 3.0 的 80B 级模型体积决定了权重不能在四张卡上完整复制。为了看清 A、B 和 ABAB 从哪里来，先从官方 checkpoint 中的完整模型权重 M 开始。

### 4.1 完整权重 M 如何拆成 A 和 B

这里用 M 表示模型的完整参数集合，而不是某一个矩阵。便于理解，把 TP2 权重抽象为：

```text
complete model weights M
│
├── storage TP rank 0 → A
└── storage TP rank 1 → B
```

A 和 B 各自都是一张 GPU 上可以参与模型执行的“本地参数集合”：二者在 TP 参数部分持有互补 shard，在 replicated 参数部分持有相同副本。只有把 A、B 中的 TP shard 按各层语义组合起来，才还原完整权重 M 的计算。

不同参数不能使用同一个机械切分方向。设官方 checkpoint 中某个线性层权重为 W：

- Column-parallel projection 沿输出维切分。概念上可写成 `W_A = W[0:O/2, :]`、`W_B = W[O/2:O, :]`。

- Row-parallel projection 沿输入维切分。概念上可写成 `W_A = W[:, 0:I/2]`、`W_B = W[:, I/2:I]`。

- Fused QKV 按完整 KV-head group 分配，不能在一个 KV group 中间截断。

- Fused gate/up 既要按输出维切分，还要保持 gate 与 up 的成对语义。

- Norm、MoE router 等没有声明 TP split type 的参数保持 replicated，而不是强行切成两半。

因此，“完整权重 M 拆成 A/B”表示的是一次按算子语义完成的 TP2 参数组织，而不是一种统一的物理切法。

### 4.2 为什么 A/B 在四卡上变成 ABAB

Denoise 使用 TP2+SP2。SP 切分的是序列和 activation，不会进一步切分模型权重；因此每一个 SP rank 都必须拥有一套完整的 TP2 rank pair。四卡 mesh 中：

```text
storage TP rank
0 1
SP rank 0 / mesh row 0 GPU0 A GPU1 B
SP rank 1 / mesh row 1 GPU2 A GPU3 B
```

物理 rank 到 storage TP rank 的关系为：

```text
storage_tp_rank = physical_rank mod 2
```

所以 rank 0、2 取得 TP rank 0 的本地参数集合 A，rank 1、3 取得 TP rank 1 的本地参数集合 B，沿物理 rank 顺序观察就是：

```text
physical rank: GPU0 GPU1 GPU2 GPU3
resident shard: A B A B
```

ABAB 不是额外设计出来的一种 checkpoint 格式，而是“storage TP2 沿每个 SP replica 复制”之后自然得到的四卡驻留结果。模型仍然从同一份官方 checkpoint 读取；每个进程根据自己的 storage TP rank 选取 A 或 B。

Denoise 激活后，TP group `[0,1]` 使用第一套 A/B，TP group `[2,3]` 使用第二套 A/B；SP group `[0,2]` 连接两个 A rank，SP group `[1,3]` 连接两个 B rank。这样才能在 SP 通信两端保持相同的 TP shard 语义。

### 4.3 从 A/B 暴露 TP4 micro view

为了让 AR 使用 TP4，每个 TP2 shard 在逻辑上继续分为两个 micro-shard：

```text
A = [a1, a2]
B = [b1, b2]
```

AR 阶段每个物理 rank 选择一个本地 micro view：

| Physical rank | Resident shard | Local micro | AR active view | Logical TP4 rank |
|---:|---|---:|---|---:|
| 0 | A | 0 | a1 | 0 |
| 1 | B | 0 | b1 | 2 |
| 2 | A | 1 | a2 | 1 |
| 3 | B | 1 | b2 | 3 |

注意区分物理执行顺序和权重的规范逻辑顺序：

```text
physical rank order: a1, b1, a2, b2
Logical TP order: a1, a2, b1, b2
```

逻辑 TP rank 由下面的关系得到：

```text
logical_tp_rank
= storage_tp_rank × micro_shard_count
+ local_micro_shard_id
```

当前 storage TP size 和 micro-shard count 都为 2，所以物理 rank 0、1、2、3 分别映射到逻辑 TP rank 0、2、1、3。

对于 row-parallel output 的 sum all-reduce，参与者顺序不会改变求和结果；但 vocabulary logits 等有序 all-gather 必须按照 `(0, 2, 1, 3)` 重新排列物理输出，才能恢复正确的词表 shard 顺序。

### 4.4 普通线性层如何切换视图

QKV 不能简单按原始 fused 维度任意切分，而是按完整 KV-head group 选择 storage shard。Fused gate/up 的官方顺序为 `[gate_all, up_all]`，初始化阶段会组织成：

```text
[gate_micro0, up_micro0, gate_micro1, up_micro1]
```

这样 AR 的一个 micro projection 可以作为连续 view 取得，而 denoise 可以在同一个常驻权重上使用两个 micro。

![从官方权重到 ABAB 常驻布局以及两阶段逻辑视图]({{ site.baseurl }}/assets/hunyuan-image3-hybrid-parallel/figure-04-weight-views.png)

*图 4：AR 需要把物理 rank 顺序映射为 canonical TP4 顺序；denoise 直接使用完整 A/B shard。*

## 5. MoE 布局为什么成为关键障碍

在工程实现中，混合并行中较难的部分不是通信组切换，而是 denoise 如何高效消费为 AR TP4 准备的 micro-shard MoE 权重。

每个 storage TP rank 上，MoE resident pack 为：

```text
W1: [micro=2, expert=64, 1536, 4096]
W2: [micro=2, expert=64, 4096, 768]
```

它们的物理线性顺序是 micro-major：先存放 micro0 的全部 64 个 experts，再存放 micro1 的全部 64 个 experts；而标准 TP2 expert view 需要按 expert 组合 micro0 与 micro1。

![MoE micro-major 物理布局与 expert-major 逻辑视图]({{ site.baseurl }}/assets/hunyuan-image3-hybrid-parallel/figure-05-moe-layout.png)

*图 5：同一 expert 的两个 micro 在物理内存中并不相邻，无法仅靠普通 view 或 stride 构造标准 TP2 expert-major 布局。*

整个 pack 是连续 tensor，但同一 expert 的两个 micro 并不是一个标准 expert-major 连续权重。普通 `view` 或 stride 变换无法在不复制数据的情况下，将 micro-major layout 伪装成每个 expert 拥有完整 intermediate dimension 的布局。

一种做法是在 denoise 前重新拼接权重，但这需要额外显存和阶段切换复制。另一种做法是把 micro0 和 micro1 分别交给一次 FlashInfer fused-MoE，再将两个结果相加。两份 GEMM1/GEMM2 是数学上必要的，但两个完整的 fused-MoE invocation 还会重复：

- route permutation；

- expert offsets；

- dispatch/finalize workspace；

- route order restoration；

- routing score；

- top-k output reduction。

真正需要消除的不是两个 micro 的 GEMM，而是重复的 dispatch 与 finalize。

## 6. Multi-micro-shard fused MoE

LightX2V 为 denoise 增加了独立接口 `lightx2v_multi_micro_fused_moe`。它不重新排列 resident weight，而是让两个 micro 共享同一套路由映射。

### 6.1 一次 routing，两个 micro 共享

下面将两次独立 FlashInfer dispatch 与 shared-routing multi-micro 路径并排展示。两侧都保留四次必要 GEMM，右侧只消除重复的 route preparation 和 finalize。

![两次 FlashInfer dispatch 与 multi-micro shared routing 的数据流对比]({{ site.baseurl }}/assets/hunyuan-image3-hybrid-parallel/figure-06-multi-micro.png)

*图 6：Multi-micro 共享路由元数据并只 finalize 一次；一次逻辑接口调用不等于一个 CUDA kernel。*

Router 产生形状为 `[num_tokens, top_k]` 的 expert id 和 routing score。算子将 expert id 展平，按 expert 排序并生成 `permuted_to_expanded`。这个 permutation 同时决定排序后 route 属于哪个原始 token，并让相同 expert 的输入在 grouped GEMM 中形成连续区间。

随后，`bincount(...).cumsum(...)` 生成 64 个 expert 的 offsets。输入只按这一 permutation 执行一次 `index_select`；micro0 和 micro1 共享相同的 permuted input、route permutation 和 offsets。

每个 micro 都执行 `grouped GEMM1 → split gate/up → SiLU(up) × gate → grouped GEMM2`。

两个 GEMM2 产生相同 shape 的 BF16 partial output。Triton 路径先建立排序逆映射，再在 finalize kernel 中按原始 token 和 top-k slot 读取两个 partial：

```text
output[token]
= Σroute routing_scale[token, route]
× (micro0_partial + micro1_partial)
```

两个 partial 先以 FP32 相加，routing score 只应用一次，top-k reduction 也只完成一次。

### 6.2 “一次调用”不等于“一个 CUDA kernel”

Denoise 上层只调用一次 LightX2V 逻辑 MoE 接口，但该接口内部仍包含：

- route sort、count 和 input indexing kernel；

- 两次 grouped GEMM1；

- SwiGLU；

- 两次 grouped GEMM2；

- permutation inverse；

- Triton finalize。

Multi-micro 路径共享一次 routing metadata，并只执行一次最终归并。它不是一个单 CUDA kernel，也不是对 `flashinfer_cutlass_fused_moe` 的复写或包装。Denoise 的自定义路径不会调用官方 FlashInfer fused-MoE；AR 每个 rank 只使用一个 micro-shard，仍然调用官方 FlashInfer CUTLASS fused-MoE。

### 6.3 工程实现

入口代码位于：

- `lightx2v/common/ops/moe/multi_micro_fused_moe.py`

- `lightx2v/models/networks/hunyuan_image3/weights/common.py`

- `lightx2v/models/networks/hunyuan_image3/infer/transformer_infer.py`

## 7. AR 与 denoise 的完整数据流

### 7.1 AR TP4

AR 不使用 SP，每个 rank 都处理完整 token 序列，并在 TP4 group 内完成输出归并、logits gather 和 logical shard reorder。

AR KV cache 始终属于 AR TP4 路径，不会在阶段边界迁移到 denoise topology。

### 7.2 Denoise TP2+SP2

Denoise 入口会沿 sequence dimension 切分 hidden states、position ids、rotary embedding 和 attention mask。Attention 可以选择 Ulysses all-to-all 或 KV All-Gather 两种 SP 实现。

每个 TP2 pair 使用完整的 A/B resident shard，MoE 走 multi-micro 路径。所有 Transformer block 完成后，再沿 SP group all-gather 恢复完整图像序列。

当前配置为 `cfg_p_size=1` 和 `cfg_mode=serial`。四张卡已经全部用于 TP2×SP2，因此 conditional 与 unconditional 两条 CFG branch 顺序执行，而不是再划出 CFG parallel rank。

![AR TP4 与 denoise TP2SP2 的完整数据流]({{ site.baseurl }}/assets/hunyuan-image3-hybrid-parallel/figure-07-phase-dataflow.png)

*图 7：AR 保留完整 token 序列并切分权重；denoise 切分 sequence activation，并支持 Ulysses 与 KV All-Gather 两种 SP attention。*

## 8. T2I 与 TI2I 如何共用这套运行时

T2I 与 TI2I 共享：

- phase-aware parallel context；

- ABAB resident layout；

- AR TP4；

- denoise TP2+SP2；

- multi-micro MoE；

- KV cache 与 serial CFG。

TI2I 还需要处理参考图像，并让 condition tensor 的生命周期跨越 AR→denoise 的 topology switch：

![TI2I 条件编码、复用与两阶段执行流程]({{ site.baseurl }}/assets/hunyuan-image3-hybrid-parallel/figure-08-ti2i-flow.png)

*图 8：图像条件只编码一次，在 AR TP4 中参与 COT 生成，随后复用到 serial CFG 和 TP2+SP2 denoise。*

## 9. 评测方法

本文使用权重加载完成后的同步推理时延作为主要指标，按阶段报告：

- AR prefill，单位 ms；

- AR decode 三轮总时间，单位 s；

- AR time per token，单位 ms；

- DiT one step，单位 ms；

- ALL，单位 s。

此外，T2I prefill 长度固定为 1251，TI2I prefill 长度固定为 6290。AR decode 各方案实际生成 token 数不同，因此 decode 总时间和 ALL 会受到 token trace 影响；故补充 AR time per token 作为更精确的指标：

```text
time per token
= three-round decode total time
/ three-round generated token count
```

### 9.1 实验环境

Hardware: 4×H200-140GB；Warmup/measurement: 2/3；weight loading and file saving are excluded.

## 10. 基础并行方案对比

### 10.1 T2I

T2I prefill 长度为 1251。

| 4 GPUs | AR prefill (ms) | AR decode, 3 rounds (s) | Time/token (ms) | Tokens | DiT one step (ms) | ALL (s) |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 68.021 | 97.191 | 47.480 | 2047 | 310.748 | 49.125 |
| TP2SP2 | 76.352 | 106.745 | 55.713 | 1916 | 288.530 | 51.158 |
| AR TP4 + DiT TP2SP2 | **66.485** | 104.250 | **46.333** | 2250 | **285.096** | 50.041 |

相对固定 TP4，混合方案：

- AR prefill 快约 2.26%；

- AR 每 token 快约 2.42%；

- DiT 单步快约 8.25%。

相对固定 TP2SP2，混合方案：

- AR prefill 快约 12.92%；

- AR 每 token 快约 16.84%；

- DiT 单步快约 1.19%。

需要注意，混合方案三轮生成了 2250 tokens，而 TP4 为 2047，因此混合方案的 decode 总时间反而更长，ALL 也不能直接用于判断纯并行效率。“AR 快 16.84%”准确指的是相对 TP2SP2 的归一化 per-token latency，而不是三轮 decode 总时间。

### 10.2 TI2I

TI2I prefill 长度为 6290。

| 4 GPUs | AR prefill (ms) | AR decode, 3 rounds (s) | Time/token (ms) | Tokens | DiT one step (ms) | ALL (s) |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 229.588 | 63.614 | 49.621 | 1282 | 357.468 | 40.731 |
| TP2SP2 | **208.489** | 78.521 | 55.871 | 1406 | 336.527 | 44.588 |
| AR TP4 + DiT TP2SP2 | 228.162 | **62.893** | **45.608** | 1379 | **336.162** | **39.410** |

相对固定 TP4，混合方案：

- prefill 快约 0.62%；

- AR 每 token 快约 8.09%；

- DiT 单步快约 5.96%；

- 本次 ALL 观测值快约 3.24%。

相对固定 TP2SP2，混合方案：

- prefill 慢约 9.44%；

- AR 每 token 快约 18.37%；

- DiT 单步基本持平并略快约 0.11%；

- 本次 ALL 观测值快约 11.61%。

TI2I 的长 prefill 中包含图像条件相关输入，因此 TP2SP2 在本次 prefill 测量中更快。混合方案的价值不是保证所有子阶段单点第一，而是同时取得更好的 AR decode per-token 和 denoise step，并改善该次端到端观测值。

### 10.3 Multi-micro MoE 消融：共享路由与两次 FlashInfer dispatch

为了单独验证 multi-micro-shard MoE 的收益，我们增加了一组 denoise 消融实验。两组实现保持完全相同的四卡拓扑、resident weights 和业务流程：AR 均使用 TP4，denoise 均使用 TP2+SP2，权重均以 storage TP2 的 ABAB 形式常驻，CFG 均串行执行。唯一变化是 denoise 如何消费每张卡上的两个 micro-shard：

- Multi-micro 路径共享一次 route permutation、expert offsets 和 permuted input，完成四次必要的 grouped GEMM 后，只执行一次 routing-scale 和 top-k finalize；

- Two-dispatch 路径对两个 active micro-shard 分别调用一次官方 `flashinfer_cutlass_fused_moe`，再在本卡相加两个输出，最后仍只执行一次 TP2 all-reduce。

T2I ：

| Denoise MoE implementation | AR prefill (ms) | AR decode, 3 rounds (s) | Time/token (ms) | Tokens | DiT one step (ms) | ALL (s) |
|---|---:|---:|---:|---:|---:|---:|
| Multi-micro shared routing | 66.485 | 104.250 | **46.333** | 2250 | **285.096** | 50.041 |
| Two FlashInfer dispatches | 67.192 | 96.064 | 47.021 | 2043 | 309.167 | **48.644** |

T2I 中，两次 FlashInfer dispatch 的 DiT 单步时延比 multi-micro 高 8.44%；等价地说，multi-micro 将单步时延从 309.167 ms 降至 285.096 ms，降低 7.79%。Two-dispatch 的 ALL 看似更低，但它三轮只生成了 2043 个 decode tokens，而 multi-micro 生成了 2250 个，不能把这 1.397 s 的差值归因于 denoise 实现。

TI2I：

| Denoise MoE implementation | AR prefill (ms) | AR decode, 3 rounds (s) | Time/token (ms) | Tokens | DiT one step (ms) | ALL (s) |
|---|---:|---:|---:|---:|---:|---:|
| Multi-micro shared routing | 228.162 | **62.893** | **45.608** | 1379 | **336.162** | **39.410** |
| Two FlashInfer dispatches | **227.934** | 64.680 | 46.599 | 1388 | 352.516 | 40.821 |

TI2I 中，两次 FlashInfer dispatch 的 DiT 单步时延比 multi-micro 高 4.86%；multi-micro 将单步时延从 352.516 ms 降至 336.162 ms，降低 4.64%。两组 decode token 数接近，分别为 1388 和 1379；在这次观测中，two-dispatch 的 ALL 也高 3.58%。

两项任务的结果共同说明，multi-micro 路径的收益不是来自减少 MoE 的 GEMM 数量。两个 micro 各自的 GEMM1 和 GEMM2 仍然必须执行；节省的是第二次完整 FlashInfer invocation 所重复的 dispatch、workspace 和 finalize 开销。

## 11. 与其他并行优化方案的对比

统一测速文档还包含 FlexTP、Flying Serving、Moebius 和 ReMP 的 LightX2V 接入结果。

### 11.1 T2I 扩展结果

| Scheme | AR prefill (ms) | AR decode, 3 rounds (s) | Time/token (ms) | Tokens | DiT step (ms) | ALL (s) |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid TP4 → TP2SP2 | 66.485 | 104.250 | **46.333** | 2250 | **285.096** | 50.041 |
| FlexTP, threshold=4096 | 86.744 | 95.725 | 53.899 | 1776 | 286.860 | 47.478 |
| Flying Serving, low load | 66.477 | 96.232 | 47.196 | 2039 | 309.755 | 48.774 |
| Flying Serving, high load | 86.107 | 93.679 | 52.747 | 1776 | 451.376 | 54.974 |
| Moebius, low load | 66.643 | 97.830 | 46.409 | 2108 | 310.476 | 49.300 |
| Moebius, high load | 325.419 | 143.777 | 63.394 | 2268 | 1804.841 | 139.599 |
| ReMP, low concurrency | 76.178 | 97.795 | 48.897 | 2000 | 312.005 | 48.789 |
| ReMP, medium/high concurrency | 216.048 | 224.398 | 115.194 | 1948 | 1122.923 | 132.600 |
| ReMP, high concurrency | 343.259 | 227.224 | 109.929 | 2067 | 1743.826 | 166.041 |

### 11.2 TI2I 扩展结果

| Scheme | AR prefill (ms) | AR decode, 3 rounds (s) | Time/token (ms) | Tokens | DiT step (ms) | ALL (s) |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid TP4 → TP2SP2 | 228.162 | 62.893 | **45.608** | 1379 | **336.162** | 39.410 |
| FlexTP, threshold=4096 | 228.266 | 64.891 | 46.886 | 1384 | 336.898 | 40.098 |
| Flying Serving, low load | 227.879 | 61.026 | 46.907 | 1301 | 356.061 | 39.777 |
| Flying Serving, high load | 331.095 | 73.283 | 53.104 | 1380 | 533.434 | 52.770 |
| Moebius, low load | 228.038 | 59.077 | 46.554 | 1269 | 355.491 | 39.119 |
| Moebius, high load | 1353.773 | 79.739 | 62.541 | 1275 | 2014.439 | 130.059 |
| ReMP, low concurrency | 239.724 | 66.404 | 47.534 | 1397 | 370.800 | 41.783 |
| ReMP, medium/high concurrency | 922.188 | 142.730 | 113.008 | 1263 | 1273.305 | 114.853 |
| ReMP, high concurrency | 1465.401 | 155.855 | 110.692 | 1408 | 2337.071 | 175.428 |

从当前数据可以得出：

1. 混合方案在 T2I 和 TI2I 中都取得了最低的 AR time/token。

2. 混合方案在两个任务中都取得了最低的 DiT one-step latency。

## 12. 配置与使用

混合并行配置的核心字段如下：

```json
{
"moe_impl": "flashinfer",
"flashinfer_multi_micro": true,
"flashinfer_multi_micro_backend": "grouped_mm",
"parallel": {
"phase_aware": true,
"storage_tensor_p_size": 2,
"ar": {
"tensor_p_size": 4,
"seq_p_size": 1
},
"denoise": {
"tensor_p_size": 2,
"seq_p_size": 2
},
"cfg_p_size": 1,
"seq_p_attn_type": "ulysses",
"cfg_mode": "serial"
}
}
```

T2I：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/hunyuan_image3/run_hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_multi_micro_flashinfer.sh
```

TI2I：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/hunyuan_image3/run_hunyuan_image3_ti2i_ar_tp4_denoise_tp2_sp2_multi_micro_flashinfer.sh
```

## 13. 总结

Hunyuan Image 3.0 的并行难点不在于选出一个全局最优的 TP 或 SP degree，而在于两个业务阶段需要不同的计算拓扑：AR 使用 TP4，denoise 使用 TP2+SP2。

LightX2V 首先将官方完整权重 M 按各层 TP 语义组织成 storage-TP2 的 A/B；由于 denoise 包含两个 SP replica，这套 A/B 在四张卡上自然形成 ABAB，并成为唯一 resident layout。方案能够成立的关键，是每张卡的 AR TP4 shard 恰好是其 denoise resident A/B shard 的一个 micro 子集。AR 激活这个子集，denoise 激活已经常驻的完整父 shard，再切换到预创建的 TP2+SP2 process groups。阶段边界只改变 active topology 和 weight view，不重新加载或搬运 resident weights。

MoE 是这套设计真正需要额外适配的部分。Multi-micro 路径保留 micro-major 存储，通过共享 route permutation、expert offsets 和最终归并，使 denoise 能直接消费两个 micro，而不必重排权重或执行两套完整 dispatch。

当前结果表明，这种设计在 T2I 和 TI2I 中同时获得了更低的 AR 每 token 时延和 DiT 单步时延。更重要的是，它给出了一条适用于统一生成模型的工程思路：不要强迫异构阶段共享同一并行策略，而应让通信拓扑和计算视图随 phase 切换，同时尽可能保持昂贵的模型状态常驻不动。

## References

1. [LightWan2.2-A14B: High-performance sparse MoE video generation](https://light-ai.top/LightX2V-BLOG/posts/LightWan22-A14B/)

2. [Parallel Mechanism of LightX2V](https://light-ai.top/LightX2V-BLOG/posts/Parallel/)

3. [LightX2V Hunyuan v16.2 branch](https://github.com/Chernobyllight/LightX2V/tree/lightx2v-hunyuan-v16.2)

4. FlexTP

5. [Flying Serving, arXiv:2602.22593](https://arxiv.org/abs/2602.22593)

6. [Moebius, arXiv:2606.26607](https://arxiv.org/abs/2606.26607)

7. [ReMP, arXiv:2606.18741](https://arxiv.org/abs/2606.18741)
