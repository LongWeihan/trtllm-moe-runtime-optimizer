# TensorRT-LLM MoE-First Runtime 项目落地实施计划（面向当前硬件）

## 1. 项目定位

### 项目标题

**MoE-Aware Runtime Scheduling Enhancement for TensorRT-LLM**

中文表达：

**面向 MoE 模型的 TensorRT-LLM Runtime 调度增强：基于 expert/rank pressure 的推理执行优化**

### 一句话定位

本项目聚焦 **MoE 模型推理的 runtime 问题**，不是泛化的 LLM serving 调度优化，更不是重写一个迷你 TensorRT-LLM。目标是在 TensorRT-LLM PyTorch backend 的真实 executor/scheduler 架构内，针对 MoE inference 中的 **expert/rank pressure、decode tail、batch straggler 和 KV 资源竞争**，实现一个可插入的 **MoE-aware scheduler enhancement**。

### 这个项目到底在解决什么

MoE 模型在 serving 时的 step latency，不仅由 token 数量决定，还受到以下因素影响：

- 请求路由到哪些 experts。
- 某些 experts / 某些 ranks 是否被打热。
- dispatch / combine / MoE auxiliary path 是否在某个 step 中被放大。
- chunked prefill 是否和高压 decode step 发生竞争。
- KV cache 资源是否让 scheduler 被迫作出更差的 admission 决策。

默认 scheduler 通常不直接感知这些 **MoE-specific runtime signals**。  
本项目的核心就是：

> **让 TensorRT-LLM 的 runtime scheduler 在 batch selection 和 admission 时感知 MoE pressure，从而缓解 MoE 推理中的 decode tail 和 straggler 问题。**

---

## 2. 你的硬件是否支持这个项目

### 当前硬件实测

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti |
| 显存 | 16380 MiB |
| 驱动 | 591.86 |
| Host CUDA | 13.1 |
| CPU | AMD Ryzen 9 7950X, 16C/32T |
| 内存 | 约 127 GB 级别 |
| Host OS | Windows 11 专业版 |
| WSL | Ubuntu-24.04, WSL2 |
| WSL Python | Python 3.12.3 |
| WSL GPU 可见性 | 已确认可用 |
| `nsys` | 已确认可用 |

### 结论

这台机器**适合做 MoE-first runtime 项目**，但前提是：

- 主目标不是本地部署大型生产级 MoE；
- 主目标是 **MoE pressure-aware runtime scheduler**；
- 本地验证方式以 **`Qwen/Qwen1.5-MoE-A2.7B-Chat` 的真实 TensorRT-LLM 路径 + routing trace + synthetic skew workload** 为主；
- TensorRT-LLM 真实 backend integration 是必须的；
- 多 GPU EP / NVLink / WideEP / DWDP 不在当前机器的主实施范围内。

换句话说，这台机器最适合做的是：

> **基于 TensorRT-LLM 真实 runtime 的单 GPU MoE-aware scheduler enhancement，并通过可控的 expert/rank pressure workload 验证其效果。**

---

## 3. 项目故事线（MoE-first 版本）

这个项目的故事线必须非常清晰：

### 不是这个故事

- 我先做一个 dense 调度项目，再顺便提一点 MoE。
- 我用 dense 模型跑 benchmark，然后加一个泛化的 chunked prefill scheduler。
- 我做的是“通用 mixed workload 优化”，MoE 只是其中一个 workload class。

### 而是这个故事

> **MoE 推理的 runtime 问题，与 dense 模型不一样。**

具体来说：

1. 同样的 token 数，不同请求可能带来不同 expert/rank pressure。
2. 默认 scheduler 在 batch 组成时，通常不感知这种压力差异。
3. 因此在 mixed serving workload 下，容易出现：
   - decode tail 放大
   - 某些 step latency 尖峰
   - 高压请求叠加导致 batch straggler
4. 所以我要在 TensorRT-LLM runtime 内部引入 **MoE pressure-aware scheduling**。

### 最终你要讲出来的是

> 我做的不是一般的 serving scheduler，而是一个针对 MoE inference runtime 的 TensorRT-LLM scheduler enhancement。它利用 expert/rank pressure signal，在不改变模型输出、不重写 kernel 的前提下，改善 MoE 推理中的 tail latency 和 batch stability。

---

## 4. 最终项目边界

## 4.1 必须坚持的边界

这个项目必须是：

- 基于 TensorRT-LLM 真实 backend。
- 基于 TensorRT-LLM 的 scheduler / resource management 接口做增强。
- 不重写 executor。
- 不重写 attention / MoE kernel。
- 不做 mock serving simulator 作为主成果。
- 不做外部“再套一层策略器”作为主实现。

也就是说，主实现应是：

```text
TensorRT-LLM PyTorch Backend
  -> CapacityScheduler enhancement
  -> MicroBatchScheduler enhancement
  -> MoE telemetry / pressure signal integration
```

## 4.2 当前机器上不作为主目标的内容

- 大模型 MoE full deployment
- 多 GPU expert parallel
- AllToAll / one-sided AllToAll
- KV transfer / disaggregated serving
- kernel 重写

这些内容可以理解、分析、在面试中讨论，但**不纳入当前机器项目的主实现目标**。

---

## 5. 本机最合理的验证思路

## 5.1 核心思想

你的主线不是“跑最大的 MoE”，而是：

> **构造和捕捉 MoE runtime pressure，然后让 TensorRT-LLM scheduler 用这些信号做更优 batch selection。**

因此，本机验证路径分成三层：

### 层 1：真实 TensorRT-LLM runtime integration

这个层必须有，负责证明：

- 你不是在做 toy scheduler；
- 你是真的在 TensorRT-LLM backend 里改 scheduler。

### 层 2：MoE pressure signal 构建

这个层是项目灵魂，负责证明：

- 你真的在做 MoE-specific optimization；
- 而不是泛化 chunked prefill 调参。

### 层 3：MoE-skew workload 验证

这个层负责证明：

- MoE-aware scheduler 在高压 workload 下比默认 scheduler 更稳；
- 重点看 tail latency 和 step variance，而不是平均吞吐 alone。

---

## 6. 本机版项目的技术主线

## 6.1 主技术问题

在 TensorRT-LLM runtime 的每个 scheduling step 中，需要决定：

- 哪些 generation requests 先执行；
- 是否插入 context chunks；
- 当前 step 还能承受多大的 prefill 压力；
- 某个请求是否会把当前 step 的 MoE pressure 推得太高；
- 当前 KV 资源是否足以支持更激进的 admission。

默认调度常见关注的是：

- fit 不 fit；
- decode 优先级；
- chunked prefill 的吞吐收益；

而本项目新增的关键问题是：

> **这个请求会不会让当前 batch 的 MoE pressure 失衡，从而放大 decode tail？**

## 6.2 核心创新点

### 创新点 1：MoE Pressure Model

为 request / batch 定义 pressure score，用于近似刻画：

- hot experts
- hot ranks
- MoE dispatch/combine 放大风险
- step latency variance 风险

pressure 可以来自：

- tiny MoE routing trace
- synthetic routing profile
- replayed expert histogram

### 创新点 2：MoE-Aware MicroBatch Scheduling

在每个 step 做 batch selection 时，除了 decode-first，还引入：

- 压力阈值控制
- 高压请求分散
- 低压 chunk opportunistic 插入

### 创新点 3：MoE-Aware Admission

在 capacity / admission 时，不只考虑 KV block，还考虑：

- 当前 decode step 的 MoE pressure
- 新请求的预估 pressure
- admission 是否会诱发后续 step 尖峰

### 创新点 4：MoE Telemetry Hook

引入 scheduler 可消费的运行时 MoE 信号，而不是只看静态 prompt length。

---

## 7. 环境与开发路线

## 7.1 主环境

**WSL2 Ubuntu 24.04**

原因：

- TensorRT-LLM 官方支持 Linux
- 你的 WSL 已确认可见 GPU
- 当前机器已具备 Python、`nvcc`、`nsys`

## 7.2 主安装路线

**TensorRT-LLM Python-only editable install**

原因：

- 当前项目重点是 Python backend scheduler
- 先避免全量 C++ 构建成本

安装原则：

- 优先使用官方 Linux pip 安装文档
- 优先使用 `TRTLLM_USE_PRECOMPILED=1 pip install -e .`

## 7.3 代码必须放在哪里

建议：

```bash
~/workspace/TensorRT-LLM
~/workspace/trtllm-moe-runtime-exp
```

不要把主仓库放在 `/mnt/c/...` 下。

---

## 8. MoE 模型与数据策略

这一节是 MoE-first 版本的关键。

## 8.1 原则

主模型现在直接固定，不再保留一个模糊的“以后再选”状态。  
本机版要做的是：

> **围绕一个真实、可下载、官方适配 TensorRT-LLM、且 16GB 显存有现实落地希望的 Qwen MoE 模型，完成 runtime scheduler enhancement 与端到端验证。**

## 8.2 模型层次

### A. 固定主模型

项目主模型固定为：

**`Qwen/Qwen1.5-MoE-A2.7B-Chat`**

主量化方案固定为：

**TensorRT-LLM `INT4 weight-only`**

这个组合是当前项目的默认执行路径，不再作为开放问题反复变动。

### B. 为什么是这个模型

锁定它的原因有四个：

1. 它是 **Qwen 的真实 MoE 模型**，不是 dense。
2. 它在 TensorRT-LLM 的 Qwen 路线里有明确支持。
3. 它的规模对你的 4060 Ti 16GB 来说最现实。
4. 它既能承担真实端到端验证，也能承担 replay trace 的来源。

因此，这个模型同时承担三种角色：

- **主执行模型**：Step 11 的真实 TensorRT-LLM 端到端验证
- **主 trace source**：离线导出 routing / expert histogram
- **主 benchmark model**：最终承载 MoE-aware scheduler 的实验

### C. 为什么量化方案定为 INT4 weight-only

本项目的正式执行路径使用：

**TRT-LLM INT4 weight-only**

原因：

1. 对 16GB 显存更现实。
2. 与 TensorRT-LLM 官方支持路径更一致。
3. 比把主线压在 GPTQ/AWQ 第三方权重上更稳。
4. 更适合作为真实 backend patch 的正式 benchmark 路线。

本项目不把 GPTQ/AWQ 作为主路线。

### D. synthetic skew profile

synthetic skew profile 仍然保留，但角色已经从“替代真实 MoE 模型”变成：

- 对真实主模型的补充压力场景；
- 用来系统性构造 hot expert / hot rank 请求；
- 用来做可控的 ablation。

### E. dense 是否还需要

不需要把 dense 作为正式项目部分。

如果后续真的出现一次最小 bring-up sanity run，也只是为了验证环境，而不是项目的一部分。

## 8.3 你到底需不需要 dense

可以完全不把 dense 作为正式项目部分。

更精确地说：

- 可以有一个极小的 bring-up sanity run；
- 但不把 dense 写进项目标题、主实验、核心结论和面试故事线。

所以在文档和实验结果里，dense 应当：

- 最多出现 1 次；
- 只用于说明环境正常；
- 不参与核心结果对比。

---

## 9. 项目实施总路线

总路线分 6 个阶段：

1. **环境与 TensorRT-LLM bring-up**
2. **TensorRT-LLM scheduler 代码路径打通**
3. **MoE pressure signal 构建**
4. **MoE-aware scheduler v1 实现**
5. **MoE-skew workload 基准与 ablation**
6. **`Qwen/Qwen1.5-MoE-A2.7B-Chat` 的 TensorRT-LLM 端到端验证**

---

## 10. 详细实施步骤

下面每一步都以“当前机器能真实推进”为前提。

---

## Step 0：建立工作区

### 做什么

在 WSL ext4 中建立项目目录。

### 目录建议

```bash
~/workspace/
  TensorRT-LLM/
  trtllm-moe-runtime-exp/
```

### 完成标准

- 主代码和 benchmark 数据都不放在 `/mnt/c`

### 交付物

- `docs/00_workspace.md`

---

## Step 1：TensorRT-LLM 环境安装与 sanity bring-up

### 做什么

按官方 Linux pip 流程装 TensorRT-LLM，并完成最小 generate sanity run。

### 重点

这一步只是把真实 backend 路打通。  
不要在这一步花很多篇幅讨论 dense 模型实验。

### 具体动作

1. 创建 Python venv
2. clone TensorRT-LLM
3. `git submodule update --init --recursive`
4. `git lfs pull`
5. 用 `TRTLLM_USE_PRECOMPILED=1 pip install -e .`
6. 跑一个最小生成脚本

### 完成标准

- `from tensorrt_llm import LLM` 成功
- 最小 generate 成功
- GPU 运行正常

### 交付物

- `docs/01_install_log.md`
- `scripts/sanity_backend.py`

---

## Step 2：打通 TensorRT-LLM scheduler 代码路径

### 做什么

识别本版本 TensorRT-LLM 中与 scheduler 相关的关键类和调用链。

### 目标

明确以下问题：

- executor loop 在哪里
- `CapacityScheduler` 在哪里
- `MicroBatchScheduler` 在哪里
- request state 在哪里维护
- 当前 step 的 context / generation request 在哪里确定

### 输出内容

做一份 codepath 笔记，至少包括：

- 关键类名
- 关键函数
- 调用关系
- 你计划插 telemetry 和 patch 的位置

### 完成标准

- 你能清楚说出要改哪些文件
- 你知道日志应该打在哪里

### 交付物

- `docs/02_codepath_notes.md`

---

## Step 3：加入最小 telemetry

### 做什么

先不改调度行为，只采集 scheduler 所需的 runtime 结构信号。

### 第一批要采什么

#### Request 级

- request id
- arrival time
- current state
- input length
- generated length
- waiting time

#### Step 级

- 当前 step 的 generation request 数
- 当前 step 的 context request 数
- 当前 step 的 prefill token 数
- 当前 step 总耗时

#### KV 级

- available blocks
- reserved blocks
- used blocks
- paused requests 数

### 这一阶段还不需要真实 MoE signal

因为你要先保证：

- telemetry 接口可用
- scheduler patch 不会引入明显开销

### 完成标准

- 默认行为完全不变
- 可以输出结构化 JSONL telemetry

### 交付物

- `scheduler/telemetry.py`
- `docs/03_runtime_telemetry.md`

---

## Step 4：构建 MoE pressure signal

这是整个项目最重要的一步之一。

### 做什么

定义并实现 `moe_pressure_score`。

### 这个 score 应该表达什么

它不需要完美模拟真实 MoE kernel time，但要能表达：

- 当前请求是否偏向某些 hot experts
- 多个请求组合后是否可能造成 rank hotspot
- 某个 batch 是否有更高的 straggler 风险

### 建议的 pressure 结构

```text
pressure(request) =
  expert_histogram
  rank_histogram
  pressure_score
  hot_expert_flags
  hot_rank_flags
```

### pressure 来源优先级

#### 方案 A：Synthetic Pressure（先做）

为请求类构造固定 profile：

- class balanced
- class hot_expert_0
- class hot_expert_1
- class hot_rank_left
- class hot_rank_right

优点：

- 完全可控
- 重复性强
- 本机可稳定推进

#### 方案 B：Replay Trace（再做）

从固定主模型 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 中离线采集：

- per-request routing histogram
- per-layer top-k expert choice

然后回放。

#### 方案 C：Live Trace（最后做）

如果后续可以从 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 的真实运行路径中在线获取 pressure，就接入 live provider。

### 统一接口

实现：

```python
pressure_provider.get(request) -> PressureInfo
```

使后续 scheduler 不关心 pressure 来源。

### 完成标准

- benchmark 可切换 synthetic / replay provider
- 你能构造至少 3 类不同压力模式

### 交付物

- `scheduler/moe_pressure.py`
- `scripts/generate_pressure_traces.py`
- `docs/04_moe_pressure_model.md`

---

## Step 5：定义 MoE-skew workloads

### 做什么

为 scheduler 设计真正的 MoE-specific benchmark。

### 为什么这一步关键

因为如果 workload 不体现 MoE pressure，  
那项目最后看起来还是一个泛化调度器。

### 至少要有这 4 类 workload

#### W1：Balanced MoE workload

特点：

- 请求 pressure 分布均匀
- 没有特别热的 experts/ranks

用途：

- 作为“无 skew”基线

#### W2：Hot-Expert workload

特点：

- 大量请求集中打到同一组 experts

用途：

- 测试 batch straggler 风险

#### W3：Hot-Rank workload

特点：

- 请求在 rank 维度上偏斜

用途：

- 测试 rank hotspot 对 step variance 的影响

#### W4：Mixed Burst workload

特点：

- 平时压力平稳
- 偶尔涌入高压请求

用途：

- 测试 tail latency 和 scheduler 稳定性

### 数据格式建议

```json
{
  "prompt": "...",
  "max_tokens": 128,
  "arrival_ms": 20,
  "pressure_class": "hot_expert_0"
}
```

### 完成标准

- 至少 4 类 workload
- 每类 workload 可复现

### 交付物

- `workloads/*.jsonl`
- `scripts/generate_workloads.py`
- `docs/05_workloads.md`

---

## Step 6：建立 MoE-first baseline

### 做什么

在不改 scheduler 行为前提下，先跑默认 TensorRT-LLM baseline。

### 注意

baseline 的重点不是“dense 模型吞吐多高”，而是：

- 不同 pressure workload 下默认 scheduler 的表现
- step latency 是否会在 skew workload 下变差
- tail latency 是否被高压请求放大

### 重点指标

- p50 / p90 / p99 TTFT
- p50 / p90 / p99 TPOT
- step latency variance
- request latency tail
- paused request count
- queue wait distribution

### 完成标准

- 你能清楚说明：默认 scheduler 在哪类 MoE pressure workload 下最脆弱

### 交付物

- `results/01_baseline/`
- `docs/06_baseline.md`

---

## Step 7：实现 MoE-aware scheduler v1

### 做什么

先做一个最小可用版本：

> **decode-first + high-pressure request dispersion**

### 逻辑

对于当前 step：

1. 优先 generation requests
2. 根据当前 batch pressure 逐个尝试加入候选请求
3. 若加入后压力超过阈值，则推迟该请求
4. 若 starvation 风险太高，则允许越过阈值

### 核心不是复杂公式

而是：

- 你真的让 TensorRT-LLM scheduler 用了 MoE pressure signal
- batch composition 因此发生变化

### 建议伪代码

```text
candidate_batch = select_generation_requests_first()

for req in remaining_candidates:
    if pressure(candidate_batch + req) <= threshold:
        candidate_batch.add(req)
    elif starvation(req) is high:
        candidate_batch.add(req)
    else:
        defer req
```

### 完成标准

在 Hot-Expert 或 Hot-Rank workload 中，至少看到一项改善：

- p99 TPOT 更低
- step latency variance 更低
- request tail 更短

### 交付物

- `scheduler/moe_microbatch_scheduler.py`
- `docs/07_moe_scheduler_v1.md`
- `results/02_scheduler_v1/`

---

## Step 8：实现 MoE-aware admission

### 做什么

把 admission / capacity decision 也纳入 MoE pressure 感知。

### 为什么要做

因为仅在 microbatch 阶段做 pressure-aware 还不够。  
如果前面 admission 太激进，可能已经把后续 step 压坏了。

### admission 要考虑的内容

- KV block cost
- waiting time / urgency
- 该请求 pressure score
- 当前 decode batch pressure
- 当前 step 是否已经接近热点状态

### score 示例

```text
admit_score =
  + urgency
  + starvation_bonus
  - kv_cost
  - pressure_cost
  - expected_tail_risk
```

### 完成标准

相对 v1，在 Mixed Burst workload 上至少表现出一项优势：

- p99 更稳
- pause 更少
- queue wait 分布更平滑

### 交付物

- `scheduler/moe_capacity_scheduler.py`
- `docs/08_moe_capacity_scheduler.md`
- `results/03_capacity_scheduler/`

---

## Step 9：加入 chunked prefill，但服务于 MoE 主线

### 做什么

如果要做 chunked prefill，也要把它写成 MoE 主线的一部分，而不是泛化吞吐技巧。

### 正确叙事

不是：

- 我做了一个通用 adaptive chunking

而是：

> **我控制 prefill chunk 的插入，避免它在高 MoE pressure decode step 中进一步放大 tail latency。**

### 具体做法

当满足下面任一条件时，减少 prefill aggressiveness：

- 当前 decode pressure 高
- 最近 step latency variance 上升
- hot-rank flag 频繁触发

### 完成标准

在高压 workload 下：

- TPOT tail 不因 prefill 插入而明显恶化
- 相对默认策略更稳

### 交付物

- `scheduler/adaptive_chunking.py`
- `docs/09_prefill_under_moe_pressure.md`
- `results/04_prefill_control/`

---

## Step 10：Replay Trace provider

### 做什么

把 synthetic pressure 升级为更接近真实 MoE 的 trace replay。

### 路径

1. 固定 trace source 为 `Qwen/Qwen1.5-MoE-A2.7B-Chat`
2. 离线采 expert histogram
3. 为请求生成 replayable `PressureInfo`
4. 用 replay provider 驱动 benchmark

### 为什么这一步重要

因为它能让项目从：

- “人工造压力场景”

升级为：

- “调度器消费接近真实 MoE routing trace 的 signal”

### 完成标准

- 能在同一 benchmark 框架里切换 synthetic / replay
- replay 路径稳定可复现

### 交付物

- `scheduler/replay_pressure_provider.py`
- `docs/10_replay_provider.md`
- `results/05_replay/`

---

## Step 11：`Qwen/Qwen1.5-MoE-A2.7B-Chat` 的 TensorRT-LLM 端到端验证（固定方案）

### 做什么

用固定主模型 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 完成真实的 TensorRT-LLM 端到端验证，并把它接到你的 scheduler patch 上。

### 这一步的地位

这一步不再是“有空再试试”的补充，而是当前项目尽量要完成的**正式里程碑**。  
它的意义是把前面已经做好的：

- TensorRT-LLM backend integration
- MoE pressure model
- replay provider
- scheduler enhancement

统一落到一个**真实、可下载、官方适配的 Qwen MoE checkpoint** 上。

### 固定执行路径

默认路径固定为：

1. 下载 `Qwen/Qwen1.5-MoE-A2.7B-Chat`
2. 按 TensorRT-LLM 官方 Qwen 路线做 checkpoint conversion
3. 使用 **INT4 weight-only** 构建正式执行路径
4. 完成单请求 generate
5. 完成小规模 benchmark
6. 把模型接入你的 MoE-aware scheduler patch
7. 在至少一个 Hot-Expert 或 Hot-Rank workload 上跑出对比结果

### 这一步至少要拿到什么

至少要完成下面 3 件事：

1. **真实模型可跑**
   - `Qwen/Qwen1.5-MoE-A2.7B-Chat` 在 TensorRT-LLM 中成功完成推理
2. **真实模型可测**
   - 至少完成一组 baseline benchmark
3. **真实模型可接入 patch**
   - 你的 scheduler patch 在该模型上能跑通一轮 workload

### 理想完成标准

理想情况下，再额外拿到下面至少一项：

- Hot-Expert workload 下默认 scheduler vs MoE-aware scheduler 对比
- Hot-Rank workload 下 step variance 对比
- replay pressure 与 live model path 的一致性检查

### 如果过程中卡住

如果 INT4 路径构建中遇到工程阻塞，这一步仍然不应被删除，而是拆成两个子里程碑：

1. **先完成模型下载、conversion、单请求 generate**
2. **再补正式 benchmark 和 scheduler 接入**

也就是说，Step 11 可以分阶段完成，但不再从计划里降级为“可选”。

### 交付物

- `docs/11_qwen15_moe_e2e.md`
- `results/06_qwen15_moe_e2e/`
- 至少一份 baseline log
- 至少一份 patched scheduler log

---

## Step 12：最终实验矩阵

### 对比对象

至少比较：

1. default TensorRT-LLM scheduler
2. v1: MoE-aware microbatch scheduler
3. v2: v1 + MoE-aware admission
4. v3: v2 + MoE-aware prefill control
5. v3 + replay pressure provider
6. v3 + `Qwen/Qwen1.5-MoE-A2.7B-Chat` real model path

### Workload 维度

至少包括：

1. Balanced
2. Hot-Expert
3. Hot-Rank
4. Mixed Burst

### 最终重点指标

- TTFT p50 / p90 / p99
- TPOT p50 / p90 / p99
- step latency variance
- queue wait distribution
- request tail latency
- starvation count
- pressure threshold trigger frequency

### 你真正想证明的结论

不是：

- 所有 workload 吞吐都大幅提升

而是：

> 在 MoE pressure 明显的 workload 下，MoE-aware scheduler 可以比默认 scheduler 更好地控制 tail latency、step variance 和 straggler 风险。

---

## 11. 仓库结构建议

```text
trtllm-moe-runtime-exp/
  docs/
    00_workspace.md
    01_install_log.md
    02_codepath_notes.md
    03_runtime_telemetry.md
    04_moe_pressure_model.md
    05_workloads.md
    06_baseline.md
    07_moe_scheduler_v1.md
    08_moe_capacity_scheduler.md
    09_prefill_under_moe_pressure.md
    10_replay_provider.md
    11_qwen15_moe_e2e.md
    final_report.md
    interview_talking_points.md
  scripts/
    sanity_backend.py
    generate_workloads.py
    generate_pressure_traces.py
    run_baseline.py
    run_ablation.py
    plot_results.py
  workloads/
    balanced.jsonl
    hot_expert.jsonl
    hot_rank.jsonl
    mixed_burst.jsonl
  scheduler/
    telemetry.py
    moe_pressure.py
    replay_pressure_provider.py
    moe_microbatch_scheduler.py
    moe_capacity_scheduler.py
    adaptive_chunking.py
  results/
    01_baseline/
    02_scheduler_v1/
    03_capacity_scheduler/
    04_prefill_control/
    05_replay/
    06_qwen15_moe_e2e/
    final_figures/
```

---

## 12. 本机版成功标准

如果做到下面这些，这个项目就已经是一个强 MoE-first 版本：

1. 你在 TensorRT-LLM 真实 backend 中改了 scheduler。
2. 你的 scheduler 使用了 `moe_pressure_score`。
3. 你设计了 Hot-Expert / Hot-Rank / Burst 这类 MoE-specific workload。
4. 你做了 default vs MoE-aware 的 ablation。
5. 你把 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 真实接进了 TensorRT-LLM 路径。
6. 你拿到了至少一类高压 workload 下的可解释收益：
   - p99 TPOT 降低
   - step latency variance 下降
   - request tail 更稳
7. 你的故事线从头到尾都在讲 MoE runtime，而不是 dense benchmark。

---

## 13. 面试讲法

### 60 秒版本

我做的是一个面向 MoE 模型的 TensorRT-LLM runtime scheduler enhancement。切入点是：MoE inference 的 step latency 不只由 token 数量决定，还受到 expert/rank pressure 影响，而默认 scheduler 对这种 MoE-specific signal 不敏感。我在 TensorRT-LLM PyTorch backend 内部实现了一个 MoE-aware scheduler，用 routing trace 或 synthetic pressure 构造 `moe_pressure_score`，并让它参与 microbatch selection、admission 和 prefill 插入控制。最终在 Hot-Expert、Hot-Rank 和 Mixed Burst workload 上，对比默认 scheduler 观察 tail latency 和 step variance 的改善。

### 你最该避免的讲法

- “我先做了一个 dense scheduler，后来顺便套到 MoE 上。”
- “我主要做 chunked prefill，MoE 只是一个场景。”
- “我没有真实改 TensorRT-LLM，只是在外面做了 trace replay。”

### 最准确的讲法

> 我针对 MoE inference runtime 中的 expert/rank pressure 问题，对 TensorRT-LLM scheduler 做了专项增强。这个增强不改变模型输出，也不重写 kernel，而是让 runtime 在 batch selection 和 admission 时显式感知 MoE pressure，从而改善高压 workload 下的 tail latency 和 batch stability。

---

## 14. 参考依据

- TensorRT-LLM Scheduler  
  https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html

- TensorRT-LLM Support Matrix  
  https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html

- TensorRT-LLM Linux Installation  
  https://nvidia.github.io/TensorRT-LLM/installation/linux.html

- TensorRT-LLM KV Cache System  
  https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html

- Scaling Expert Parallelism in TensorRT-LLM, Part 1  
  https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html

- Scaling Expert Parallelism in TensorRT-LLM, Part 2  
  https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog08_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html

- Scaling Expert Parallelism in TensorRT-LLM, Part 3  
  https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html

---

## 15. 最后一条原则

这份计划最重要的不是“把 MoE 做大”，而是：

> **把 MoE-specific runtime problem 做实。**

只要你做到了下面这三点，故事线就成立：

1. **MoE pressure 是项目主角。**
2. **TensorRT-LLM backend integration 是主实现。**
3. **高压 workload 下的 tail/stability 改善是主结果。**

这样这个项目就会明显比“一个泛化 serving scheduler + 少量 MoE 讨论”更锋利，也更接近你最初想要的方向。
