# TensorRT-LLM MoE Runtime 架构优化项目方案

## 1. 项目标题

**MoE-Aware Runtime Scheduling Enhancement for TensorRT-LLM**

中文表达：

**面向 MoE 模型的 TensorRT-LLM Runtime 调度增强：基于 expert/rank pressure 的推理执行优化**

## 2. 一句话定位

本项目不是重新实现一个 LLM serving system，也不是在 TensorRT-LLM 外部简单加一层路由策略，而是在 TensorRT-LLM PyTorch backend 的真实 executor、KV cache manager 和 scheduler 架构内，实现一个面向 MoE 推理的 runtime-level 优化补充：让调度器在 batch selection 和 admission 时显式感知 **expert/rank pressure、decode tail 风险、KV 资源竞争和 prefill 插入代价**，从而改善 MoE 推理中的 straggler 和 tail latency 问题。

## 3. 岗位匹配点

该项目主打 NVIDIA AI Computing Software Development Intern 中的 **TensorRT-LLM - Inference Optimization** track。

岗位要求中的关键词和本项目的对应关系：

| 岗位关键词 | 项目对应能力 |
|---|---|
| TensorRT-LLM inference pipelines | 基于 TensorRT-LLM PyTorch backend 的 executor/scheduler 路径做增强 |
| Analyze and optimize model execution | 分析 prefill/decode/MoE/communication/KV cache 对执行路径的影响 |
| Scalability | 面向多 GPU MoE expert parallelism、decode batch、KV cache 资源压力建模 |
| Memory use | 结合 KV cache block reservation、memory pressure、chunked prefill admission 做调度 |
| Python / PyTorch | 在 PyTorch backend 中实现自定义 scheduler、telemetry、benchmark harness |
| Collaborate across framework and research teams | 项目同时涉及 runtime 架构、MoE 模型特性、GPU profiling、serving workload 建模 |

项目面试叙事应强调：

> 我没有试图单点打败 TensorRT-LLM 已高度优化的 kernel，而是研究 TensorRT-LLM runtime 在 MoE workload 下的执行架构，把 expert/rank pressure、KV cache、prefill/decode scheduling 和 tail-latency risk 统一纳入 scheduler/resource model，实现一个可插入现有 TensorRT-LLM backend 的 runtime 优化补充。

### 固定主模型与验证路径

当前敲定的主验证路径为：

- **主模型：`Qwen/Qwen1.5-MoE-A2.7B-Chat`**
- **主量化方案：TensorRT-LLM `INT4 weight-only`**

这个固定模型承担三种角色：

1. **主执行模型**：真实 TensorRT-LLM 端到端验证
2. **主 trace source**：离线导出 routing / expert histogram
3. **主 benchmark model**：最终承载 MoE-aware scheduler 的正式实验

因此，这份架构方案的落地版本不是一个抽象的“未来某个 MoE 模型”，而是围绕 `Qwen/Qwen1.5-MoE-A2.7B-Chat + TRT-LLM INT4 weight-only` 的具体实现。

## 4. 为什么不是外层策略，也不是仿制 TensorRT-LLM

### 不做的版本

下面这种方式不适合作为主项目：

```text
requests
  -> 自己写的 toy scheduler
  -> mock KV cache
  -> mock prefill/decode
  -> fake performance metrics
```

这会变成对 TensorRT-LLM / vLLM 的简化模仿，面试中很容易被质疑：

- 是否真的理解 TensorRT-LLM 内部执行路径？
- 是否只是在 trace replay 上验证了一个策略？
- 是否能迁移到真实 backend？
- 是否绕开了真正困难的 runtime resource management？

### 本项目要做的版本

本项目应该基于 TensorRT-LLM 真实组件进行增强：

```text
TensorRT-LLM PyTorch Backend
  |
  +-- PyExecutor / executor loop
  +-- KVCacheManager / block resource estimation
  +-- CapacityScheduler
  +-- MicroBatchScheduler
  +-- active request state
  +-- MoE layer telemetry hooks
  +-- trtllm-bench / trtllm-serve evaluation
```

项目原则：

| 模块 | 项目做法 |
|---|---|
| Executor | 不重写，使用 TensorRT-LLM 现有 PyTorch executor |
| KV cache manager | 不重写，只读取或扩展 resource estimation / block usage 信息 |
| Attention / MoE kernels | 不重写主 kernel，只 profile 并消费 telemetry |
| Scheduler | 重点增强，基于 `CapacityScheduler` / `MicroBatchScheduler` 做 MoE-aware 调度 |
| Request lifecycle | 使用 TensorRT-LLM 原生 request state |
| Benchmark | 使用 `trtllm-bench` / `trtllm-serve`，避免自造不可信 benchmark |
| Simulator | 只能作为算法预验证，不作为主成果 |

因此，本项目的性质是：

> 对 TensorRT-LLM runtime scheduling/resource management 的优化补充，而不是把 TensorRT-LLM 中间部分抽出来模仿一遍。

## 5. 背景判断

TensorRT-LLM 在底层 kernel 和 runtime 上已经非常强，尤其是：

- MoE auxiliary kernels。
- MoE TopK / routing 优化。
- Expert parallelism。
- Expert Parallelism Load Balancer, EPLB。
- AllToAll / one-sided AllToAll。
- PDL kernel overlap。
- KV cache reuse / offloading / prioritized eviction。
- overlap scheduler。
- disaggregated serving 和 KV cache transmission。

因此，学生项目不适合把主线放在：

- 写一个更快的通用 MoE GEMM。
- 重写 KV cache manager。
- 重写 AllToAll。
- 在 TensorRT-LLM 外面做一个简单 router。
- 单独写一个 toy TopK kernel 然后声称优化了 TensorRT-LLM。

更有价值的空间在于：

> 在 TensorRT-LLM 已有高性能 kernel 和 KV cache 能力之上，进一步改进 runtime 如何组织 **MoE-specific workload** 的执行。

尤其是 MoE serving 中，prefill、decode、KV cache、expert/rank pressure、dispatch/combine 路径和 host-side scheduling 之间存在复杂耦合。固定策略往往无法同时优化 TTFT、TPOT、step stability 和 memory pressure。

## 6. 核心问题定义

当前 TensorRT-LLM scheduler 主要关注：

- 当前有哪些 active requests。
- 哪些 request 可以 fit into KV cache。
- 哪些 context / generation requests 被选入本轮 forward。
- 如何通过 inflight batching 和 overlap scheduler 提高 GPU 利用率。

本项目希望进一步探索：

1. **Prefill 和 decode 的异构性**
   - prefill 更偏大块 compute / attention / KV 写入。
   - decode 更偏小步迭代，对 TPOT 和 latency tail 更敏感。
   - MoE decode 还可能引入 expert dispatch 和 communication pressure。

2. **KV cache resource 不只是能不能放下**
   - 不同 request 的 KV block 需求不同。
   - chunked prefill 会改变 block reservation 粒度。
   - 过度激进 admission 可能提高吞吐但恶化 TPOT 或引发 pause。

3. **MoE pressure 不只是 model 内部问题**
   - expert imbalance 和 AllToAll 时间会影响每个 decode step。
   - EPLB 解决 expert placement，但 request batching 仍会影响短期 expert/rank pressure。
   - scheduler 如果完全不感知 MoE pressure，可能把多个 MoE-heavy 请求组合成 straggler batch。

4. **Prefill 插入需要显式受 MoE pressure 约束**
   - prefill chunk 在 dense 场景下可能主要是吞吐问题，在 MoE 场景下还可能放大高压 decode step 的 tail。
   - scheduler 输出不应只是 request list，而应体现当前 step 对 MoE pressure 的控制意图。

## 7. 项目目标

实现一个 TensorRT-LLM PyTorch backend 内部的 **MoE-aware runtime scheduler**，使其在每个 scheduling step 中同时考虑：

- KV cache block capacity。
- request state: context init / context running / generation。
- prefill chunk size。
- decode latency target。
- memory pressure。
- MoE expert/rank pressure。
- step-level tail-latency risk。
- throughput@latency SLA。

目标不是在所有场景下都超过官方默认配置，而是在以下 workload 中展示明确收益或清晰边界：

- Balanced MoE workload。
- Hot-Expert workload。
- Hot-Rank workload。
- Mixed Burst workload。
- Repeated-prefix under MoE pressure workload。

此外，项目落地版本要求在固定主模型 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 上完成真实 TensorRT-LLM 端到端验证，而不是停留在 synthetic pressure 或 trace replay。

## 8. 系统架构

```text
Client / Benchmark Workload
  |
  v
trtllm-bench / trtllm-serve
  |
  v
TensorRT-LLM PyTorch Backend
  |
  +-- PyExecutor
        |
        +-- Request State Queue
        |
        +-- KVCacheManager
        |     +-- block usage
        |     +-- needed blocks
        |     +-- reusable blocks
        |     +-- memory pressure
        |
        +-- MoE Telemetry Collector
        |     +-- expert histogram
        |     +-- per-rank routed tokens
        |     +-- MoE kernel duration
        |     +-- AllToAll / dispatch / combine timing
        |
        +-- MoE-Aware CapacityScheduler
        |     +-- chunk-level KV reservation
        |     +-- memory pressure guard
        |     +-- pause / admit decision
        |
        +-- MoE-Aware MicroBatchScheduler
              +-- decode-first scheduling
              +-- opportunistic prefill chunk insertion
              +-- MoE pressure-aware request grouping
              +-- SLA-aware batch selection
```

## 9. 主要技术贡献

### 9.1 Runtime Resource Model

为每个 request 和每个 scheduling step 建立轻量资源模型。

Request-level features:

```text
request_features:
  input_len
  generated_len
  remaining_output_budget
  shared_prefix_len
  estimated_kv_blocks
  current_state
  waiting_time
  deadline_or_sla_target
  historical_moe_pressure_score
```

Runtime-level features:

```text
runtime_state:
  available_kv_blocks
  reserved_kv_blocks
  active_decode_batch_size
  pending_prefill_tokens
  recent_tpot
  recent_ttft
  gpu_memory_pressure
  moe_rank_pressure_vector
  alltoall_duration_moving_average
```

MoE pressure score:

```text
moe_pressure(batch)
  = max_rank_tokens / mean_rank_tokens
  + lambda_1 * expert_load_cv
  + lambda_2 * recent_alltoall_time
  + lambda_3 * hot_rank_penalty
```

该模型不改变模型输出，只影响 request admission 和 batch composition。落地时，`historical_moe_pressure_score` 的正式来源优先级为：

1. synthetic pressure profile
2. 基于 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 导出的 replay trace
3. 如条件允许，再接入 live runtime signal

### 9.2 MoE-Aware CapacityScheduler

在 TensorRT-LLM `CapacityScheduler` 层实现增强。

默认 capacity scheduler 通常关注 request 是否能被 KV cache 资源容纳。本项目扩展为：

- 不仅判断完整 request 是否能放下，也判断 prefill chunk 是否能安全放下。
- 根据当前 decode pressure 控制 prefill chunk admission。
- 在 KV cache 紧张时避免引入长 prompt request 造成 decode request 被 pause。
- 在 repeated-prefix workload 下优先 admission 可复用 KV cache 的 request。
- 在 MoE pressure 高时减少 MoE-heavy context chunk 与 decode step 的竞争。

示意逻辑：

```text
for request in pending_requests:
    kv_need = estimate_kv_blocks(request, chunk_size)
    mem_cost = estimate_memory_pressure(kv_need)
    moe_cost = estimate_moe_pressure(request)
    latency_risk = estimate_tpot_violation_if_admitted(request)

    admit_score =
        alpha * request_urgency
      + beta  * kv_reuse_gain
      - gamma * mem_cost
      - delta * moe_cost
      - eta   * latency_risk

    admit requests by score under resource constraints
```

### 9.3 MoE-Aware MicroBatchScheduler

在 TensorRT-LLM `MicroBatchScheduler` 层实现 batch selection 优化。

目标：

- decode-first，保护 TPOT。
- 在 decode batch 较轻时插入 prefill chunk，提高 GPU 利用率。
- 避免把多个 expert-skew request 放在同一 step，降低 hot-rank straggler。
- 根据 SLA urgency 调整请求优先级。

示意策略：

```text
1. Select generation requests first to satisfy TPOT target.
2. Estimate current decode batch MoE pressure.
3. If pressure and memory allow, select context chunks.
4. Prefer chunks with:
   - high KV reuse gain
   - low expected MoE pressure
   - high waiting-time urgency
5. Emit context_requests and generation_requests for current forward step.
```

### 9.4 Chunked Prefill and Decode Co-Scheduling

项目重点不是简单启用 chunked prefill，而是研究 chunk 粒度如何影响：

- KV block reservation。
- TTFT。
- TPOT。
- GPU utilization。
- MoE dispatch pressure。
- prefill/decode overlap。

实验变量：

- fixed chunk size。
- adaptive chunk size。
- decode-pressure-aware chunk size。
- memory-pressure-aware chunk size。
- MoE-pressure-aware chunk size。

Adaptive chunk size 示例：

```text
if recent_tpot > target_tpot:
    reduce prefill_chunk_size
elif gpu_util_low and kv_blocks_available:
    increase prefill_chunk_size
elif moe_pressure_high:
    avoid MoE-heavy prefill chunks
else:
    use default chunk size
```

### 9.5 MoE Telemetry Hook

为 scheduler 提供低开销 MoE 运行时信号。

优先实现 Python/CUDA 低侵入 telemetry，而不是改动核心 MoE kernel。

可采集指标：

- per-layer expert histogram。
- per-rank routed token count。
- expert load coefficient of variation。
- hot expert / hot rank moving average。
- MoE auxiliary kernel time。
- AllToAll / dispatch / combine duration。

实现层级：

1. 第一阶段：synthetic pressure 与结构化 replay provider。
2. 第二阶段：从 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 路径离线导出 routing / expert histogram。
3. 第三阶段：在 PyTorch backend MoE module 周围增加轻量 hook。
4. 第四阶段：可选实现 CUDA extension，用于低开销统计 expert ids，避免 CPU sync。

### 9.6 Optional: Telemetry-Oriented CUDA Kernel

kernel 支线不作为项目主成果，而是服务于 runtime scheduler。

合适的 kernel 目标：

- expert id histogram。
- token-per-rank counting。
- hot expert detection。
- routing metadata summary。

不建议主打：

- 重新实现完整 MoE GEMM。
- 重写 TensorRT-LLM TopK。
- 重写 AllToAll。

因为这些路径 TensorRT-LLM 已经高度优化，学生项目很难形成可信主线。

## 10. 关键算法设计

### 10.1 Scheduler Score

```text
score(request, runtime_state)
  = + w1 * urgency(request)
    + w2 * kv_reuse_gain(request)
    + w3 * starvation_bonus(request)
    - w4 * kv_block_cost(request)
    - w5 * expected_decode_latency_risk(request)
    - w6 * moe_pressure_cost(request)
    - w7 * transfer_or_comm_pressure(request)
```

其中：

- `urgency`: request 等待时间、SLA deadline。
- `kv_reuse_gain`: shared prefix / reusable block 预估收益。
- `kv_block_cost`: 对 KV cache pool 的压力。
- `expected_decode_latency_risk`: 对当前 decode TPOT 的潜在影响。
- `moe_pressure_cost`: 对 expert/rank imbalance 的潜在影响。
- `transfer_or_comm_pressure`: KV transfer 或 AllToAll 压力。

### 10.2 Decode-First with Opportunistic Prefill

```text
if decode_requests_exist:
    schedule decode requests first
    estimate remaining compute/memory budget
    if budget allows:
        schedule selected prefill chunks
else:
    schedule prefill chunks to maximize throughput
```

核心思想：

- decode step 对用户可见 latency 更敏感。
- prefill chunk 可以作为填充 GPU 空隙的 workload。
- prefill 不能无脑插入，否则会恶化 TPOT tail。

### 10.3 MoE Pressure-Aware Grouping

```text
candidate_batch = []
for request in sorted_candidates:
    new_pressure = estimate_pressure(candidate_batch + request)
    if new_pressure < pressure_threshold:
        candidate_batch.append(request)
    else:
        defer request unless starvation risk is high
```

注意：

- 该策略不修改 router 结果，不影响模型输出。
- 它只影响哪些 request 同时进入同一个 forward step。
- 与 EPLB 互补：EPLB 管 expert placement，本项目管 request/batch placement。

## 11. 实现路径

### Phase 0: TensorRT-LLM 内部路径熟悉

目标：

- 跑通 TensorRT-LLM PyTorch backend。
- 理解 PyExecutor scheduling loop。
- 找到 `CapacityScheduler` / `MicroBatchScheduler` 接入点。
- 跑通 `trtllm-bench`。
- 记录 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 的 conversion / build / runtime 接入点。

产出：

- `docs/codepath_notes.md`
- 关键类和调用链图。

### Phase 1: 强 baseline 建立

必须比较强 baseline，而不是只和 naive 实现比。

Baseline:

- TensorRT-LLM default scheduler。
- TensorRT-LLM overlap scheduler enabled。
- `GUARANTEED_NO_EVICT`。
- `MAX_UTILIZATION`。
- fixed chunked prefill configs。

固定模型路径：

- `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- TensorRT-LLM `INT4 weight-only`

指标：

- TTFT p50/p90/p99。
- TPOT p50/p90/p99。
- output tokens/sec/GPU。
- request throughput。
- GPU utilization。
- peak HBM。
- KV block usage。

产出：

- `docs/baseline_report.md`
- `benchmarks/baseline/*.json`

### Phase 2: Telemetry and Bottleneck Attribution

实现 MoE 和 KV 相关 telemetry。

采集：

- KV block available/reserved/used。
- active request state。
- context/generation request count。
- per-step prefill tokens。
- per-step decode tokens。
- expert load histogram。
- per-rank MoE pressure。
- MoE / AllToAll kernel timing。
- replay trace 与 live path 的一致性检查。

产出：

- `scheduler/telemetry.py`
- `docs/runtime_telemetry_report.md`
- timeline 可视化。

### Phase 3: MoE-Aware CapacityScheduler

实现第一版可插拔 scheduler。

功能：

- memory-pressure-aware admission。
- chunk-level resource estimation。
- repeated-prefix aware priority。
- decode pressure guard。
- MoE pressure guard。
- Hot-Expert / Hot-Rank workload 下的 pressure-threshold 控制。

产出：

- `scheduler/moe_capacity_scheduler.py`
- ablation:
  - default
  - KV/memory-aware only
  - decode-pressure-aware
  - MoE-pressure-aware

### Phase 4: MoE-Aware MicroBatchScheduler

实现 batch selection 逻辑。

功能：

- decode-first。
- opportunistic prefill。
- adaptive chunk size。
- MoE pressure-aware grouping。
- SLA urgency。

产出：

- `scheduler/moe_microbatch_scheduler.py`
- `scheduler/replay_pressure_provider.py`
- `docs/scheduler_design.md`
- MoE-skew workload benchmark。

### Phase 5: `Qwen/Qwen1.5-MoE-A2.7B-Chat` 端到端验证与最终评估

Workloads:

1. **Balanced MoE Workload**
   - pressure 分布均衡。
   - 测试默认策略在无 skew 情况下是否已足够好。

2. **Hot-Expert Workload**
   - 请求集中打到同一组 experts。
   - 测试 batch straggler 与 TPOT tail。

3. **Hot-Rank Workload**
   - 请求在 rank 维度明显偏斜。
   - 测试 rank hotspot 对 step variance 的影响。

4. **Mixed Burst Workload**
   - 平时压力平稳，偶尔涌入高压请求。
   - 测试 p99 latency、queue wait 与 starvation。

5. **Repeated-Prefix under MoE Pressure**
   - 在共享前缀基础上叠加不同 pressure class。
   - 测试 KV-aware admission 与 MoE pressure-aware grouping 的相互作用。

产出：

- `docs/final_report.md`
- `docs/qwen15_moe_e2e.md`
- performance tables。
- Nsight / trace 截图。
- ablation study。

### Phase 6 Optional: Low-Overhead MoE Telemetry CUDA Extension

如果时间充足，实现一个小型 CUDA extension：

- 输入：expert ids / rank ids。
- 输出：histogram / pressure score。
- 目标：减少 telemetry 对 runtime 的影响。

产出：

- `kernels/moe_telemetry/`
- microbenchmark。
- scheduler overhead 对比。

## 12. 评估指标

### User-facing Metrics

- TTFT p50/p90/p99。
- TPOT p50/p90/p99。
- end-to-end request latency。
- throughput under SLA。
- timeout / SLA violation rate。

### System Metrics

- output tokens/sec/GPU。
- GPU utilization。
- SM active。
- HBM peak usage。
- KV cache block usage。
- KV cache eviction / reuse。
- number of paused requests。
- prefill/decode worker idle time。

### MoE Metrics

- expert load CV。
- max-rank / mean-rank token ratio。
- hot expert frequency。
- AllToAll duration。
- dispatch/combine duration。
- MoE auxiliary kernel duration。

### Scheduler Metrics

- scheduler overhead per step。
- number of admitted context chunks。
- decode starvation count。
- prefill starvation count。
- adaptive chunk size distribution。
- pressure guard trigger frequency。

## 13. 预期结果表达

不要承诺所有场景都提升。更成熟的表达是：

```text
在 Balanced MoE workload 下，TensorRT-LLM 默认配置已经非常强，本项目策略应保持接近默认性能，不引入明显 regression。

在 Hot-Expert workload 下，通过 MoE pressure-aware grouping 与 decode-first scheduling，降低 p90/p99 TPOT，并抑制高压 batch straggler。

在 Hot-Rank workload 下，通过 pressure-threshold batching 和 admission 控制，降低 step latency variance。

在 Mixed Burst workload 下，通过 MoE-aware admission 和 prefill 控制，改善 queue wait 与 request tail。

在 Repeated-Prefix under MoE Pressure workload 下，通过 KV-aware admission 提高 cache reuse 对 TTFT 的贡献，同时避免高压请求叠加。
```

面试中最可信的说法：

> 这个项目不是为了证明默认 TensorRT-LLM 不够好，而是为了研究在 MoE pressure 明显的 serving 场景中，runtime scheduler 是否可以进一步利用 workload 结构，在不改变模型输出和不重写 kernel 的前提下优化 tail latency 和 batch stability。

## 14. Go / No-Go 判断

为了避免项目做成空泛优化，需要设置阶段性判断。

| 判断点 | 如果成立 | 项目动作 |
|---|---|---|
| scheduler overhead 超过 decode step 明显比例 | 策略太重 | 简化 cost model，减少 per-step 计算 |
| MoE pressure 与 TPOT tail 无明显相关性 | MoE-aware grouping 价值低 | 降级为 telemetry 分析，不做主优化 |
| KV reuse workload 中默认策略已接近最优 | KV-aware admission 空间小 | 转向 chunked prefill / decode co-scheduling |
| chunked prefill 插入导致 TPOT 恶化 | 策略过激 | 引入 decode pressure guard |
| balanced workload 出现 regression | 策略泛化差 | 增加 fallback to default scheduler |
| TensorRT-LLM 接口改动成本过高 | 集成风险高 | 保留最小侵入 patch，外部 wrapper 只用于实验入口 |

## 15. 风险与备选方案

### 风险 1: 没有足够 GPU 跑大 MoE

备选：

- 固定使用 `Qwen/Qwen1.5-MoE-A2.7B-Chat` + TRT-LLM INT4 weight-only。
- 先完成 conversion / build / 单请求 generate，再补 benchmark 与 patch 接入。
- 使用 replay trace 预验证 MoE pressure model。
- 如 live path 一时不稳定，先在 replay provider 上完成大部分 ablation。

### 风险 2: TensorRT-LLM 内部改动过复杂

备选：

- 先实现自定义 scheduler 的最小 patch。
- 保留外部 workload generator 和 benchmark harness。
- 不重写 executor，只在 scheduler 接口内做增强。

### 风险 3: 默认 TensorRT-LLM 已足够好

备选：

- 强调 workload-specific improvement。
- 做 ablation 证明哪些 workload 有收益，哪些 workload 应 fallback。
- 把“不该优化的场景”也作为工程判断结果。

### 风险 4: MoE telemetry overhead 太高

备选：

- 降低采样频率。
- 使用 moving average。
- 只采集关键层。
- optional CUDA extension。

## 16. 推荐仓库结构

```text
trtllm-moe-runtime-optimization/
  README.md
  docs/
    project_plan.md
    codepath_notes.md
    scheduler_design.md
    baseline_report.md
    runtime_telemetry_report.md
    qwen15_moe_e2e.md
    final_report.md
    interview_talking_points.md
  configs/
    qwen15_moe_int4wo.yaml
    trtllm_default.yaml
    trtllm_max_utilization.yaml
    moe_scheduler.yaml
    workload_balanced_moe.yaml
    workload_hot_expert.yaml
    workload_hot_rank.yaml
    workload_mixed_burst.yaml
    workload_repeated_prefix_moe.yaml
  scripts/
    run_trtllm_bench.py
    launch_trtllm_serve.py
    collect_runtime_metrics.py
    generate_pressure_traces.py
    parse_traces.py
    plot_results.py
  scheduler/
    telemetry.py
    resource_model.py
    replay_pressure_provider.py
    moe_capacity_scheduler.py
    moe_microbatch_scheduler.py
    adaptive_chunking.py
  tensorrt_llm_patch/
    README.md
    patch.diff
  kernels/
    moe_telemetry/
      README.md
      benchmark.py
      src/
  benchmarks/
    baseline/
    ablation/
    end_to_end/
  results/
    figures/
    tables/
    qwen15_moe_e2e/
```

## 17. 面试讲法

### 60 秒版本

我做的是一个 TensorRT-LLM runtime 架构优化项目，目标是优化 MoE 模型在高压 serving workload 下的 tail latency 和 batch stability。我没有重新实现 executor、KV cache 或 MoE kernel，而是基于 TensorRT-LLM PyTorch backend 的 `CapacityScheduler` 和 `MicroBatchScheduler` 做增强。核心思路是把 KV cache block pressure、decode TPOT、MoE expert/rank pressure 和 prefill 插入代价纳入同一个 runtime resource model，让 scheduler 在每个 step 更合理地选择 context chunks 和 generation requests。项目的主验证模型固定为 `Qwen/Qwen1.5-MoE-A2.7B-Chat`，正式执行路径是 TensorRT-LLM `INT4 weight-only`。

### 深入版本

TensorRT-LLM 的底层 kernel 已经很强，所以我没有选择重写 MoE GEMM 或 KV cache manager。我的切入点是 runtime scheduling：在 MoE serving 中，同样的 token 数量会因为 router 选择不同而形成不同的 expert/rank pressure，进而放大 decode tail 和 step variance。默认 scheduler 主要做资源 fit 和 batching，而我扩展了 resource model，使它能感知 KV block、prefill chunk、decode latency risk 和 MoE pressure。这个优化不改变模型输出，也不替换 TensorRT-LLM 高性能 kernel，而是作为现有 runtime 的补充。为了让项目落地，我把真实端到端验证固定在 `Qwen/Qwen1.5-MoE-A2.7B-Chat + TRT-LLM INT4 weight-only` 这条路径上。

### 面试官可能追问

**Q: 你是不是在外面加了一个策略层？**

A: 不是主线。外部 workload generator 只用于实验入口。核心实现是在 TensorRT-LLM PyTorch backend 的 scheduler 接口内，增强 `CapacityScheduler` 和 `MicroBatchScheduler` 的资源模型和 batch selection。

**Q: 你是不是重新实现了 TensorRT-LLM 的一部分？**

A: 没有。我复用 TensorRT-LLM 的 executor、KV cache manager、attention kernel、MoE kernel 和 request lifecycle，只在 scheduler/resource model/telemetry 层做补充。

**Q: 为什么不直接优化 kernel？**

A: TensorRT-LLM 的 MoE kernel、TopK、AllToAll 和 EPLB 已经高度优化。对这个岗位来说，我认为更有价值的是理解真实 inference runtime 中 compute、memory、communication 和 scheduling 的耦合关系，并在现有架构内做可验证的优化。

**Q: 如何证明不是 toy project？**

A: 用 TensorRT-LLM 真实 backend、真实 scheduler 接口、`trtllm-bench`/`trtllm-serve`、强 baseline、Nsight profiling 和 end-to-end 指标证明，而不是只用 mock simulator。

## 18. 简历 Bullet

- Built a MoE-aware runtime scheduling enhancement for TensorRT-LLM PyTorch backend by extending capacity and microbatch scheduling with KV cache pressure, decode latency risk, and expert/rank pressure signals.
- Designed a lightweight runtime resource model for MoE serving workloads, jointly modeling KV block reservation, prefill/decode co-scheduling, SLA urgency, and pressure-aware batching.
- Implemented telemetry and replay-pressure hooks to collect per-step KV usage, prefill/decode composition, expert load imbalance, and MoE timing signals for scheduler ablation.
- Validated the scheduler on `Qwen/Qwen1.5-MoE-A2.7B-Chat` with TensorRT-LLM `INT4 weight-only`, evaluating Balanced / Hot-Expert / Hot-Rank / Mixed Burst workloads with TTFT/TPOT tail latency and step-variance metrics.

## 19. 参考资料

- TensorRT-LLM Scheduler  
  https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html

- TensorRT-LLM Supported Models  
  https://nvidia.github.io/TensorRT-LLM/latest/models/supported-models.html

- TensorRT-LLM KV Cache System  
  https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html

- TensorRT-LLM Qwen Example README  
  https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md

- Qwen1.5-MoE-A2.7B-Chat Model Card  
  https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat

- TensorRT-LLM Performance Analysis  
  https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-analysis.html

- Scaling Expert Parallelism in TensorRT-LLM, Part 1  
  https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html

- Scaling Expert Parallelism in TensorRT-LLM, Part 2  
  https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog08_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html

- Scaling Expert Parallelism in TensorRT-LLM, Part 3  
  https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html

## 20. 最终项目边界

最终项目应该被定义为：

> 一个基于 TensorRT-LLM PyTorch backend 的 MoE-aware runtime scheduler enhancement，重点优化 MoE pressure 明显的 workload 下的 prefill/decode co-scheduling、KV cache resource management 和 pressure-aware batching，并以 `Qwen/Qwen1.5-MoE-A2.7B-Chat + TRT-LLM INT4 weight-only` 作为固定主验证路径。

不应该被定义为：

> 一个外部 router、一个 mock serving simulator、一个单独 kernel benchmark，或者一个重新实现的迷你 TensorRT-LLM。

这个边界非常重要，因为它决定了项目在面试中是“岗位相关的 runtime 架构优化”，还是“看起来相关但实际脱离 TensorRT-LLM 的外围实验”。
