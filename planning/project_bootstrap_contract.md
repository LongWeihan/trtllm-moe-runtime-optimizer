# Project Bootstrap Contract

## 1. 文件目的

这份文件用于在项目正式开工前，固定以下事项：

- 项目边界
- 架构设计
- 仓库结构
- 产物约定
- 命名规范
- 结果与文档契约
- 里程碑定义

它是本项目的**启动合同**。

## 2. 项目身份

### 项目名称

**MoE-Aware Runtime Scheduling Enhancement for TensorRT-LLM**

### 固定目标

在 TensorRT-LLM PyTorch backend 内部，实现一个面向 MoE inference 的 runtime scheduler enhancement，让调度器在 batch selection 和 admission 时显式感知：

- expert/rank pressure
- decode tail 风险
- KV 资源竞争
- prefill 插入代价

并在固定主模型：

**`Qwen/Qwen1.5-MoE-A2.7B-Chat + TensorRT-LLM INT4 weight-only`**

上完成真实端到端验证。

## 3. 固定技术决策

### 3.1 环境

- Host：Windows 11
- 主执行环境：WSL2 Ubuntu 24.04
- 主开发位置：WSL ext4 文件系统
- 主推理框架：TensorRT-LLM PyTorch backend

### 3.2 模型

- 主模型：`Qwen/Qwen1.5-MoE-A2.7B-Chat`
- 主量化：TensorRT-LLM `INT4 weight-only`
- 主验证路径：真实 TensorRT-LLM 端到端路径

### 3.3 实现边界

允许增强：

- `CapacityScheduler`
- `MicroBatchScheduler`
- telemetry / replay-pressure provider
- workload generator
- benchmark harness

不允许重写：

- executor 主体
- KV cache manager 主体
- attention kernel
- MoE kernel
- 通信层

### 3.4 项目主线

项目主线固定为：

> **MoE-first runtime scheduler enhancement**

而不是：

- dense-first benchmark 项目
- kernel 优化项目
- 外部 router 项目
- 分布式多 GPU serving 项目

## 4. 非目标

以下内容明确不作为当前项目的主交付物：

1. 多 GPU EP / WideEP / DWDP
2. disaggregated serving / KV transfer
3. AllToAll 优化
4. 通用 MoE GEMM 重写
5. dense 模型主线 benchmark
6. 一个新的 toy serving system

## 5. 架构设计定稿

项目的规范架构如下：

```text
Workload Generator
  |
  +-- Balanced MoE workload
  +-- Hot-Expert workload
  +-- Hot-Rank workload
  +-- Mixed Burst workload
  +-- Repeated-Prefix under MoE Pressure
  v

TensorRT-LLM Benchmark / Serve Entry
  |
  v
TensorRT-LLM PyTorch Backend
  |
  +-- PyExecutor
  +-- Request State Queue
  +-- KVCacheManager
  +-- CapacityScheduler (patched)
  +-- MicroBatchScheduler (patched)
  +-- Telemetry Hook
  +-- Pressure Provider Interface
         +-- synthetic provider
         +-- replay provider
         +-- optional live provider
  v

Results + Logs + Reports
```

### 5.1 核心运行时数据流

```text
request
  -> request features
  -> pressure provider
  -> scheduler score
  -> batch selection / admission
  -> TensorRT-LLM forward step
  -> telemetry output
  -> result aggregation
```

### 5.2 固定 workload 集合

正式 benchmark 只允许使用以下 5 类主 workload：

1. `Balanced MoE`
2. `Hot-Expert`
3. `Hot-Rank`
4. `Mixed Burst`
5. `Repeated-Prefix under MoE Pressure`

如需新增 workload，必须：

- 写入 `docs/decision_log.md`
- 同时更新 `todolist.md`

## 6. 规范仓库结构

项目正式仓库应遵循如下结构：

```text
trtllm-moe-runtime-exp/
  README.md
  docs/
    00_workspace.md
    01_env_fingerprint.md
    02_install_log.md
    03_backend_sanity.md
    04_model_download.md
    05_qwen_trtllm_path.md
    06_conversion_log.md
    07_int4wo_build.md
    08_qwen15_single_request.md
    09_qwen15_bench_sanity.md
    10_codepath_notes.md
    11_runtime_telemetry.md
    12_moe_trace_extraction.md
    13_synthetic_pressure.md
    14_replay_pressure.md
    15_pressure_score_spec.md
    16_workload_spec.md
    17_baseline_default.md
    18_baseline_strong.md
    19_baseline_readout.md
    20_scheduler_v1_design.md
    21_scheduler_v1_synthetic.md
    22_scheduler_v1_replay.md
    23_capacity_scheduler.md
    24_prefill_control.md
    25_scheduler_v2_ablation.md
    26_patch_on_qwen15.md
    27_qwen15_e2e_eval.md
    28_replay_vs_live.md
    29_low_overhead_telemetry.md
    artifact_index.md
    blockers.md
    decision_log.md
    run_log.md
    final_report.md
    interview_talking_points.md
    resume_bullets.md
    reproducibility_checklist.md
  scripts/
    sanity_backend.py
    generate_workloads.py
    generate_pressure_traces.py
    run_baseline.py
    run_ablation.py
    plot_results.py
  scheduler/
    telemetry.py
    moe_pressure.py
    replay_pressure_provider.py
    moe_microbatch_scheduler.py
    moe_capacity_scheduler.py
    adaptive_chunking.py
    resource_model.py
  workloads/
    balanced_moe.jsonl
    hot_expert.jsonl
    hot_rank.jsonl
    mixed_burst.jsonl
    repeated_prefix_moe.jsonl
  artifacts/
    model_conversion/
    qwen15_moe_int4wo/
    moe_traces/
  logs/
    install/
    build/
    runs/
  results/
    00_qwen15_sanity/
    01_qwen15_bench_sanity/
    02_telemetry/
    03_baseline_default/
    04_baseline_strong/
    05_v1_synthetic/
    06_v1_replay/
    07_v2_ablation/
    08_patch_qwen15/
    09_qwen15_e2e_eval/
```

## 7. 产物契约

每种活动都必须产生固定类型的产物。

### 7.1 安装类任务

必须留下：

- 安装命令
- 版本信息
- 错误与修复记录

### 7.2 模型类任务

必须留下：

- 模型来源
- 下载路径
- conversion 命令
- build 命令
- 结果路径

### 7.3 benchmark 类任务

必须留下：

- workload 文件
- 配置
- 原始结果
- 汇总结果
- 简要结论

### 7.4 调度器 patch 类任务

必须留下：

- 修改文件列表
- 逻辑说明
- 对应 benchmark 结果

## 8. 命名规范

### 8.1 实验命名

统一格式：

```text
<date>_<model>_<quant>_<workload>_<variant>
```

示例：

```text
2026-04-23_qwen15moe_int4wo_hotexpert_default
2026-04-23_qwen15moe_int4wo_hotexpert_moev1
```

### 8.2 结果目录命名

- baseline 结果进入 `results/03_*` 或 `results/04_*`
- ablation 结果进入 `results/05_*` / `results/06_*` / `results/07_*`
- 真实端到端对比进入 `results/09_qwen15_e2e_eval/`

### 8.3 文档命名

- 一律采用数字前缀保证阅读顺序
- 最终类文档不使用临时文件名如 `new.md`、`draft2.md`

## 9. 指标契约

正式报告中的核心指标固定为：

### 9.1 User-facing

- TTFT p50 / p90 / p99
- TPOT p50 / p90 / p99
- end-to-end request latency

### 9.2 Runtime

- step latency variance
- queue wait distribution
- paused request count
- scheduler overhead per step

### 9.3 MoE-specific

- expert load CV
- max-rank / mean-rank token ratio
- hot expert frequency
- pressure threshold trigger frequency

### 9.4 系统

- GPU utilization
- KV block usage
- peak memory / HBM usage

## 10. 基线契约

正式对比至少包含以下 baseline：

1. TensorRT-LLM default scheduler
2. overlap / 官方推荐强 baseline
3. v1: MoE-aware microbatch scheduler
4. v2: v1 + MoE-aware admission
5. v3: v2 + prefill under pressure control

最终对比必须覆盖真实主模型路径，不能只停留在 synthetic 或 replay。

## 11. 决策与 blocker 记录契约

### `docs/decision_log.md`

用于记录：

- 为什么改某个阈值
- 为什么新增/删除 workload
- 为什么某个方案被放弃

每条记录至少包括：

- 日期
- 决策项
- 备选项
- 最终决定
- 原因

### `docs/blockers.md`

用于记录：

- 当前阻塞点
- 已尝试路径
- 失败原因
- 下一步建议

## 12. 里程碑定义

### M1：环境可用

定义：

- TensorRT-LLM 安装成功
- 最小 backend sanity 成功

### M2：固定主模型路径可用

定义：

- `Qwen/Qwen1.5-MoE-A2.7B-Chat` 下载成功
- conversion 成功
- INT4 WO 构建成功
- 单请求生成成功

### M3：baseline 完整

定义：

- 五类 workload baseline 跑完
- baseline 读数与 go/no-go 结论生成

### M4：scheduler patch 可运行

定义：

- v1 和 v2 patch 都能跑通
- 至少一轮 ablation 完成

### M5：真实端到端验证完成

定义：

- patched scheduler 在真实主模型路径上跑通
- 至少一个高压 workload 有可解释改进

### M6：项目可交付

定义：

- 最终报告
- 面试稿
- 简历 bullet
- 可复现性检查

## 13. Definition of Done

单个任务完成的定义：

1. 有代码或命令执行
2. 有产物落盘
3. 有结果或日志
4. `todolist.md` 已更新

整个项目完成的定义：

1. M1-M6 全部达成
2. 真实主模型端到端验证完成
3. 至少一个高压 workload 拿到可解释收益
4. 没有未记录的关键 blocker

## 14. 开工顺序

正式开工时，执行顺序固定为：

1. 阅读本文件
2. 阅读 `codex_execution_discipline.md`
3. 阅读 `todolist.md`
4. 从 `T00` 开始执行

不允许跳过引导文档直接开写代码。
