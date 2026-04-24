# TODO List

## 1. 使用方式

这份 `todolist.md` 是本项目的**执行主清单**。  
后续 Codex 自动推进项目时，必须按以下规则使用：

1. 只从**依赖已满足**的任务里选择下一个任务执行。
2. 每完成一个任务，必须同步更新本文件中的状态、完成时间、产物路径和备注。
3. 未产生产物、未完成验证、未写入结果的任务，不得标记为 `DONE`。
4. 若发生阻塞，必须把任务状态改为 `BLOCKED`，并在 `project_bootstrap_contract.md` 约定的位置补充 blocker 记录。
5. 如果任务需要拆分，必须新增子任务，不能直接跳过。

## 2. 状态图例

- `TODO`：未开始
- `IN_PROGRESS`：正在进行
- `DONE`：已完成且有验证与产物
- `BLOCKED`：被阻塞
- `SKIPPED`：明确放弃，并已记录原因

## 3. 固定约束

- 主模型固定为：`Qwen/Qwen1.5-MoE-A2.7B-Chat`
- 主量化路径固定为：TensorRT-LLM `INT4 weight-only`
- 主环境固定为：WSL2 Ubuntu 24.04
- 主实现固定为：TensorRT-LLM PyTorch backend 内部 `CapacityScheduler` / `MicroBatchScheduler` 增强
- 主故事线固定为：**MoE-first runtime scheduler enhancement**
- 项目完成前，**不得**把 dense 模型路径升级为主线
- 项目完成前，**不得**把 synthetic-only 结果当作最终端到端成果
- `Step 11` 对应的真实模型端到端验证是**正式里程碑**，不是装饰项

## 4. Phase 0 - 项目引导与骨架

### T00 建立工作区与仓库骨架

- 状态：`DONE`
- 依赖：无
- 目标：
  - 在 WSL ext4 中建立项目工作区
  - 建立约定的仓库目录结构
- 完成标准：
  - 工作区位于 `~/workspace/...`
  - 目录结构与 `project_bootstrap_contract.md` 一致
- 产物：
  - 目录树
  - `docs/00_workspace.md`
- 备注：
  - 不允许把主仓库放在 `/mnt/c/...`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - Windows 工作区建立于 `C:\26spring\nv项目\full_version\trtllm-moe-runtime-exp`
    - WSL 工作区建立于 `/home/a/trtllm-moe-runtime-exp-full`
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/00_workspace.md`

### T01 建立项目跟踪文档

- 状态：`DONE`
- 依赖：T00
- 目标：
  - 建立 `docs/decision_log.md`
  - 建立 `docs/blockers.md`
  - 建立 `docs/run_log.md`
- 完成标准：
  - 三份文件存在
  - 模板结构完整
- 产物：
  - `docs/decision_log.md`
  - `docs/blockers.md`
  - `docs/run_log.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 三份跟踪文档均已创建并写入初始模板
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/decision_log.md`
    - `full_version/trtllm-moe-runtime-exp/docs/blockers.md`
    - `full_version/trtllm-moe-runtime-exp/docs/run_log.md`

### T02 建立结果目录与命名模板

- 状态：`DONE`
- 依赖：T00
- 目标：
  - 建立 baseline、ablation、end-to-end 结果目录
  - 建立统一的文件命名规则
- 完成标准：
  - `results/` 和 `logs/` 目录就绪
  - 命名规则写入 `docs/artifact_index.md`
- 产物：
  - `docs/artifact_index.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - `results/`、`logs/` 和分层子目录均已建立
    - 命名规则已写入索引
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/artifact_index.md`

## 5. Phase 1 - 环境与 TensorRT-LLM Bring-up

### T10 采集环境指纹

- 状态：`DONE`
- 依赖：T00
- 目标：
  - 记录 GPU、驱动、CUDA、Python、WSL、磁盘与 `nsys`
- 完成标准：
  - 关键版本全部记录
  - 结论能判断是否满足主线要求
- 产物：
  - `docs/01_env_fingerprint.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 环境指纹已采集，确认 `4060 Ti 16GB + WSL2 Ubuntu 24.04` 适合单 GPU MoE runtime 项目
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/01_env_fingerprint.md`

### T11 安装 TensorRT-LLM 开发环境

- 状态：`DONE`
- 依赖：T10
- 目标：
  - 创建 Python venv
  - clone TensorRT-LLM
  - 使用 `TRTLLM_USE_PRECOMPILED=1 pip install -e .`
- 完成标准：
  - `from tensorrt_llm import LLM` 成功
- 产物：
  - `docs/02_install_log.md`
  - `logs/install/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - `/home/a/trtllm-moe-runtime-exp-full/venv` 已创建
    - TensorRT-LLM `v1.2.1` editable install 完成
    - `from tensorrt_llm import LLM` 路径可用
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/02_install_log.md`
    - `/home/a/trtllm-moe-runtime-exp-full/logs/install/`

### T12 跑通最小 backend sanity

- 状态：`DONE`
- 依赖：T11
- 目标：
  - 跑通一个最小生成脚本
  - 确认 GPU 正常工作
- 完成标准：
  - 单请求 generate 成功
  - 无明显环境错误
- 产物：
  - `scripts/sanity_backend.py`
  - `docs/03_backend_sanity.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 最小 backend import 成功
    - sanity 结果已落盘
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/scripts/sanity_backend.py`
    - `full_version/trtllm-moe-runtime-exp/docs/03_backend_sanity.md`
    - `full_version/trtllm-moe-runtime-exp/results/00_qwen15_sanity/backend_import.json`

## 6. Phase 2 - 固定主模型路径打通

### T20 下载固定主模型

- 状态：`DONE`
- 依赖：T12
- 目标：
  - 下载 `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- 完成标准：
  - 模型完整下载
  - 路径记录清楚
- 产物：
  - `docs/04_model_download.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 固定主模型已在 full workspace 中落盘
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/04_model_download.md`

### T21 记录官方 Qwen TRT-LLM 路径

- 状态：`DONE`
- 依赖：T20
- 目标：
  - 阅读并记录 TensorRT-LLM 官方 Qwen example 的 conversion/build 流程
- 完成标准：
  - 关键命令、脚本入口、模型支持说明记录清楚
- 产物：
  - `docs/05_qwen_trtllm_path.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 已记录官方 Qwen example 的支持矩阵与 conversion/build 命令
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/05_qwen_trtllm_path.md`

### T22 完成 checkpoint conversion

- 状态：`DONE`
- 依赖：T21
- 目标：
  - 将固定主模型转换到 TensorRT-LLM 可消费格式
- 完成标准：
  - conversion 成功
  - 中间产物路径明确
- 产物：
  - `docs/06_conversion_log.md`
  - `artifacts/model_conversion/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - INT4 weight-only checkpoint conversion 成功
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/06_conversion_log.md`
    - `/home/a/trtllm-moe-runtime-exp-full/artifacts/model_conversion/qwen15_moe_int4wo_checkpoint`

### T23 完成 INT4 weight-only 构建

- 状态：`DONE`
- 依赖：T22
- 目标：
  - 完成 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 的 TRT-LLM `INT4 weight-only` 构建
- 完成标准：
  - engine 或对应执行产物构建成功
  - 可进入运行阶段
- 产物：
  - `docs/07_int4wo_build.md`
  - `artifacts/qwen15_moe_int4wo/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - `Qwen1.5-MoE-A2.7B-Chat` 的 TRT-LLM INT4 WO engine 构建成功
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/07_int4wo_build.md`
    - `/home/a/trtllm-moe-runtime-exp-full/artifacts/qwen15_moe_int4wo`

### T24 完成真实主模型单请求推理

- 状态：`DONE`
- 依赖：T23
- 目标：
  - 用固定主模型完成至少一条单请求推理
- 完成标准：
  - 生成结果有效
  - 路径稳定可复现
- 产物：
  - `docs/08_qwen15_single_request.md`
  - `results/00_qwen15_sanity/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 真实单请求推理成功
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/08_qwen15_single_request.md`
    - `full_version/trtllm-moe-runtime-exp/results/00_qwen15_sanity/qwen15_single_request.txt`

### T25 完成真实主模型小规模 benchmark sanity

- 状态：`DONE`
- 依赖：T24
- 目标：
  - 用 `trtllm-bench` 或等价路径完成小规模 benchmark
- 完成标准：
  - 至少有一组 latency / throughput 数据
- 产物：
  - `docs/09_qwen15_bench_sanity.md`
  - `results/01_qwen15_bench_sanity/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 小规模 benchmark sanity 成功，已获得 latency / throughput 初始读数
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/09_qwen15_bench_sanity.md`
    - `full_version/trtllm-moe-runtime-exp/results/01_qwen15_bench_sanity/balanced_sanity.json`

## 7. Phase 3 - 代码路径与最小 telemetry

### T30 梳理 TensorRT-LLM scheduler 代码路径

- 状态：`DONE`
- 依赖：T12
- 目标：
  - 找出 PyExecutor、CapacityScheduler、MicroBatchScheduler、request state 的关键调用点
- 完成标准：
  - 有可追踪的调用链说明
- 产物：
  - `docs/10_codepath_notes.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 已定位 `BindMicroBatchScheduler` 与 `executor_request_to_llm_request` 为主 patch seam
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/10_codepath_notes.md`

### T31 加入最小 runtime telemetry

- 状态：`DONE`
- 依赖：T30
- 目标：
  - 不改变默认行为，只增加 step/request/KV 结构化 telemetry
- 完成标准：
  - telemetry JSONL 输出成功
  - overhead 不明显
- 产物：
  - `scheduler/telemetry.py`
  - `docs/11_runtime_telemetry.md`
  - `results/02_telemetry/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 结构化 telemetry JSONL 已成功输出
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/scheduler/telemetry.py`
    - `full_version/trtllm-moe-runtime-exp/docs/11_runtime_telemetry.md`
    - `full_version/trtllm-moe-runtime-exp/results/02_telemetry/balanced_v1_probe_telemetry.jsonl`

### T32 加入 MoE 相关 trace 提取接口

- 状态：`DONE`
- 依赖：T24, T30
- 目标：
  - 为 replay provider 准备 routing / expert histogram 提取接口
- 完成标准：
  - 至少可以离线导出 request 级 pressure 所需信息
- 产物：
  - `docs/12_moe_trace_extraction.md`
  - `artifacts/moe_traces/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - replay trace 导出脚本与 provider 已建立
    - 已生成 hot-expert replay trace 样例
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/12_moe_trace_extraction.md`
    - `full_version/trtllm-moe-runtime-exp/artifacts/moe_traces/hot_expert_replay_trace.jsonl`

## 8. Phase 4 - MoE Pressure System

### T40 实现 synthetic pressure provider

- 状态：`DONE`
- 依赖：T31
- 目标：
  - 提供 balanced / hot_expert / hot_rank 等 synthetic pressure
- 完成标准：
  - provider 可被 benchmark 调用
- 产物：
  - `scheduler/moe_pressure.py`
  - `docs/13_synthetic_pressure.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - balanced / hot_expert / hot_rank synthetic pressure 已可供 benchmark 调用
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/scheduler/moe_pressure.py`
    - `full_version/trtllm-moe-runtime-exp/docs/13_synthetic_pressure.md`

### T41 实现 replay pressure provider

- 状态：`DONE`
- 依赖：T32, T40
- 目标：
  - 从 `Qwen/Qwen1.5-MoE-A2.7B-Chat` trace 构建 replay provider
- 完成标准：
  - synthetic / replay 可在同一 benchmark 框架中切换
- 产物：
  - `scheduler/replay_pressure_provider.py`
  - `docs/14_replay_pressure.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - replay provider 已能加载 trace 并覆写 workload 中的 pressure metadata
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/scheduler/replay_pressure_provider.py`
    - `full_version/trtllm-moe-runtime-exp/docs/14_replay_pressure.md`

### T42 固化 pressure score 定义

- 状态：`DONE`
- 依赖：T40, T41
- 目标：
  - 给出 request/batch 级 pressure score 的正式定义
- 完成标准：
  - score 定义、阈值、使用位置和局限性写清楚
- 产物：
  - `docs/15_pressure_score_spec.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - request / batch 级 pressure contract 已固定
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/15_pressure_score_spec.md`

## 9. Phase 5 - MoE-specific Workloads

### T50 实现 Balanced MoE workload

- 状态：`DONE`
- 依赖：T40
- 目标：
  - 生成 pressure 分布均衡的 workload
- 完成标准：
  - JSONL 可复现
- 产物：
  - `workloads/balanced_moe.jsonl`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - Balanced MoE workload 已生成
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/workloads/balanced_moe.jsonl`

### T51 实现 Hot-Expert workload

- 状态：`DONE`
- 依赖：T40
- 目标：
  - 生成集中打热某组 experts 的 workload
- 完成标准：
  - JSONL 可复现
- 产物：
  - `workloads/hot_expert.jsonl`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - Hot-Expert workload 已生成
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/workloads/hot_expert.jsonl`

### T52 实现 Hot-Rank workload

- 状态：`DONE`
- 依赖：T40
- 目标：
  - 生成 rank 维度偏斜明显的 workload
- 完成标准：
  - JSONL 可复现
- 产物：
  - `workloads/hot_rank.jsonl`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - Hot-Rank workload 已生成
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/workloads/hot_rank.jsonl`

### T53 实现 Mixed Burst workload

- 状态：`DONE`
- 依赖：T40
- 目标：
  - 生成平时平稳、偶尔涌入高压请求的 workload
- 完成标准：
  - JSONL 可复现
- 产物：
  - `workloads/mixed_burst.jsonl`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - Mixed Burst workload 已生成
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/workloads/mixed_burst.jsonl`

### T54 实现 Repeated-Prefix under MoE Pressure workload

- 状态：`DONE`
- 依赖：T40
- 目标：
  - 生成共享前缀下叠加不同 pressure class 的 workload
- 完成标准：
  - JSONL 可复现
- 产物：
  - `workloads/repeated_prefix_moe.jsonl`
  - `docs/16_workload_spec.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - Repeated-Prefix under MoE Pressure workload 已生成
    - workload 规范已写入文档
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/workloads/repeated_prefix_moe.jsonl`
    - `full_version/trtllm-moe-runtime-exp/docs/16_workload_spec.md`

## 10. Phase 6 - Baseline

### T60 跑 default scheduler baseline

- 状态：`DONE`
- 依赖：T25, T50, T51, T52, T53, T54
- 目标：
  - 在固定主模型上跑默认 scheduler baseline
- 完成标准：
  - 五类 workload 至少各有一份 baseline 结果
- 产物：
  - `results/03_baseline_default/`
  - `docs/17_baseline_default.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 五类 workload 的 default baseline 已全部跑完
    - default baseline 结果已用于后续 replay trace 与 v1/v2 对比
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/results/03_baseline_default/`
    - `full_version/trtllm-moe-runtime-exp/docs/17_baseline_default.md`

### T61 跑强 baseline

- 状态：`DONE`
- 依赖：T60
- 目标：
  - 运行 default / overlap / `GUARANTEED_NO_EVICT` / `MAX_UTILIZATION`
- 完成标准：
  - 强 baseline 对比表可生成
- 产物：
  - `results/04_baseline_strong/`
  - `docs/18_baseline_strong.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - `GUARANTEED_NO_EVICT`、`MAX_UTILIZATION`、overlap 均已在五类 workload 上完成
    - 强 baseline 对比表已生成
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/results/04_baseline_strong/`
    - `full_version/trtllm-moe-runtime-exp/results/compare_tables/baseline_compare.md`
    - `full_version/trtllm-moe-runtime-exp/docs/18_baseline_strong.md`

### T62 输出 baseline 读数与 go/no-go 结论

- 状态：`DONE`
- 依赖：T61
- 目标：
  - 判断 MoE pressure 是否与 tail / variance 有明显关联
- 完成标准：
  - 给出下一阶段是否继续主线优化的结论
- 产物：
  - `docs/19_baseline_readout.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 给出 `GO` 结论
    - 结论为 generic strong baselines 不足以解决 hot workload tail，需要进入 MoE-aware scheduler 主线
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/19_baseline_readout.md`

## 11. Phase 7 - Scheduler v1

### T70 实现 MoE-aware microbatch scheduler v1

- 状态：`DONE`
- 依赖：T42, T30
- 目标：
  - 实现 decode-first + high-pressure request dispersion
- 完成标准：
  - patch 可运行
- 产物：
  - `scheduler/moe_microbatch_scheduler.py`
  - `docs/20_scheduler_v1_design.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - v1 scheduler patch 与最小 runtime resource model 已在 full workspace 中落盘
    - 可在真实 TRT engine 路径上通过 external step planning 运行
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/scheduler/moe_microbatch_scheduler.py`
    - `full_version/trtllm-moe-runtime-exp/scheduler/resource_model.py`
    - `full_version/trtllm-moe-runtime-exp/docs/20_scheduler_v1_design.md`

### T71 跑 v1 synthetic ablation

- 状态：`DONE`
- 依赖：T70, T50, T51, T52, T53, T54
- 目标：
  - 在 synthetic pressure 上做第一轮 ablation
- 完成标准：
  - 至少一个高压 workload 出现可解释收益
- 产物：
  - `results/05_v1_synthetic/`
  - `docs/21_scheduler_v1_synthetic.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 五类 workload 的 synthetic ablation 已完成
    - `Hot-Expert`、`Hot-Rank`、`Mixed Burst`、`Repeated-Prefix` 均出现可解释收益
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/results/05_v1_synthetic/`
    - `full_version/trtllm-moe-runtime-exp/docs/21_scheduler_v1_synthetic.md`

### T72 跑 v1 replay ablation

- 状态：`DONE`
- 依赖：T70, T41
- 目标：
  - 在 replay pressure 上验证 v1
- 完成标准：
  - replay 结果可复现
- 产物：
  - `results/06_v1_replay/`
  - `docs/22_scheduler_v1_replay.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - replay trace 已从真实 workload + baseline 结果导出
    - replay 结果方向与 synthetic 高度一致
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/artifacts/moe_traces/`
    - `full_version/trtllm-moe-runtime-exp/results/06_v1_replay/`
    - `full_version/trtllm-moe-runtime-exp/docs/22_scheduler_v1_replay.md`

## 12. Phase 8 - Scheduler v2

### T80 实现 MoE-aware capacity / admission

- 状态：`DONE`
- 依赖：T70, T42
- 目标：
  - 把 MoE pressure 引入 admission / capacity path
- 完成标准：
  - 代码可运行
  - 有明确 score 逻辑
- 产物：
  - `scheduler/moe_capacity_scheduler.py`
  - `docs/23_capacity_scheduler.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - capacity / admission 逻辑已实现
    - dynamic pressure budget、shared-prefix bonus、hot-rank penalty 已进入 score path
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/scheduler/moe_capacity_scheduler.py`
    - `full_version/trtllm-moe-runtime-exp/docs/23_capacity_scheduler.md`

### T81 实现 prefill under MoE pressure 控制

- 状态：`DONE`
- 依赖：T80
- 目标：
  - 让 prefill 插入受 decode pressure / MoE pressure 约束
- 完成标准：
  - chunked prefill 不再是无条件激进插入
- 产物：
  - `scheduler/adaptive_chunking.py`
  - `docs/24_prefill_control.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - adaptive chunking / prefill control 已实现
    - repeated-prefix 与 hot-request count 已纳入 batch-shape 调整
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/scheduler/adaptive_chunking.py`
    - `full_version/trtllm-moe-runtime-exp/docs/24_prefill_control.md`

### T82 跑 v2 完整 ablation

- 状态：`DONE`
- 依赖：T80, T81
- 目标：
  - 比较 v1 / v2 / baseline
- 完成标准：
  - 结果表、图、结论完整
- 产物：
  - `results/07_v2_ablation/`
  - `docs/25_scheduler_v2_ablation.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 五类 workload 的 v2 synthetic / replay 结果已完成
    - v2 成为 full-version 主结果：在 `Balanced`、`Mixed Burst`、`Repeated-Prefix` 上表现最完整
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/results/07_v2_ablation/`
    - `full_version/trtllm-moe-runtime-exp/docs/25_scheduler_v2_ablation.md`

## 13. Phase 9 - Step 11 正式里程碑

### T90 在真实主模型上接入 scheduler patch

- 状态：`DONE`
- 依赖：T24, T70
- 目标：
  - 让 patch 在 `Qwen/Qwen1.5-MoE-A2.7B-Chat` 实际执行路径上跑通
- 完成标准：
  - patched path 能完成至少一轮 workload
- 产物：
  - `docs/26_patch_on_qwen15.md`
  - `results/08_patch_qwen15/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - patch 已在固定主模型的真实执行路径上完成 patched run
    - telemetry 已同步产出
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/26_patch_on_qwen15.md`
    - `full_version/trtllm-moe-runtime-exp/results/08_patch_qwen15/`

### T91 完成真实主模型 baseline vs patched 对比

- 状态：`DONE`
- 依赖：T90, T61, T82
- 目标：
  - 在固定主模型上跑出最终对比
- 完成标准：
  - 至少一个 Hot-Expert 或 Hot-Rank workload 有可解释改进
- 产物：
  - `docs/27_qwen15_e2e_eval.md`
  - `results/09_qwen15_e2e_eval/`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - `Hot-Expert` 与 `Hot-Rank` 的 baseline vs patched 对比均已完成
    - 至少一个高压 workload 已获得可解释改进
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/27_qwen15_e2e_eval.md`
    - `full_version/trtllm-moe-runtime-exp/results/09_qwen15_e2e_eval/`

### T92 做 replay vs real path 一致性检查

- 状态：`DONE`
- 依赖：T91, T72
- 目标：
  - 对比 replay 结论与 live path 结论
- 完成标准：
  - 一致性与偏差被记录
- 产物：
  - `docs/28_replay_vs_live.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - replay 与 milestone e2e 结论的方向和量级已记录
    - 偏差主要来自 replay 信号来源的近似性，而非运行时测量不一致
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/28_replay_vs_live.md`

## 14. Phase 10 - 可选强化项

### T100 低开销 MoE telemetry 扩展

- 状态：`SKIPPED`
- 依赖：T91
- 目标：
  - 如果 telemetry overhead 明显，再做轻量扩展
- 完成标准：
  - overhead 对比可量化
- 产物：
  - `kernels/moe_telemetry/`
  - `docs/29_low_overhead_telemetry.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - telemetry overhead 不是主矛盾，故不进入 kernel 级强化
    - 已记录跳过原因
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/29_low_overhead_telemetry.md`

## 15. Phase 11 - 项目收尾与包装

### T110 产出最终报告

- 状态：`DONE`
- 依赖：T91, T92
- 目标：
  - 汇总方法、实验、图表、结论和局限性
- 完成标准：
  - 最终报告可独立阅读
- 产物：
  - `docs/final_report.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 最终报告已写出 full-version 的完整方法、结果、局限性与下一步
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/final_report.md`

### T111 产出面试讲稿

- 状态：`DONE`
- 依赖：T110
- 目标：
  - 形成 60 秒、3 分钟、深挖问答三个版本
- 完成标准：
  - 面试稿与项目结果一致
- 产物：
  - `docs/interview_talking_points.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 面试讲稿已和 full-version 结果对齐
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/interview_talking_points.md`

### T112 产出简历 bullet

- 状态：`DONE`
- 依赖：T110
- 目标：
  - 生成 3-5 条高密度简历 bullet
- 完成标准：
  - 与真实结果一致
- 产物：
  - `docs/resume_bullets.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 简历 bullet 已与 full-version 主结果对齐
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/resume_bullets.md`

### T113 做 reproducibility pass

- 状态：`DONE`
- 依赖：T110
- 目标：
  - 检查命令、路径、日志、图表和结论是否可复现
- 完成标准：
  - 没有缺文件、缺命令、缺结果索引的问题
- 产物：
  - `docs/reproducibility_checklist.md`
  - 完成时间：`2026-04-24`
  - 关键结果：
    - 复现命令、结果索引、文档入口已补齐
  - 产物路径：
    - `full_version/trtllm-moe-runtime-exp/docs/reproducibility_checklist.md`

## 16. 项目完成定义

整个项目只有在满足下面所有条件时才可标记为“完成”：

1. TensorRT-LLM backend 环境跑通。
2. `Qwen/Qwen1.5-MoE-A2.7B-Chat + TRT-LLM INT4 weight-only` 路径跑通。
3. synthetic pressure、replay pressure、真实主模型路径都已打通。
4. 至少完成一轮 default vs patched 的真实主模型对比。
5. 至少在一个高压 workload 上拿到可解释收益。
6. 最终报告、面试稿、简历 bullet 和结果索引都存在。

## 17. 当前执行入口

正式开工时，Codex 必须从以下入口开始：

1. 先读 `project_bootstrap_contract.md`
2. 再读 `codex_execution_discipline.md`
3. 再读本文件 `todolist.md`
4. 从 `T00` 开始推进
