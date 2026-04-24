# Codex Execution Discipline

## 1. 目的

这份文件是把项目“完整交给 Codex 自动推进”时的执行纪律。  
Codex 不是自由探索，而是按**固定边界、固定主线、固定产物要求**推进项目。

## 2. 项目总目标

项目目标固定为：

> 在 TensorRT-LLM PyTorch backend 内部实现一个面向 MoE inference 的 runtime scheduler enhancement，让调度器显式感知 expert/rank pressure、decode tail 风险、KV 资源竞争和 prefill 插入代价，并在固定主模型 `Qwen/Qwen1.5-MoE-A2.7B-Chat + TensorRT-LLM INT4 weight-only` 上完成端到端验证。

## 3. 单一事实来源

Codex 执行时，必须按以下优先级理解项目：

1. `project_bootstrap_contract.md`  
   用于固定边界、架构、仓库结构、产物约定、命名规则、DoD
2. `todolist.md`  
   用于固定任务顺序、依赖、完成标准与执行状态
3. `trtllm_moe_runtime_4060ti_implementation_plan_moe_first.md`  
   用于落地实施思路和里程碑解释
4. `trtllm_moe_runtime_architecture_optimization_plan.md`  
   用于架构动机、方法设计和面试叙事

若文件之间出现冲突：

- **边界、模型、仓库、产物约定** 以 `project_bootstrap_contract.md` 为准
- **任务顺序与是否完成** 以 `todolist.md` 为准
- **方法叙事与动机** 以两份项目方案文档为准

## 4. 不可变更项

除非用户明确重新决策，否则 Codex **不得**擅自改变以下事项：

1. 主模型：`Qwen/Qwen1.5-MoE-A2.7B-Chat`
2. 主量化路径：TensorRT-LLM `INT4 weight-only`
3. 主环境：WSL2 Ubuntu 24.04
4. 主实现边界：TensorRT-LLM backend 内部 scheduler/resource model 增强
5. 主故事线：MoE-first runtime scheduler enhancement

Codex 不得擅自：

- 把 dense 模型升级为主线
- 把 synthetic-only 结果包装成最终端到端成果
- 把项目转成外部 router / mock simulator
- 把项目主线改成 kernel 重写
- 把项目主线改成多 GPU / disaggregated serving

## 5. 执行总原则

### 原则 1：先打通真实路径，再谈优化

Codex 必须先完成：

- TensorRT-LLM 环境
- 固定主模型 conversion/build/runtime
- baseline

之后才进入 scheduler patch。

### 原则 2：每一步都必须有产物

Codex 不得只说“完成了”。  
每个任务都必须至少对应：

- 一份文档
- 一份日志
- 一份结果目录
- 或一份代码变更

### 原则 3：没有验证，不算完成

任务只有同时满足下面三条时才能标记为 `DONE`：

1. 代码或命令执行过
2. 结果被保存
3. 结论被写入文档

### 原则 4：优先最小可运行改动

Codex 每次应做最小必要改动，避免：

- 大规模无关重构
- 顺手做美化
- 在未验证前同时改很多路径

### 原则 5：真实模型端到端验证是正式里程碑

`Step 11` 对应的真实主模型端到端验证，不得被降级为“有空再试试”。

## 6. Codex 的标准执行循环

每轮执行必须遵循以下循环：

1. 阅读 `todolist.md`，找到**第一个依赖已满足且状态为 `TODO`** 的任务
2. 阅读该任务对应的完成标准与产物要求
3. 只修改与该任务直接相关的文件
4. 运行必要命令
5. 保存结果、日志、文档
6. 更新 `todolist.md`
7. 如有重要取舍，写入 `docs/decision_log.md`
8. 再进入下一个任务

不允许：

- 跳过依赖
- 批量声称完成多个任务但没有对应证据
- 不更新 `todolist.md`

## 7. 状态更新纪律

Codex 必须维护 `todolist.md` 的真实状态。

### 开始任务时

- 把状态改为 `IN_PROGRESS`

### 完成任务时

- 把状态改为 `DONE`
- 在任务下补充：
  - 完成时间
  - 关键结果
  - 产物路径

### 被阻塞时

- 把状态改为 `BLOCKED`
- 在 `docs/blockers.md` 中写明：
  - 阻塞点
  - 已尝试的方法
  - 下一步需要什么条件

## 8. 何时必须停下来记录，而不是硬闯

出现下面情况时，Codex 不得“闷头继续试到天荒地老”，而是必须记录并收口：

1. Hugging Face 下载需要用户登录或权限
2. TensorRT-LLM 安装与当前 CUDA / Python 版本出现系统级冲突
3. `Qwen/Qwen1.5-MoE-A2.7B-Chat` 的 INT4 WO 构建在合理尝试后仍不可达
4. 同一问题尝试 3 种路径仍无进展
5. benchmark 持续 OOM 且无法通过 batch/sequence 配置收敛
6. scheduler patch 破坏默认路径稳定性

此时必须：

- 收集错误
- 记录已尝试方案
- 明确当前最优下一步

## 9. 证据纪律

Codex 不得出现以下行为：

- 没跑命令却说“已跑通”
- 没有 benchmark 结果却说“有提升”
- 没有真实模型结果却说“端到端验证完成”
- 只在对话里说结论，不把结论写入文件

Codex 完成关键节点时必须留下：

### 环境节点

- 版本记录
- 安装命令
- sanity 输出摘要

### 模型节点

- 下载路径
- conversion/build 命令
- 生成示例

### benchmark 节点

- workload 文件
- 结果文件
- 汇总表或图

### patch 节点

- 修改文件列表
- 核心逻辑说明
- 回归验证结果

## 10. 范围控制纪律

Codex 可以优化，但不能擅自换题。

### 允许的灵活性

- 微调具体文件组织
- 调整任务拆分粒度
- 增加必要的中间脚本
- 增加为了验证所需的小型辅助文档

### 不允许的漂移

- 从 MoE-first 漂移成 dense-first
- 从 runtime scheduler 漂移成 kernel 项目
- 从 TensorRT-LLM 内部增强漂移成外部策略层
- 从单 GPU 可落地项目漂移成依赖远端大机器的方案

## 11. 完成定义纪律

Codex 只有在下面全部满足后，才可以对外宣称项目“完成”：

1. 固定主模型真实端到端路径打通
2. baseline 完整
3. scheduler patch 至少一轮真实主模型对比完成
4. 至少一个高压 workload 有可解释收益
5. 最终报告存在
6. 面试讲稿存在
7. `todolist.md` 中所有必做任务都已 `DONE`

## 12. 对用户的沟通纪律

Codex 向用户汇报时应：

- 短
- 准
- 有证据

优先汇报：

1. 当前在做哪个任务
2. 是否完成
3. 核心产物是什么
4. 是否有 blocker

不要用空泛表述：

- “差不多好了”
- “应该可以”
- “大概有提升”

## 13. 最后一条纪律

Codex 的目标不是“尽快写很多代码”，而是：

> **按固定项目边界，把一个 MoE-first 的 TensorRT-LLM runtime scheduler 项目真正推进到可以验收、可以讲、可以复现。**

如果某一步做不到，正确动作不是装作做到了，而是把问题写清楚，把现场收干净，把下一步变得明确。
