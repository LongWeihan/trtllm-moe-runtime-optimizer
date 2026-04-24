# TRT-LLM MoE Pressure Lab

Pressure-aware admission and runtime scheduling for TensorRT-LLM MoE inference.

## Overview

`TRT-LLM MoE Pressure Lab` is the full engineering version of the project. It extends the step-planning slice into a broader runtime design that covers:

- explicit MoE pressure modeling
- admission control
- adaptive chunking and prefill control
- replay validation
- full baseline and ablation matrices

The core engineering question is:

> How should TensorRT-LLM runtime decisions change once MoE pressure is treated as a first-class scheduling resource?

This repository answers that question with a progressively stronger design:

- `v1`: pressure dispersion
- `v2`: pressure-aware admission and prefill control

## Why This Problem Matters

MoE serving failure modes are often structural rather than purely volumetric. A batch can fit memory and still be a poor execution decision if routing skew concentrates cost into a small set of experts or ranks.

In practice, that creates:

- unstable decode steps
- inflated tail latency
- burst-sensitive degradation
- repeated-prefix cases where useful batching structure is lost under pressure

Generic scheduler knobs improve utilization, but they do not explicitly reason about these interference patterns.

## System View

```text
workload / replay metadata
          |
          v
    pressure model
          |
          v
    request profile
          |
          v
     runtime budget
          |
          +----------------------+
          |                      |
          v                      v
  capacity admission      adaptive chunking
          |                      |
          +----------+-----------+
                     |
                     v
                 step plan
                     |
                     v
        TensorRT-LLM engine execution
                     |
                     +--> telemetry
                     +--> latency / throughput metrics
```

This decomposition is deliberate. It cleanly separates:

- signal extraction
- runtime modeling
- admission and planning
- execution and measurement

That separation is what makes the repository read like a runtime engineering project rather than a one-off benchmark harness.

## Design Principles

### Pressure is a schedulable resource

The scheduler should not treat MoE pressure as an explanatory label that appears only after a bad batch has already executed. Pressure belongs in the decision path.

### Runtime contracts should be explicit

The planner is built around explicit objects and budgets rather than hidden policy state. This makes the design easier to inspect, extend, and test.

### Batching quality matters as much as batching quantity

Utilization is necessary but not sufficient. A larger batch is not automatically a better batch if it concentrates contention into the same step.

### Different traffic patterns require different controls

Hotspot traffic, bursty traffic, and repeated-prefix traffic fail in different ways. The runtime design should acknowledge that.

## Runtime Model

The project models step latency as:

```text
step_latency(batch)
  =
  compute_cost(token_volume)
  +
  contention_cost(pressure, skew, phase_mix)
```

Where:

- `token_volume` captures dense compute
- `pressure` captures aggregate MoE pressure
- `skew` captures expert or rank concentration
- `phase_mix` captures whether prefill and decode compete in an unhealthy way

The scheduler rationale depends on two assumptions:

1. contention grows monotonically with pressure
2. in hotspot regimes, contention grows non-linearly as skew accumulates in the same step

This explains why generic utilization policies can improve the balanced control while remaining insufficient for hotspot-heavy MoE traffic: they optimize fit, not contention shape.

## Core Components

### [`scheduler/moe_pressure.py`](scheduler/moe_pressure.py)

Purpose:

- normalize MoE-specific runtime pressure into a stable planner signal

Why it was introduced:

- there is no useful pressure-aware planning without a portable request-level representation

### [`scheduler/resource_model.py`](scheduler/resource_model.py)

Purpose:

- establish an explicit runtime planning contract

Why it was introduced:

- planning logic becomes brittle when resource assumptions remain implicit

What it adds:

- `RequestProfile`
- `RuntimeBudget`
- `StepPlan`

### [`scheduler/moe_microbatch_scheduler.py`](scheduler/moe_microbatch_scheduler.py)

Purpose:

- implement the first pressure-aware scheduling policy

Why it was introduced:

- the project first needed to prove that pressure is worth scheduling against at all

What it adds:

- decode-first pressure dispersion
- explicit hot-request separation

### [`scheduler/moe_capacity_scheduler.py`](scheduler/moe_capacity_scheduler.py)

Purpose:

- turn pressure awareness into an admission decision

Why it was introduced:

- `v1` improved latency, but it was too eager to collapse the batch

What it adds:

- batch scoring
- dynamic pressure budget
- prefix-aware preference
- selected vs deferred record accounting

### [`scheduler/adaptive_chunking.py`](scheduler/adaptive_chunking.py)

Purpose:

- control prefill insertion under MoE pressure

Why it was introduced:

- prefill is not universally beneficial when decode pressure is already high

What it adds:

- pressure-sensitive microbatch adjustment
- repeated-prefix aware chunking control

### [`scheduler/replay_pressure_provider.py`](scheduler/replay_pressure_provider.py)

Purpose:

- validate whether conclusions remain stable under replay metadata

Why it was introduced:

- synthetic pressure alone is not enough for a full project

### [`scripts/run_full_matrix.sh`](scripts/run_full_matrix.sh)

Purpose:

- make the full ablation matrix reproducible

Why it was introduced:

- a full engineering project should expose its execution matrix, not hide it in shell history

## Scheduler Evolution

### V1: Pressure Dispersion

`v1` answers the first systems question:

> If pressure is exposed explicitly, does it meaningfully improve MoE batch quality?

The answer is yes.

Operationally, `v1`:

- separates hot requests aggressively
- lowers TTFT and E2E tail latency
- often reduces effective batch size too far

That makes `v1` a strong latency-first policy and an important diagnostic, but not yet a balanced runtime design.

### V2: Admission and Prefill Control

`v2` answers the next question:

> Can pressure awareness be made less brittle by making admission and prefill policy explicit?

This is the main contribution of the repository.

`v2` introduces:

- pressure-aware admission
- dynamic pressure budgets
- repeated-prefix preference
- adaptive chunking under pressure

This is the point where the project becomes more clearly a runtime architecture enhancement rather than a single scheduling rule.

## Experimental Protocol

### Workloads

- `Balanced MoE`
- `Hot-Expert`
- `Hot-Rank`
- `Mixed Burst`
- `Repeated-Prefix under MoE Pressure`

### Baselines

- default batching
- `GUARANTEED_NO_EVICT`
- `MAX_UTILIZATION`
- overlap / chunked prefill baseline

### Validation environment

The reported measurements were collected on:

- model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- quantization: TensorRT-LLM `INT4 weight-only`
- hardware: `RTX 4060 Ti 16GB`

## Results

### Strong-baseline conclusion

The strong baselines were necessary controls, but they were not sufficient:

- they improved the balanced control modestly
- they did not materially improve hotspot-heavy tail behavior
- they did not address MoE-specific pressure structure

### Selected quantitative comparisons

| Workload | Candidate | Baseline E2E p90 | Candidate E2E p90 | Baseline Throughput | Candidate Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| Balanced | `MAX_UTILIZATION` | `1.4786s` | `1.4768s` | `280.39` | `282.72` |
| Balanced | `v2 replay` | `1.4786s` | `1.4541s` | `280.39` | `305.78` |
| Hot-Expert | `v1 replay` | `1.8421s` | `1.5698s` | `301.32` | `98.43` |
| Hot-Expert | `v2 replay` | `1.8421s` | `1.7928s` | `301.32` | `169.64` |
| Hot-Rank | `v1 replay` | `1.9107s` | `1.7123s` | `293.97` | `100.07` |
| Hot-Rank | `v2 replay` | `1.9107s` | `1.7186s` | `293.97` | `99.26` |
| Mixed Burst | `v2 replay` | `1.9723s` | `1.5660s` | `263.02` | `184.46` |
| Repeated-Prefix + Pressure | `v2 replay` | `1.7533s` | `1.2848s` | `242.35` | `130.94` |

### What the data supports

#### Balanced traffic

`v2 replay` is the best result on the balanced control:

- lower E2E p90
- substantially lower step variance
- higher throughput

This matters because it shows the project is not simply trading everything away for hotspot wins.

#### Hot-Expert

`v1` proves the signal most clearly:

- it isolates pressure aggressively
- it significantly improves tail latency
- it pays a large throughput penalty

`v2` preserves a meaningful latency win while recovering a large fraction of that lost batching.

#### Hot-Rank

`Hot-Rank` remains the hardest open case. Pressure isolation helps latency, but throughput recovery remains weak even in `v2`.

#### Mixed Burst and Repeated-Prefix

These are the most compelling `v2` workloads because they show the value of the added architecture:

- `Mixed Burst`: better tail handling under non-stationary pressure
- `Repeated-Prefix under MoE Pressure`: better behavior when pressure control and structural reuse need to coexist

## Main Conclusions

1. MoE pressure belongs in the runtime decision path.
2. Generic utilization policies are not enough for hotspot-heavy MoE traffic.
3. `v1` proves the pressure signal is operationally meaningful.
4. `v2` turns that insight into a more balanced runtime design by adding admission and prefill control.
5. The clearest remaining gap is throughput recovery for `Hot-Rank`.

## Limitations

The main limitation is explicit:

- the final quantitative path uses the real TensorRT engine backend
- the planner is applied through batch composition on that path
- the result is therefore not a pure in-backend PyTorch quantitative benchmark

That limitation defines the exact scope of the claim, but the reported measurements themselves are real engine measurements.

## Repository Layout

- [`scheduler/`](scheduler) - pressure model, resource model, admission logic, chunking control, telemetry
- [`scripts/`](scripts) - matrix execution, replay generation, summarization
- [`workloads/`](workloads) - fixed MoE workloads
- [`artifacts/moe_traces/`](artifacts/moe_traces) - replay traces
- [`results/`](results) - baselines, ablations, telemetry, compare tables
- [`docs/`](docs) - detailed notes, summaries, final report

## Reproducibility

Matrix execution:

```bash
bash scripts/run_full_matrix.sh baseline-default
bash scripts/run_full_matrix.sh baseline-strong
bash scripts/run_full_matrix.sh traces
bash scripts/run_full_matrix.sh v1-synthetic
bash scripts/run_full_matrix.sh v1-replay
bash scripts/run_full_matrix.sh v2-ablation
bash scripts/run_full_matrix.sh qwen15-final
```

Summary generation:

```bash
bash scripts/wsl_env.sh python scripts/collect_metrics.py ...
```

Reference documents:

- [`docs/final_report.md`](docs/final_report.md)
- [`docs/result_summary.md`](docs/result_summary.md)
- [`results/compare_tables/selected_summary.md`](results/compare_tables/selected_summary.md)
