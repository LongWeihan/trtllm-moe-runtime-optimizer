# TRT-LLM MoE Pressure Lab

MoE pressure modeling, admission planning, and runtime scheduling ablations for TensorRT-LLM on a fixed `Qwen/Qwen1.5-MoE-A2.7B-Chat` `INT4 weight-only` engine path.

## What This Project Does

This repo is the full ablation-oriented version of the project. It goes beyond a single scheduler patch and treats MoE pressure as part of the runtime decision path:

- explicit `RequestProfile -> RuntimeBudget -> StepPlan`
- `v1` pressure-dispersion scheduler
- `v2` admission and adaptive chunking
- five fixed MoE workloads
- strong baselines, replay validation, and real end-to-end milestone runs

## Fixed Setup

- Model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- Quantization: TensorRT-LLM `INT4 weight-only`
- Hardware: `RTX 4060 Ti 16GB`
- Main files:
  - [`scheduler/resource_model.py`](scheduler/resource_model.py)
  - [`scheduler/moe_microbatch_scheduler.py`](scheduler/moe_microbatch_scheduler.py)
  - [`scheduler/moe_capacity_scheduler.py`](scheduler/moe_capacity_scheduler.py)
  - [`scheduler/adaptive_chunking.py`](scheduler/adaptive_chunking.py)
  - [`scripts/run_full_matrix.sh`](scripts/run_full_matrix.sh)

## Key Results

### Strong-baseline takeaway

Generic knobs such as `MAX_UTILIZATION` and overlap tuning helped the balanced control a little, but they did not materially solve the hot MoE tails.

### Main result table

| Workload | Baseline E2E p90 | Candidate E2E p90 | Throughput (`tok/s`) | Note |
| --- | ---: | ---: | ---: | --- |
| Balanced | `1.4786s` | `1.4541s` (`v2 replay`) | `280.39 -> 305.78` | best balanced-control result |
| Hot-Expert | `1.8421s` | `1.7928s` (`v2 replay`) | `301.32 -> 169.64` | throughput partially recovered vs `v1` |
| Hot-Rank | `1.9107s` | `1.7186s` (`v2 replay`) | `293.97 -> 99.26` | hardest remaining case |
| Mixed Burst | `1.9723s` | `1.5660s` (`v2 replay`) | `263.02 -> 184.46` | strongest burst-traffic result |
| Repeated-Prefix + MoE Pressure | `1.7533s` | `1.2848s` (`v2 replay`) | `242.35 -> 130.94` | strongest structure-aware result |

### Why `v1` still matters

`v1` is the clearest proof that the pressure signal is useful:

- Hot-Expert `TTFT p90`: `0.0740s -> 0.0107s`
- Hot-Expert `E2E p90`: `1.8421s -> 1.5698s`

But `v1` pays for that by collapsing the batch too aggressively. `v2` is the more balanced runtime design.

## Repository Layout

- [`scheduler/`](scheduler) - runtime resource model, admission logic, adaptive chunking, pressure model
- [`scripts/`](scripts) - matrix runner, baseline and patched drivers, trace generation, metrics collection
- [`workloads/`](workloads) - balanced, hot, bursty, and repeated-prefix MoE workloads
- [`artifacts/moe_traces/`](artifacts/moe_traces) - replay traces
- [`results/`](results) - baselines, ablations, telemetry, compare tables
- [`docs/`](docs) - phase-by-phase notes, final report, result summary
- [`planning/`](planning) - execution contract, todolist, and project design docs

## Reproduce

Full matrix:

```bash
bash scripts/run_full_matrix.sh baseline-default
bash scripts/run_full_matrix.sh baseline-strong
bash scripts/run_full_matrix.sh traces
bash scripts/run_full_matrix.sh v1-synthetic
bash scripts/run_full_matrix.sh v1-replay
bash scripts/run_full_matrix.sh v2-ablation
bash scripts/run_full_matrix.sh qwen15-final
```

Summaries:

```bash
bash scripts/wsl_env.sh python scripts/collect_metrics.py ...
```

Useful entry points:

- [`docs/final_report.md`](docs/final_report.md)
- [`docs/result_summary.md`](docs/result_summary.md)
- [`results/compare_tables/selected_summary.md`](results/compare_tables/selected_summary.md)

## Caveat

The final quantitative benchmark uses the real TensorRT engine backend with the planner externalized into batch composition. The runtime seam work is real, but the final measured comparison is not a pure in-backend PyTorch quantitative benchmark.

## Project Notes

- Final report: [`docs/final_report.md`](docs/final_report.md)
- Result summary: [`docs/result_summary.md`](docs/result_summary.md)
- Planning docs: [`planning/`](planning)
