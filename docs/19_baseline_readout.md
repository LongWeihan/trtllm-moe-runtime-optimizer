# 19 Baseline Readout

Inputs:

- `docs/17_baseline_default.md`
- `docs/18_baseline_strong.md`
- `results/compare_tables/baseline_compare.md`

## Go / No-Go

Decision: **GO**

## Why continue

The baseline phase showed that MoE pressure was associated with worse tail behavior, and the generic strong baselines were not enough to address it.

Signals:

1. `Hot-Expert` and `Hot-Rank` both had materially worse E2E p90 than the balanced control.
2. `Mixed Burst` had the highest step-latency variance among the fixed workloads.
3. `Repeated-Prefix under MoE Pressure` exposed a useful combination of KV locality and pressure skew.
4. `MAX_UTILIZATION` and overlap tuning did not materially fix the hot-workload tails.

## Project implication

The project should not stop at generic scheduling knobs.  
It should continue into:

- explicit pressure-aware batch formation
- admission / capacity control
- repeated-prefix aware planning

That directly motivated `v1` and later `v2`.
