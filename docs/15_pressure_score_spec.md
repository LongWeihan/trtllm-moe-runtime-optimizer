# 15 Pressure Score Spec

## Goal

Define a small but explicit pressure contract for the full-version scheduler.

## Request-level fields

Each request may carry:

- `pressure_class`
- `pressure_score`
- `pressure_group`
- optional replay metadata:
  - `expert_histogram`
  - `rank_histogram`
  - `observed_e2e`
  - `observed_ttft`
  - `observed_tpot`

## Score interpretation

### `balanced = 1.0`

- no expected MoE hotspot
- baseline control

### `hot_expert = 2.2`

- expected expert concentration
- higher batch straggler risk

### `hot_rank = 2.6`

- expected rank concentration
- higher step-variance risk

## How the score is used

### v1

- pressure budget gates how many requests can coexist in one step

### v2

- pressure score still contributes to the batch fit check
- capacity scheduling also introduces request ranking and KV-reuse preference

## Current limitation

The score is manually calibrated and workload-aware. It is designed for scheduler experiments, not as a universal MoE latency oracle.
