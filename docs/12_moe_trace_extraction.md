# 12 MoE Trace Extraction

## Goal

Create a replay-compatible pressure trace format that the scheduler evaluation pipeline can consume.

## Implementation

- [scripts/generate_pressure_traces.py](C:/26spring\nv项目/full_version/trtllm-moe-runtime-exp/scripts/generate_pressure_traces.py)
- [scheduler/replay_pressure_provider.py](C:/26spring\nv项目/full_version/trtllm-moe-runtime-exp/scheduler/replay_pressure_provider.py)

## Full-version replay format

Each trace row stores:

- `request_id`
- `pressure_class`
- `pressure_score`
- `pressure_group`
- `source_kind`
- `expert_histogram`
- `rank_histogram`
- optional observed metrics:
  - `observed_e2e`
  - `observed_ttft`
  - `observed_tpot`

## Current artifact

- [hot_expert_replay_trace.jsonl](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/artifacts/moe_traces/hot_expert_replay_trace.jsonl)

## Important honesty note

The real TensorRT engine path on this machine does not expose live expert routing directly to Python. So the current replay trace is an **offline replay format** built from:

- the real Qwen workload records
- the real benchmark result file
- synthetic expert/rank histograms aligned with the assigned pressure class

This is enough to support replay-pressure evaluation, while keeping the limitation explicit.
