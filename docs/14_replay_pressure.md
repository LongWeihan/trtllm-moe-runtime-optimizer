# 14 Replay Pressure

## Implementation

- [scheduler/replay_pressure_provider.py](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/scheduler/replay_pressure_provider.py)
- [scripts/generate_pressure_traces.py](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/scripts/generate_pressure_traces.py)

## Current replay artifact

- [hot_expert_replay_trace.jsonl](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/artifacts/moe_traces/hot_expert_replay_trace.jsonl)

## Role in the project

Replay pressure sits between pure synthetic pressure and full live routing telemetry:

- more structured than synthetic-only labels
- still reproducible and easy to swap into the same benchmark harness
- compatible with request-level historical metrics

## Limitation

The current replay format is an offline replay representation, not a direct live expert-routing dump from the TRT engine.
