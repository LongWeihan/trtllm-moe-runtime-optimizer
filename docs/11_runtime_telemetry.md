# 11 Runtime Telemetry

## Goal

Add minimal structured telemetry without changing scheduler behavior.

## Implementation

- [scheduler/telemetry.py](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/scheduler/telemetry.py)

Recorded fields:

- `step_name`
- `timestamp`
- `context_request_ids`
- `generation_request_ids`
- `num_context_requests`
- `num_generation_requests`
- `planned_total_tokens`
- `planned_total_pressure`
- `deferred_request_ids`
- `notes`

## Probe run

Artifact:

- [balanced_v1_probe_telemetry.jsonl](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/results/02_telemetry/balanced_v1_probe_telemetry.jsonl)

Companion result:

- [balanced_v1_probe.json](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/results/02_telemetry/balanced_v1_probe.json)

## What this proves

The full-version runtime can emit structured per-step scheduling telemetry on the real Qwen engine path.

## Limitation

This is intentionally still a minimal telemetry layer. It does not directly expose true expert routing from the built TRT engine.
