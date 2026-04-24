# 13 Synthetic Pressure

## Implementation

- [scheduler/moe_pressure.py](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/scheduler/moe_pressure.py)

## Supported classes

- `balanced`
- `hot_expert`
- `hot_rank`

## Default scores

- `balanced = 1.0`
- `hot_expert = 2.2`
- `hot_rank = 2.6`

## Purpose

Synthetic pressure is the controllable pressure source used for:

- workload generation
- first-pass scheduler ablations
- stable reproduction of hot-expert and hot-rank scenarios

It is not claimed to be a kernel-time predictor. It is a scheduler input signal.
