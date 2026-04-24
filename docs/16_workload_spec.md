# 16 Workload Spec

## Generator

- [scripts/generate_workloads.py](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/scripts/generate_workloads.py)

## Fixed workload set

### 1. `Balanced MoE`

- file:
  - [balanced_moe.jsonl](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/workloads/balanced_moe.jsonl)
- role:
  - balanced control
  - non-regression check

### 2. `Hot-Expert`

- file:
  - [hot_expert.jsonl](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/workloads/hot_expert.jsonl)
- role:
  - concentrated expert pressure
  - batch straggler and tail-latency probe

### 3. `Hot-Rank`

- file:
  - [hot_rank.jsonl](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/workloads/hot_rank.jsonl)
- role:
  - rank-skew probe
  - step-variance probe

### 4. `Mixed Burst`

- file:
  - [mixed_burst.jsonl](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/workloads/mixed_burst.jsonl)
- role:
  - mostly balanced traffic with bursty hot pressure phases

### 5. `Repeated-Prefix under MoE Pressure`

- file:
  - [repeated_prefix_moe.jsonl](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/workloads/repeated_prefix_moe.jsonl)
- role:
  - combines shared-prefix structure with different pressure classes

## Shared record fields

Each workload record includes:

- `request_id`
- `prompt`
- `max_tokens`
- `pressure_class`
- `pressure_score`
- `pressure_group`
- `workload_kind`

Repeated-prefix records also include:

- `shared_prefix_id`

## Generation

Current full-version generation command:

```bash
python scripts/generate_workloads.py --output-dir workloads --count 12
```
