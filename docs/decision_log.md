# Decision Log

## 2026-04-24

### D001 - Full-version workspace layout

- Options:
  - reuse the 24h workspace directly
  - create a separate full-version workspace
- Decision:
  - create a separate workspace at `full_version/trtllm-moe-runtime-exp` and `/home/a/trtllm-moe-runtime-exp-full`
- Reason:
  - keeps the full-version run isolated from the 24h slice
  - allows independent docs, results, and task tracking

### D002 - Keep the quantitative path on the real TensorRT engine backend

- Options:
  - force a pure in-backend PyTorch quantitative benchmark
  - keep the real quantized TensorRT engine path and externalize the planner logic
- Decision:
  - keep the real quantized TensorRT engine path and apply the same `resource_model -> step_plan` logic in external batch composition
- Reason:
  - preserves the fixed `Qwen1.5-MoE + INT4 WO` path
  - avoids a misleading fallback to a non-matching benchmark setup
  - keeps the final measurement real on this machine

### D003 - Promote v2 as the main full-version result

- Options:
  - keep `v1` as the headline because its latency gains are largest
  - promote `v2` because it better reflects runtime architecture enhancement
- Decision:
  - promote `v2` as the main full-version result
- Reason:
  - `v1` proves the pressure signal but is too throughput-destructive
  - `v2` adds admission and prefill control
  - `v2` gives the best balanced story on `Balanced`, `Mixed Burst`, and `Repeated-Prefix under MoE Pressure`

### D004 - Keep Hot-Expert as the official Step 11 milestone

- Options:
  - use `Hot-Expert` as the final milestone
  - use `Hot-Rank` as the final milestone
- Decision:
  - use `Hot-Expert`
- Reason:
  - it maps more directly to the MoE pressure story
  - it keeps a more balanced v2 tradeoff than `Hot-Rank`
