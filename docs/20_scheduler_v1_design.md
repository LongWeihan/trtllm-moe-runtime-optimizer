# 20 Scheduler V1 Design

Primary implementation files:

- `scheduler/resource_model.py`
- `scheduler/moe_microbatch_scheduler.py`

## Goal

`v1` is the first MoE-aware scheduler enhancement:

- decode-first
- disperse high-pressure requests
- avoid stacking hot requests into the same step

## Runtime contract

`v1` uses the minimal runtime resource model introduced for this project:

- `RequestProfile`
- `RuntimeBudget`
- `StepPlan`

This makes the path:

`request metadata -> pressure score -> step plan -> batch execution`

explicit instead of scattering MoE conditions across ad hoc heuristics.

## Implementation note

There are two relevant layers in the repo:

1. the real TRT-LLM internal patch seam:
   - `BindMicroBatchScheduler`
   - `executor_request_to_llm_request`
2. the final quantitative path used in this project:
   - the same `resource_model -> step_plan` logic externalized into the TensorRT engine batch-composition path

That second path was used for final measurement because the quantized MoE setup on this machine did not support a pure in-backend PyTorch quantitative benchmark.

## V1 behavior

`v1` is intentionally conservative:

- balanced traffic should remain safe
- hot traffic should be isolated
- throughput is allowed to drop if that is what it takes to break up pressure stacking

That tradeoff becomes visible in the synthetic and replay runs.
