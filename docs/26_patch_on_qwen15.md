# 26 Patch On Qwen15

Artifacts:

- `results/08_patch_qwen15/`
- `results/02_telemetry/`

## What this milestone means

This milestone records that the scheduler patch path was exercised against the fixed real model:

- `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- TensorRT-LLM `INT4 weight-only`

## Patch seam

The internal seam explored in the repo remains:

- `BindMicroBatchScheduler`
- `executor_request_to_llm_request`

## Quantitative execution path

For the final quantitative runs on this machine, the planner was applied on the real TensorRT engine batch path through:

- `scripts/run_patched.py`
- `scheduler/resource_model.py`
- `scheduler/moe_capacity_scheduler.py`
- `scheduler/adaptive_chunking.py`

That kept the model path real and the planner path real, while avoiding the quantized-backend limitation described elsewhere in the docs.

## Recorded patched run

The concrete patched Qwen run captured here is:

- `results/08_patch_qwen15/hot_expert_moe_v2_hot_expert_patch.json`

It is paired with telemetry in:

- `results/02_telemetry/hot_expert_moe_v2_hot_expert_patch_telemetry.jsonl`
