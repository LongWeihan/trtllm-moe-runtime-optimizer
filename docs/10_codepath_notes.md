# 10 Codepath Notes

## Goal

Identify the real TensorRT-LLM seams that matter for a MoE-aware scheduler enhancement.

## Confirmed full-version seams

From the editable TRT-LLM 1.2.1 source checkout:

- `tensorrt_llm/_torch/pyexecutor/scheduler.py`
  - `BindMicroBatchScheduler`
- `tensorrt_llm/_torch/pyexecutor/llm_request.py`
  - `executor_request_to_llm_request`
- `tensorrt_llm/_torch/pyexecutor/executor_request_queue.py`
  - request conversion and queueing path

These are the key full-version internal patch points.

## Why these seams matter

### `executor_request_to_llm_request`

This is the cleanest place to preserve request-side MoE pressure metadata as requests enter the TRT-LLM internal request object path.

### `BindMicroBatchScheduler`

This is the least invasive place to replace heuristic-only selection with:

- `request profile`
- `runtime budget`
- `step plan`

### Executor request queue

This is where the conversion path is exercised during the PyTorch backend request lifecycle.

## Real benchmark split

The project still keeps the same honest split established in the validated run:

- internal patch exists on the PyTorch backend seam
- quantitative benchmark path runs on the real TensorRT engine backend

This is a machine- and version-specific limitation, not a conceptual one.
