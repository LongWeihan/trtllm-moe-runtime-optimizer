# 03 Backend Sanity

## Goal

Confirm that the full-version editable TensorRT-LLM environment can import the backend API.

## Script

- [scripts/sanity_backend.py](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/scripts/sanity_backend.py)

## Result

Sanity import succeeded.

Evidence:

- [backend_import.json](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/results/00_qwen15_sanity/backend_import.json)
- `/home/a/trtllm-moe-runtime-exp-full/logs/install/sanity_backend_clean.log`

Observed payload:

```json
{
  "tensorrt_llm_version": "1.2.1",
  "llm_class": "LLM"
}
```

## Notes

The import path still emits two non-blocking warnings on this machine:

- `Python.h` missing in temporary C compilation inside auxiliary probing code
- Triton/FLA fallback warning

These warnings do not prevent the TensorRT-LLM backend from importing or the later real-engine path from running.
