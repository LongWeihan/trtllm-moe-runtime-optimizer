# Blockers

## Active

None currently.

## Resolved

### B001 - `libpython3.12.so` missing during TRT-LLM import

- Symptom:
  - editable install completed, but `import tensorrt_llm` failed on missing `libpython3.12.so`
- Fix:
  - symlinked `/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0` to `/home/a/trtllm-moe-runtime-exp-full/venv/lib/libpython3.12.so`

### B002 - MPI runtime missing

- Symptom:
  - `mpi4py` failed because `libmpi.so` was unavailable to the venv
- Fix:
  - installed user-space `openmpi==5.0.10`

### B003 - CUDA 13 runtime loader gap in editable environment

- Symptom:
  - TRT-LLM import failed on `libcublasLt.so.13`
- Fix:
  - synced the known-good `nvidia/cu13/lib` runtime library set from the previously validated workspace into the full-version venv

This is an environment workaround specific to this machine.

### B004 - Pure in-backend PyTorch quantitative path did not match the fixed quantized MoE benchmark path

- Symptom:
  - the fixed full-version benchmark path needed the real quantized TensorRT engine, but the in-backend PyTorch quantitative path was not the right vehicle for that setup on this machine
- Fix:
  - kept the real engine path and externalized the same planner logic into `scripts/run_patched.py`

This is a project constraint rather than a correctness issue, and it is called out explicitly in the final report.
