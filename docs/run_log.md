# Run Log

## 2026-04-24

- Created the full-version Windows workspace at `C:\26spring\nv项目\full_version\trtllm-moe-runtime-exp`.
- Created the full-version WSL execution workspace at `/home/a/trtllm-moe-runtime-exp-full`.
- Created the full-version result, log, script, scheduler, workload, and artifact directories.
- Seed-copied the validated scheduler and script files from the 24h workspace as a starting point for the full run.
- Created the full-version editable venv at `/home/a/trtllm-moe-runtime-exp-full/venv`.
- Created a separate TensorRT-LLM source checkout at `/home/a/trtllm-moe-runtime-exp-full/src/TensorRT-LLM` and checked out `v1.2.1`.
- Completed `TRTLLM_USE_PRECOMPILED=1 pip install -e .` in the full-version environment.
- Resolved the full-version import chain by fixing:
  - `libpython3.12.so`
  - user-space MPI
  - CUDA 13 runtime library visibility
- Verified backend import success and saved the sanity payload to `results/00_qwen15_sanity/backend_import.json`.
- Materialized the fixed Qwen MoE checkpoint into the full workspace.
- Completed official Qwen conversion and TensorRT-LLM `INT4 weight-only` engine build.
- Ran the full baseline matrix:
  - default baselines on 5 workloads
  - strong baselines on 5 workloads
- Generated replay traces from baseline results.
- Ran the full scheduler matrix:
  - `v1 synthetic`
  - `v1 replay`
  - `v2 synthetic`
  - `v2 replay`
- Completed the fixed-model patch milestone and final end-to-end compares for:
  - `Hot-Expert`
  - `Hot-Rank`
- Generated compare tables under `results/compare_tables/`.
- Wrote the full-version report, interview notes, resume bullets, and reproducibility checklist.
