# Reproducibility Checklist

## Environment

- fixed WSL workspace: `/home/a/trtllm-moe-runtime-exp-full`
- fixed Windows workspace: `C:\26spring\nv项目\full_version\trtllm-moe-runtime-exp`
- fixed venv: `/home/a/trtllm-moe-runtime-exp-full/venv`

## Fixed model path

- model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- engine: `/home/a/trtllm-moe-runtime-exp-full/artifacts/qwen15_moe_int4wo`
- tokenizer: `/home/a/trtllm-moe-runtime-exp-full/hf/Qwen1.5-MoE-A2.7B-Chat`

## Workload generation

```bash
cd /mnt/c/26spring/nv项目/full_version/trtllm-moe-runtime-exp
bash scripts/wsl_env.sh python scripts/generate_workloads.py --output-dir workloads --count 12
```

## Matrix execution

```bash
cd /mnt/c/26spring/nv项目/full_version/trtllm-moe-runtime-exp
bash scripts/run_full_matrix.sh baseline-default
bash scripts/run_full_matrix.sh baseline-strong
bash scripts/run_full_matrix.sh traces
bash scripts/run_full_matrix.sh v1-synthetic
bash scripts/run_full_matrix.sh v1-replay
bash scripts/run_full_matrix.sh v2-ablation
bash scripts/run_full_matrix.sh qwen15-final
```

## Summary generation

```bash
cd /mnt/c/26spring/nv项目/full_version/trtllm-moe-runtime-exp
bash scripts/wsl_env.sh python scripts/collect_metrics.py \
  --inputs results/03_baseline_default results/04_baseline_strong \
  --output-json results/compare_tables/baseline_compare.json \
  --output-md results/compare_tables/baseline_compare.md
```

```bash
cd /mnt/c/26spring/nv项目/full_version/trtllm-moe-runtime-exp
bash scripts/wsl_env.sh python scripts/collect_metrics.py \
  --inputs results/03_baseline_default results/05_v1_synthetic results/06_v1_replay results/07_v2_ablation results/09_qwen15_e2e_eval \
  --output-json results/compare_tables/scheduler_compare.json \
  --output-md results/compare_tables/scheduler_compare.md
```

## Final document set

The final deliverables that should exist after a successful rerun:

- `docs/final_report.md`
- `docs/result_summary.md`
- `docs/reproducibility_checklist.md`
- `results/compare_tables/selected_summary.md`

## Known caveat

The final quantitative path uses the real TensorRT engine backend with the project planner externalized into batch composition.  
That caveat is part of the intended reproduced result and should remain stated explicitly.
