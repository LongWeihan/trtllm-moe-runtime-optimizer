# Artifact Index

## Naming convention

Primary evaluation name format:

`<date>_<model>_<quant>_<workload>_<variant>`

Example:

- `2026-04-24_qwen15moe_int4wo_hotexpert_default`
- `2026-04-24_qwen15moe_int4wo_hotexpert_moev1`

## Result directories

- `results/00_qwen15_sanity/`
- `results/01_qwen15_bench_sanity/`
- `results/02_telemetry/`
- `results/03_baseline_default/`
- `results/04_baseline_strong/`
- `results/05_v1_synthetic/`
- `results/06_v1_replay/`
- `results/07_v2_ablation/`
- `results/08_patch_qwen15/`
- `results/09_qwen15_e2e_eval/`

## Log directories

- `logs/install/`
- `logs/build/`
- `logs/runs/`

## Artifact directories

- `artifacts/model_conversion/`
- `artifacts/qwen15_moe_int4wo/`
- `artifacts/moe_traces/`
