# 17 Baseline Default

Fixed default baseline results were collected on all five workloads with:

- model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- runtime path: TensorRT-LLM `INT4 weight-only`
- batch size: `4`
- policy: `GUARANTEED_NO_EVICT`

Result files live in:

- `results/03_baseline_default/`
- `results/compare_tables/selected_summary.md`

## Default baseline readout

### Balanced MoE

- TTFT p90: `0.0809s`
- E2E p90: `1.4786s`
- TPOT p90: `0.0115s`
- throughput: `280.39 tok/s`

### Hot-Expert

- TTFT p90: `0.0740s`
- E2E p90: `1.8421s`
- TPOT p90: `0.0114s`
- throughput: `301.32 tok/s`

### Hot-Rank

- TTFT p90: `0.0803s`
- E2E p90: `1.9107s`
- TPOT p90: `0.0112s`
- throughput: `293.97 tok/s`

### Mixed Burst

- TTFT p90: `0.0878s`
- E2E p90: `1.9723s`
- TPOT p90: `0.0141s`
- step latency std: `296.38ms`

### Repeated-Prefix under MoE Pressure

- TTFT p90: `0.0862s`
- E2E p90: `1.7533s`
- TPOT p90: `0.0140s`
- throughput: `242.35 tok/s`

## Takeaway

The fixed default baseline was stable and reproducible across all five workloads.  
The most interesting pressure-sensitive baselines were:

- `Hot-Expert`
- `Hot-Rank`
- `Mixed Burst`
- `Repeated-Prefix under MoE Pressure`

Those became the main optimization targets for the next phases.
