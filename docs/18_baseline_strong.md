# 18 Baseline Strong

Strong baselines were collected in:

- `results/04_baseline_strong/`
- `results/compare_tables/baseline_compare.md`
- `results/compare_tables/selected_summary.md`

Variants:

- `GUARANTEED_NO_EVICT`
- `MAX_UTILIZATION`
- `overlap` via `enable_chunked_prefill`

## Summary

### Balanced MoE

`MAX_UTILIZATION` was the best strong baseline on the balanced control:

- TTFT p90: `0.0809s -> 0.0737s`
- E2E p90: `1.4786s -> 1.4768s`
- throughput: `280.39 -> 282.72 tok/s`

This is a useful control result because it shows the default path was not already globally optimal.

### Hot-Expert

Strong baselines did not solve the MoE pressure problem:

- `MAX_UTILIZATION` changed TTFT p90 only from `0.0740s` to `0.0734s`
- E2E p90 slightly worsened: `1.8421s -> 1.8643s`
- throughput stayed effectively flat: `301.32 -> 300.21 tok/s`

### Hot-Rank

The same pattern held:

- TTFT p90: `0.0803s -> 0.0738s`
- E2E p90: `1.9107s -> 1.9273s`
- throughput: `293.97 -> 293.50 tok/s`

### Mixed Burst

`MAX_UTILIZATION` slightly improved TTFT but did not improve the tail:

- TTFT p90: `0.0878s -> 0.0725s`
- E2E p90 worsened: `1.9723s -> 1.9917s`
- step std also did not improve

### Repeated-Prefix under MoE Pressure

The strong baselines were also weak on the repeated-prefix case:

- `MAX_UTILIZATION` slightly improved TTFT and E2E
- throughput stayed almost unchanged
- no strong pressure-aware isolation effect was visible

## Takeaway

The strong baselines were real and worth checking, but they stayed generic:

- they helped a little on the balanced control
- they did not materially address high-pressure MoE batches
- they did not produce the kind of tail-latency change needed for the main project story

That was the go signal for a MoE-aware scheduler path.
