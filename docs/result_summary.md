# Result Summary

Machine-readable tables:

- `results/compare_tables/selected_summary.json`
- `results/compare_tables/selected_summary.md`

## Headline

The full-version project completed the intended arc:

- real `Qwen/Qwen1.5-MoE-A2.7B-Chat` path
- full TensorRT-LLM `INT4 weight-only` conversion and build
- five MoE-specific workloads
- generic strong baselines
- `v1` and `v2`
- synthetic and replay validation
- real end-to-end milestone on the fixed model path

## Strong baseline takeaway

Generic strong baselines were not enough.

They helped a little on the balanced control, but did not materially improve the hot MoE tails.

## V1 takeaway

`v1` is a strong latency-first isolation policy.

It produced very large tail-latency wins on:

- `Hot-Expert`
- `Hot-Rank`

But it did so by collapsing the effective batch size and paying a very large throughput cost.

## V2 takeaway

`v2` is the main full-version result because it starts to balance the tradeoff.

### Balanced MoE

- E2E p90: `1.4786s -> 1.4541s`
- throughput: `280.39 -> 305.78 tok/s`

### Hot-Expert

- E2E p90: `1.8421s -> 1.7928s`
- throughput: `301.32 -> 169.64 tok/s`

### Mixed Burst

- E2E p90: `1.9723s -> 1.5660s`
- step std: `296.38ms -> 168.58ms`

### Repeated-Prefix under MoE Pressure

- E2E p90: `1.7533s -> 1.2848s`
- step std: `195.04ms -> 109.51ms`

## Bottom line

The full-version project shows a more interesting conclusion than the compact slice:

- `v1` proves the pressure signal is useful
- `v2` shows that adding admission and prefill control makes the result feel like a real runtime architecture enhancement
- the hardest unsolved case is still `Hot-Rank`, where throughput recovery remains weak
