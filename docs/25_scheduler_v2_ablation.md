# 25 Scheduler V2 Ablation

Artifacts:

- `results/07_v2_ablation/`
- `results/compare_tables/selected_summary.md`

## Headline

`v2` is the main full-version result.

Compared with `v1`, it keeps the same pressure-aware story but begins to recover throughput in the workloads where the conservative isolation policy was too expensive.

## Balanced MoE

`v2 replay` was the best result on the balanced control:

- E2E p90: `1.4786s -> 1.4541s`
- step std: `233.71ms -> 140.22ms`
- throughput: `280.39 -> 305.78 tok/s`

That is a meaningful improvement over both default and `v1`.

## Hot-Expert

`v1 replay` was the strongest latency optimizer:

- TTFT p90: `0.0740s -> 0.0107s`
- E2E p90: `1.8421s -> 1.5698s`
- throughput: `301.32 -> 98.43 tok/s`

`v2 replay` softened the policy:

- TTFT p90: `0.0740s -> 0.0660s`
- E2E p90: `1.8421s -> 1.7928s`
- throughput: `301.32 -> 169.64 tok/s`

So `v2` gives up some latency win to recover a large chunk of throughput.

## Hot-Rank

`Hot-Rank` remained the hardest case.

Both `v1` and `v2` mostly serialize the batch:

- `v1 replay` throughput: `100.07 tok/s`
- `v2 replay` throughput: `99.26 tok/s`

The latency win is strong, but `v2` did not recover batching here.

## Mixed Burst

This is one of the best `v2` stories:

- E2E p90: `1.9723s -> 1.5660s`
- TPOT p90: `0.0141s -> 0.0115s`
- step std: `296.38ms -> 168.58ms`
- throughput: `263.02 -> 184.46 tok/s`

`v2` outperformed `v1 replay` on E2E and lost less throughput.

## Repeated-Prefix under MoE Pressure

This is the other strongest `v2` story:

- E2E p90: `1.7533s -> 1.2848s`
- TPOT p90: `0.0140s -> 0.0115s`
- step std: `195.04ms -> 109.51ms`
- throughput: `242.35 -> 130.94 tok/s`

That result is exactly why the full-version project kept the repeated-prefix workload instead of trimming back to the compact slice.

## Conclusion

`v1` proved the pressure signal.  
`v2` made it more like a runtime-architecture enhancement:

- explicit capacity decision
- dynamic pressure budget
- adaptive prefill / chunking

It is still not throughput-optimal, but it is clearly more balanced than `v1`.
