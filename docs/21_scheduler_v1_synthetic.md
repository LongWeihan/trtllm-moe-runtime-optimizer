# 21 Scheduler V1 Synthetic

Artifacts:

- `results/05_v1_synthetic/`
- `results/compare_tables/scheduler_compare.md`

## Headline

`v1` worked exactly like a latency-first isolation policy.

## Balanced control

`v1` was safe on the balanced control:

- TTFT p90: `0.0809s -> 0.0765s`
- step std: `233.71ms -> 188.14ms`
- throughput: `280.39 -> 282.94 tok/s`

## Hot-Expert

The pressure-isolation effect was strong:

- TTFT p90: `0.0740s -> 0.0118s`
- E2E p90: `1.8421s -> 1.5735s`
- TPOT p90: `0.0114s -> 0.0099s`
- throughput: `301.32 -> 99.52 tok/s`

The cost is obvious: the batch collapses from `3 x 4-request steps` to `12 x 1-request steps`.

## Hot-Rank

The same pattern held:

- TTFT p90: `0.0803s -> 0.0110s`
- E2E p90: `1.9107s -> 1.7298s`
- throughput: `293.97 -> 99.50 tok/s`

## Mixed Burst

`v1` was also effective on bursty pressure:

- E2E p90: `1.9723s -> 1.6103s`
- TPOT p90: `0.0141s -> 0.0115s`
- step std: `296.38ms -> 172.82ms`

## Repeated-Prefix under MoE Pressure

`v1` improved the repeated-prefix case too:

- E2E p90: `1.7533s -> 1.4198s`
- TPOT p90: `0.0140s -> 0.0114s`

## Conclusion

`v1` proved that the MoE pressure signal was useful.  
But it was too conservative for throughput-sensitive serving.
