# 22 Scheduler V1 Replay

Artifacts:

- `results/06_v1_replay/`
- replay traces in `artifacts/moe_traces/`

## Purpose

Replay mode checks whether the scheduler still behaves sensibly when pressure metadata comes from an offline trace representation rather than the synthetic label directly embedded in the workload.

## Result

Replay tracked synthetic surprisingly closely.

### Balanced MoE

- TTFT p90: `0.0809s -> 0.0772s`
- throughput: `280.39 -> 280.62 tok/s`

### Hot-Expert

- TTFT p90: `0.0740s -> 0.0107s`
- E2E p90: `1.8421s -> 1.5698s`
- throughput: `301.32 -> 98.43 tok/s`

### Hot-Rank

- TTFT p90: `0.0803s -> 0.0114s`
- E2E p90: `1.9107s -> 1.7123s`
- throughput: `293.97 -> 100.07 tok/s`

### Mixed Burst

- E2E p90: `1.9723s -> 1.6128s`
- step std: `296.38ms -> 170.40ms`

### Repeated-Prefix under MoE Pressure

- E2E p90: `1.7533s -> 1.4127s`
- TPOT p90: `0.0140s -> 0.0113s`

## Interpretation

Replay mode preserved the same direction as synthetic mode on all core workloads.  
That was enough evidence to keep replay as the full-version validation path.

## Caveat

The replay trace is still an offline approximation.  
It is built from real workload records and real TensorRT-engine result JSON, but the expert / rank histograms are synthetic stand-ins, not live router export from the engine.
