# 28 Replay Vs Live

Inputs:

- replay runs in `results/06_v1_replay/` and `results/07_v2_ablation/`
- official end-to-end milestone in `results/09_qwen15_e2e_eval/`

## What is being compared

In this project, `replay` means:

- offline pressure metadata attached to real workload records
- real TensorRT engine execution
- real latency and throughput measurement

So the replay-vs-live question here is not:

- fake simulator vs real engine

It is:

- offline replay pressure metadata vs the same real-engine milestone packaging

## Consistency result

The direction of the result was consistent.

### Hot-Expert

`v2 replay` and the final `hot_expert_patched` milestone are numerically identical in direction and magnitude because the milestone artifact is the re-staged real engine run:

- TTFT p90 delta: `-0.0080s`
- E2E p90 delta: `-0.0493s`
- throughput delta: `-131.68 tok/s`

### Hot-Rank

The same is true for the hot-rank milestone:

- TTFT p90 delta: `-0.0688s`
- E2E p90 delta: `-0.1921s`
- throughput delta: `-194.71 tok/s`

## The real remaining gap

The remaining gap is not the runtime measurement.  
It is the signal provenance:

- the execution path is real
- the latency and throughput are real
- but the replay pressure metadata is still offline and approximate rather than live expert-routing export from the engine

That limitation is acceptable for this project and is already called out in the final report.
