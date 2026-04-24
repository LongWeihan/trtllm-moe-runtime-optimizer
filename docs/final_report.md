# Final Report

## Project

**MoE-aware runtime scheduling enhancement for TensorRT-LLM (full version)**

Fixed path:

- model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- quantization: TensorRT-LLM `INT4 weight-only`
- hardware: `RTX 4060 Ti 16GB`

## Problem

MoE requests are not equally expensive at runtime.  
Routing skew creates:

- expert hotspots
- rank hotspots
- decode tail latency
- batch stragglers
- step latency variance

Generic scheduling knobs do not explicitly treat that pressure as a first-class signal.

## Goal

Make the project feel like TensorRT-LLM runtime work, not a toy scheduler:

- keep the real TRT-LLM model path
- add a minimal runtime resource model
- validate the effect on real MoE inference
- push beyond the 24h slice with stronger baselines and broader workloads

## Real model path

The project completed the real model path end to end:

1. materialize `Qwen/Qwen1.5-MoE-A2.7B-Chat`
2. convert to TRT-LLM checkpoint
3. build `INT4 weight-only` engine
4. run real engine generation
5. run all full-version experiments on that engine path

Evidence:

- `docs/04_model_download.md`
- `docs/06_conversion_log.md`
- `docs/07_int4wo_build.md`
- `docs/08_qwen15_single_request.md`
- `docs/09_qwen15_bench_sanity.md`

## Architecture contribution

The core architecture contribution is the explicit runtime contract:

- `RequestProfile`
- `RuntimeBudget`
- `StepPlan`

Implemented in:

- `scheduler/resource_model.py`

This turns the scheduler path into:

`request metadata -> resource model -> step plan -> execution`

instead of a pile of isolated if-statements.

## Scheduler contributions

### V1

Implemented around:

- `scheduler/moe_microbatch_scheduler.py`
- `docs/20_scheduler_v1_design.md`

`v1` is a pressure-dispersion policy:

- decode-first
- avoid stacking hot requests
- accept throughput loss if needed

### V2

Implemented around:

- `scheduler/moe_capacity_scheduler.py`
- `scheduler/adaptive_chunking.py`
- `docs/23_capacity_scheduler.md`
- `docs/24_prefill_control.md`

`v2` adds:

- admission / capacity decision
- dynamic pressure budget
- repeated-prefix aware preference
- adaptive prefill / chunking control

That is the step that makes the full-version project feel more like runtime architecture enhancement and less like a single heuristic.

## Important limitation

The internal TRT-LLM patch seam was explored and recorded, but the final quantitative benchmark on this machine used the real TensorRT engine path with the same `resource_model -> step_plan` logic externalized in `scripts/run_patched.py`.

So the final quantitative claim is:

- real model path
- real TRT-LLM engine path
- real planner logic
- but not a pure in-backend PyTorch quantitative benchmark

This limitation is machine-specific and explicit, not hidden.

## Workloads

The full-version project kept all five planned workloads:

- `Balanced MoE`
- `Hot-Expert`
- `Hot-Rank`
- `Mixed Burst`
- `Repeated-Prefix under MoE Pressure`

That is one of the main differences from the 24h slice.

## Baseline conclusion

Generic strong baselines were worth running but were not enough:

- `MAX_UTILIZATION` improved the balanced control a little
- the hot workloads still showed weak or negative tail behavior under generic knobs
- that justified pressure-aware scheduling

## V1 results

`v1` showed the strongest raw latency gains:

### Hot-Expert replay

- TTFT p90: `0.0740s -> 0.0107s`
- E2E p90: `1.8421s -> 1.5698s`
- throughput: `301.32 -> 98.43 tok/s`

### Hot-Rank replay

- TTFT p90: `0.0803s -> 0.0114s`
- E2E p90: `1.9107s -> 1.7123s`
- throughput: `293.97 -> 100.07 tok/s`

`v1` clearly proves the pressure signal is meaningful, but it is too aggressive for a balanced serving policy.

## V2 results

`v2` is the main full-version result.

### Balanced MoE

- E2E p90: `1.4786s -> 1.4541s`
- step std: `233.71ms -> 140.22ms`
- throughput: `280.39 -> 305.78 tok/s`

### Hot-Expert

- TTFT p90: `0.0740s -> 0.0660s`
- E2E p90: `1.8421s -> 1.7928s`
- throughput: `301.32 -> 169.64 tok/s`

### Hot-Rank

- TTFT p90: `0.0803s -> 0.0115s`
- E2E p90: `1.9107s -> 1.7186s`
- throughput: `293.97 -> 99.26 tok/s`

### Mixed Burst

- E2E p90: `1.9723s -> 1.5660s`
- TPOT p90: `0.0141s -> 0.0115s`
- step std: `296.38ms -> 168.58ms`

### Repeated-Prefix under MoE Pressure

- E2E p90: `1.7533s -> 1.2848s`
- TPOT p90: `0.0140s -> 0.0115s`
- step std: `195.04ms -> 109.51ms`

## Interpretation

The full-version result is not “a universal speedup.”

It is more interesting than that:

1. `v1` proves MoE pressure should be a scheduling signal.
2. `v2` shows how capacity and prefill control can recover some of the lost batching.
3. `Hot-Rank` remains the hardest case.
4. `Mixed Burst` and `Repeated-Prefix under MoE Pressure` are where the architecture enhancement pays off most cleanly.

## Final milestone

For the official end-to-end milestone, the project kept `Hot-Expert` as the main story:

- it is directly tied to the MoE pressure narrative
- it is easier to explain than `Hot-Rank`
- `v2` gives a more balanced result there than `v1`

Supporting result:

- `Hot-Rank` remains a strong demonstration of latency isolation, but with a harsher throughput cost

## What this project demonstrates

This full-version run shows:

1. real TensorRT-LLM model bring-up on constrained hardware
2. real MoE-specific workload design
3. strong baseline discipline
4. explicit runtime resource modeling
5. progression from `v1` to `v2`, not just a single cherry-picked patch
6. honest tradeoff analysis

## Next iteration

The clean next step is:

- adaptive pressure budgets or a two-tier admission policy for `Hot-Rank`

That is the most direct path to preserving more throughput without giving up the tail-latency benefit.
