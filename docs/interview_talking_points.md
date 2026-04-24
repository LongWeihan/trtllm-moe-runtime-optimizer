# Interview Talking Points

## 60-second version

I built a full TensorRT-LLM MoE runtime optimization project around `Qwen/Qwen1.5-MoE-A2.7B-Chat` on a single `4060 Ti 16GB`. I completed the real HF-to-TRT-LLM `INT4 weight-only` path, then focused on the runtime layer instead of kernels. I introduced a minimal runtime resource model with `RequestProfile`, `RuntimeBudget`, and `StepPlan`, and used that to drive MoE-aware scheduling under five workloads: `Balanced`, `Hot-Expert`, `Hot-Rank`, `Mixed Burst`, and `Repeated-Prefix under MoE Pressure`. I ran generic strong baselines, then compared a `v1` pressure-isolation policy against a `v2` admission plus prefill-control policy. `v1` gave the biggest tail-latency wins but collapsed throughput; `v2` recovered part of the batching and was especially strong on mixed-burst and repeated-prefix workloads.

## 3-minute version

I wanted the project to look like TensorRT-LLM inference optimization work rather than a toy scheduler or a kernel side project. So I fixed the model path to `Qwen/Qwen1.5-MoE-A2.7B-Chat + TensorRT-LLM INT4 weight-only`, and I treated the real conversion/build path as non-negotiable.

Then I focused on the MoE runtime problem. Different requests create different expert or rank pressure, but generic scheduler knobs do not explicitly represent that pressure. I introduced a minimal runtime resource model with `RequestProfile`, `RuntimeBudget`, and `StepPlan`. That gave me an explicit contract from request metadata to batch plan.

The full-version project goes beyond the 24h slice in three ways. First, I kept five workloads instead of trimming down to three, so I could see pressure under balanced, hot, bursty, and repeated-prefix traffic. Second, I ran strong baselines like `MAX_UTILIZATION` and overlap tuning before claiming I needed a custom approach. Third, I built two versions of the planner. `v1` is a pure pressure-isolation policy, and it dramatically improves tail latency on `Hot-Expert` and `Hot-Rank`, but at a huge throughput cost. `v2` adds admission control and adaptive chunking. It is still not perfect, but it starts to recover throughput, and the clearest wins show up on `Mixed Burst` and `Repeated-Prefix under MoE Pressure`.

The main caveat is that, on this machine, the final quantitative path uses the real TensorRT engine backend with the same planning logic externalized into batch composition, rather than a pure in-backend PyTorch quantitative benchmark. I keep that limitation explicit. I think the value is still strong because the model path, engine path, workloads, and tradeoffs are all real.

## Deep-dive prompts

### Why is this more than a scheduler heuristic?

Because the project adds an explicit runtime resource model and then makes the planner consume it. The flow is `request metadata -> runtime budget -> step plan -> execution`. That is closer to runtime architecture work than a few isolated conditions in a sort function.

### Why did you keep five workloads in the full version?

Because the full version was meant to answer more than one question. `Hot-Expert` and `Hot-Rank` validate the direct MoE-pressure story. `Mixed Burst` checks whether the planner behaves well under realistic traffic. `Repeated-Prefix under MoE Pressure` is where pressure control and KV-related structure start interacting.

### What did the strong baselines tell you?

They were good controls. `MAX_UTILIZATION` helped the balanced case a little, but it did not materially fix the hot-workload tails. That gave me evidence that the gap was really about pressure awareness, not just a missing generic tuning knob.

### What is the best single result from the full version?

For architecture value, I would point to `Repeated-Prefix under MoE Pressure` with `v2 replay`:

- E2E p90: `1.7533s -> 1.2848s`
- step std: `195.04ms -> 109.51ms`

For the clearest MoE-specific story, I would point to the official `Hot-Expert` end-to-end milestone:

- TTFT p90: `0.0740s -> 0.0660s`
- E2E p90: `1.8421s -> 1.7928s`

### What is still unsolved?

`Hot-Rank`. The planner can isolate it and improve latency, but throughput recovery is still weak. That is the next engineering target.
