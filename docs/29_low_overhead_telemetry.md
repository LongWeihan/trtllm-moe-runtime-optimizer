# 29 Low-Overhead Telemetry

Status: `SKIPPED`

## Reason

This item was optional in the full-version plan.

By the time the project reached the final evaluation:

- telemetry overhead was not the dominant bottleneck
- the main open problem was still the latency / throughput tradeoff under hot workloads
- a low-overhead telemetry kernel would not have changed the main conclusion

## Decision

Keep the existing JSONL telemetry path and spend the project budget on:

- strong baselines
- synthetic vs replay validation
- `v1` / `v2` comparisons
- final Qwen end-to-end evaluation
