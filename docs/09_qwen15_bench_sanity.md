# 09 Qwen15 Bench Sanity

## Goal

Obtain a small, reproducible latency / throughput reading on the real full-version engine before starting scheduler evaluations.

## Workload

- `workloads/sanity/balanced_moe.jsonl`
- 4 requests
- microbatch size: `2`

## Command path

The benchmark sanity used the local baseline harness:

```bash
python scripts/run_baseline.py \
  --backend trt \
  --model /home/a/trtllm-moe-runtime-exp-full/artifacts/qwen15_moe_int4wo \
  --tokenizer /home/a/trtllm-moe-runtime-exp-full/hf/Qwen1.5-MoE-A2.7B-Chat \
  --workload workloads/sanity/balanced_moe.jsonl \
  --output results/01_qwen15_bench_sanity/balanced_sanity.json \
  --microbatch-size 2
```

## Result

Artifact:

- [balanced_sanity.json](C:/26spring/nv项目/full_version/trtllm-moe-runtime-exp/results/01_qwen15_bench_sanity/balanced_sanity.json)

Observed summary:

- requests: `4`
- batches: `2`
- avg batch wall time: `1517.01 ms`
- total output tokens: `432`
- throughput: `142.39 tok/s`
- TTFT max/p90-on-4-samples: `0.3633 s`
- E2E max/p90-on-4-samples: `1.4708 s`
- TPOT max/p90-on-4-samples: `0.0117 s`

## Conclusion

The real engine path is benchmarkable in the full-version workspace, so the project can now move into codepath, telemetry, workload, and baseline phases.
