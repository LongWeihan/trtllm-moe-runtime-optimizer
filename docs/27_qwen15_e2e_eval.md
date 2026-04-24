# 27 Qwen15 E2E Eval

Final evaluation artifacts:

- `results/09_qwen15_e2e_eval/hot_expert_baseline.json`
- `results/09_qwen15_e2e_eval/hot_expert_patched.json`
- `results/09_qwen15_e2e_eval/hot_rank_baseline.json`
- `results/09_qwen15_e2e_eval/hot_rank_patched.json`

Summary table:

- `results/compare_tables/selected_summary.md`

## Official end-to-end compare

### Hot-Expert

- baseline TTFT p90: `0.0740s`
- patched TTFT p90: `0.0660s`
- baseline E2E p90: `1.8421s`
- patched E2E p90: `1.7928s`
- baseline TPOT p90: `0.0114s`
- patched TPOT p90: `0.0112s`
- throughput: `301.32 -> 169.64 tok/s`

This became the official `Step 11` milestone result because it keeps the story directly tied to MoE pressure while showing a more balanced tradeoff than `v1`.

### Hot-Rank

- baseline TTFT p90: `0.0803s`
- patched TTFT p90: `0.0115s`
- baseline E2E p90: `1.9107s`
- patched E2E p90: `1.7186s`
- throughput: `293.97 -> 99.26 tok/s`

This is still a valid win, but it is more extreme and less balanced than the `Hot-Expert` final result.

## Evaluation takeaway

On the real fixed model path:

- at least one hot MoE workload improved
- the result is explainable
- the project can show both benefit and tradeoff honestly
