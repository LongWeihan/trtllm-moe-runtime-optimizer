# Resume Bullets

- Built a **full TensorRT-LLM MoE runtime optimization project** on `Qwen/Qwen1.5-MoE-A2.7B-Chat`, completing the real **HF checkpoint -> TRT-LLM conversion -> INT4 weight-only engine build -> engine inference** path on a single `RTX 4060 Ti 16GB`.
- Designed and implemented a minimal **runtime resource model** for MoE scheduling, introducing `RequestProfile`, `RuntimeBudget`, and `StepPlan` so scheduling decisions consumed explicit pressure and token budgets instead of ad hoc heuristics.
- Extended the project from a `v1` pressure-isolation scheduler into a `v2` admission plus adaptive-chunking design, making the work look more like **runtime architecture enhancement** than a one-off scheduling rule.
- Built five **MoE-specific workloads** (`Balanced`, `Hot-Expert`, `Hot-Rank`, `Mixed Burst`, `Repeated-Prefix under MoE Pressure`) and benchmarked them against strong baselines including `MAX_UTILIZATION` and overlap tuning.
- On the full-version `v2` replay path, improved **E2E p90 from `1.9723s` to `1.5660s` on `Mixed Burst`** and **from `1.7533s` to `1.2848s` on `Repeated-Prefix under MoE Pressure`**, while identifying `Hot-Rank` throughput recovery as the main open problem.
