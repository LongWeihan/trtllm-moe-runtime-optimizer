#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="/home/a/trtllm-moe-runtime-exp-full/artifacts/qwen15_moe_int4wo"
TOKENIZER_DIR="/home/a/trtllm-moe-runtime-exp-full/hf/Qwen1.5-MoE-A2.7B-Chat"
WSL_ENV_SH="$PROJECT_ROOT/scripts/wsl_env.sh"

WORKLOADS=(
  "balanced_moe"
  "hot_expert"
  "hot_rank"
  "mixed_burst"
  "repeated_prefix_moe"
)

run_baseline_variant() {
  local workload_name="$1"
  local output_dir="$2"
  local variant_name="$3"
  shift 3

  mkdir -p "$PROJECT_ROOT/$output_dir" "$PROJECT_ROOT/logs/${output_dir}"
  echo "[baseline:$variant_name] ${workload_name}"
  bash "$WSL_ENV_SH" python "$PROJECT_ROOT/scripts/run_baseline.py" \
    --backend trt \
    --model "$ENGINE_DIR" \
    --tokenizer "$TOKENIZER_DIR" \
    --workload "$PROJECT_ROOT/workloads/${workload_name}.jsonl" \
    --output "$PROJECT_ROOT/${output_dir}/${workload_name}_${variant_name}.json" \
    --microbatch-size 4 \
    --variant-name "$variant_name" \
    "$@" \
    > "$PROJECT_ROOT/logs/${output_dir}/${workload_name}_${variant_name}.log" 2>&1
  tail -n 5 "$PROJECT_ROOT/logs/${output_dir}/${workload_name}_${variant_name}.log"
}

run_patched_variant() {
  local workload_name="$1"
  local output_dir="$2"
  local variant_name="$3"
  local planner="$4"
  local pressure_source="$5"
  shift 5

  mkdir -p "$PROJECT_ROOT/$output_dir" "$PROJECT_ROOT/logs/${output_dir}" "$PROJECT_ROOT/results/02_telemetry"
  echo "[patched:$variant_name:$pressure_source] ${workload_name}"
  bash "$WSL_ENV_SH" python "$PROJECT_ROOT/scripts/run_patched.py" \
    --backend trt \
    --model "$ENGINE_DIR" \
    --tokenizer "$TOKENIZER_DIR" \
    --workload "$PROJECT_ROOT/workloads/${workload_name}.jsonl" \
    --output "$PROJECT_ROOT/${output_dir}/${workload_name}_${variant_name}.json" \
    --telemetry-output "$PROJECT_ROOT/results/02_telemetry/${workload_name}_${variant_name}_telemetry.jsonl" \
    --microbatch-size 4 \
    --planner "$planner" \
    --pressure-source "$pressure_source" \
    --variant-name "$variant_name" \
    "$@" \
    > "$PROJECT_ROOT/logs/${output_dir}/${workload_name}_${variant_name}.log" 2>&1
  tail -n 5 "$PROJECT_ROOT/logs/${output_dir}/${workload_name}_${variant_name}.log"
}

generate_trace() {
  local workload_name="$1"
  local result_json="$2"

  mkdir -p "$PROJECT_ROOT/artifacts/moe_traces"
  echo "[trace] ${workload_name}"
  bash "$WSL_ENV_SH" python "$PROJECT_ROOT/scripts/generate_pressure_traces.py" \
    --workload "$PROJECT_ROOT/workloads/${workload_name}.jsonl" \
    --result-json "$PROJECT_ROOT/${result_json}" \
    --output "$PROJECT_ROOT/artifacts/moe_traces/${workload_name}_replay_trace.jsonl" \
    > "$PROJECT_ROOT/logs/phase_replay_${workload_name}.log" 2>&1
  tail -n 3 "$PROJECT_ROOT/logs/phase_replay_${workload_name}.log"
}

phase="${1:-}"
if [[ -z "$phase" ]]; then
  echo "usage: $0 <phase>" >&2
  exit 2
fi

case "$phase" in
  baseline-default)
    for workload_name in "${WORKLOADS[@]}"; do
      run_baseline_variant \
        "$workload_name" \
        "results/03_baseline_default" \
        "default" \
        --capacity-scheduler-policy GUARANTEED_NO_EVICT
    done
    ;;

  baseline-strong)
    for workload_name in "${WORKLOADS[@]}"; do
      run_baseline_variant \
        "$workload_name" \
        "results/04_baseline_strong" \
        "guaranteed_no_evict" \
        --capacity-scheduler-policy GUARANTEED_NO_EVICT
      run_baseline_variant \
        "$workload_name" \
        "results/04_baseline_strong" \
        "max_utilization" \
        --capacity-scheduler-policy MAX_UTILIZATION
      run_baseline_variant \
        "$workload_name" \
        "results/04_baseline_strong" \
        "overlap" \
        --capacity-scheduler-policy GUARANTEED_NO_EVICT \
        --enable-chunked-prefill
    done
    ;;

  traces)
    for workload_name in "${WORKLOADS[@]}"; do
      generate_trace "$workload_name" "results/03_baseline_default/${workload_name}_default.json"
    done
    ;;

  v1-synthetic)
    for workload_name in "${WORKLOADS[@]}"; do
      run_patched_variant \
        "$workload_name" \
        "results/05_v1_synthetic" \
        "moe_v1_synthetic" \
        "v1" \
        "synthetic"
    done
    ;;

  v1-replay)
    for workload_name in "${WORKLOADS[@]}"; do
      run_patched_variant \
        "$workload_name" \
        "results/06_v1_replay" \
        "moe_v1_replay" \
        "v1" \
        "replay" \
        --trace-path "$PROJECT_ROOT/artifacts/moe_traces/${workload_name}_replay_trace.jsonl"
    done
    ;;

  v2-ablation)
    for workload_name in "${WORKLOADS[@]}"; do
      run_patched_variant \
        "$workload_name" \
        "results/07_v2_ablation" \
        "moe_v2_synthetic" \
        "v2" \
        "synthetic" \
        --enable-adaptive-chunking \
        --enable-chunked-prefill \
        --capacity-scheduler-policy MAX_UTILIZATION
      run_patched_variant \
        "$workload_name" \
        "results/07_v2_ablation" \
        "moe_v2_replay" \
        "v2" \
        "replay" \
        --trace-path "$PROJECT_ROOT/artifacts/moe_traces/${workload_name}_replay_trace.jsonl" \
        --enable-adaptive-chunking \
        --enable-chunked-prefill \
        --capacity-scheduler-policy MAX_UTILIZATION
    done
    ;;

  qwen15-final)
    mkdir -p "$PROJECT_ROOT/results/08_patch_qwen15" "$PROJECT_ROOT/results/09_qwen15_e2e_eval"

    run_patched_variant \
      "hot_expert" \
      "results/08_patch_qwen15" \
      "moe_v2_hot_expert_patch" \
      "v2" \
      "replay" \
      --trace-path "$PROJECT_ROOT/artifacts/moe_traces/hot_expert_replay_trace.jsonl" \
      --enable-adaptive-chunking \
      --enable-chunked-prefill \
      --capacity-scheduler-policy MAX_UTILIZATION

    cp "$PROJECT_ROOT/results/03_baseline_default/hot_expert_default.json" \
      "$PROJECT_ROOT/results/09_qwen15_e2e_eval/hot_expert_baseline.json"
    cp "$PROJECT_ROOT/results/07_v2_ablation/hot_expert_moe_v2_replay.json" \
      "$PROJECT_ROOT/results/09_qwen15_e2e_eval/hot_expert_patched.json"

    cp "$PROJECT_ROOT/results/03_baseline_default/hot_rank_default.json" \
      "$PROJECT_ROOT/results/09_qwen15_e2e_eval/hot_rank_baseline.json"
    cp "$PROJECT_ROOT/results/07_v2_ablation/hot_rank_moe_v2_replay.json" \
      "$PROJECT_ROOT/results/09_qwen15_e2e_eval/hot_rank_patched.json"
    ;;

  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac
