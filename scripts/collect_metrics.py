from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def iter_result_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(
                path
                for path in sorted(item.glob("*.json"))
                if path.name != "manifest.json"
            )
        elif item.is_file():
            files.append(item)
    return files


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


def metric_value(record: dict, key: str) -> float:
    raw = record.get("metrics_dict", {}).get(key, 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def summarize_payload(label: str, payload: dict) -> dict:
    records = payload.get("records", [])
    batch_plan = payload.get("batch_plan", [])
    ttft = [metric_value(record, "MetricNames.TTFT") for record in records]
    e2e = [metric_value(record, "MetricNames.E2E") for record in records]
    tpot = [metric_value(record, "MetricNames.TPOT") for record in records]
    queue_time = [metric_value(record, "MetricNames.REQUEST_QUEUE_TIME") for record in records]
    output_token_counts = [len(record.get("output_token_ids", []) or []) for record in records]
    batch_wall_ms = [float(batch.get("batch_wall_ms", 0.0)) for batch in batch_plan]
    total_batch_wall_s = sum(batch_wall_ms) / 1000.0
    total_output_tokens = sum(output_token_counts)

    pressure_classes = [
        str(record.get("pressure_class", "unknown"))
        for record in records
    ]
    pressure_histogram: dict[str, int] = {}
    for key in pressure_classes:
        pressure_histogram[key] = pressure_histogram.get(key, 0) + 1

    return {
        "label": label,
        "mode": payload.get("mode"),
        "backend": payload.get("backend"),
        "workload": payload.get("workload"),
        "variant_name": payload.get("variant_name"),
        "planner": payload.get("planner"),
        "pressure_source": payload.get("pressure_source"),
        "num_requests": len(records),
        "num_batches": len(batch_plan),
        "avg_requests_per_batch": (len(records) / len(batch_plan)) if batch_plan else 0.0,
        "avg_batch_ms": statistics.mean(batch_wall_ms) if batch_wall_ms else 0.0,
        "step_latency_std_ms": statistics.pstdev(batch_wall_ms) if len(batch_wall_ms) > 1 else 0.0,
        "step_latency_var_ms2": statistics.pvariance(batch_wall_ms) if len(batch_wall_ms) > 1 else 0.0,
        "total_batch_wall_s": total_batch_wall_s,
        "total_output_tokens": total_output_tokens,
        "throughput_tok_s": (total_output_tokens / total_batch_wall_s) if total_batch_wall_s > 0 else 0.0,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p90_s": percentile(ttft, 0.90),
        "ttft_p99_s": percentile(ttft, 0.99),
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p90_s": percentile(e2e, 0.90),
        "e2e_p99_s": percentile(e2e, 0.99),
        "tpot_p50_s": percentile(tpot, 0.50),
        "tpot_p90_s": percentile(tpot, 0.90),
        "tpot_p99_s": percentile(tpot, 0.99),
        "queue_p50_s": percentile(queue_time, 0.50),
        "queue_p90_s": percentile(queue_time, 0.90),
        "queue_p99_s": percentile(queue_time, 0.99),
        "pressure_histogram": pressure_histogram,
    }


def compare_pair(lhs: dict, rhs: dict) -> dict:
    def delta(key: str) -> float:
        return float(rhs.get(key, 0.0)) - float(lhs.get(key, 0.0))

    def pct(key: str) -> float:
        base = float(lhs.get(key, 0.0))
        if base == 0.0:
            return 0.0
        return delta(key) / base * 100.0

    return {
        "baseline_label": lhs["label"],
        "candidate_label": rhs["label"],
        "delta_ttft_p90_s": delta("ttft_p90_s"),
        "delta_e2e_p90_s": delta("e2e_p90_s"),
        "delta_tpot_p90_s": delta("tpot_p90_s"),
        "delta_step_latency_std_ms": delta("step_latency_std_ms"),
        "delta_throughput_tok_s": delta("throughput_tok_s"),
        "pct_ttft_p90": pct("ttft_p90_s"),
        "pct_e2e_p90": pct("e2e_p90_s"),
        "pct_tpot_p90": pct("tpot_p90_s"),
        "pct_step_latency_std": pct("step_latency_std_ms"),
        "pct_throughput_tok_s": pct("throughput_tok_s"),
    }


def build_markdown(summary_by_label: dict[str, dict], comparisons: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Collected Metrics")
    lines.append("")
    lines.append("| Label | Requests | Batches | Avg Req/Batch | Avg Batch ms | Step Std ms | TTFT p90 s | E2E p90 s | TPOT p90 s | Throughput tok/s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, summary in summary_by_label.items():
        lines.append(
            f"| {label} | {summary['num_requests']} | {summary['num_batches']} | "
            f"{summary['avg_requests_per_batch']:.2f} | {summary['avg_batch_ms']:.2f} | "
            f"{summary['step_latency_std_ms']:.2f} | {summary['ttft_p90_s']:.4f} | "
            f"{summary['e2e_p90_s']:.4f} | {summary['tpot_p90_s']:.4f} | "
            f"{summary['throughput_tok_s']:.2f} |"
        )
    if comparisons:
        lines.append("")
        lines.append("## Comparisons")
        lines.append("")
        lines.append("| Baseline | Candidate | TTFT p90 delta s | E2E p90 delta s | TPOT p90 delta s | Step Std delta ms | Throughput delta tok/s |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for item in comparisons:
            lines.append(
                f"| {item['baseline_label']} | {item['candidate_label']} | "
                f"{item['delta_ttft_p90_s']:.4f} | {item['delta_e2e_p90_s']:.4f} | "
                f"{item['delta_tpot_p90_s']:.4f} | {item['delta_step_latency_std_ms']:.2f} | "
                f"{item['delta_throughput_tok_s']:.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="comparison in the form baseline_label:candidate_label",
    )
    args = parser.parse_args()

    summary_by_label: dict[str, dict] = {}
    for path in iter_result_files(args.inputs):
        payload = load_payload(path)
        label = path.stem
        summary_by_label[label] = summarize_payload(label, payload)

    comparisons: list[dict] = []
    for raw in args.compare:
        baseline_label, candidate_label = raw.split(":", 1)
        if baseline_label in summary_by_label and candidate_label in summary_by_label:
            comparisons.append(compare_pair(summary_by_label[baseline_label], summary_by_label[candidate_label]))

    output_payload = {
        "summary_by_label": summary_by_label,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    args.output_md.write_text(build_markdown(summary_by_label, comparisons), encoding="utf-8")
    print(json.dumps(output_payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
