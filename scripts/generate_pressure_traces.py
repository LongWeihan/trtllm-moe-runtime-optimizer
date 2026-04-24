from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_result_metrics(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics_by_id: dict[int, dict] = {}
    for record in payload.get("records", []):
        metrics = record.get("metrics_dict", {})
        metrics_by_id[int(record["request_id"])] = {
            "observed_e2e": metrics.get("MetricNames.E2E"),
            "observed_ttft": metrics.get("MetricNames.TTFT"),
            "observed_tpot": metrics.get("MetricNames.TPOT"),
        }
    return metrics_by_id


def synthetic_histograms(pressure_class: str, group: str | None) -> tuple[dict[str, float], dict[str, float]]:
    if pressure_class == "hot_expert":
        return (
            {"expert_a": 0.72, "expert_b": 0.18, "expert_tail": 0.10},
            {"rank_0": 0.58, "rank_1": 0.22, "rank_tail": 0.20},
        )
    if pressure_class == "hot_rank":
        return (
            {"expert_left": 0.34, "expert_right": 0.33, "expert_tail": 0.33},
            {"rank_0": 0.76, "rank_1": 0.14, "rank_tail": 0.10},
        )
    prefix_weight = 0.18 if group and "shared_prefix" in group else 0.0
    return (
        {"expert_a": 0.34 + prefix_weight, "expert_b": 0.33, "expert_tail": 0.33 - prefix_weight},
        {"rank_0": 0.38, "rank_1": 0.31, "rank_tail": 0.31},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-kind", type=str, default="qwen15_replay")
    args = parser.parse_args()

    workload_records = load_jsonl(args.workload)
    metrics_by_id = load_result_metrics(args.result_json) if args.result_json else {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in workload_records:
            expert_histogram, rank_histogram = synthetic_histograms(
                str(record.get("pressure_class", "balanced")),
                record.get("pressure_group"),
            )
            metrics = metrics_by_id.get(int(record["request_id"]), {})
            payload = {
                "request_id": int(record["request_id"]),
                "pressure_class": record.get("pressure_class", "balanced"),
                "pressure_score": float(record.get("pressure_score", 1.0)),
                "pressure_group": record.get("pressure_group"),
                "source_kind": args.source_kind,
                "expert_histogram": expert_histogram,
                "rank_histogram": rank_histogram,
                **metrics,
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
