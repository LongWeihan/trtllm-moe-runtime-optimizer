from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(path: Path) -> dict:
    payload = load_result(path)
    batches = payload.get("batch_plan", [])
    records = payload.get("records", [])
    total_tokens = sum(len(record.get("output_token_ids", [])) for record in records)
    total_wall_s = sum(float(batch.get("batch_wall_ms", 0.0)) for batch in batches) / 1000.0
    return {
        "file": path.name,
        "num_requests": len(records),
        "num_batches": len(batches),
        "throughput_tok_s": (total_tokens / total_wall_s) if total_wall_s else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = [summarize(path) for path in args.inputs]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Plot-Free Result Summary",
        "",
        "| File | Requests | Batches | Throughput tok/s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['file']} | {row['num_requests']} | {row['num_batches']} | {row['throughput_tok_s']:.2f} |"
        )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
