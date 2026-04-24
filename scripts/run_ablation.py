from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_command(args: argparse.Namespace, variant: str, workload: Path, output: Path) -> list[str]:
    base = [
        sys.executable,
        str(PROJECT_ROOT / ("scripts/run_patched.py" if variant.startswith("moe") else "scripts/run_baseline.py")),
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--workload",
        str(workload),
        "--output",
        str(output),
        "--microbatch-size",
        str(args.microbatch_size),
    ]
    if variant == "default":
        base += [
            "--capacity-scheduler-policy",
            "GUARANTEED_NO_EVICT",
        ]
    elif variant == "max_util":
        base += [
            "--capacity-scheduler-policy",
            "MAX_UTILIZATION",
        ]
    elif variant == "overlap":
        base += [
            "--capacity-scheduler-policy",
            "GUARANTEED_NO_EVICT",
            "--enable-chunked-prefill",
        ]
    elif variant == "moe_v1":
        base += [
            "--planner",
            "v1",
            "--variant-name",
            variant,
        ]
    elif variant == "moe_v2":
        base += [
            "--planner",
            "v2",
            "--variant-name",
            variant,
            "--enable-adaptive-chunking",
        ]
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("trt", "torch"), default="trt")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"variants": {}}
    for variant in args.variants:
        output_path = args.output_dir / f"{variant}.json"
        command = build_command(args, variant, args.workload, output_path)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        manifest["variants"][variant] = {
            "output": str(output_path),
            "command": command,
        }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
