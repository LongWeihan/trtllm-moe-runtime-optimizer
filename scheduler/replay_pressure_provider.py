from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReplayTraceRecord:
    request_id: int
    pressure_class: str
    pressure_score: float
    pressure_group: str | None
    source_kind: str | None
    expert_histogram: dict[str, float]
    rank_histogram: dict[str, float]
    observed_e2e: float | None = None
    observed_ttft: float | None = None
    observed_tpot: float | None = None


class ReplayPressureProvider:
    def __init__(self, trace_path: str | Path) -> None:
        self.trace_path = Path(trace_path)
        self._records = self._load_records(self.trace_path)

    @staticmethod
    def _load_records(path: Path) -> dict[int, ReplayTraceRecord]:
        records: dict[int, ReplayTraceRecord] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            payload = json.loads(line)
            records[int(payload["request_id"])] = ReplayTraceRecord(
                request_id=int(payload["request_id"]),
                pressure_class=str(payload["pressure_class"]),
                pressure_score=float(payload["pressure_score"]),
                pressure_group=payload.get("pressure_group"),
                source_kind=payload.get("source_kind"),
                expert_histogram=dict(payload.get("expert_histogram", {})),
                rank_histogram=dict(payload.get("rank_histogram", {})),
                observed_e2e=(
                    float(payload["observed_e2e"])
                    if payload.get("observed_e2e") is not None
                    else None
                ),
                observed_ttft=(
                    float(payload["observed_ttft"])
                    if payload.get("observed_ttft") is not None
                    else None
                ),
                observed_tpot=(
                    float(payload["observed_tpot"])
                    if payload.get("observed_tpot") is not None
                    else None
                ),
            )
        return records

    def apply(self, workload_records: list[dict]) -> list[dict]:
        enriched: list[dict] = []
        for record in workload_records:
            req_id = int(record["request_id"])
            trace = self._records.get(req_id)
            if trace is None:
                enriched.append(dict(record))
                continue
            merged = dict(record)
            merged.update(
                {
                    "pressure_class": trace.pressure_class,
                    "pressure_score": trace.pressure_score,
                    "pressure_group": trace.pressure_group,
                    "trace_source_kind": trace.source_kind,
                    "expert_histogram": trace.expert_histogram,
                    "rank_histogram": trace.rank_histogram,
                    "observed_e2e": trace.observed_e2e,
                    "observed_ttft": trace.observed_ttft,
                    "observed_tpot": trace.observed_tpot,
                }
            )
            enriched.append(merged)
        return enriched
