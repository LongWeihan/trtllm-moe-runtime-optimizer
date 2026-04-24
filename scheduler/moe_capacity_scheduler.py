from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .moe_pressure import normalize_pressure_class


@dataclass(slots=True)
class CapacityDecision:
    selected_records: list[dict]
    deferred_records: list[dict]
    selected_request_ids: list[int]
    deferred_request_ids: list[int]
    selected_total_pressure: float
    selected_total_tokens: int
    score_table: dict[int, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def _record_score(record: dict, *, prefer_kv_reuse: bool = True) -> float:
    pressure_score = float(record.get("pressure_score", 1.0))
    max_tokens = int(record.get("max_tokens", 0))
    shared_prefix_bonus = 0.45 if prefer_kv_reuse and record.get("shared_prefix_id") else 0.0
    hot_rank_penalty = 0.30 if normalize_pressure_class(record.get("pressure_class")).value == "hot_rank" else 0.0
    return shared_prefix_bonus - (pressure_score * 0.35) - (max_tokens / 1024.0) - hot_rank_penalty


def build_capacity_decision(
    pending_records: list[dict],
    *,
    max_batch_size: int,
    max_num_tokens: int | None,
    base_pressure_budget: float,
    allow_hot_pair: bool = True,
    prefer_kv_reuse: bool = True,
) -> CapacityDecision:
    max_num_tokens = None if max_num_tokens in (None, 0) else int(max_num_tokens)
    score_table = {
        int(record["request_id"]): _record_score(record, prefer_kv_reuse=prefer_kv_reuse)
        for record in pending_records
    }

    sorted_records = sorted(
        pending_records,
        key=lambda record: (
            score_table[int(record["request_id"])],
            -int(record.get("request_id", 0)),
        ),
        reverse=True,
    )

    selected: list[dict] = []
    deferred: list[dict] = []
    total_pressure = 0.0
    total_tokens = 0
    dynamic_pressure_budget = base_pressure_budget

    for record in sorted_records:
        pressure_class = normalize_pressure_class(record.get("pressure_class"))
        pressure_score = float(record.get("pressure_score", 1.0))
        token_cost = max(1, int(record.get("max_tokens", 1)))

        if (
            allow_hot_pair
            and len(selected) == 1
            and normalize_pressure_class(selected[0].get("pressure_class")) == pressure_class
            and pressure_class.value != "balanced"
        ):
            dynamic_pressure_budget = max(dynamic_pressure_budget, 4.6)

        if len(selected) >= max_batch_size:
            deferred.append(record)
            continue
        if max_num_tokens is not None and total_tokens + token_cost > max_num_tokens:
            deferred.append(record)
            continue
        if total_pressure + pressure_score > dynamic_pressure_budget and selected:
            deferred.append(record)
            continue

        selected.append(record)
        total_pressure += pressure_score
        total_tokens += token_cost

    return CapacityDecision(
        selected_records=selected,
        deferred_records=deferred,
        selected_request_ids=[int(record["request_id"]) for record in selected],
        deferred_request_ids=[int(record["request_id"]) for record in deferred],
        selected_total_pressure=total_pressure,
        selected_total_tokens=total_tokens,
        score_table=score_table,
        notes={
            "dynamic_pressure_budget": dynamic_pressure_budget,
            "prefer_kv_reuse": prefer_kv_reuse,
            "allow_hot_pair": allow_hot_pair,
        },
    )
