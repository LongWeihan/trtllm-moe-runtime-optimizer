from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChunkingDecision:
    scheduler_max_tokens: int | None
    effective_microbatch_size: int
    prefill_penalty: float
    notes: dict


def compute_chunking_decision(
    pending_records: list[dict],
    *,
    default_microbatch_size: int,
    default_scheduler_max_tokens: int | None,
) -> ChunkingDecision:
    repeated_prefix_requests = sum(1 for record in pending_records if record.get("shared_prefix_id"))
    hot_requests = sum(1 for record in pending_records if record.get("pressure_class") != "balanced")
    prefill_penalty = 0.0
    effective_max_tokens = default_scheduler_max_tokens
    effective_microbatch_size = default_microbatch_size

    if repeated_prefix_requests > 0:
        prefill_penalty += 0.2
    if hot_requests >= max(1, len(pending_records) // 2):
        prefill_penalty += 0.35
        effective_microbatch_size = max(1, default_microbatch_size - 1)
    if default_scheduler_max_tokens not in (None, 0) and hot_requests > 0:
        effective_max_tokens = max(64, int(default_scheduler_max_tokens * (1.0 - prefill_penalty)))

    return ChunkingDecision(
        scheduler_max_tokens=effective_max_tokens,
        effective_microbatch_size=effective_microbatch_size,
        prefill_penalty=prefill_penalty,
        notes={
            "repeated_prefix_requests": repeated_prefix_requests,
            "hot_requests": hot_requests,
        },
    )
