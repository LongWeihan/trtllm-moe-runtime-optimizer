# 24 Prefill Control

Primary implementation file:

- `scheduler/adaptive_chunking.py`

## Goal

The full-version project adds a small prefill control layer so chunked prefill is not treated as an always-good knob.

## Logic

The adaptive decision considers:

- hot request count
- repeated-prefix structure
- default microbatch size
- current scheduler token budget

It then adjusts:

- effective microbatch size
- effective scheduler max tokens
- a prefill penalty recorded into the `StepPlan` notes

## Why this matters

This was the cheapest way to make the runtime feel more architecture-aware without rewriting the executor:

- `v1` isolated pressure but often destroyed throughput
- `v2` combines admission and chunking so hot batches can stay smaller while repeated-prefix cases still retain useful structure

## Where it helped most

The clearest benefits showed up on:

- `Mixed Burst`
- `Repeated-Prefix under MoE Pressure`

Those are the two workloads where pressure control and batch-shape control mattered together.
