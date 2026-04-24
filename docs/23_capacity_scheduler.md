# 23 Capacity Scheduler

Primary implementation file:

- `scheduler/moe_capacity_scheduler.py`

## Goal

`v2` extends the project from pure pressure dispersion into admission / capacity control.

## Core idea

Instead of only asking:

- which requests are hot?

`v2` also asks:

- which requests fit the current pressure budget?
- can a controlled hot pair be admitted?
- should repeated-prefix requests get a reuse bonus?

## Current scoring logic

The file implements a lightweight ranking score using:

- pressure penalty
- request token cost
- shared-prefix bonus
- extra penalty for `hot_rank`

The planner then constructs a `CapacityDecision`:

- selected records
- deferred records
- selected total pressure
- selected total tokens
- dynamic pressure budget

## Why this matters

This is the part that pushes the project closer to runtime architecture work rather than a single heuristic:

- resource modeling is explicit
- the planner returns a real admission decision
- the selected batch is no longer just a sorted subset from `v1`

## Observed effect

`v2` was especially useful on:

- `Hot-Expert`
- `Mixed Burst`
- `Repeated-Prefix under MoE Pressure`

because it recovers some batching without giving up all pressure awareness.
