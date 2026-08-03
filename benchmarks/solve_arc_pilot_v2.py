#!/usr/bin/env python3
"""Corrected, training-gated solver for the frozen ARC-AGI-2 pilot.

This version preserves the original solver for auditability and changes one waterfall
rule: a cap is always drawn above a horizontal color-2 run struck by an active ray,
even when an exit lane is bordered by another color-2 run.
"""
from __future__ import annotations

import json
from typing import Any

from solve_arc_pilot import (
    PILOT_DIR,
    denoise_stamp_transform,
    dump_json,
    marker_stack_transform,
    runs_of_value,
)

Grid = list[list[int]]


def waterfall_transform_v2(source: Grid) -> Grid:
    height, width = len(source), len(source[0])
    output = [row[:] for row in source]
    active = {c for c, value in enumerate(source[0]) if value == 6}

    for row_index in range(1, height):
        active |= {c for c, value in enumerate(source[row_index]) if value == 6}
        hit_runs: dict[tuple[int, int], set[int]] = {}
        unaffected = set(active)

        for left, right in runs_of_value(source[row_index], 2):
            hits = {c for c in active if left <= c <= right}
            if hits:
                hit_runs[(left, right)] = hits
                unaffected -= hits

        new_active = set(unaffected)
        for left, right in hit_runs:
            if left - 1 >= 0:
                new_active.add(left - 1)
            if right + 1 < width:
                new_active.add(right + 1)

            # Training pair 36a08778/3 establishes that this cap remains present
            # even when an exit cell is itself bounded by a neighboring 2-run.
            for column in range(max(0, left - 1), min(width - 1, right + 1) + 1):
                if output[row_index - 1][column] == 7:
                    output[row_index - 1][column] = 6

        active = new_active
        for column in active:
            if output[row_index][column] == 7:
                output[row_index][column] = 6

    return output


def choose_solver(task: dict[str, Any]):
    values = {
        value
        for pair in task["train"]
        for grid_name in ("input", "output")
        for row in pair[grid_name]
        for value in row
    }
    first_input = task["train"][0]["input"]
    if values <= {2, 6, 7} and 6 in values:
        return "waterfall-v2", waterfall_transform_v2
    if len(first_input) == 30 and len(first_input[0]) == 30 and 2 in first_input[-1] and 3 in first_input[-1]:
        return "marker-stack", marker_stack_transform
    if 0 in values and 8 in values:
        return "denoise-stamp", denoise_stamp_transform
    raise ValueError(f"No solver signature matched values={sorted(values)}")


def main() -> None:
    manifest = json.loads((PILOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    predictions: dict[str, Any] = {
        "candidate": "Anamnesis AGI Candidate v0.3.0 — reproducible pilot solver v2",
        "notes": (
            "Each rule must reproduce every supplied training pair exactly. "
            "Attempt 1 is generated before hidden-output scoring; attempt 2 is left empty "
            "except where the separate evidence-bounded revision step fills it."
        ),
        "tasks": {},
        "validation": {},
    }

    for task_id in manifest["task_ids"]:
        task = json.loads((PILOT_DIR / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        solver_name, solver = choose_solver(task)
        checks = []
        for index, pair in enumerate(task["train"]):
            exact = solver(pair["input"]) == pair["output"]
            checks.append({"train_index": index, "exact": exact})
            if not exact:
                raise SystemExit(f"{task_id}: {solver_name} failed training pair {index}; refusing test prediction")

        predictions["tasks"][task_id] = {
            "attempt_1": [solver(pair["input"]) for pair in task["test"]],
            "attempt_2": [],
        }
        predictions["validation"][task_id] = {
            "solver": solver_name,
            "all_training_pairs_exact": True,
            "train_checks": checks,
        }

    dump_json(PILOT_DIR / "predictions.json", predictions)


if __name__ == "__main__":
    main()
