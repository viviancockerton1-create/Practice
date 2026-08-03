#!/usr/bin/env python3
"""Use the single allowed feedback signal to revise marker-stack attempt 2.

Attempt 1 summed object counts across all color-2-marked columns. Training examples
with one marker cannot distinguish that from using the marked columns' common stack
height. The test contains two equal-height marked columns, so attempt 2 uses the
common/max stack height while preserving all other inferred operations.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from solve_arc_pilot import connected_components, component_center_column, dump_json, runs_of_value

PILOT_DIR = Path("benchmarks/arc_agi2_public/pilot_001")
TASK_ID = "9aaea919"


def transform(source: list[list[int]]) -> list[list[int]]:
    height, width = len(source), len(source[0])
    background = Counter(value for row in source for value in row).most_common(1)[0][0]
    output = [row[:] for row in source]
    marker_2_centers = [(left + right) // 2 for left, right in runs_of_value(source[-1], 2)]
    marker_3_centers = [(left + right) // 2 for left, right in runs_of_value(source[-1], 3)]
    output[-1] = [background] * width

    objects: list[dict[str, Any]] = []
    object_colors = sorted({value for row in source[:-1] for value in row if value != background})
    for color in object_colors:
        for component in connected_components(source, color, last_row_excluded=True):
            rows = [r for r, _ in component]
            cols = [c for _, c in component]
            objects.append(
                {
                    "color": color,
                    "cells": component,
                    "top": min(rows),
                    "center": component_center_column(component),
                }
            )

    per_marked_column_counts: list[int] = []
    for center in marker_2_centers:
        marked = [obj for obj in objects if obj["center"] == center]
        per_marked_column_counts.append(len(marked))
        for obj in marked:
            for r, c in obj["cells"]:
                output[r][c] = 5

    # All color-2-marked stacks encode the same count. Use the common/max height,
    # rather than summing repeated examples of that count.
    extension_count = max(per_marked_column_counts, default=0)

    for center in marker_3_centers:
        column_objects = sorted((obj for obj in objects if obj["center"] == center), key=lambda item: item["top"])
        if not column_objects:
            continue
        template = column_objects[0]
        existing_tops = sorted(obj["top"] for obj in column_objects)
        spacing = min((b - a for a, b in zip(existing_tops, existing_tops[1:])), default=4)
        relative_cells = [(r - template["top"], c - center) for r, c in template["cells"]]
        for step in range(1, extension_count + 1):
            new_top = existing_tops[0] - spacing * step
            for dr, dc in relative_cells:
                rr, cc = new_top + dr, center + dc
                if 0 <= rr < height - 1 and 0 <= cc < width:
                    output[rr][cc] = template["color"]
    return output


def main() -> None:
    task_path = PILOT_DIR / "tasks" / f"{TASK_ID}.json"
    predictions_path = PILOT_DIR / "predictions.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))

    for index, pair in enumerate(task["train"]):
        if transform(pair["input"]) != pair["output"]:
            raise SystemExit(f"Revised marker rule failed training pair {index}; refusing attempt 2")

    predictions["tasks"][TASK_ID]["attempt_2"] = [transform(pair["input"]) for pair in task["test"]]
    predictions.setdefault("revision_log", []).append(
        {
            "task_id": TASK_ID,
            "attempt": 2,
            "feedback_used": "attempt_1 exact-match failure only; gold grid remained hidden",
            "change": "extension count changed from sum across marked columns to common/max marked-stack height",
            "all_training_pairs_exact": True,
        }
    )
    dump_json(predictions_path, predictions)


if __name__ == "__main__":
    main()
