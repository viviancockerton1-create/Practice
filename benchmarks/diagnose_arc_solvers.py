#!/usr/bin/env python3
"""Record exact training-pair failures for the frozen ARC pilot solvers."""
from __future__ import annotations

import json

from solve_arc_pilot_v2 import PILOT_DIR, choose_solver, dump_json


def mismatch_summary(predicted, expected):
    rows = max(len(predicted), len(expected))
    mismatch_count = 0
    examples = []
    shape_match = bool(predicted and expected and len(predicted) == len(expected) and len(predicted[0]) == len(expected[0]))
    for r in range(rows):
        prow = predicted[r] if r < len(predicted) else []
        erow = expected[r] if r < len(expected) else []
        cols = max(len(prow), len(erow))
        for c in range(cols):
            pv = prow[c] if c < len(prow) else None
            ev = erow[c] if c < len(erow) else None
            if pv != ev:
                mismatch_count += 1
                if len(examples) < 20:
                    examples.append({"row": r, "column": c, "predicted": pv, "expected": ev})
    return {"shape_match": shape_match, "mismatch_count": mismatch_count, "first_mismatches": examples}


def main() -> None:
    manifest = json.loads((PILOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    report = {"solver_version": "v2", "all_training_pairs_exact": True, "tasks": {}}
    for task_id in manifest["task_ids"]:
        task = json.loads((PILOT_DIR / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        try:
            solver_name, solver = choose_solver(task)
        except Exception as exc:
            report["all_training_pairs_exact"] = False
            report["tasks"][task_id] = {"solver_selection_error": repr(exc)}
            continue
        checks = []
        for index, pair in enumerate(task["train"]):
            predicted = solver(pair["input"])
            exact = predicted == pair["output"]
            check = {"train_index": index, "exact": exact}
            if not exact:
                report["all_training_pairs_exact"] = False
                check.update(mismatch_summary(predicted, pair["output"]))
            checks.append(check)
        report["tasks"][task_id] = {"solver": solver_name, "checks": checks}
    dump_json(PILOT_DIR / "solver_diagnostic.json", report)


if __name__ == "__main__":
    main()
