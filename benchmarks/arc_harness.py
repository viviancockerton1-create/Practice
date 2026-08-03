#!/usr/bin/env python3
"""Prepare and score a contamination-controlled ARC-AGI-2 public pilot.

The prepare step writes demonstration pairs and test inputs only. Gold test outputs
remain in the separately cloned official repository and are used only by score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PILOT_DIR = Path("benchmarks/arc_agi2_public/pilot_001")
SEED = "anamnesis-v0.3.0-pilot-001"
TASK_COUNT = 3
DATASET_COMMIT = "f3283f727488ad98fe575ea6a5ac981e4a188e49"


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(arc_root: Path) -> None:
    evaluation = arc_root / "data" / "evaluation"
    files = sorted(evaluation.glob("*.json"))
    if len(files) < TASK_COUNT:
        raise SystemExit(f"Expected ARC evaluation files in {evaluation}; found {len(files)}")

    rng_seed = int(hashlib.sha256(SEED.encode()).hexdigest()[:16], 16)
    chosen = sorted(random.Random(rng_seed).sample(files, TASK_COUNT), key=lambda p: p.stem)

    tasks_dir = PILOT_DIR / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_ids: list[str] = []
    for source in chosen:
        task = json.loads(source.read_text(encoding="utf-8"))
        task_id = source.stem
        task_ids.append(task_id)
        sanitized = {
            "task_id": task_id,
            "train": task["train"],
            "test": [{"input": pair["input"]} for pair in task["test"]],
            "instructions": {
                "goal": "Infer the transformation from train pairs and predict every test output grid.",
                "allowed_attempts_per_test_input": 2,
                "grid_values": "integers 0 through 9",
                "success": "A task is solved only if every test input is exactly correct within two attempts.",
            },
        }
        dump_json(tasks_dir / f"{task_id}.json", sanitized)

    manifest = {
        "benchmark": "ARC-AGI-2 public evaluation pilot",
        "dataset_repository": "arcprize/ARC-AGI-2",
        "dataset_commit": DATASET_COMMIT,
        "selection_seed_sha256": hashlib.sha256(SEED.encode()).hexdigest(),
        "selection_method": "deterministic sample without replacement from sorted public evaluation filenames",
        "task_count": TASK_COUNT,
        "task_ids": task_ids,
        "answer_exposure": "Gold test outputs were removed from committed task files.",
        "contamination_warning": "The source set is public; this is not private or independent AGI evidence.",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(PILOT_DIR / "manifest.json", manifest)

    predictions = PILOT_DIR / "predictions.json"
    if not predictions.exists():
        dump_json(
            predictions,
            {
                "candidate": "Anamnesis AGI Candidate v0.3.0 — in-chat cognitive loop",
                "notes": "Fill attempts without reading official test outputs. Each attempt is a list of output grids, one per test input.",
                "tasks": {task_id: {"attempt_1": [], "attempt_2": []} for task_id in task_ids},
            },
        )


def valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    return all(
        len(row) == width
        and all(isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9 for cell in row)
        for row in value
    )


def score(arc_root: Path) -> None:
    manifest_path = PILOT_DIR / "manifest.json"
    predictions_path = PILOT_DIR / "predictions.json"
    if not manifest_path.exists() or not predictions_path.exists():
        raise SystemExit("Run prepare and provide predictions.json before scoring")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    reports: list[dict[str, Any]] = []
    solved_tasks = 0
    total_test_inputs = 0
    solved_test_inputs = 0

    for task_id in manifest["task_ids"]:
        official = json.loads((arc_root / "data" / "evaluation" / f"{task_id}.json").read_text(encoding="utf-8"))
        expected = [pair["output"] for pair in official["test"]]
        submitted = predictions.get("tasks", {}).get(task_id, {})
        attempt_1 = submitted.get("attempt_1", [])
        attempt_2 = submitted.get("attempt_2", [])

        input_results = []
        for index, gold in enumerate(expected):
            first = attempt_1[index] if index < len(attempt_1) else None
            second = attempt_2[index] if index < len(attempt_2) else None
            first_valid = valid_grid(first)
            second_valid = valid_grid(second)
            first_exact = bool(first_valid and first == gold)
            second_exact = bool(second_valid and second == gold)
            solved = first_exact or second_exact
            total_test_inputs += 1
            solved_test_inputs += int(solved)
            input_results.append(
                {
                    "test_index": index,
                    "attempt_1_valid_grid": first_valid,
                    "attempt_1_exact": first_exact,
                    "attempt_2_valid_grid": second_valid,
                    "attempt_2_exact": second_exact,
                    "solved_within_two_attempts": solved,
                }
            )

        task_solved = bool(input_results and all(item["solved_within_two_attempts"] for item in input_results))
        solved_tasks += int(task_solved)
        reports.append({"task_id": task_id, "task_solved": task_solved, "test_inputs": input_results})

    report = {
        "benchmark": manifest["benchmark"],
        "candidate": predictions.get("candidate"),
        "dataset_commit": manifest["dataset_commit"],
        "task_count": len(manifest["task_ids"]),
        "tasks_solved": solved_tasks,
        "task_accuracy": solved_tasks / len(manifest["task_ids"]) if manifest["task_ids"] else 0.0,
        "test_inputs_solved": solved_test_inputs,
        "test_input_accuracy": solved_test_inputs / total_test_inputs if total_test_inputs else 0.0,
        "scoring": "Exact grid match; each test input may be solved by attempt 1 or attempt 2; all test inputs required for task success.",
        "evidence_class": "developer-run public pilot; not independent and not AGI proof",
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "tasks": reports,
    }
    dump_json(PILOT_DIR / "score.json", report)

    lines = [
        "# ARC-AGI-2 Public Pilot Result",
        "",
        f"- Candidate: {report['candidate']}",
        f"- Dataset commit: `{report['dataset_commit']}`",
        f"- Tasks solved: **{solved_tasks}/{report['task_count']}**",
        f"- Task accuracy: **{report['task_accuracy']:.1%}**",
        f"- Test inputs solved: **{solved_test_inputs}/{total_test_inputs}**",
        f"- Test-input accuracy: **{report['test_input_accuracy']:.1%}**",
        "- Evidence class: developer-run public pilot; not independent and not AGI proof",
        "",
        "## Per-task results",
        "",
    ]
    for item in reports:
        lines.append(f"- `{item['task_id']}`: {'SOLVED' if item['task_solved'] else 'NOT SOLVED'}")
    (PILOT_DIR / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("--arc-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.arc_root)
    else:
        score(args.arc_root)


if __name__ == "__main__":
    main()
