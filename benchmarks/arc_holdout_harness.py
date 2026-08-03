#!/usr/bin/env python3
"""Run a frozen-solver ARC-AGI-2 public holdout without post-selection adaptation.

Protocol:
1. Verify the Git blob IDs of the frozen solver and its dependency.
2. Deterministically select unseen public-evaluation tasks, excluding the pilot tasks.
3. Commit only training pairs and test inputs; gold outputs remain in the pinned dataset.
4. Generate predictions with the unchanged frozen solver. Unsupported or training-gate
   failures receive no prediction and count as unsolved.
5. Score exact matches in a separate command against the untouched official outputs.

This is a developer-run public holdout, not a private or independent AGI evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solve_arc_pilot_v2 import choose_solver

ROOT = Path("benchmarks/arc_agi2_public/holdout_001")
SEED = "anamnesis-v0.3.0-frozen-holdout-001"
TASK_COUNT = 20
DATASET_COMMIT = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
EXCLUDED_TASKS = {"36a08778", "9aaea919", "de809cff"}
FROZEN_BLOBS = {
    "benchmarks/solve_arc_pilot_v2.py": "a58edcfab637a7fe9f9e9f3e3b34ec2b69d72b8f",
    "benchmarks/solve_arc_pilot.py": "3d6feff44dd86da3c7ed5ef18c3983a65704bf69",
}


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_frozen_solver() -> dict[str, Any]:
    checks = {}
    for name, expected in FROZEN_BLOBS.items():
        actual = git_blob_sha(Path(name))
        checks[name] = {"expected_blob": expected, "actual_blob": actual, "match": actual == expected}
        if actual != expected:
            raise SystemExit(f"Frozen solver mismatch for {name}: expected {expected}, got {actual}")
    return checks


def prepare(arc_root: Path) -> None:
    frozen_checks = verify_frozen_solver()
    evaluation = arc_root / "data" / "evaluation"
    files = [path for path in sorted(evaluation.glob("*.json")) if path.stem not in EXCLUDED_TASKS]
    if len(files) < TASK_COUNT:
        raise SystemExit(f"Expected at least {TASK_COUNT} eligible tasks; found {len(files)}")

    rng_seed = int(hashlib.sha256(SEED.encode()).hexdigest()[:16], 16)
    chosen = sorted(random.Random(rng_seed).sample(files, TASK_COUNT), key=lambda path: path.stem)
    tasks_dir = ROOT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_ids = []
    for source in chosen:
        task = json.loads(source.read_text(encoding="utf-8"))
        task_id = source.stem
        task_ids.append(task_id)
        sanitized = {
            "task_id": task_id,
            "train": task["train"],
            "test": [{"input": pair["input"]} for pair in task["test"]],
            "instructions": {
                "allowed_attempts_per_test_input": 1,
                "success": "Every test output for the task must exactly match.",
            },
        }
        dump_json(tasks_dir / f"{task_id}.json", sanitized)

    dump_json(
        ROOT / "manifest.json",
        {
            "benchmark": "ARC-AGI-2 frozen-solver public holdout",
            "dataset_commit": DATASET_COMMIT,
            "selection_seed_sha256": hashlib.sha256(SEED.encode()).hexdigest(),
            "selection_method": "deterministic sample without replacement from sorted public evaluation tasks, excluding pilot IDs",
            "task_count": TASK_COUNT,
            "task_ids": task_ids,
            "excluded_pilot_task_ids": sorted(EXCLUDED_TASKS),
            "solver_frozen_before_selection": True,
            "frozen_blob_checks": frozen_checks,
            "no_post_selection_modification_rule": True,
            "answer_exposure": "Gold test outputs excluded from holdout task files and unavailable to prediction command.",
            "evidence_class": "developer-run public holdout; not independent or private AGI proof",
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def predict() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    verify_frozen_solver()
    output: dict[str, Any] = {
        "candidate": "Anamnesis AGI Candidate v0.3.0 — frozen ARC solver v2",
        "frozen_blobs": FROZEN_BLOBS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
    }

    for task_id in manifest["task_ids"]:
        task = json.loads((ROOT / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        record: dict[str, Any] = {"status": None, "solver": None, "training_checks": [], "attempt_1": []}
        try:
            solver_name, solver = choose_solver(task)
            record["solver"] = solver_name
        except Exception as exc:
            record["status"] = "unsupported_signature"
            record["error"] = repr(exc)
            output["tasks"][task_id] = record
            continue

        all_exact = True
        for index, pair in enumerate(task["train"]):
            exact = solver(pair["input"]) == pair["output"]
            record["training_checks"].append({"train_index": index, "exact": exact})
            all_exact &= exact

        if not all_exact:
            record["status"] = "rejected_by_training_gate"
            output["tasks"][task_id] = record
            continue

        record["status"] = "predicted"
        record["attempt_1"] = [solver(pair["input"]) for pair in task["test"]]
        output["tasks"][task_id] = record

    dump_json(ROOT / "predictions.json", output)


def valid_grid(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(isinstance(row, list) and row for row in value)
        and all(len(row) == len(value[0]) for row in value)
        and all(isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9 for row in value for cell in row)
    )


def score(arc_root: Path) -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    predictions = json.loads((ROOT / "predictions.json").read_text(encoding="utf-8"))

    reports = []
    tasks_solved = 0
    predicted_tasks = 0
    training_gate_rejections = 0
    unsupported_tasks = 0
    total_test_inputs = 0
    solved_test_inputs = 0

    for task_id in manifest["task_ids"]:
        official = json.loads((arc_root / "data" / "evaluation" / f"{task_id}.json").read_text(encoding="utf-8"))
        expected = [pair["output"] for pair in official["test"]]
        submitted = predictions["tasks"][task_id]
        status = submitted["status"]
        attempts = submitted.get("attempt_1", [])
        if status == "predicted":
            predicted_tasks += 1
        elif status == "rejected_by_training_gate":
            training_gate_rejections += 1
        elif status == "unsupported_signature":
            unsupported_tasks += 1

        input_results = []
        for index, gold in enumerate(expected):
            prediction = attempts[index] if index < len(attempts) else None
            exact = bool(valid_grid(prediction) and prediction == gold)
            total_test_inputs += 1
            solved_test_inputs += int(exact)
            input_results.append({"test_index": index, "valid_grid": valid_grid(prediction), "exact": exact})

        task_solved = bool(input_results and all(item["exact"] for item in input_results))
        tasks_solved += int(task_solved)
        reports.append(
            {
                "task_id": task_id,
                "status": status,
                "solver": submitted.get("solver"),
                "task_solved": task_solved,
                "test_inputs": input_results,
            }
        )

    total_tasks = len(manifest["task_ids"])
    report = {
        "benchmark": manifest["benchmark"],
        "dataset_commit": manifest["dataset_commit"],
        "candidate": predictions["candidate"],
        "frozen_blobs": predictions["frozen_blobs"],
        "total_tasks": total_tasks,
        "predicted_tasks": predicted_tasks,
        "unsupported_tasks": unsupported_tasks,
        "training_gate_rejections": training_gate_rejections,
        "coverage": predicted_tasks / total_tasks if total_tasks else 0.0,
        "tasks_solved": tasks_solved,
        "task_accuracy": tasks_solved / total_tasks if total_tasks else 0.0,
        "test_inputs_solved": solved_test_inputs,
        "total_test_inputs": total_test_inputs,
        "test_input_accuracy": solved_test_inputs / total_test_inputs if total_test_inputs else 0.0,
        "evidence_class": manifest["evidence_class"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "tasks": reports,
    }
    dump_json(ROOT / "score.json", report)

    lines = [
        "# ARC-AGI-2 Frozen-Solver Holdout Result",
        "",
        f"- Candidate: {report['candidate']}",
        f"- Dataset commit: `{report['dataset_commit']}`",
        f"- Frozen solver coverage: **{predicted_tasks}/{total_tasks} ({report['coverage']:.1%})**",
        f"- Unsupported signatures: **{unsupported_tasks}**",
        f"- Training-gate rejections: **{training_gate_rejections}**",
        f"- Tasks solved: **{tasks_solved}/{total_tasks} ({report['task_accuracy']:.1%})**",
        f"- Test inputs solved: **{solved_test_inputs}/{total_test_inputs} ({report['test_input_accuracy']:.1%})**",
        f"- Evidence class: {report['evidence_class']}",
        "",
        "## Per-task results",
        "",
    ]
    for item in reports:
        lines.append(f"- `{item['task_id']}` — {item['status']} — {'SOLVED' if item['task_solved'] else 'NOT SOLVED'}")
    (ROOT / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "predict", "score"])
    parser.add_argument("--arc-root", type=Path)
    args = parser.parse_args()
    if args.command in {"prepare", "score"} and args.arc_root is None:
        raise SystemExit("--arc-root is required")
    if args.command == "prepare":
        prepare(args.arc_root)
    elif args.command == "predict":
        predict()
    else:
        score(args.arc_root)


if __name__ == "__main__":
    main()
