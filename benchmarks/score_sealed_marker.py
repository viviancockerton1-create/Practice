#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PILOT = Path("benchmarks/arc_agi2_public/pilot_001")
TASK_ID = "9aaea919"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-root", type=Path, required=True)
    args = parser.parse_args()

    sealed = json.loads((PILOT / "sealed_attempt2_9aaea919.json").read_text(encoding="utf-8"))
    official = json.loads((args.arc_root / "data" / "evaluation" / f"{TASK_ID}.json").read_text(encoding="utf-8"))
    gold = official["test"][0]["output"]
    prediction = [[int(cell) for cell in row] for row in sealed["prediction_rows"]]
    exact = prediction == gold
    result = {
        "task_id": TASK_ID,
        "attempt": 2,
        "dataset_commit": "f3283f727488ad98fe575ea6a5ac981e4a188e49",
        "sealed_commit": "25f930468bda44c18261b5000b8235ba386b74f1",
        "valid_rectangular_grid": bool(prediction and all(len(row) == len(prediction[0]) for row in prediction)),
        "predicted_shape": [len(prediction), len(prediction[0]) if prediction else 0],
        "gold_shape": [len(gold), len(gold[0]) if gold else 0],
        "exact_match": exact,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "developer-run public ARC-AGI-2 pilot; not independent AGI proof"
    }
    output = PILOT / "sealed_attempt2_score.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
