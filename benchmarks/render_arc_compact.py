#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

PILOT_DIR = Path("benchmarks/arc_agi2_public/pilot_001")


def rows(grid):
    return ["".join(str(cell) for cell in row) for row in grid]


def emit_grid(lines, title, grid):
    lines.append(title)
    lines.extend(rows(grid))
    lines.append("")


def main():
    out_dir = PILOT_DIR / "compact"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((PILOT_DIR / "tasks").glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        lines = [f"TASK {task['task_id']}", "Digits are ARC colors 0-9.", ""]
        for i, pair in enumerate(task["train"]):
            emit_grid(lines, f"TRAIN {i} INPUT {len(pair['input'])}x{len(pair['input'][0])}", pair["input"])
            emit_grid(lines, f"TRAIN {i} OUTPUT {len(pair['output'])}x{len(pair['output'][0])}", pair["output"])
        for i, pair in enumerate(task["test"]):
            emit_grid(lines, f"TEST {i} INPUT {len(pair['input'])}x{len(pair['input'][0])}", pair["input"])
            lines.append(f"TEST {i} OUTPUT: predict exactly, up to two attempts")
            lines.append("")
        (out_dir / f"{task['task_id']}.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
