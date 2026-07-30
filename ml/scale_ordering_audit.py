"""Audit FPB small/average/big ordering from saved SCALE evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
from typing import Any


FPB_SIZE_ID = re.compile(
    r"^fpb:test:(.+)_(small|average|big)_(?:RGB|RGb|rgb|Depth)_"
)
SIZES = ("small", "average", "big")


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        match = FPB_SIZE_ID.match(str(row["id"]))
        if match is None:
            continue
        grouped.setdefault(match.group(1), {}).setdefault(match.group(2), []).append(
            float(row["p50_g"])
        )
    triads = {
        food: {
            size: statistics.median(values)
            for size, values in by_size.items()
        }
        for food, by_size in grouped.items()
        if set(by_size) == set(SIZES)
    }
    failures = {
        food: predictions
        for food, predictions in triads.items()
        if not (
            predictions["small"]
            < predictions["average"]
            < predictions["big"]
        )
    }
    pairwise_correct = sum(
        int(predictions["small"] < predictions["average"])
        + int(predictions["average"] < predictions["big"])
        + int(predictions["small"] < predictions["big"])
        for predictions in triads.values()
    )
    return {
        "complete_triads": len(triads),
        "ordered_triads": len(triads) - len(failures),
        "pairwise_comparisons": 3 * len(triads),
        "pairwise_correct": pairwise_correct,
        "failed_families": dict(sorted(failures.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = {}
    for path in args.evaluations:
        payload = json.loads(path.read_text(encoding="utf-8"))
        output[str(path)] = audit(payload["paired"])
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
