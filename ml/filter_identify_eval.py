"""Filter a saved IDENTIFY evaluation to IDs in a frozen JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def filter_evaluation(
    payload: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed = {str(row["id"]) for row in manifest_rows}
    paired = [
        row
        for row in payload["paired_results"]
        if str(row["id"]) in allowed
    ]
    found = {str(row["id"]) for row in paired}
    missing = allowed - found
    if missing:
        raise ValueError(
            f"{len(missing)} manifest IDs are absent from IDENTIFY evaluation: "
            f"{sorted(missing)[:5]}"
        )
    statuses = [str(row["status"]) for row in paired]
    result = {
        **payload,
        "samples": len(paired),
        "evaluation_seconds": None,
        "parse_failures": statuses.count("parse_error"),
        "schema_failures": statuses.count("schema_error"),
        "repaired_predictions": sum(
            "evaluation_repair" in row for row in paired
        ),
        "refusals_correct": statuses.count("refusal_correct"),
        "refusals_missed": statuses.count("refusal_missed"),
        "false_refusals": statuses.count("false_refusal"),
        "paired_results": paired,
        "filtered_from_samples": int(payload["samples"]),
    }
    result["repair_rate"] = (
        result["repaired_predictions"] / len(paired) if paired else None
    )
    result["post_repair_rejection_rate"] = (
        (result["parse_failures"] + result["schema_failures"]) / len(paired)
        if paired
        else None
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    manifest_rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = filter_evaluation(payload, manifest_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "paired_results"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
