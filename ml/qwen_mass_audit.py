"""Score total mass implied by Qwen's per-item gram predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _total_mass(payload: dict[str, Any]) -> float:
    if "total_mass_g" in payload:
        value = float(payload["total_mass_g"])
    else:
        value = sum(float(item["estimated_grams"]) for item in payload["items"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("mass must be finite and positive")
    return value


def audit(paired: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    rejected = 0
    for row in paired:
        if row.get("status") != "ok":
            rejected += 1
            continue
        try:
            truth = _total_mass(row["ground_truth"])
            prediction = _total_mass(row["prediction"])
        except (KeyError, TypeError, ValueError, OverflowError):
            rejected += 1
            continue
        rows.append(
            {
                "id": str(row["id"]),
                "group_id": str(row.get("group_id") or row["id"]),
                "truth_mass_g": truth,
                "qwen_mass_g": prediction,
                "absolute_error_g": abs(prediction - truth),
                "absolute_percentage_error": abs(prediction - truth) / truth,
                "underestimated": prediction < truth,
            }
        )
    if not rows:
        raise ValueError("evaluation has no valid Qwen mass predictions")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    return {
        "schema_version": 1,
        "records": len(rows),
        "groups": len(groups),
        "rejected_records": rejected,
        "record_mass_mae_g": statistics.fmean(
            row["absolute_error_g"] for row in rows
        ),
        "equal_group_mass_mae_g": statistics.fmean(
            statistics.fmean(row["absolute_error_g"] for row in group)
            for group in groups.values()
        ),
        "equal_group_mass_mape": statistics.fmean(
            statistics.fmean(
                row["absolute_percentage_error"] for row in group
            )
            for group in groups.values()
        ),
        "equal_group_underestimation_rate": statistics.fmean(
            statistics.fmean(row["underestimated"] for row in group)
            for group in groups.values()
        ),
        "paired": rows,
    }


def compare_scale(
    qwen_audit: dict[str, Any],
    scale_paired: list[dict[str, Any]],
) -> dict[str, Any]:
    scale_by_id = {str(row["id"]): row for row in scale_paired}
    rows = []
    for qwen in qwen_audit["paired"]:
        scale = scale_by_id.get(str(qwen["id"]))
        if scale is None:
            continue
        scale_error = abs(
            float(scale["p50_g"]) - float(qwen["truth_mass_g"])
        )
        rows.append(
            {
                **qwen,
                "scale_mass_g": float(scale["p50_g"]),
                "scale_absolute_error_g": scale_error,
                "scale_minus_qwen_absolute_error_g": (
                    scale_error - float(qwen["absolute_error_g"])
                ),
            }
        )
    if not rows:
        raise ValueError("no shared Qwen and SCALE mass records")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)
    return {
        "records": len(rows),
        "groups": len(groups),
        "qwen_equal_group_mass_mae_g": statistics.fmean(
            statistics.fmean(float(row["absolute_error_g"]) for row in group)
            for group in groups.values()
        ),
        "scale_equal_group_mass_mae_g": statistics.fmean(
            statistics.fmean(
                float(row["scale_absolute_error_g"]) for row in group
            )
            for group in groups.values()
        ),
        "paired_scale_minus_qwen_mae_g": statistics.fmean(
            statistics.fmean(
                float(row["scale_minus_qwen_absolute_error_g"])
                for row in group
            )
            for group in groups.values()
        ),
        "scale_win_rate": statistics.fmean(
            float(row["scale_absolute_error_g"])
            < float(row["absolute_error_g"])
            for row in rows
        ),
        "paired": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale-evaluation",
        type=Path,
        help="Optional SCALE evaluation JSON for an exact shared-ID comparison.",
    )
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = audit(evaluation.get("paired_results") or evaluation.get("paired") or [])
    if args.scale_evaluation is not None:
        scale = json.loads(args.scale_evaluation.read_text(encoding="utf-8"))
        result["scale_comparison"] = compare_scale(
            result,
            scale.get("paired") or scale.get("paired_results") or [],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: (
                    {nested: value for nested, value in value.items() if nested != "paired"}
                    if key == "scale_comparison"
                    else value
                )
                for key, value in result.items()
                if key != "paired"
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
