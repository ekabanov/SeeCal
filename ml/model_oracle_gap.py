"""Measure the kcal gap attributable to a real IDENTIFY model.

The model and oracle assemblies must use the same SCALE predictions and
resolver.  Incomplete or missing model rows are reported as coverage failures
and never silently disappear from the denominator; kcal MAE is conditional on
the mutually complete set because an incomplete assembly has no numeric kcal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def _group_mean(rows: list[dict[str, Any]], value) -> float:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(float(value(row)))
    return statistics.fmean(
        statistics.fmean(values) for values in groups.values()
    )


def compare_model_to_oracle(
    *,
    model_assembly: dict[str, Any],
    oracle_audit: dict[str, Any],
    oracle_variant: str = "p50_mass_bucketed_shares",
    scope: str = "all_complete",
) -> dict[str, Any]:
    oracle_rows = oracle_audit["paired"][oracle_variant]
    if scope == "baseline_shared_clean":
        allowed = set(
            oracle_audit["baseline_comparison"]["variants"][oracle_variant][
                "shared_eligible_ids"
            ]
        )
    elif scope == "all_complete":
        allowed = {
            str(row["id"]) for row in oracle_rows if row["complete"]
        }
    else:
        raise ValueError(f"unsupported scope: {scope}")
    oracle_by_id = {
        str(row["id"]): row
        for row in oracle_rows
        if row["complete"] and str(row["id"]) in allowed
    }
    model_by_id = {
        str(row["id"]): row
        for row in model_assembly["paired"]
        if str(row["id"]) in allowed
    }
    shared_complete_ids = sorted(
        record_id
        for record_id in oracle_by_id
        if record_id in model_by_id and model_by_id[record_id]["complete"]
    )
    shared_rows = [
        {
            "id": record_id,
            "group_id": str(oracle_by_id[record_id]["group_id"]),
            "model_absolute_error": float(
                model_by_id[record_id]["calorie_absolute_error"]
            ),
            "oracle_absolute_error": float(
                oracle_by_id[record_id]["calorie_absolute_error"]
            ),
        }
        for record_id in shared_complete_ids
    ]
    model_mae = (
        _group_mean(shared_rows, lambda row: row["model_absolute_error"])
        if shared_rows
        else None
    )
    oracle_mae = (
        _group_mean(shared_rows, lambda row: row["oracle_absolute_error"])
        if shared_rows
        else None
    )
    return {
        "schema_version": 1,
        "scope": scope,
        "oracle_variant": oracle_variant,
        "oracle_scope_predictions": len(oracle_by_id),
        "model_present_predictions": len(model_by_id),
        "model_complete_predictions": sum(
            row["complete"] for row in model_by_id.values()
        ),
        "shared_complete_predictions": len(shared_rows),
        "model_completion_rate_on_oracle_scope": (
            sum(row["complete"] for row in model_by_id.values())
            / len(oracle_by_id)
            if oracle_by_id
            else None
        ),
        "model_kcal_mae_shared_complete": model_mae,
        "oracle_kcal_mae_shared_complete": oracle_mae,
        "model_minus_oracle_kcal_mae": (
            model_mae - oracle_mae
            if model_mae is not None and oracle_mae is not None
            else None
        ),
        "paired": shared_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-assembly", type=Path, required=True)
    parser.add_argument("--oracle-audit", type=Path, required=True)
    parser.add_argument(
        "--oracle-variant",
        default="p50_mass_bucketed_shares",
    )
    parser.add_argument(
        "--scope",
        choices=("all_complete", "baseline_shared_clean"),
        default="all_complete",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_model_to_oracle(
        model_assembly=json.loads(
            args.model_assembly.read_text(encoding="utf-8")
        ),
        oracle_audit=json.loads(
            args.oracle_audit.read_text(encoding="utf-8")
        ),
        oracle_variant=args.oracle_variant,
        scope=args.scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "paired"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
