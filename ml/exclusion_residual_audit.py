"""Measure the signed visible-label/exclusion residual on train and validation.

This audit uses only measured Nutrition5K truth.  It gives the true total mass
to the visible ground-truth labels, preserves their exact measured shares and
calorie densities, and compares the resulting assembly with the full meal
calories.  No model, resolver profile, or frozen test row participates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


CONDIMENT_TOKENS = {
    "aioli",
    "butter",
    "dressing",
    "gravy",
    "margarine",
    "mayonnaise",
    "oil",
    "pesto",
    "sauce",
    "vinaigrette",
}
VISIBLE_PRONE_TOKENS = CONDIMENT_TOKENS | {
    "burger",
    "caesar",
    "greens",
    "lettuce",
    "salad",
    "sandwich",
    "slaw",
    "taco",
}
ROUNDING_TOLERANCE_KCAL = 0.5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _completion(row: dict[str, Any]) -> dict[str, Any]:
    content = row["messages"][-1]["content"]
    text = content[0]["text"] if isinstance(content, list) else content
    return json.loads(text)


def _full_truth_by_id(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in _read_jsonl(path):
        record_id = Path(row["images"][0]).parent.name
        truth = _completion(row)
        truth["total_mass_g"] = sum(
            float(item["estimated_grams"]) for item in truth["items"]
        )
        if record_id in output:
            raise ValueError(f"duplicate full-truth record id: {record_id}")
        output[record_id] = truth
    return output


def _token_match(name: str, tokens: set[str]) -> bool:
    words = (
        name.lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace("-", " ")
        .replace("/", " ")
        .split()
    )
    return bool(set(words) & tokens)


def _item_key(item: dict[str, Any]) -> tuple[str, float, float]:
    return (
        str(item["name"]).strip().lower(),
        round(float(item["estimated_grams"]), 3),
        round(float(item.get("calories", 0)), 3),
    )


def _excluded_items(
    full_items: list[dict[str, Any]],
    visible_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    remaining = Counter(_item_key(item) for item in visible_items)
    excluded = []
    for item in full_items:
        key = _item_key(item)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            excluded.append(item)
    if any(remaining.values()):
        raise ValueError("visible items do not form a multiset subset of full truth")
    return excluded


def residual_rows(
    identify_manifest: Path,
    full_truth_manifest: Path,
    *,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return per-dish residuals and manifest accounting for one split."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation; frozen test is forbidden")
    full_by_id = _full_truth_by_id(full_truth_manifest)
    rows = []
    manifest_rows = _read_jsonl(identify_manifest)
    skipped_without_full_truth = 0
    skipped_nonpositive_truth = 0
    for manifest_row in manifest_rows:
        record_id = str(manifest_row["id"])
        visible = manifest_row.get("evaluation_ground_truth")
        full = full_by_id.get(record_id)
        if visible is None or full is None:
            skipped_without_full_truth += 1
            continue
        visible_items = visible.get("items") or []
        full_items = full.get("items") or []
        visible_mass = sum(
            float(item["estimated_grams"]) for item in visible_items
        )
        visible_kcal = sum(float(item["calories"]) for item in visible_items)
        total_mass = float(full["total_mass_g"])
        truth_kcal = float(full["total_calories"])
        if min(visible_mass, total_mass, truth_kcal) <= 0:
            skipped_nonpositive_truth += 1
            continue
        excluded = _excluded_items(full_items, visible_items)
        oracle_kcal = total_mass * visible_kcal / visible_mass
        residual = oracle_kcal - truth_kcal
        visible_names = [str(item["name"]) for item in visible_items]
        excluded_names = [str(item["name"]) for item in excluded]
        rows.append(
            {
                "id": record_id,
                "split": split,
                "visible_item_count": len(visible_items),
                "full_item_count": len(full_items),
                "visible_mass_g": visible_mass,
                "total_mass_g": total_mass,
                "visible_kcal": visible_kcal,
                "truth_kcal": truth_kcal,
                "truth_density_oracle_kcal": oracle_kcal,
                "signed_residual_kcal": residual,
                "absolute_residual_kcal": abs(residual),
                "signed_residual_fraction": residual / truth_kcal,
                "hidden_mass_fraction": max(0.0, 1 - visible_mass / total_mass),
                "hidden_kcal_fraction": max(0.0, 1 - visible_kcal / truth_kcal),
                "visible_sauce_dressing_prone": any(
                    _token_match(name, VISIBLE_PRONE_TOKENS)
                    for name in visible_names
                ),
                "excluded_condiment_or_fat": any(
                    _token_match(name, CONDIMENT_TOKENS)
                    for name in excluded_names
                ),
                "visible_names": visible_names,
                "excluded_names": excluded_names,
            }
        )
    return rows, {
        "manifest_rows": len(manifest_rows),
        "full_truth_rows": len(full_by_id),
        "scored_rows": len(rows),
        "skipped_without_matching_nutrition_truth": skipped_without_full_truth,
        "skipped_nonpositive_truth": skipped_nonpositive_truth,
    }


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * probability) - 1),
    )
    return ordered[rank]


def _summary_from_errors(errors: list[float]) -> dict[str, Any]:
    if not errors:
        raise ValueError("cannot summarize an empty error set")
    absolute = [abs(value) for value in errors]
    mean = statistics.fmean(errors)
    variance = statistics.pvariance(errors)
    mse = statistics.fmean(value * value for value in errors)
    ordered_absolute = sorted(absolute, reverse=True)

    def top_share(fraction: float) -> float:
        count = max(1, math.ceil(len(absolute) * fraction))
        denominator = sum(absolute)
        return (
            sum(ordered_absolute[:count]) / denominator
            if denominator > 0
            else 0.0
        )

    return {
        "n": len(errors),
        "mean_signed_kcal": mean,
        "median_signed_kcal": statistics.median(errors),
        "standard_deviation_kcal": math.sqrt(variance),
        "mae_kcal": statistics.fmean(absolute),
        "median_absolute_error_kcal": statistics.median(absolute),
        "rmse_kcal": math.sqrt(mse),
        "mse_bias_variance_decomposition": {
            "mse_kcal_squared": mse,
            "bias_squared_kcal_squared": mean * mean,
            "variance_kcal_squared": variance,
            "bias_squared_fraction": mean * mean / mse if mse > 0 else 0.0,
            "variance_fraction": variance / mse if mse > 0 else 0.0,
        },
        "direction": {
            "rounding_tolerance_kcal": ROUNDING_TOLERANCE_KCAL,
            "material_underestimate_rate": sum(
                value < -ROUNDING_TOLERANCE_KCAL for value in errors
            )
            / len(errors),
            "material_overestimate_rate": sum(
                value > ROUNDING_TOLERANCE_KCAL for value in errors
            )
            / len(errors),
            "within_rounding_tolerance_rate": sum(
                abs(value) <= ROUNDING_TOLERANCE_KCAL for value in errors
            )
            / len(errors),
            "underestimate_rate_among_material_residuals": (
                sum(value < -ROUNDING_TOLERANCE_KCAL for value in errors)
                / sum(
                    abs(value) > ROUNDING_TOLERANCE_KCAL for value in errors
                )
                if any(
                    abs(value) > ROUNDING_TOLERANCE_KCAL for value in errors
                )
                else 0.0
            ),
        },
        "signed_quantiles_kcal": {
            f"p{round(100 * probability)}": _quantile(errors, probability)
            for probability in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        },
        "absolute_quantiles_kcal": {
            f"p{round(100 * probability)}": _quantile(absolute, probability)
            for probability in (0.5, 0.8, 0.9, 0.95, 0.99)
        },
        "tail_mass": {
            "absolute_error_over_20_kcal_rate": sum(
                value > 20 for value in absolute
            )
            / len(absolute),
            "absolute_error_over_50_kcal_rate": sum(
                value > 50 for value in absolute
            )
            / len(absolute),
            "absolute_error_over_100_kcal_rate": sum(
                value > 100 for value in absolute
            )
            / len(absolute),
            "top_10pct_share_of_total_absolute_error": top_share(0.10),
            "top_5pct_share_of_total_absolute_error": top_share(0.05),
        },
    }


def _slice(
    rows: list[dict[str, Any]],
    predicate,
) -> dict[str, Any] | None:
    selected = [row for row in rows if predicate(row)]
    if not selected:
        return None
    return _summary_from_errors(
        [float(row["signed_residual_kcal"]) for row in selected]
    )


def summarize_split(
    rows: list[dict[str, Any]],
    *,
    additive_correction_kcal: float = 0.0,
) -> dict[str, Any]:
    errors = [
        float(row["signed_residual_kcal"]) + additive_correction_kcal
        for row in rows
    ]
    output = _summary_from_errors(errors)
    output["mean_hidden_mass_fraction"] = statistics.fmean(
        float(row["hidden_mass_fraction"]) for row in rows
    )
    output["mean_hidden_kcal_fraction"] = statistics.fmean(
        float(row["hidden_kcal_fraction"]) for row in rows
    )
    if additive_correction_kcal:
        output["additive_correction_kcal"] = additive_correction_kcal
        return output
    output["feature_slices"] = {
        "visible_item_count": {
            "1": _slice(rows, lambda row: row["visible_item_count"] == 1),
            "2": _slice(rows, lambda row: row["visible_item_count"] == 2),
            "3_to_4": _slice(
                rows, lambda row: 3 <= row["visible_item_count"] <= 4
            ),
            "5_plus": _slice(rows, lambda row: row["visible_item_count"] >= 5),
        },
        "visible_sauce_dressing_prone": {
            "true": _slice(
                rows, lambda row: row["visible_sauce_dressing_prone"]
            ),
            "false": _slice(
                rows, lambda row: not row["visible_sauce_dressing_prone"]
            ),
        },
        "diagnostic_excluded_condiment_or_fat": {
            "true": _slice(
                rows, lambda row: row["excluded_condiment_or_fat"]
            ),
            "false": _slice(
                rows, lambda row: not row["excluded_condiment_or_fat"]
            ),
        },
    }
    output["top_absolute_residuals"] = sorted(
        rows,
        key=lambda row: -float(row["absolute_residual_kcal"]),
    )[:25]
    return output


def _coverage(errors: Iterable[float], half_width: float) -> float:
    values = list(errors)
    return sum(abs(value) <= half_width for value in values) / len(values)


def _item_count_bin(row: dict[str, Any]) -> str:
    count = int(row["visible_item_count"])
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3_to_4"
    return "5_plus"


def run_audit(
    *,
    identify_train: Path,
    full_train: Path,
    identify_validation: Path,
    full_validation: Path,
) -> dict[str, Any]:
    train_rows, train_accounting = residual_rows(
        identify_train, full_train, split="train"
    )
    validation_rows, validation_accounting = residual_rows(
        identify_validation, full_validation, split="validation"
    )
    overlap = sorted(
        {str(row["id"]) for row in train_rows}
        & {str(row["id"]) for row in validation_rows}
    )
    if overlap:
        raise ValueError(f"train/validation overlap: {overlap[:5]}")

    train = summarize_split(train_rows)
    validation = summarize_split(validation_rows)
    correction = -float(train["mean_signed_kcal"])
    corrected_validation = summarize_split(
        validation_rows,
        additive_correction_kcal=correction,
    )
    correction_guard = {
        "absolute_mean_bias_improves": abs(
            float(corrected_validation["mean_signed_kcal"])
        )
        < abs(float(validation["mean_signed_kcal"])),
        "mae_does_not_worsen": float(corrected_validation["mae_kcal"])
        <= float(validation["mae_kcal"]),
        "rmse_does_not_worsen": float(corrected_validation["rmse_kcal"])
        <= float(validation["rmse_kcal"]),
    }
    correction_guard["passes_all"] = all(correction_guard.values())

    train_absolute = [
        float(row["absolute_residual_kcal"]) for row in train_rows
    ]
    validation_errors = [
        float(row["signed_residual_kcal"]) for row in validation_rows
    ]
    intervals = {}
    for probability in (0.8, 0.9, 0.95):
        half_width = float(_quantile(train_absolute, probability))
        intervals[f"train_p{round(100 * probability)}"] = {
            "half_width_kcal": half_width,
            "validation_coverage": _coverage(validation_errors, half_width),
        }
    keyed_intervals = {}
    for key in ("1", "2", "3_to_4", "5_plus"):
        train_bin = [
            row for row in train_rows if _item_count_bin(row) == key
        ]
        validation_bin = [
            row for row in validation_rows if _item_count_bin(row) == key
        ]
        half_width = _quantile(
            [float(row["absolute_residual_kcal"]) for row in train_bin],
            0.9,
        )
        keyed_intervals[key] = {
            "train_rows": len(train_bin),
            "validation_rows": len(validation_bin),
            "train_p90_half_width_kcal": half_width,
            "validation_coverage": (
                _coverage(
                    (
                        float(row["signed_residual_kcal"])
                        for row in validation_bin
                    ),
                    float(half_width),
                )
                if half_width is not None and validation_bin
                else None
            ),
        }
    keyed_covered = sum(
        abs(float(row["signed_residual_kcal"]))
        <= float(
            keyed_intervals[_item_count_bin(row)][
                "train_p90_half_width_kcal"
            ]
        )
        for row in validation_rows
        if keyed_intervals[_item_count_bin(row)][
            "train_p90_half_width_kcal"
        ]
        is not None
    )
    keyed_eligible = sum(
        keyed_intervals[_item_count_bin(row)][
            "train_p90_half_width_kcal"
        ]
        is not None
        for row in validation_rows
    )

    return {
        "schema_version": 1,
        "scope": {
            "datasets": ["Nutrition5K train", "Nutrition5K validation"],
            "frozen_test_rows_used": 0,
            "model_predictions_used": 0,
            "resolver_profiles_used": 0,
            "definition": (
                "true total mass redistributed over exact visible measured "
                "item shares and calorie densities, minus full meal calories"
            ),
            "residual_sign": "assembled prediction minus full truth",
        },
        "accounting": {
            "train": train_accounting,
            "validation": validation_accounting,
            "train_validation_overlap": 0,
        },
        "train": train,
        "validation": validation,
        "train_derived_additive_debiasing": {
            "correction_kcal_added_to_assembly": correction,
            "provenance_requirement": (
                "If adopted, expose as estimated exclusion correction; never "
                "fold it silently into an ingredient profile."
            ),
            "validation_after_correction": corrected_validation,
            "acceptance_guard": correction_guard,
        },
        "floor_derived_uniform_intervals": intervals,
        "floor_derived_item_count_p90_intervals": {
            "bins": keyed_intervals,
            "validation_overall_coverage": keyed_covered
            / keyed_eligible,
            "validation_eligible_rows": keyed_eligible,
            "policy_status": (
                "candidate uncertainty term only; not wired into product"
            ),
        },
        "feature_policy_note": (
            "visible_sauce_dressing_prone is observable at inference; "
            "excluded_condiment_or_fat uses hidden truth and is diagnostic only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identify-train", type=Path, required=True)
    parser.add_argument("--full-train", type=Path, required=True)
    parser.add_argument("--identify-validation", type=Path, required=True)
    parser.add_argument("--full-validation", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    payload = run_audit(
        identify_train=args.identify_train,
        full_train=args.full_train,
        identify_validation=args.identify_validation,
        full_validation=args.full_validation,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "train": {
                    key: payload["train"][key]
                    for key in ("n", "mean_signed_kcal", "mae_kcal", "rmse_kcal")
                },
                "validation": {
                    key: payload["validation"][key]
                    for key in ("n", "mean_signed_kcal", "mae_kcal", "rmse_kcal")
                },
                "correction_guard": payload[
                    "train_derived_additive_debiasing"
                ]["acceptance_guard"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
