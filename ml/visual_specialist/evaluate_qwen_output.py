"""Evaluate ingredient naming and item/total consistency in Qwen eval JSON."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


TOTAL_FIELDS = ("total_calories", "protein_g", "fat_g", "carbs_g")
ITEM_FIELDS = ("calories", "protein_g", "fat_g", "carbs_g")


def normalize_name(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def evaluate_output(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paired = payload["paired_results"]
    item_counts: list[float] = []
    target_item_counts: list[float] = []
    consistency: dict[str, list[float]] = {
        field: [] for field in TOTAL_FIELDS
    }
    item_sum_truth_error: dict[str, list[float]] = {
        field: [] for field in TOTAL_FIELDS
    }
    true_positive = 0
    false_positive = 0
    false_negative = 0
    valid_item_schemas = 0
    for row in paired:
        if row.get("status") != "ok":
            continue
        prediction = row["prediction"]
        target = row["ground_truth"]
        items = prediction.get("items")
        target_items = target.get("items")
        if not isinstance(items, list) or not isinstance(target_items, list):
            continue
        if not all(isinstance(item, dict) for item in items):
            continue
        valid_item_schemas += 1
        item_counts.append(float(len(items)))
        target_item_counts.append(float(len(target_items)))
        predicted_names = {
            normalize_name(item.get("name", ""))
            for item in items
            if normalize_name(item.get("name", ""))
        }
        target_names = {
            normalize_name(item.get("name", ""))
            for item in target_items
            if normalize_name(item.get("name", ""))
        }
        true_positive += len(predicted_names & target_names)
        false_positive += len(predicted_names - target_names)
        false_negative += len(target_names - predicted_names)
        for total_field, item_field in zip(TOTAL_FIELDS, ITEM_FIELDS):
            try:
                item_sum = sum(float(item[item_field]) for item in items)
                declared = float(prediction[total_field])
                truth = float(target[total_field])
            except (KeyError, TypeError, ValueError):
                continue
            consistency[total_field].append(abs(declared - item_sum))
            item_sum_truth_error[total_field].append(abs(truth - item_sum))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "schema_version": 1,
        "source": str(path),
        "records": len(paired),
        "valid_predictions": sum(row.get("status") == "ok" for row in paired),
        "valid_item_schemas": valid_item_schemas,
        "predicted_item_count": {
            "mean": statistics.mean(item_counts) if item_counts else None,
            "median": statistics.median(item_counts) if item_counts else None,
            "p95": _percentile(item_counts, 0.95),
        },
        "target_item_count": {
            "mean": statistics.mean(target_item_counts)
            if target_item_counts else None,
            "median": statistics.median(target_item_counts)
            if target_item_counts else None,
            "p95": _percentile(target_item_counts, 0.95),
        },
        "normalized_exact_name_micro": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "declared_vs_item_sum_mae": {
            field: statistics.mean(errors) if errors else None
            for field, errors in consistency.items()
        },
        "item_sum_vs_ground_truth_mae": {
            field: statistics.mean(errors) if errors else None
            for field, errors in item_sum_truth_error.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_output(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
