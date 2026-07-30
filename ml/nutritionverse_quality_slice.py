"""Derive a NutritionVerse nutrition-quality slice from official Train only.

The primary official-Val benchmark is never altered. This tool freezes a
separate diagnostic slice by detecting gross semantic/nutrient incompatibility
on official-Train labels, then removing entire affected scenes from a supplied
evaluation manifest. It does not inspect model outputs or official-Val
nutrition values when choosing exclusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from factored_pipeline.contract import normalize_name


ANIMAL_FAMILIES = {
    "fish",
    "pork",
    "poultry",
    "processed_meat",
    "red_meat",
    "shellfish",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def derive_excluded_labels(
    training_manifest: Path,
    taxonomy_path: Path,
    *,
    minimum_occurrences: int = 5,
) -> dict[str, dict[str, Any]]:
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))["entries"]
    profiles: dict[str, list[dict[str, float]]] = {}
    for row in _rows(training_manifest):
        for item in row["evaluation_ground_truth"]["items"]:
            grams = float(item["estimated_grams"])
            if grams <= 0:
                continue
            name = normalize_name(str(item["name"]))
            profiles.setdefault(name, []).append(
                {
                    "protein_per_100g": 100 * float(item["protein_g"]) / grams,
                    "fat_per_100g": 100 * float(item["fat_g"]) / grams,
                    "carbs_per_100g": 100 * float(item["carbs_g"]) / grams,
                    "source_kcal_per_100g": (
                        100 * float(item["calories"]) / grams
                    ),
                }
            )
    excluded = {}
    for name, values in profiles.items():
        family = (taxonomy.get(name) or {}).get("family")
        medians = {
            key: statistics.median(row[key] for row in values)
            for key in values[0]
        }
        # A whole/meat-labelled animal food with a majority-carbohydrate,
        # low-protein profile is a gross template mismatch, not preparation
        # variance. Thresholds deliberately do not reject breaded meats.
        incompatible = (
            family in ANIMAL_FAMILIES
            and len(values) >= minimum_occurrences
            and medians["carbs_per_100g"] > 45
            and medians["protein_per_100g"] < 15
        )
        if incompatible:
            excluded[name] = {
                "family": family,
                "occurrences": len(values),
                "median_profile": medians,
                "reason": "animal_label_with_majority_carbohydrate_low_protein_profile",
            }
    return excluded


def build_quality_slice(
    training_manifest: Path,
    evaluation_manifest: Path,
    taxonomy_path: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    excluded = derive_excluded_labels(training_manifest, taxonomy_path)
    rows = _rows(evaluation_manifest)
    excluded_groups = {
        str(row.get("group_id") or row["id"])
        for row in rows
        if any(
            normalize_name(str(item["name"])) in excluded
            for item in row["evaluation_ground_truth"]["items"]
        )
    }
    kept = [
        row
        for row in rows
        if str(row.get("group_id") or row["id"]) not in excluded_groups
    ]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept),
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "selection_source": str(training_manifest),
        "evaluation_source": str(evaluation_manifest),
        "excluded_labels": excluded,
        "excluded_groups": len(excluded_groups),
        "input_groups": len(
            {str(row.get("group_id") or row["id"]) for row in rows}
        ),
        "input_records": len(rows),
        "output_groups": len(
            {str(row.get("group_id") or row["id"]) for row in kept}
        ),
        "output_records": len(kept),
        "output_manifest": str(output_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = build_quality_slice(
        args.training_manifest,
        args.evaluation_manifest,
        args.taxonomy,
        args.output_manifest,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
