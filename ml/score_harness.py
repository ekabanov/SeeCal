"""Score hard mistakes, resolution, and mass with paired dish IDs.

Examples (run from ``ml/``):

  .venv/bin/python score_harness.py audit \
    --eval runs/visual-specialist/conditioned-eval/4b-test325.json \
    --database datasets/fdc/seecal-nutrition.sqlite \
    --output runs/factored/e0-v8.json

  .venv/bin/python score_harness.py mass \
    --eval runs/visual-specialist/conditioned-eval/4b-test325.json \
    --scale-predictions runs/visual-specialist/deployment-predictions/test.jsonl \
    --output runs/factored/e2-mass.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from factored_pipeline.eval_taxonomy import EvaluationTaxonomy
from factored_pipeline.resolver import SQLiteNutritionResolver
from factored_pipeline.scoring import score_dish, summarize_dishes


CONDIMENT_TERMS = {
    "butter",
    "dressing",
    "mayo",
    "mayonnaise",
    "oil",
    "vinaigrette",
}


def _has_condiment(payload: dict[str, Any] | None) -> bool:
    return any(
        CONDIMENT_TERMS & set(str(item.get("name", "")).lower().replace("-", " ").split())
        for item in (payload or {}).get("items", [])
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(
    eval_path: Path,
    database: Path,
    taxonomy_path: Path,
) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    paired = payload.get("paired_results") or payload.get("paired")
    if not paired:
        raise ValueError("eval JSON has no paired per-item results; re-run eval.sh once")
    taxonomy = EvaluationTaxonomy(taxonomy_path)
    resolver = SQLiteNutritionResolver(database)
    try:
        rows = [
            {
                "id": row["id"],
                "group_id": row.get("group_id") or row["id"],
                "hallucinated_condiment": (
                    _has_condiment(row.get("prediction"))
                    and not _has_condiment(row.get("ground_truth"))
                ),
                **score_dish(
                    row["ground_truth"],
                    row["prediction"],
                    resolver,
                    taxonomy=taxonomy,
                ),
            }
            for row in paired
            if row.get("status") == "ok"
        ]
    finally:
        resolver.close()
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    group_summary = {
        "groups": len(groups),
        "predictions": len(rows),
        "hmr_mean": statistics.fmean(
            statistics.fmean(row["hmr"] for row in group)
            for group in groups.values()
        ),
        "idr_mean": statistics.fmean(
            statistics.fmean(row["idr"] for row in group)
            for group in groups.values()
        ),
        "idp_mean": statistics.fmean(
            statistics.fmean(row["idp"] for row in group)
            for group in groups.values()
        ),
        "hallucinated_condiment_rate_equal_dish_weight": statistics.fmean(
            statistics.fmean(row["hallucinated_condiment"] for row in group)
            for group in groups.values()
        ),
        "hmr_per_view_p90": _percentile(
            [row["hmr"] for row in rows],
            0.90,
        ),
    }
    return {
        "schema_version": 1,
        "eval_taxonomy": taxonomy.version,
        "summary": summarize_dishes(rows),
        "group_summary": group_summary,
        "paired": rows,
    }


def resolution_audit(eval_path: Path, database: Path) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    paired = payload.get("paired_results") or payload.get("paired")
    if not paired:
        raise ValueError("eval JSON has no paired results")
    resolver = SQLiteNutritionResolver(database)
    rows = []
    try:
        for dish in paired:
            for item in (dish.get("prediction") or {}).get("items", []):
                result = resolver.resolve(str(item.get("name", "")))
                if "estimated_grams" in item:
                    weight = float(item["estimated_grams"])
                    weight_unit = "grams"
                elif "grams" in item:
                    weight = float(item["grams"])
                    weight_unit = "grams"
                elif "portion_units" in item:
                    weight = float(item["portion_units"])
                    weight_unit = "portion_units"
                else:
                    weight = float(item.get("share_pct", 0))
                    weight_unit = "share_pct"
                rows.append(
                    {
                        "dish_id": dish["id"],
                        "group_id": dish.get("group_id") or dish["id"],
                        "name": item.get("name", ""),
                        "weight": max(0.0, weight),
                        "weight_unit": weight_unit,
                        "rung": result.rung,
                        "score": result.score,
                        "fdc_id": result.profile.fdc_id if result.profile else None,
                        "resolved_name": result.profile.name if result.profile else None,
                    }
                )
    finally:
        resolver.close()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["rung"]] = counts.get(row["rung"], 0) + 1
    rung12 = counts.get("exact_alias", 0) + counts.get("fuzzy", 0)
    total_weight = sum(row["weight"] for row in rows)
    rung12_weight = sum(
        row["weight"]
        for row in rows
        if row["rung"] in {"exact_alias", "fuzzy"}
    )
    return {
        "schema_version": 1,
        "items": len(rows),
        "rung_1_2_rate": rung12 / len(rows) if rows else None,
        "rung_1_2_item_rate": rung12 / len(rows) if rows else None,
        "rung_1_2_weighted_rate": (
            rung12_weight / total_weight if total_weight else None
        ),
        "weight_units": sorted({row["weight_unit"] for row in rows}),
        "rungs": dict(sorted(counts.items())),
        "rows": rows,
    }


def mass_audit(eval_path: Path, scale_predictions: Path) -> dict[str, Any]:
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    paired = evaluation.get("paired_results") or evaluation.get("paired")
    if not paired:
        raise ValueError("eval JSON has no paired results")
    scale_by_dish = {}
    for row in _read_jsonl(scale_predictions):
        parts = row["id"].split(":")
        if len(parts) >= 3 and parts[-1] != "overhead":
            continue
        dish_id = parts[1] if len(parts) >= 2 else row["id"]
        scale_by_dish[dish_id] = row
    rows = []
    for row in paired:
        if row.get("status") != "ok" or row["id"] not in scale_by_dish:
            continue
        truth_mass = sum(
            float(item["estimated_grams"]) for item in row["ground_truth"]["items"]
        )
        qwen_mass = sum(
            float(item["estimated_grams"]) for item in row["prediction"]["items"]
        )
        scale = scale_by_dish[row["id"]]["numeric"]["mass_g"]
        p10, p50, p90 = float(scale["p10"]), float(scale["p50"]), float(scale["p90"])
        rows.append(
            {
                "id": row["id"],
                "truth_mass_g": truth_mass,
                "qwen_mass_g": qwen_mass,
                "scale_p10_g": p10,
                "scale_p50_g": p50,
                "scale_p90_g": p90,
                "qwen_absolute_error_g": abs(qwen_mass - truth_mass),
                "scale_absolute_error_g": abs(p50 - truth_mass),
                "scale_interval_covered": p10 <= truth_mass <= p90,
            }
        )
    if not rows:
        raise ValueError("no shared dish IDs between eval and SCALE predictions")
    return {
        "schema_version": 1,
        "paired_dishes": len(rows),
        "qwen_mass_mae_g": statistics.fmean(row["qwen_absolute_error_g"] for row in rows),
        "scale_mass_mae_g": statistics.fmean(row["scale_absolute_error_g"] for row in rows),
        "paired_scale_minus_qwen_mae_g": statistics.fmean(
            row["scale_absolute_error_g"] - row["qwen_absolute_error_g"] for row in rows
        ),
        "scale_p10_p90_coverage": statistics.fmean(
            row["scale_interval_covered"] for row in rows
        ),
        "scale_mass_mape": statistics.fmean(
            row["scale_absolute_error_g"] / row["truth_mass_g"]
            for row in rows
            if not math.isclose(row["truth_mass_g"], 0)
        ),
        "paired": rows,
    }


def _scale_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    rows = payload.get("paired")
    if not isinstance(rows, list):
        raise ValueError("SCALE JSON must contain a paired array")
    return rows


def _scale_keys(record_id: str) -> set[str]:
    keys = {record_id}
    parts = record_id.split(":")
    if len(parts) >= 3 and parts[0] == "nutrition5k":
        keys.add(parts[1])
    return keys


def assemble_audit(
    identify_path: Path,
    scale_path: Path,
    database: Path,
    taxonomy_path: Path,
) -> dict[str, Any]:
    identification = json.loads(identify_path.read_text(encoding="utf-8"))
    paired = identification.get("paired_results")
    if not paired:
        raise ValueError("IDENTIFY JSON has no paired_results")
    scale_by_id: dict[str, dict[str, Any]] = {}
    for row in _scale_rows(scale_path):
        for key in _scale_keys(str(row["id"])):
            scale_by_id[key] = row
    expected_group_counts: dict[str, int] = {}
    for row in paired:
        if row["id"] in scale_by_id:
            group_id = str(row.get("group_id") or row["id"])
            expected_group_counts[group_id] = expected_group_counts.get(group_id, 0) + 1
    taxonomy = EvaluationTaxonomy(taxonomy_path)
    resolver = SQLiteNutritionResolver(database)
    output_rows = []
    try:
        for row in paired:
            if row.get("status") != "ok" or row["id"] not in scale_by_id:
                continue
            scale = scale_by_id[row["id"]]
            total_mass = float(scale["p50_g"])
            prediction_items = []
            complete = True
            rungs = []
            for item in row["prediction"]["items"]:
                resolution = resolver.resolve(str(item["name"]))
                rungs.append(resolution.rung)
                grams = total_mass * float(item["share_pct"]) / 100
                assembled = {
                    "name": item["name"],
                    "share_pct": item["share_pct"],
                    "estimated_grams": grams,
                }
                if resolution.profile is None:
                    complete = False
                else:
                    profile = resolution.profile
                    assembled.update(
                        {
                            "calories": grams / 100 * profile.kcal_per_100g,
                            "protein_g": grams / 100 * profile.protein_per_100g,
                            "fat_g": grams / 100 * profile.fat_per_100g,
                            "carbs_g": grams / 100 * profile.carbs_per_100g,
                        }
                    )
                prediction_items.append(assembled)
            totals = {
                key: sum(float(item.get(key, 0)) for item in prediction_items)
                for key in ("calories", "protein_g", "fat_g", "carbs_g")
            }
            prediction = {
                "total_calories": totals["calories"],
                "protein_g": totals["protein_g"],
                "fat_g": totals["fat_g"],
                "carbs_g": totals["carbs_g"],
                "items": prediction_items,
            }
            scored = score_dish(
                row["ground_truth"],
                prediction,
                resolver,
                taxonomy=taxonomy,
            )
            output_rows.append(
                {
                    "id": row["id"],
                    "group_id": row.get("group_id") or row["id"],
                    "complete": complete,
                    "resolution_rungs": rungs,
                    "scale": scale,
                    "prediction": prediction,
                    **scored,
                }
            )
    finally:
        resolver.close()
    if not output_rows:
        raise ValueError("no shared successful IDENTIFY/SCALE records")
    complete_rows = [row for row in output_rows if row["complete"]]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in output_rows:
        groups.setdefault(row["group_id"], []).append(row)
    complete_group_ids = sorted(
        group_id
        for group_id, rows in groups.items()
        if len(rows) == expected_group_counts.get(group_id, 0)
        and all(row["complete"] for row in rows)
    )
    eligible_group_ids = sorted(
        group_id
        for group_id, rows in groups.items()
        if len(rows) == expected_group_counts.get(group_id, 0)
        and all(row["complete"] and row["tier1_clean"] for row in rows)
    )
    return {
        "schema_version": 1,
        "eval_taxonomy": taxonomy.version,
        "expected_predictions": sum(expected_group_counts.values()),
        "expected_groups": len(expected_group_counts),
        "predictions": len(output_rows),
        "groups": len(groups),
        "complete_predictions": len(complete_rows),
        "complete_groups": len(complete_group_ids),
        "eligible_groups": len(eligible_group_ids),
        "complete_group_ids": complete_group_ids,
        "eligible_group_ids": eligible_group_ids,
        "group_hmr_mean": statistics.fmean(
            statistics.fmean(item["hmr"] for item in rows)
            for rows in groups.values()
        ),
        "group_idr_mean": statistics.fmean(
            statistics.fmean(item["idr"] for item in rows)
            for rows in groups.values()
        ),
        "complete_group_conditional_kcal_mae": (
            statistics.fmean(
                statistics.fmean(
                    item["calorie_absolute_error"]
                    for item in groups[group_id]
                )
                for group_id in eligible_group_ids
            )
            if eligible_group_ids
            else None
        ),
        "complete_summary": (
            summarize_dishes(complete_rows) if complete_rows else None
        ),
        "paired": output_rows,
    }


def compare_assemblies(
    left_path: Path,
    right_path: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Compare conditional kcal only on the mutually eligible dish set."""

    left_payload = json.loads(left_path.read_text(encoding="utf-8"))
    right_payload = json.loads(right_path.read_text(encoding="utf-8"))
    left_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    right_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for row in left_payload["paired"]:
        left_by_group.setdefault(str(row["group_id"]), {})[str(row["id"])] = row
    for row in right_payload["paired"]:
        right_by_group.setdefault(str(row["group_id"]), {})[str(row["id"])] = row
    left_eligible = set(left_payload.get("eligible_group_ids") or [])
    right_eligible = set(right_payload.get("eligible_group_ids") or [])
    if not left_eligible:
        left_eligible = {
            group_id
            for group_id, rows in left_by_group.items()
            if all(row.get("complete") and row.get("tier1_clean") for row in rows.values())
        }
    if not right_eligible:
        right_eligible = {
            group_id
            for group_id, rows in right_by_group.items()
            if all(row.get("complete") and row.get("tier1_clean") for row in rows.values())
        }
    shared_groups = sorted(left_eligible & right_eligible)
    if not shared_groups:
        raise ValueError("no mutually complete, Tier-1-clean dish groups")
    paired = []
    group_rows = []
    for group_id in shared_groups:
        left_ids = set(left_by_group[group_id])
        right_ids = set(right_by_group[group_id])
        if left_ids != right_ids:
            raise ValueError(f"view-ID mismatch in mutually eligible group {group_id}")
        left_errors = []
        right_errors = []
        for record_id in sorted(left_ids):
            left_error = float(
                left_by_group[group_id][record_id]["calorie_absolute_error"]
            )
            right_error = float(
                right_by_group[group_id][record_id]["calorie_absolute_error"]
            )
            left_errors.append(left_error)
            right_errors.append(right_error)
            paired.append(
                {
                    "id": record_id,
                    "group_id": group_id,
                    "left_absolute_error": left_error,
                    "right_absolute_error": right_error,
                    "right_minus_left": right_error - left_error,
                }
            )
        group_rows.append(
            {
                "group_id": group_id,
                "left_absolute_error": statistics.fmean(left_errors),
                "right_absolute_error": statistics.fmean(right_errors),
            }
        )
    differences = [
        row["right_absolute_error"] - row["left_absolute_error"]
        for row in group_rows
    ]
    rng = random.Random(seed)
    bootstrap = sorted(
        statistics.fmean(
            differences[rng.randrange(len(differences))]
            for _ in differences
        )
        for _ in range(bootstrap_samples)
    )
    lower = bootstrap[math.floor(0.025 * (bootstrap_samples - 1))]
    upper = bootstrap[math.ceil(0.975 * (bootstrap_samples - 1))]
    return {
        "schema_version": 1,
        "eligibility": "intersection_complete_and_tier1_clean",
        "left": {
            "predictions": int(left_payload["predictions"]),
            "groups": int(left_payload["groups"]),
            "complete_predictions": int(left_payload["complete_predictions"]),
            "complete_groups": int(left_payload["complete_groups"]),
        },
        "right": {
            "predictions": int(right_payload["predictions"]),
            "groups": int(right_payload["groups"]),
            "complete_predictions": int(right_payload["complete_predictions"]),
            "complete_groups": int(right_payload["complete_groups"]),
        },
        "shared_eligible_predictions": len(paired),
        "shared_eligible_groups": len(group_rows),
        "left_conditional_kcal_mae": statistics.fmean(
            row["left_absolute_error"] for row in group_rows
        ),
        "right_conditional_kcal_mae": statistics.fmean(
            row["right_absolute_error"] for row in group_rows
        ),
        "paired_right_minus_left_kcal_mae": statistics.fmean(differences),
        "paired_bootstrap_95_ci": [lower, upper],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "paired": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "resolve"):
        command = subparsers.add_parser(name)
        command.add_argument("--eval", type=Path, required=True)
        command.add_argument("--database", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "audit":
            command.add_argument("--taxonomy", type=Path, required=True)
    mass = subparsers.add_parser("mass")
    mass.add_argument("--eval", type=Path, required=True)
    mass.add_argument("--scale-predictions", type=Path, required=True)
    mass.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--identify-eval", type=Path, required=True)
    assemble.add_argument("--scale-predictions", type=Path, required=True)
    assemble.add_argument("--database", type=Path, required=True)
    assemble.add_argument("--taxonomy", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare-assemblies")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.eval, args.database, args.taxonomy)
    elif args.command == "resolve":
        result = resolution_audit(args.eval, args.database)
    elif args.command == "mass":
        result = mass_audit(args.eval, args.scale_predictions)
    elif args.command == "assemble":
        result = assemble_audit(
            args.identify_eval,
            args.scale_predictions,
            args.database,
            args.taxonomy,
        )
    else:
        result = compare_assemblies(
            args.left,
            args.right,
            bootstrap_samples=args.bootstrap_samples,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "paired" and key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
