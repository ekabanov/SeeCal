"""Measure the factored pipeline's error floor with perfect visible labels.

The oracle uses visible-truth names and shares, the real resolver, true or
SCALE mass, and the production assembly arithmetic. It never runs a VLM.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any

from factored_pipeline.eval_taxonomy import EvaluationTaxonomy
from factored_pipeline.resolver import SQLiteNutritionResolver
from factored_pipeline.scoring import score_dish
from score_harness import _scale_keys, _scale_rows, assemble_audit


RUNG12 = {"exact_alias", "fuzzy"}


def regret_decomposition(
    *,
    total_kcal_mae: float,
    true_mass_floor_kcal_mae: float,
) -> dict[str, Any]:
    if total_kcal_mae < 0 or true_mass_floor_kcal_mae < 0:
        raise ValueError("calorie MAE values must be non-negative")
    mass_attributable = total_kcal_mae - true_mass_floor_kcal_mae
    floor_fraction = (
        true_mass_floor_kcal_mae / total_kcal_mae
        if total_kcal_mae > 0
        else (0.0 if true_mass_floor_kcal_mae == 0 else math.inf)
    )
    return {
        "total_c1_kcal_mae": total_kcal_mae,
        "true_mass_floor_kcal_mae": true_mass_floor_kcal_mae,
        "mass_attributable_excess_kcal_mae": mass_attributable,
        "non_mass_fraction_of_total": floor_fraction,
        "binding_constraint_by_half_total_rule": (
            "identify_resolve"
            if floor_fraction > 0.5
            else "scale"
        ),
    }


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


def _full_truth_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    output = {}
    for row in _read_jsonl(path):
        image = Path(row["images"][0])
        record_id = image.parent.name
        truth = _completion(row)
        truth["total_mass_g"] = sum(
            float(item["estimated_grams"]) for item in truth["items"]
        )
        output[record_id] = truth
    return output


def _oracle_items(
    truth: dict[str, Any],
    *,
    bucketed: bool,
) -> list[dict[str, Any]]:
    if bucketed:
        return [
            {"name": str(item["name"]), "share_pct": float(item["share_pct"])}
            for item in truth["items"]
        ]
    total_visible_mass = sum(
        float(item["estimated_grams"]) for item in truth["items"]
    )
    if total_visible_mass <= 0:
        raise ValueError("visible truth has no positive mass")
    return [
        {
            "name": str(item["name"]),
            "share_pct": 100 * float(item["estimated_grams"]) / total_visible_mass,
        }
        for item in truth["items"]
    ]


def _group_values(
    rows: list[dict[str, Any]],
    value,
) -> list[float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(float(value(row)))
    return [statistics.fmean(values) for values in groups.values()]


def _variant_summary(
    assembly: dict[str, Any],
    truth_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    complete = [row for row in assembly["paired"] if row["complete"]]
    signed = _group_values(
        complete,
        lambda row: (
            float(row["prediction"]["total_calories"])
            - float(truth_by_id[row["id"]]["total_calories"])
        )
        / float(truth_by_id[row["id"]]["total_calories"]),
    )
    absolute_kcal = _group_values(
        complete,
        lambda row: abs(
            float(row["prediction"]["total_calories"])
            - float(truth_by_id[row["id"]]["total_calories"])
        ),
    )
    return {
        "predictions": assembly["predictions"],
        "groups": assembly["groups"],
        "complete_predictions": assembly["complete_predictions"],
        "complete_groups": assembly["complete_groups"],
        "complete_group_rate": (
            assembly["complete_groups"] / assembly["groups"]
            if assembly["groups"]
            else None
        ),
        "equal_group_kcal_mae": statistics.fmean(absolute_kcal)
        if absolute_kcal
        else None,
        "median_signed_kcal_error_fraction": statistics.median(signed)
        if signed
        else None,
        "mean_signed_kcal_error_fraction": statistics.fmean(signed)
        if signed
        else None,
        "signed_kcal_error_fraction_p10": _quantile(signed, 0.1),
        "signed_kcal_error_fraction_p90": _quantile(signed, 0.9),
    }


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[rank]


def _resolution_summary(
    assembly: dict[str, Any],
    truth_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item_count = rung12_items = 0
    total_mass = rung12_mass = 0.0
    rung_counts: dict[str, int] = {}
    for row in assembly["paired"]:
        truth_items = truth_by_id[row["id"]]["items"]
        for item, rung in zip(truth_items, row["resolution_rungs"]):
            mass = float(item["estimated_grams"])
            item_count += 1
            total_mass += mass
            rung_counts[rung] = rung_counts.get(rung, 0) + 1
            if rung in RUNG12:
                rung12_items += 1
                rung12_mass += mass
    return {
        "items": item_count,
        "rungs": dict(sorted(rung_counts.items())),
        "rung_1_2_item_rate": rung12_items / item_count if item_count else None,
        "rung_1_2_visible_mass_rate": (
            rung12_mass / total_mass if total_mass else None
        ),
    }


def floor_attribution(
    *,
    exact_assembly: dict[str, Any],
    bucketed_assembly: dict[str, Any],
    truth_by_id: dict[str, dict[str, Any]],
    resolver: SQLiteNutritionResolver | None = None,
) -> dict[str, Any]:
    """Attribute the true-mass floor with an explicit additive waterfall.

    The ordering is: visible-label/truth-density residual, rung-1/2 density
    mismatch, rung-3+ density mismatch, then share bucketing.  Absolute-error
    interactions make any waterfall order-dependent, so the order is recorded
    rather than presenting the components as causal Shapley values.
    """
    bucketed_by_id = {
        str(row["id"]): row
        for row in bucketed_assembly["paired"]
        if row["complete"]
    }
    rows = []
    mismatch_by_name: dict[str, dict[str, Any]] = {}
    for row in exact_assembly["paired"]:
        if not row["complete"] or str(row["id"]) not in bucketed_by_id:
            continue
        truth = truth_by_id[str(row["id"])]
        truth_items = truth["items"]
        prediction_items = row["prediction"]["items"]
        rungs = row["resolution_rungs"]
        if not (
            len(truth_items) == len(prediction_items) == len(rungs)
        ):
            raise ValueError(f"oracle item alignment failed for {row['id']}")
        truth_density_prediction = 0.0
        rung12_density_delta = 0.0
        rung3_density_delta = 0.0
        rung3_predicted_kcal = 0.0
        usable = True
        for truth_item, prediction_item, rung in zip(
            truth_items,
            prediction_items,
            rungs,
        ):
            truth_grams = float(truth_item.get("estimated_grams", 0))
            if truth_grams <= 0 or "calories" not in truth_item:
                usable = False
                break
            assembled_grams = float(prediction_item["estimated_grams"])
            truth_density_kcal_per_g = (
                float(truth_item["calories"]) / truth_grams
            )
            truth_density_kcal = (
                assembled_grams * truth_density_kcal_per_g
            )
            predicted_kcal = float(prediction_item["calories"])
            density_delta = predicted_kcal - truth_density_kcal
            truth_density_prediction += truth_density_kcal
            if rung in RUNG12:
                rung12_density_delta += density_delta
                name = str(truth_item["name"])
                aggregate = mismatch_by_name.setdefault(
                    name,
                    {
                        "name": name,
                        "rung": rung,
                        "occurrences": 0,
                        "absolute_kcal_contribution": 0.0,
                        "signed_kcal_contribution": 0.0,
                        "truth_density_kcal_per_100g_sum": 0.0,
                    },
                )
                aggregate["occurrences"] += 1
                aggregate["absolute_kcal_contribution"] += abs(density_delta)
                aggregate["signed_kcal_contribution"] += density_delta
                aggregate["truth_density_kcal_per_100g_sum"] += (
                    100 * truth_density_kcal_per_g
                )
            else:
                rung3_density_delta += density_delta
                rung3_predicted_kcal += predicted_kcal
        if not usable:
            continue
        rows.append(
            {
                "id": str(row["id"]),
                "group_id": str(row["group_id"]),
                "truth_kcal": float(truth["total_calories"]),
                "exact_resolver_kcal": float(
                    row["prediction"]["total_calories"]
                ),
                "bucketed_resolver_kcal": float(
                    bucketed_by_id[str(row["id"])]["prediction"][
                        "total_calories"
                    ]
                ),
                "truth_density_kcal": truth_density_prediction,
                "rung12_density_delta_kcal": rung12_density_delta,
                "rung3_density_delta_kcal": rung3_density_delta,
                "rung3_predicted_kcal": rung3_predicted_kcal,
            }
        )
    if not rows:
        raise ValueError("no complete rows with item-level calorie truth")

    def equal_group_mae(prediction) -> float:
        return statistics.fmean(
            _group_values(
                rows,
                lambda row: abs(
                    float(prediction(row)) - float(row["truth_kcal"])
                ),
            )
        )

    truth_density_mae = equal_group_mae(
        lambda row: row["truth_density_kcal"]
    )
    rung3_corrected_mae = equal_group_mae(
        lambda row: (
            row["exact_resolver_kcal"]
            - row["rung3_density_delta_kcal"]
        )
    )
    exact_mae = equal_group_mae(lambda row: row["exact_resolver_kcal"])
    bucketed_mae = equal_group_mae(
        lambda row: row["bucketed_resolver_kcal"]
    )
    waterfall = {
        "visible_label_and_exclusion_residual_kcal_mae": truth_density_mae,
        "rung_1_2_density_mismatch_kcal_mae": (
            rung3_corrected_mae - truth_density_mae
        ),
        "rung_3_plus_density_mismatch_kcal_mae": (
            exact_mae - rung3_corrected_mae
        ),
        "share_bucketing_kcal_mae": bucketed_mae - exact_mae,
    }
    for aggregate in mismatch_by_name.values():
        occurrences = int(aggregate["occurrences"])
        aggregate["mean_truth_density_kcal_per_100g"] = (
            aggregate.pop("truth_density_kcal_per_100g_sum") / occurrences
        )
        aggregate["absolute_kcal_contribution_equal_group_mean"] = (
            float(aggregate["absolute_kcal_contribution"])
            / len({str(row["group_id"]) for row in rows})
        )
        if resolver is not None:
            resolution = resolver.resolve(str(aggregate["name"]))
            profile = resolution.profile
            aggregate["resolved_profile"] = (
                {
                    "fdc_id": profile.fdc_id,
                    "name": profile.name,
                    "category": profile.category,
                    "kcal_per_100g": profile.kcal_per_100g,
                }
                if profile is not None
                else None
            )
    top_mismatches = sorted(
        mismatch_by_name.values(),
        key=lambda row: -float(row["absolute_kcal_contribution"]),
    )
    return {
        "scope": "complete_groups_with_item_calorie_truth",
        "groups": len({str(row["group_id"]) for row in rows}),
        "predictions": len(rows),
        "waterfall_order": [
            "visible_label_and_exclusion_residual",
            "rung_1_2_density_mismatch",
            "rung_3_plus_density_mismatch",
            "share_bucketing",
        ],
        "waterfall_order_note": (
            "Absolute-error interactions make the attribution order-dependent; "
            "the listed components telescope exactly to the measured floor."
        ),
        "waterfall_kcal_mae": waterfall,
        "waterfall_sum_kcal_mae": sum(waterfall.values()),
        "measured_true_mass_bucketed_floor_kcal_mae": bucketed_mae,
        "rung_3_plus": {
            "affected_predictions": sum(
                row["rung3_predicted_kcal"] > 0 for row in rows
            ),
            "predicted_kcal_contribution_equal_group_mean": statistics.fmean(
                row["rung3_predicted_kcal"] for row in rows
            ),
        },
        "top_rung_1_2_density_mismatches": top_mismatches[:20],
    }


def _baseline_comparison(
    baseline_eval: Path,
    assemblies: dict[str, dict[str, Any]],
    *,
    database: Path,
    taxonomy: Path,
) -> dict[str, Any]:
    payload = json.loads(baseline_eval.read_text(encoding="utf-8"))
    paired = payload.get("paired_results") or payload.get("paired") or []
    frozen_taxonomy = EvaluationTaxonomy(taxonomy)
    resolver = SQLiteNutritionResolver(database)
    baseline_by_id = {}
    try:
        for row in paired:
            if row.get("status") != "ok":
                continue
            scored = score_dish(
                row["ground_truth"],
                row["prediction"],
                resolver,
                taxonomy=frozen_taxonomy,
            )
            baseline_by_id[str(row["id"])] = {
                **scored,
                "calorie_absolute_error": abs(
                    float(row["prediction"]["total_calories"])
                    - float(row["ground_truth"]["total_calories"])
                ),
            }
    finally:
        resolver.close()
    baseline_eligible = {
        record_id
        for record_id, row in baseline_by_id.items()
        if row["tier1_clean"]
    }
    variants = {}
    for name, assembly in assemblies.items():
        oracle_by_id = {
            str(row["id"]): row
            for row in assembly["paired"]
            if row["complete"] and row["tier1_clean"]
        }
        shared = sorted(baseline_eligible & set(oracle_by_id))
        variants[name] = {
            "shared_eligible_dishes": len(shared),
            "shared_eligible_ids": shared,
            "baseline_kcal_mae": (
                statistics.fmean(
                    baseline_by_id[record_id]["calorie_absolute_error"]
                    for record_id in shared
                )
                if shared
                else None
            ),
            "oracle_kcal_mae": (
                statistics.fmean(
                    oracle_by_id[record_id]["calorie_absolute_error"]
                    for record_id in shared
                )
                if shared
                else None
            ),
        }
    return {
        "baseline_eval": str(baseline_eval),
        "baseline_tier1_clean_dishes": len(baseline_eligible),
        "variants": variants,
        "regret_decomposition": regret_decomposition(
            total_kcal_mae=float(
                variants["p50_mass_bucketed_shares"]["oracle_kcal_mae"]
            ),
            true_mass_floor_kcal_mae=float(
                variants["true_mass_bucketed_shares"]["oracle_kcal_mae"]
            ),
        ),
    }


def run_oracle(
    *,
    identify_manifest: Path,
    scale_predictions: Path,
    database: Path,
    taxonomy: Path,
    full_truth_manifest: Path | None = None,
    baseline_eval: Path | None = None,
) -> dict[str, Any]:
    manifest_rows = _read_jsonl(identify_manifest)
    full_truth = _full_truth_by_id(full_truth_manifest)
    scale_by_key: dict[str, dict[str, Any]] = {}
    for scale in _scale_rows(scale_predictions):
        for key in _scale_keys(str(scale["id"])):
            scale_by_key[key] = scale

    truth_by_id: dict[str, dict[str, Any]] = {}
    shared: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for row in manifest_rows:
        record_id = str(row["id"])
        scale = scale_by_key.get(record_id)
        if scale is None:
            continue
        truth = deepcopy(row["evaluation_ground_truth"])
        if "total_mass_g" not in truth:
            if record_id not in full_truth:
                raise ValueError(f"missing full measured truth for {record_id}")
            truth["total_mass_g"] = full_truth[record_id]["total_mass_g"]
        truth_by_id[record_id] = truth
        shared.append((row, truth, scale))
    if not shared:
        raise ValueError("no shared oracle truth and SCALE predictions")

    identify_payloads = {}
    for share_mode, bucketed in (("exact", False), ("bucketed", True)):
        identify_payloads[share_mode] = {
            "schema_version": 1,
            "paired_results": [
                {
                    "id": str(row["id"]),
                    "group_id": str(row.get("group_id") or row["id"]),
                    "status": "ok",
                    "ground_truth": truth,
                    "prediction": {
                        "not_food": False,
                        "container": str(truth.get("container", "other")),
                        "items": _oracle_items(truth, bucketed=bucketed),
                    },
                }
                for row, truth, _ in shared
            ],
        }

    scale_payloads: dict[str, dict[str, Any]] = {}
    for mass_mode in ("true", "p10", "p50", "p90"):
        paired = []
        for row, truth, scale in shared:
            mass = (
                float(truth["total_mass_g"])
                if mass_mode == "true"
                else float(scale[f"{mass_mode}_g"])
            )
            paired.append(
                {
                    "id": str(row["id"]),
                    "group_id": str(row.get("group_id") or row["id"]),
                    "source": scale.get("source", "oracle"),
                    "target_mass_g": float(truth["total_mass_g"]),
                    "p10_g": mass,
                    "p50_g": mass,
                    "p90_g": mass,
                }
            )
        scale_payloads[mass_mode] = {"schema_version": 1, "paired": paired}
    for retained_error_fraction in (0.25, 0.5, 0.75):
        paired = []
        for row, truth, scale in shared:
            true_mass = float(truth["total_mass_g"])
            current_mass = float(scale["p50_g"])
            mass = true_mass + retained_error_fraction * (
                current_mass - true_mass
            )
            paired.append(
                {
                    "id": str(row["id"]),
                    "group_id": str(row.get("group_id") or row["id"]),
                    "source": scale.get("source", "oracle"),
                    "target_mass_g": true_mass,
                    "p10_g": mass,
                    "p50_g": mass,
                    "p90_g": mass,
                }
            )
        key = f"p50_error_{round(100 * retained_error_fraction)}pct"
        scale_payloads[key] = {"schema_version": 1, "paired": paired}

    assemblies = {}
    with tempfile.TemporaryDirectory(prefix="seecal-oracle-") as temporary:
        root = Path(temporary)
        identify_paths = {}
        scale_paths = {}
        for name, payload in identify_payloads.items():
            path = root / f"identify-{name}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            identify_paths[name] = path
        for name, payload in scale_payloads.items():
            path = root / f"scale-{name}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            scale_paths[name] = path
        for mass_mode, share_mode in (
            ("true", "exact"),
            ("true", "bucketed"),
            ("p50_error_25pct", "bucketed"),
            ("p50_error_50pct", "bucketed"),
            ("p50_error_75pct", "bucketed"),
            ("p50", "exact"),
            ("p50", "bucketed"),
            ("p10", "bucketed"),
            ("p90", "bucketed"),
        ):
            key = f"{mass_mode}_mass_{share_mode}_shares"
            assemblies[key] = assemble_audit(
                identify_paths[share_mode],
                scale_paths[mass_mode],
                database,
                taxonomy,
            )

    summaries = {
        key: _variant_summary(assembly, truth_by_id)
        for key, assembly in assemblies.items()
    }
    bucketed = assemblies["true_mass_bucketed_shares"]
    hidden_mass_fractions = _group_values(
        bucketed["paired"],
        lambda row: max(
            0.0,
            1
            - sum(
                float(item["estimated_grams"])
                for item in truth_by_id[row["id"]]["items"]
            )
            / float(truth_by_id[row["id"]]["total_mass_g"]),
        ),
    )
    hidden_kcal_fractions = _group_values(
        bucketed["paired"],
        lambda row: max(
            0.0,
            1
            - sum(
                float(item.get("calories", 0))
                for item in truth_by_id[row["id"]]["items"]
            )
            / float(truth_by_id[row["id"]]["total_calories"]),
        ),
    )
    low_by_id = {
        row["id"]: float(row["prediction"]["total_calories"])
        for row in assemblies["p10_mass_bucketed_shares"]["paired"]
        if row["complete"]
    }
    high_by_id = {
        row["id"]: float(row["prediction"]["total_calories"])
        for row in assemblies["p90_mass_bucketed_shares"]["paired"]
        if row["complete"]
    }
    interval_rows = [
        row
        for row in bucketed["paired"]
        if row["id"] in low_by_id and row["id"] in high_by_id
    ]
    interval_coverage = _group_values(
        interval_rows,
        lambda row: low_by_id[row["id"]]
        <= float(truth_by_id[row["id"]]["total_calories"])
        <= high_by_id[row["id"]],
    )
    result = {
        "schema_version": 1,
        "identify_manifest": str(identify_manifest),
        "full_truth_manifest": str(full_truth_manifest)
        if full_truth_manifest
        else None,
        "scale_predictions": str(scale_predictions),
        "shared_predictions": len(shared),
        "shared_groups": len(
            {str(row.get("group_id") or row["id"]) for row, _, _ in shared}
        ),
        "resolution": _resolution_summary(bucketed, truth_by_id),
        "visible_exclusion": {
            "median_excluded_mass_fraction": statistics.median(
                hidden_mass_fractions
            ),
            "mean_excluded_mass_fraction": statistics.fmean(
                hidden_mass_fractions
            ),
            "median_excluded_kcal_fraction": statistics.median(
                hidden_kcal_fractions
            ),
            "mean_excluded_kcal_fraction": statistics.fmean(
                hidden_kcal_fractions
            ),
        },
        "variants": summaries,
        "regret_decomposition": regret_decomposition(
            total_kcal_mae=float(
                summaries["p50_mass_bucketed_shares"]["equal_group_kcal_mae"]
            ),
            true_mass_floor_kcal_mae=float(
                summaries["true_mass_bucketed_shares"]["equal_group_kcal_mae"]
            ),
        ),
        "share_bucket_cost_equal_group_kcal_mae": (
            summaries["true_mass_bucketed_shares"]["equal_group_kcal_mae"]
            - summaries["true_mass_exact_shares"]["equal_group_kcal_mae"]
        ),
        "scale_kcal_interval_coverage_equal_group": statistics.fmean(
            interval_coverage
        )
        if interval_coverage
        else None,
        "paired": {
            key: assembly["paired"] for key, assembly in assemblies.items()
        },
    }
    attribution_resolver = SQLiteNutritionResolver(database)
    try:
        result["true_mass_floor_attribution"] = floor_attribution(
            exact_assembly=assemblies["true_mass_exact_shares"],
            bucketed_assembly=assemblies["true_mass_bucketed_shares"],
            truth_by_id=truth_by_id,
            resolver=attribution_resolver,
        )
    finally:
        attribution_resolver.close()
    if baseline_eval is not None:
        result["baseline_comparison"] = _baseline_comparison(
            baseline_eval,
            assemblies,
            database=database,
            taxonomy=taxonomy,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identify-manifest", type=Path, required=True)
    parser.add_argument("--full-truth-manifest", type=Path)
    parser.add_argument("--scale-predictions", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument(
        "--baseline-eval",
        type=Path,
        help="Optional monolithic baseline for a shared Tier-1-clean comparison.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_oracle(
        identify_manifest=args.identify_manifest,
        full_truth_manifest=args.full_truth_manifest,
        scale_predictions=args.scale_predictions,
        database=args.database,
        taxonomy=args.taxonomy,
        baseline_eval=args.baseline_eval,
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
