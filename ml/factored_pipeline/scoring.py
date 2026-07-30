"""Tier-1/Tier-2 metrics for paired monolith or factored predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any

from .contract import normalize_name
from .eval_taxonomy import EvaluationTaxonomy
from .resolver import Resolution, SQLiteNutritionResolver, lexical_score


def _number(item: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in item:
            try:
                value = float(item[key])
                return value if math.isfinite(value) else default
            except (TypeError, ValueError):
                return default
    return default


def _dominant_macro(values: tuple[float, float, float]) -> int:
    return max(range(3), key=values.__getitem__)


@dataclass(frozen=True)
class MatchedName:
    kind: str
    score: float
    prediction_index: int | None


def match_kind(
    truth_name: str,
    prediction_name: str,
    truth_resolution: Resolution,
    prediction_resolution: Resolution,
    *,
    soft_threshold: float = 0.72,
    hard_threshold: float = 0.42,
) -> tuple[str, float]:
    if normalize_name(truth_name) == normalize_name(prediction_name):
        return "exact", 1.0
    truth = truth_resolution.profile
    prediction = prediction_resolution.profile
    if truth and prediction and truth.fdc_id is not None and truth.fdc_id == prediction.fdc_id:
        return "exact", 1.0
    if truth and prediction and truth.category and truth.category == prediction.category:
        return "soft", 1.0
    score = lexical_score(truth_name, prediction_name)
    if score >= soft_threshold:
        return "exact", score
    if score >= hard_threshold:
        return "soft", score
    return "hard", score


def score_dish(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    resolver: SQLiteNutritionResolver,
    *,
    taxonomy: EvaluationTaxonomy | None = None,
) -> dict[str, Any]:
    truth_items = list(ground_truth.get("items") or [])
    prediction_items = list(prediction.get("items") or [])
    truth_mass = sum(
        max(0.0, _number(item, "estimated_grams", "grams", "share_pct"))
        for item in truth_items
    )
    prediction_mass = sum(
        max(0.0, _number(item, "estimated_grams", "grams", "share_pct"))
        for item in prediction_items
    )
    prediction_resolutions = [
        resolver.resolve(str(item.get("name", ""))) for item in prediction_items
    ]
    truth_resolutions = [resolver.resolve(str(item.get("name", ""))) for item in truth_items]

    unused_truth = set(range(len(truth_items)))
    unused_prediction = set(range(len(prediction_items)))
    assignments: dict[int, MatchedName] = {}
    pair_kinds: dict[tuple[int, int], tuple[str, float]] = {}
    for truth_index, truth_item in enumerate(truth_items):
        for prediction_index, prediction_item in enumerate(prediction_items):
            truth_name = str(truth_item.get("name", ""))
            prediction_name = str(prediction_item.get("name", ""))
            if taxonomy is not None:
                pair_kinds[(truth_index, prediction_index)] = taxonomy.match_kind(
                    truth_name,
                    prediction_name,
                )
            else:
                pair_kinds[(truth_index, prediction_index)] = match_kind(
                    truth_name,
                    prediction_name,
                    truth_resolutions[truth_index],
                    prediction_resolutions[prediction_index],
                )

    # Match all exact names before considering softer alternatives. The old
    # per-truth greedy walk could consume a prediction as a soft match for a
    # larger truth item before reaching its exact ground-truth counterpart
    # (for example sausage consuming a bacon prediction before bacon itself).
    for desired_kind in ("exact", "soft"):
        edges = []
        for truth_index in unused_truth:
            truth_mass_value = max(
                0.0,
                _number(
                    truth_items[truth_index],
                    "estimated_grams",
                    "grams",
                    "share_pct",
                ),
            )
            for prediction_index in unused_prediction:
                kind, score = pair_kinds[(truth_index, prediction_index)]
                if kind == desired_kind:
                    edges.append(
                        (
                            -truth_mass_value,
                            -score,
                            truth_index,
                            prediction_index,
                        )
                    )
        for _, negative_score, truth_index, prediction_index in sorted(edges):
            if (
                truth_index not in unused_truth
                or prediction_index not in unused_prediction
            ):
                continue
            assignments[truth_index] = MatchedName(
                desired_kind,
                -negative_score,
                prediction_index,
            )
            unused_truth.remove(truth_index)
            unused_prediction.remove(prediction_index)

    # Remaining pairs are hard mismatches regardless of which names are
    # coupled. Pair the largest masses first so extra predictions—not arbitrary
    # greedy ordering—define hallucinated mass.
    remaining_truth = sorted(
        unused_truth,
        key=lambda index: -_number(
            truth_items[index],
            "estimated_grams",
            "grams",
            "share_pct",
        ),
    )
    remaining_prediction = sorted(
        unused_prediction,
        key=lambda index: -_number(
            prediction_items[index],
            "estimated_grams",
            "grams",
            "share_pct",
        ),
    )
    for truth_index, prediction_index in zip(
        remaining_truth,
        remaining_prediction,
    ):
        _, score = pair_kinds[(truth_index, prediction_index)]
        assignments[truth_index] = MatchedName("hard", score, prediction_index)
        unused_truth.remove(truth_index)
        unused_prediction.remove(prediction_index)

    exact_truth_mass = soft_truth_mass = hard_truth_mass = 0.0
    exact_prediction_mass = 0.0
    exact_share_errors: list[float] = []
    for truth_index, truth_item in enumerate(truth_items):
        grams = max(
            0.0, _number(truth_item, "estimated_grams", "grams", "share_pct")
        )
        match = assignments.get(truth_index)
        if match is None:
            hard_truth_mass += grams
            continue
        prediction_index = match.prediction_index
        assert prediction_index is not None
        if match.kind == "exact":
            exact_truth_mass += grams
            exact_prediction_mass += max(
                0.0,
                _number(
                    prediction_items[prediction_index],
                    "estimated_grams",
                    "grams",
                    "share_pct",
                ),
            )
            if "share_pct" in truth_item and "share_pct" in prediction_items[prediction_index]:
                exact_share_errors.append(
                    abs(
                        float(truth_item["share_pct"])
                        - float(prediction_items[prediction_index]["share_pct"])
                    )
                )
        elif match.kind == "soft":
            soft_truth_mass += grams
        else:
            hard_truth_mass += grams

    hallucinated_mass = sum(
        max(
            0.0,
            _number(
                prediction_items[index],
                "estimated_grams",
                "grams",
                "share_pct",
            ),
        )
        for index in unused_prediction
    )
    hard_denominator = truth_mass + hallucinated_mass
    hmr = (
        (hard_truth_mass + hallucinated_mass) / hard_denominator
        if hard_denominator
        else 0.0
    )

    density_violations = 0
    atwater_violations = 0
    emitted = 0
    for item, resolution in zip(prediction_items, prediction_resolutions):
        grams = _number(item, "estimated_grams", "grams")
        if grams <= 0:
            continue
        emitted += 1
        kcal = _number(item, "calories", "kcal")
        protein = _number(item, "protein_g", "protein")
        fat = _number(item, "fat_g", "fat")
        carbs = _number(item, "carbs_g", "carbs")
        if resolution.profile:
            profile = resolution.profile
            implied_density = kcal / grams * 100
            reference_density = profile.kcal_per_100g
            density_bad = (
                reference_density == 0
                and implied_density > 0.5
            ) or (
                reference_density > 0
                and not 0.5 * reference_density <= implied_density <= 2.0 * reference_density
            )
            predicted_dominant = _dominant_macro((protein, fat, carbs))
            reference_dominant = _dominant_macro(
                (
                    profile.protein_per_100g,
                    profile.fat_per_100g,
                    profile.carbs_per_100g,
                )
            )
            if density_bad or predicted_dominant != reference_dominant:
                density_violations += 1
        atwater = 4 * protein + 9 * fat + 4 * carbs
        if abs(atwater - kcal) > 0.15 * kcal:
            atwater_violations += 1

    return {
        "hmr": hmr,
        "hard_mass_g": hard_truth_mass,
        "hallucinated_mass_g": hallucinated_mass,
        "truth_mass_g": truth_mass,
        "prediction_mass_g": prediction_mass,
        "idr": exact_truth_mass / truth_mass if truth_mass else 0.0,
        "soft_recall": soft_truth_mass / truth_mass if truth_mass else 0.0,
        "idp": exact_prediction_mass / prediction_mass if prediction_mass else 0.0,
        "share_mae": (
            statistics.fmean(exact_share_errors) if exact_share_errors else None
        ),
        "dvr": density_violations / emitted if emitted else 0.0,
        "air": atwater_violations / emitted if emitted else 0.0,
        "resolution_rungs": [resolution.rung for resolution in prediction_resolutions],
        "tier1_clean": hmr == 0 and density_violations == 0 and atwater_violations == 0,
        "calorie_absolute_error": abs(
            _number(prediction, "total_calories", "calories")
            - _number(ground_truth, "total_calories", "calories")
        ),
    }


def summarize_dishes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero dishes")
    clean_errors = [row["calorie_absolute_error"] for row in rows if row["tier1_clean"]]
    share_errors = [row["share_mae"] for row in rows if row["share_mae"] is not None]
    rung_counts: dict[str, int] = {}
    for row in rows:
        for rung in row["resolution_rungs"]:
            rung_counts[rung] = rung_counts.get(rung, 0) + 1
    return {
        "dishes": len(rows),
        "hmr_mean": statistics.fmean(row["hmr"] for row in rows),
        "idr_mean": statistics.fmean(row["idr"] for row in rows),
        "idp_mean": statistics.fmean(row["idp"] for row in rows),
        "share_mae": statistics.fmean(share_errors) if share_errors else None,
        "dvr_mean": statistics.fmean(row["dvr"] for row in rows),
        "air_mean": statistics.fmean(row["air"] for row in rows),
        "tier1_clean_dishes": len(clean_errors),
        "conditional_kcal_mae": statistics.fmean(clean_errors) if clean_errors else None,
        "resolution_rungs": dict(sorted(rung_counts.items())),
    }
