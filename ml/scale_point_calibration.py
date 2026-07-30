"""Select and apply leakage-safe source-specific SCALE point calibration.

Model families are selected by deterministic group-fold cross-validation on
the SCALE validation manifest only. A candidate must improve equal-group MAE
over identity by a configured minimum before it is selected. The fitted
monotone transform is applied to all three quantile heads, after which
source-specific conformal margins are recomputed on the same calibration
groups. Official test manifests are application-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from visual_specialist.scale import (
    binomial_wilson_interval,
    group_balanced_source_margins,
)


CANDIDATES = ("identity", "scale", "offset", "affine", "log_affine")


def _group_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts: dict[str, int] = {}
    for row in rows:
        group = str(row["group_id"])
        counts[group] = counts.get(group, 0) + 1
    return np.asarray(
        [1.0 / counts[str(row["group_id"])] for row in rows],
        dtype=np.float64,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    index = int(
        np.searchsorted(
            np.cumsum(ordered_weights),
            ordered_weights.sum() / 2,
            side="left",
        )
    )
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def fit_candidate(name: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    if name not in CANDIDATES:
        raise ValueError(f"unknown point-calibration candidate: {name}")
    if not rows:
        raise ValueError("cannot fit point calibration on no rows")
    prediction = np.asarray([row["p50_g"] for row in rows], dtype=np.float64)
    target = np.asarray([row["target_mass_g"] for row in rows], dtype=np.float64)
    weights = _group_weights(rows)
    if name == "identity":
        return {"slope": 1.0, "intercept": 0.0}
    if name == "scale":
        positive = prediction > 0
        slope = _weighted_median(
            target[positive] / prediction[positive],
            weights[positive] * prediction[positive],
        )
        return {"slope": slope, "intercept": 0.0}
    if name == "offset":
        return {
            "slope": 1.0,
            "intercept": _weighted_median(target - prediction, weights),
        }
    if name == "log_affine":
        prediction = np.log1p(prediction)
        target = np.log1p(target)
    design = np.column_stack((prediction, np.ones(len(prediction))))
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    slope, intercept = np.linalg.lstsq(
        weighted_design,
        weighted_target,
        rcond=None,
    )[0]
    return {"slope": float(slope), "intercept": float(intercept)}


def transform_value(name: str, parameters: dict[str, float], value: float) -> float:
    slope = float(parameters["slope"])
    intercept = float(parameters["intercept"])
    if name == "log_affine":
        transformed = math.expm1(slope * math.log1p(max(0.0, value)) + intercept)
    else:
        transformed = slope * value + intercept
    return max(0.0, transformed)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)
    equal_group_coverage = statistics.fmean(
        statistics.fmean(bool(row["covered"]) for row in group)
        for group in groups.values()
    )
    return {
        "records": len(rows),
        "groups": len(groups),
        "mass_mae_g": statistics.fmean(row["absolute_error_g"] for row in rows),
        "mass_mape": statistics.fmean(
            row["absolute_percentage_error"] for row in rows
        ),
        "median_absolute_error_g": statistics.median(
            row["absolute_error_g"] for row in rows
        ),
        "p10_p90_coverage": statistics.fmean(
            bool(row["covered"]) for row in rows
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
        "equal_group_p10_p90_coverage": equal_group_coverage,
        "equal_group_p10_p90_coverage_wilson_95_ci": binomial_wilson_interval(
            equal_group_coverage,
            len(groups),
        ),
    }


def _fold(group_id: str, *, folds: int, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    return int(digest, 16) % folds


def cross_validate(
    name: str,
    rows: list[dict[str, Any]],
    *,
    folds: int,
    seed: str,
) -> dict[str, float]:
    predictions = []
    for row in rows:
        held_out = _fold(str(row["group_id"]), folds=folds, seed=seed)
        training = [
            candidate
            for candidate in rows
            if _fold(
                str(candidate["group_id"]),
                folds=folds,
                seed=seed,
            )
            != held_out
        ]
        parameters = fit_candidate(name, training)
        predictions.append(
            transform_value(name, parameters, float(row["p50_g"]))
        )
    scored = []
    for row, prediction in zip(rows, predictions):
        target = float(row["target_mass_g"])
        scored.append(
            {
                **row,
                "absolute_error_g": abs(prediction - target),
                "absolute_percentage_error": (
                    abs(prediction - target) / target if target else 0.0
                ),
                "covered": False,
            }
        )
    summary = summarize_rows(scored)
    return {
        "equal_group_mass_mae_g": summary["equal_group_mass_mae_g"],
        "equal_group_mass_mape": summary["equal_group_mass_mape"],
    }


def transform_rows(
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    apply_interval_margin: bool,
) -> list[dict[str, Any]]:
    source_models = calibration["sources"]
    default_source = str(calibration["runtime_default_source"])
    margins = calibration["interval_calibration"]["source_margins_g"]
    fallback_margin = float(calibration["interval_calibration"]["fallback_margin_g"])
    output = []
    for source_row in rows:
        row = dict(source_row)
        source = str(row.get("source", "unknown"))
        model = source_models.get(source) or source_models[default_source]
        name = str(model["selected"])
        parameters = model["parameters"]
        p10 = transform_value(name, parameters, float(row["p10_g"]))
        p50 = transform_value(name, parameters, float(row["p50_g"]))
        p90 = transform_value(name, parameters, float(row["p90_g"]))
        margin = (
            float(margins.get(source, fallback_margin))
            if apply_interval_margin
            else 0.0
        )
        target = float(row["target_mass_g"])
        row.update(
            {
                "p10_g": max(0.0, p10 - margin),
                "p50_g": p50,
                "p90_g": p90 + margin,
                "absolute_error_g": abs(p50 - target),
                "absolute_percentage_error": (
                    abs(p50 - target) / target if target else 0.0
                ),
                "covered": max(0.0, p10 - margin) <= target <= p90 + margin,
                "point_calibration_source": (
                    source if source in source_models else default_source
                ),
                "interval_margin_g": margin,
            }
        )
        output.append(row)
    return output


def fit_calibration(
    evaluation_path: Path,
    *,
    folds: int,
    seed: str,
    minimum_relative_improvement: float,
    target_coverage: float,
    runtime_default_source: str,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    rows = payload["paired"]
    sources = {}
    for source in sorted({str(row["source"]) for row in rows}):
        selected_rows = [row for row in rows if str(row["source"]) == source]
        candidates = {
            name: cross_validate(name, selected_rows, folds=folds, seed=seed)
            for name in CANDIDATES
        }
        best = min(
            CANDIDATES,
            key=lambda name: candidates[name]["equal_group_mass_mae_g"],
        )
        identity_mae = candidates["identity"]["equal_group_mass_mae_g"]
        improvement = (
            (
                identity_mae - candidates[best]["equal_group_mass_mae_g"]
            )
            / identity_mae
            if identity_mae > 0
            else 0.0
        )
        selected = (
            best if improvement >= minimum_relative_improvement else "identity"
        )
        sources[source] = {
            "groups": len({str(row["group_id"]) for row in selected_rows}),
            "records": len(selected_rows),
            "candidates": candidates,
            "best_candidate": best,
            "best_relative_improvement": improvement,
            "selected": selected,
            "selection_reason": (
                "cross_validated_improvement_passed_threshold"
                if selected != "identity"
                else "improvement_below_threshold"
            ),
            "parameters": fit_candidate(selected, selected_rows),
        }
    if runtime_default_source not in sources:
        raise ValueError(
            f"runtime default source is absent from calibration: "
            f"{runtime_default_source}"
        )
    calibration = {
        "schema_version": 1,
        "method": "source-specific-group-cv-point-and-conformal",
        "selection_manifest": {
            "path": str(evaluation_path),
            "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        },
        "folds": folds,
        "fold_seed": seed,
        "minimum_relative_improvement": minimum_relative_improvement,
        "target_coverage": target_coverage,
        "runtime_default_source": runtime_default_source,
        "sources": sources,
    }
    transformed = transform_rows(
        rows,
        {
            **calibration,
            "interval_calibration": {
                "source_margins_g": {},
                "fallback_margin_g": 0.0,
            },
        },
        apply_interval_margin=False,
    )
    margins = group_balanced_source_margins(
        transformed,
        target_coverage=target_coverage,
    )
    calibration["interval_calibration"] = {
        "method": "equal-group-weight-finite-group-rank-after-point-transform",
        "source_margins_g": margins,
        "fallback_margin_g": max(margins.values()),
    }
    calibration["selection_metrics"] = summarize_rows(
        transform_rows(
            rows,
            calibration,
            apply_interval_margin=True,
        )
    )
    return calibration


def apply_calibration(
    evaluation_path: Path,
    calibration_path: Path,
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    rows = transform_rows(
        evaluation["paired"],
        calibration,
        apply_interval_margin=True,
    )
    return {
        "schema_version": 1,
        "source_evaluation": str(evaluation_path),
        "point_calibration": str(calibration_path),
        "metrics": summarize_rows(rows),
        "paired": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--evaluation", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--folds", type=int, default=5)
    fit.add_argument("--seed", default="20260729-scale-point-v1")
    fit.add_argument("--minimum-relative-improvement", type=float, default=0.02)
    fit.add_argument("--target-coverage", type=float, default=0.8)
    fit.add_argument(
        "--runtime-default-source",
        default="nutritionverse-real-v2",
    )
    apply = subparsers.add_parser("apply")
    apply.add_argument("--evaluation", type=Path, required=True)
    apply.add_argument("--calibration", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        result = fit_calibration(
            args.evaluation,
            folds=args.folds,
            seed=args.seed,
            minimum_relative_improvement=args.minimum_relative_improvement,
            target_coverage=args.target_coverage,
            runtime_default_source=args.runtime_default_source,
        )
    else:
        result = apply_calibration(args.evaluation, args.calibration)
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
