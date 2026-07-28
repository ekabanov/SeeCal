"""Failure-aware paired calorie comparison across specialist and Gemini runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def load_predictions(path: Path) -> dict[str, tuple[float, float]]:
    if path.suffix == ".log":
        return load_infer_log(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "paired_results" in payload:
        result: dict[str, tuple[float, float]] = {}
        for row in payload["paired_results"]:
            if row.get("status") != "ok":
                continue
            prediction = row["prediction"]
            target = row["ground_truth"]
            result[row["id"]] = (
                float(target["total_calories"]),
                float(prediction["total_calories"]),
            )
        return result

    paired = payload["paired"]
    result: dict[str, tuple[float, float]] = {}
    for row in paired:
        if "prediction" not in row:
            continue
        dish_id = row.get("dish_id") or row["id"].split(":")[1]
        prediction = row["prediction"]
        target = row["truth"] if "truth" in row else row["target"]
        if "total_calories" in prediction:
            predicted_calories = float(prediction["total_calories"])
            actual_calories = float(target["total_calories"])
        else:
            predicted_calories = float(prediction["calories"]["p50"])
            actual_calories = float(target["calories"])
        result[dish_id] = (actual_calories, predicted_calories)
    return result


def load_infer_log(path: Path) -> dict[str, tuple[float, float]]:
    """Load the first evaluation block from an infer.py log."""
    pattern = re.compile(
        r"\[\d+/\d+\]\s+(?P<id>dish_[^/]+)/\S+"
        r".*?cal err:\s*(?P<error>\d+(?:\.\d+)?)\s+kcal"
        r".*?pred:\s*(?P<pred>-?\d+(?:\.\d+)?)"
        r"\s+gt:\s*(?P<gt>-?\d+(?:\.\d+)?)"
    )
    result: dict[str, tuple[float, float]] = {}
    in_evaluation = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Evaluating on "):
            if in_evaluation:
                break
            in_evaluation = True
            continue
        if in_evaluation and line.startswith("Results on "):
            break
        if not in_evaluation:
            continue
        match = pattern.search(line)
        if match:
            # Historical logs round pred/gt for display but retain absolute
            # error to one decimal. Encode a synthetic prediction on either
            # side of the rounded target so the absolute error is preserved.
            target = float(match.group("gt"))
            result[match.group("id")] = (
                target,
                target + float(match.group("error")),
            )
    if not result:
        raise ValueError(f"no inference predictions found in {path}")
    return result


def compare(
    baseline: dict[str, tuple[float, float]],
    candidate: dict[str, tuple[float, float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError("prediction sets have no shared dish IDs")
    baseline_actual = np.array([baseline[key][0] for key in shared])
    candidate_actual = np.array([candidate[key][0] for key in shared])
    baseline_prediction = np.array([baseline[key][1] for key in shared])
    candidate_prediction = np.array([candidate[key][1] for key in shared])
    if not np.allclose(
        baseline_actual, candidate_actual, atol=0.51
    ):
        raise ValueError("ground-truth values disagree across prediction sets")
    baseline_error = np.abs(baseline_prediction - baseline_actual)
    candidate_error = np.abs(candidate_prediction - candidate_actual)
    actual = candidate_actual
    delta = candidate_error - baseline_error
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(shared), size=(bootstrap_samples, len(shared))
    )
    bootstrap = delta[indices].mean(axis=1)
    bins = [
        ("0-100", 0, 100),
        ("100-250", 100, 250),
        ("250-500", 250, 500),
        ("500+", 500, np.inf),
    ]
    by_calorie_bin = {}
    for name, low, high in bins:
        mask = (actual >= low) & (actual < high)
        by_calorie_bin[name] = {
            "n": int(mask.sum()),
            "baseline_mae": float(baseline_error[mask].mean()) if mask.any() else None,
            "candidate_mae": float(candidate_error[mask].mean()) if mask.any() else None,
            "mean_delta": float(delta[mask].mean()) if mask.any() else None,
        }
    return {
        "shared_dishes": len(shared),
        "baseline_mae": float(baseline_error.mean()),
        "candidate_mae": float(candidate_error.mean()),
        "mean_delta_candidate_minus_baseline": float(delta.mean()),
        "median_delta_candidate_minus_baseline": float(np.median(delta)),
        "candidate_win_rate": float((candidate_error < baseline_error).mean()),
        "ties": int((candidate_error == baseline_error).sum()),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_mean_delta_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "by_calorie_bin": by_calorie_bin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        **compare(
            load_predictions(args.baseline),
            load_predictions(args.candidate),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
