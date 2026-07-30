"""Audit USDA typical-portion priors before using them as a SCALE fallback.

This is an oracle-identification diagnostic: names and relative shares come
from ground truth, not a model. The audit compares several deterministic ways
of turning per-item USDA portion weights into total meal mass, then measures
simple fusions with the frozen SCALE Probe B output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

import yaml

from factored_pipeline.resolver import SQLiteNutritionResolver


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def portion_estimates(
    weighted_portions: list[tuple[float, float]],
) -> dict[str, float]:
    """Return predeclared total-mass estimators from (share, portion_g)."""
    values = [
        (float(share), float(portion))
        for share, portion in weighted_portions
        if share > 0 and portion > 0
    ]
    if not values:
        return {}
    dominant_share, dominant_portion = max(values)
    return {
        "median_portion_div_share": statistics.median(
            portion / share for share, portion in values
        ),
        "sum_item_portions": sum(portion for _, portion in values),
        "dominant_portion_div_share": dominant_portion / dominant_share,
        "least_squares_portion_fit": (
            sum(share * portion for share, portion in values)
            / sum(share * share for share, _ in values)
        ),
        "share_weighted_portion": (
            sum(share * portion for share, portion in values)
            / sum(share for share, _ in values)
        ),
    }


def _scale_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload["paired"]}


def _n5k_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    truth_mass = {
        str(row["group_id"]): float(row["total_mass_g"])
        for row in _read_jsonl(args.nutrition5k_scale_manifest)
    }
    rows = []
    for row in _read_jsonl(args.nutrition5k_identify_manifest):
        truth = row["evaluation_ground_truth"]
        if truth["not_food"]:
            continue
        rows.append(
            {
                "id": str(row["id"]),
                "scale_id": f"nutrition5k:{row['group_id']}:overhead",
                "group_id": str(row["group_id"]),
                "container": str(truth["container"]),
                "truth_mass_g": truth_mass[str(row["group_id"])],
                "items": truth["items"],
            }
        )
    return rows


def _nutritionverse_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row["id"]),
            "scale_id": str(row["id"]),
            "group_id": str(row["group_id"]),
            "container": str(row["evaluation_ground_truth"]["container"]),
            "truth_mass_g": float(
                row["evaluation_ground_truth"]["total_mass_g"]
            ),
            "items": row["evaluation_ground_truth"]["items"],
        }
        for row in _read_jsonl(args.nutritionverse_manifest)
    ]


def _fpb_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    names = yaml.safe_load(args.fpb_data_yaml.read_text(encoding="utf-8"))[
        "names"
    ]
    output = []
    for row in _read_jsonl(args.fpb_manifest):
        stem = str(row["id"]).split("fpb:test:", 1)[-1]
        items = []
        for line in (args.fpb_label_dir / f"{stem}.txt").read_text(
            encoding="utf-8"
        ).splitlines():
            fields = line.split()
            if not fields:
                continue
            weight = float(fields[-1])
            if weight <= 0:
                raise ValueError(f"clean FPB manifest contains {weight=}")
            items.append(
                {
                    "name": str(names[int(fields[0])]).replace("_", " "),
                    "estimated_grams": weight,
                }
            )
        output.append(
            {
                "id": str(row["id"]),
                "scale_id": str(row["id"]),
                "group_id": str(row["group_id"]),
                "container": "unknown",
                "truth_mass_g": float(row["total_mass_g"]),
                "items": items,
            }
        )
    return output


def _group_metrics(
    rows: list[dict[str, Any]], prediction_key: str
) -> dict[str, float | int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if prediction_key in row:
            grouped[str(row["group_id"])].append(row)
    group_errors = [
        {
            "absolute": statistics.fmean(
                abs(
                    float(row[prediction_key])
                    - float(row["truth_mass_g"])
                )
                for row in values
            ),
            "percentage": statistics.fmean(
                abs(
                    float(row[prediction_key])
                    - float(row["truth_mass_g"])
                )
                / float(row["truth_mass_g"])
                for row in values
            ),
            "signed": statistics.fmean(
                float(row[prediction_key]) - float(row["truth_mass_g"])
                for row in values
            ),
        }
        for values in grouped.values()
    ]
    if not group_errors:
        return {"groups": 0}
    return {
        "groups": len(group_errors),
        "equal_group_mass_mae_g": statistics.fmean(
            row["absolute"] for row in group_errors
        ),
        "equal_group_mass_mape": statistics.fmean(
            row["percentage"] for row in group_errors
        ),
        "mean_signed_error_g": statistics.fmean(
            row["signed"] for row in group_errors
        ),
        "median_group_mean_absolute_error_g": statistics.median(
            row["absolute"] for row in group_errors
        ),
    }


def _audit_domain(
    rows: list[dict[str, Any]],
    *,
    scale: dict[str, dict[str, Any]],
    resolver: SQLiteNutritionResolver,
) -> dict[str, Any]:
    item_count = resolved_portion_count = 0
    scored = []
    for row in rows:
        total_visible = sum(
            float(item["estimated_grams"]) for item in row["items"]
        )
        weighted_portions = []
        for item in row["items"]:
            item_count += 1
            resolution = resolver.resolve(str(item["name"]))
            portion = (
                resolution.profile.typical_portion_g
                if resolution.profile
                else None
            )
            if portion is None or portion <= 0:
                continue
            resolved_portion_count += 1
            weighted_portions.append(
                (
                    float(item["estimated_grams"]) / total_visible,
                    float(portion),
                )
            )
        estimates = portion_estimates(weighted_portions)
        scale_row = scale.get(str(row["scale_id"]))
        if scale_row is None:
            continue
        output = {
            **row,
            **{f"prior_{key}": value for key, value in estimates.items()},
            "scale_p50": float(scale_row["p50_g"]),
        }
        # Sum of named serving portions is the most literal meal prior. These
        # fusions are diagnostics, not accepted production behavior.
        prior = estimates.get("sum_item_portions")
        if prior is not None:
            scale_p50 = float(scale_row["p50_g"])
            output["scale_p50_portion_eligible"] = scale_p50
            output["fused_arithmetic"] = (scale_p50 + prior) / 2
            output["fused_geometric"] = math.sqrt(scale_p50 * prior)
            disagrees = (
                prior < float(scale_row["p10_g"]) * 0.5
                or prior > float(scale_row["p90_g"]) * 2
            )
            output["fallback_prior_on_current_disagreement"] = (
                prior if disagrees else scale_p50
            )
            output["current_disagreement"] = disagrees
        legacy_prior = estimates.get("median_portion_div_share")
        if legacy_prior is not None:
            output["legacy_disagreement"] = (
                legacy_prior < float(scale_row["p10_g"]) * 0.5
                or legacy_prior > float(scale_row["p90_g"]) * 2
            )
        scored.append(output)

    prediction_keys = ["scale_p50", "scale_p50_portion_eligible"]
    prediction_keys.extend(
        f"prior_{name}"
        for name in (
            "median_portion_div_share",
            "sum_item_portions",
            "dominant_portion_div_share",
            "least_squares_portion_fit",
            "share_weighted_portion",
        )
    )
    prediction_keys.extend(
        (
            "fused_arithmetic",
            "fused_geometric",
            "fallback_prior_on_current_disagreement",
        )
    )
    disagreement = [
        bool(row["current_disagreement"])
        for row in scored
        if "current_disagreement" in row
    ]
    legacy_disagreement = [
        bool(row["legacy_disagreement"])
        for row in scored
        if "legacy_disagreement" in row
    ]
    return {
        "records": len(scored),
        "groups": len({str(row["group_id"]) for row in scored}),
        "containers": dict(
            sorted(
                {
                    container: sum(row["container"] == container for row in scored)
                    for container in {str(row["container"]) for row in scored}
                }.items()
            )
        ),
        "item_portion_coverage": (
            resolved_portion_count / item_count if item_count else None
        ),
        "current_disagreement_rate": (
            statistics.fmean(disagreement) if disagreement else None
        ),
        "legacy_disagreement_rate": (
            statistics.fmean(legacy_disagreement)
            if legacy_disagreement
            else None
        ),
        "metrics": {
            key: _group_metrics(scored, key) for key in prediction_keys
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    resolver = SQLiteNutritionResolver(args.database)
    try:
        domains = {
            "nutrition5k": _audit_domain(
                _n5k_rows(args),
                scale=_scale_by_id(args.nutrition5k_scale_evaluation),
                resolver=resolver,
            ),
            "nutritionverse": _audit_domain(
                _nutritionverse_rows(args),
                scale=_scale_by_id(args.nutritionverse_scale_evaluation),
                resolver=resolver,
            ),
            "fpb": _audit_domain(
                _fpb_rows(args),
                scale=_scale_by_id(args.fpb_scale_evaluation),
                resolver=resolver,
            ),
        }
    finally:
        resolver.close()
    result = {
        "schema_version": 1,
        "warning": (
            "Oracle names/shares are used. USDA portion weights describe "
            "reference servings, not observed portions, and are not calibrated."
        ),
        "domains": domains,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("datasets/fdc/seecal-nutrition.sqlite"),
    )
    parser.add_argument(
        "--nutrition5k-identify-manifest",
        type=Path,
        default=Path("finetune_data_id_v2/test.jsonl"),
    )
    parser.add_argument(
        "--nutrition5k-scale-manifest",
        type=Path,
        default=Path("datasets/scale_v2_nc_1024/test-nutrition5k.jsonl"),
    )
    parser.add_argument(
        "--nutritionverse-manifest",
        type=Path,
        default=Path("datasets/nutritionverse-real/eval-official-val-v2.jsonl"),
    )
    parser.add_argument(
        "--fpb-manifest",
        type=Path,
        default=Path("datasets/food-portion-benchmark/scale-zero-shot/test.jsonl"),
    )
    parser.add_argument(
        "--fpb-data-yaml",
        type=Path,
        default=Path(
            "/Users/jevgenikabanov/.cache/huggingface/hub/"
            "datasets--issai--Food_Portion_Benchmark/snapshots/"
            "53fcacf4b9dbe24c1c6ffa5a2cdb9d8c502e482f/"
            "FPB_Dataset/RGB/data.yaml"
        ),
    )
    parser.add_argument(
        "--fpb-label-dir",
        type=Path,
        default=Path(
            "/Users/jevgenikabanov/.cache/huggingface/hub/"
            "datasets--issai--Food_Portion_Benchmark/snapshots/"
            "53fcacf4b9dbe24c1c6ffa5a2cdb9d8c502e482f/"
            "FPB_Dataset/RGB/test/labels"
        ),
    )
    base = Path("runs/factored/scale-v2-probe-b-nv-1024")
    parser.add_argument(
        "--nutrition5k-scale-evaluation",
        type=Path,
        default=base / "eval-nutrition5k-overhead-calibrated.json",
    )
    parser.add_argument(
        "--nutritionverse-scale-evaluation",
        type=Path,
        default=base / "eval-nutritionverse-real-calibrated.json",
    )
    parser.add_argument(
        "--fpb-scale-evaluation",
        type=Path,
        default=base / "eval-fpb-test-zero-shot.json",
    )
    parser.add_argument(
        "--output", type=Path, default=base / "portion-prior-audit.json"
    )
    args = parser.parse_args()
    result = build(args)
    print(
        json.dumps(
            {
                domain: {
                    "portion_coverage": values["item_portion_coverage"],
                    "disagreement_rate": values["current_disagreement_rate"],
                    "metrics": values["metrics"],
                }
                for domain, values in result["domains"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
