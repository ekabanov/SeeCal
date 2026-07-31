"""Validate semantics-first prepared aliases without mutating the main database."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
from typing import Any

from factored_pipeline.contract import normalize_name
from oracle_assembly_audit import run_oracle


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


def _full_mass_by_id(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    output = {}
    for row in _read_jsonl(path):
        truth = _completion(row)
        output[Path(row["images"][0]).parent.name] = sum(
            float(item["estimated_grams"]) for item in truth["items"]
        )
    return output


def _true_mass_scale(
    manifest: Path,
    full_truth: Path | None,
) -> dict[str, Any]:
    full_mass = _full_mass_by_id(full_truth)
    paired = []
    for row in _read_jsonl(manifest):
        truth = row.get("evaluation_ground_truth")
        if truth is None:
            continue
        record_id = str(row["id"])
        mass = truth.get("total_mass_g", full_mass.get(record_id))
        if mass is None:
            continue
        paired.append(
            {
                "id": record_id,
                "group_id": str(row.get("group_id") or record_id),
                "source": "true_mass_alias_validation",
                "target_mass_g": float(mass),
                "p10_g": float(mass),
                "p50_g": float(mass),
                "p90_g": float(mass),
            }
        )
    return {"schema_version": 1, "paired": paired}


def _candidates(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["candidate_fdc_id"] = (
            int(row["candidate_fdc_id"])
            if row["candidate_fdc_id"].strip()
            else None
        )
    return rows


def _profile(database: Path, fdc_id: int) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT fdc_id, name, category, data_type, kcal_per_100g,
                   protein_per_100g, fat_per_100g, carbs_per_100g
            FROM foods WHERE fdc_id = ?
            """,
            (fdc_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError(f"candidate FDC ID not in local database: {fdc_id}")
    return dict(row)


def _copy_with_aliases(
    source: Path,
    destination: Path,
    candidates: list[dict[str, Any]],
) -> None:
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.executemany(
            "INSERT OR REPLACE INTO aliases VALUES (?,?,?,?)",
            [
                (
                    normalize_name(str(row["name"])),
                    int(row["candidate_fdc_id"]),
                    300,
                    "semantics_first_prepared_v1",
                )
                for row in candidates
                if row["candidate_fdc_id"] is not None
            ],
        )
        destination_connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?,?)",
            (
                "reviewed_aliases:semantics_first_prepared_v1",
                str(
                    sum(
                        row["candidate_fdc_id"] is not None
                        for row in candidates
                    )
                ),
            ),
        )
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _observed_density(
    manifest: Path,
    names: set[str],
) -> dict[str, dict[str, Any]]:
    values: dict[str, list[tuple[float, float]]] = {
        name: [] for name in names
    }
    for row in _read_jsonl(manifest):
        truth = row.get("evaluation_ground_truth")
        if truth is None:
            continue
        for item in truth.get("items", []):
            name = str(item["name"])
            grams = float(item["estimated_grams"])
            if name in values and grams > 0:
                values[name].append(
                    (100 * float(item["calories"]) / grams, grams)
                )
    output = {}
    for name, observations in values.items():
        densities = [density for density, _ in observations]
        total_mass = sum(mass for _, mass in observations)
        output[name] = {
            "occurrences": len(observations),
            "median_kcal_per_100g": (
                statistics.median(densities) if densities else None
            ),
            "mean_kcal_per_100g": (
                statistics.fmean(densities) if densities else None
            ),
            "mass_weighted_kcal_per_100g": (
                sum(density * mass for density, mass in observations)
                / total_mass
                if total_mass
                else None
            ),
        }
    return output


def _affected_groups(manifest: Path, names: set[str]) -> set[str]:
    groups = set()
    for row in _read_jsonl(manifest):
        truth = row.get("evaluation_ground_truth")
        if truth is None:
            continue
        if any(str(item["name"]) in names for item in truth.get("items", [])):
            groups.add(str(row.get("group_id") or row["id"]))
    return groups


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        mode: {
            key: result["variants"][mode][key]
            for key in (
                "groups",
                "complete_groups",
                "complete_group_rate",
                "equal_group_kcal_mae",
                "mean_signed_kcal_error_fraction",
                "median_signed_kcal_error_fraction",
            )
        }
        for mode in (
            "true_mass_exact_shares",
            "true_mass_bucketed_shares",
        )
    }


def _affected_mae(
    result: dict[str, Any],
    groups: set[str],
) -> float | None:
    rows = [
        row
        for row in result["paired"]["true_mass_exact_shares"]
        if row["complete"] and str(row["group_id"]) in groups
    ]
    by_group: dict[str, list[float]] = {}
    for row in rows:
        by_group.setdefault(str(row["group_id"]), []).append(
            float(row["calorie_absolute_error"])
        )
    return (
        statistics.fmean(
            statistics.fmean(errors) for errors in by_group.values()
        )
        if by_group
        else None
    )


def _score_dataset(
    *,
    manifest: Path,
    full_truth: Path | None,
    scale_path: Path,
    database: Path,
    taxonomy: Path,
) -> dict[str, Any]:
    return run_oracle(
        identify_manifest=manifest,
        full_truth_manifest=full_truth,
        scale_predictions=scale_path,
        database=database,
        taxonomy=taxonomy,
    )


def run_audit(
    *,
    candidates_path: Path,
    database: Path,
    taxonomy: Path,
    n5k_manifest: Path,
    n5k_full_truth: Path,
    nv_quality_manifest: Path,
) -> dict[str, Any]:
    candidate_rows = _candidates(candidates_path)
    candidates_with_profiles = [
        row for row in candidate_rows if row["candidate_fdc_id"] is not None
    ]
    names = {str(row["name"]) for row in candidate_rows}
    train_manifest = n5k_manifest.parent / "train.jsonl"
    observations = {
        "nutrition5k_train": _observed_density(train_manifest, names),
        "nutrition5k_validation": _observed_density(n5k_manifest, names),
        "nutritionverse_quality": _observed_density(
            nv_quality_manifest, names
        ),
    }
    for row in candidate_rows:
        fdc_id = row["candidate_fdc_id"]
        row["candidate_profile"] = (
            _profile(database, int(fdc_id)) if fdc_id is not None else None
        )
        row["observed_density"] = {
            split: values[str(row["name"])]
            for split, values in observations.items()
        }

    datasets = {
        "nutrition5k_validation": {
            "manifest": n5k_manifest,
            "full_truth": n5k_full_truth,
        },
        "nutritionverse_quality": {
            "manifest": nv_quality_manifest,
            "full_truth": None,
        },
    }
    with tempfile.TemporaryDirectory(
        prefix="seecal-prepared-alias-audit-"
    ) as temporary:
        root = Path(temporary)
        scale_paths = {}
        current_results = {}
        for dataset_name, config in datasets.items():
            scale_path = root / f"{dataset_name}-true-mass.json"
            scale_path.write_text(
                json.dumps(
                    _true_mass_scale(config["manifest"], config["full_truth"])
                ),
                encoding="utf-8",
            )
            scale_paths[dataset_name] = scale_path
            current_results[dataset_name] = _score_dataset(
                manifest=config["manifest"],
                full_truth=config["full_truth"],
                scale_path=scale_path,
                database=database,
                taxonomy=taxonomy,
            )

        arms: dict[str, list[dict[str, Any]]] = {
            str(row["name"]): [row] for row in candidates_with_profiles
        }
        arms["combined_provisional"] = [
            row
            for row in candidates_with_profiles
            if str(row["decision"]).startswith("provisional")
        ]
        arm_results = {}
        for arm_name, arm_candidates in arms.items():
            candidate_database = root / f"{arm_name.replace(' ', '-')}.sqlite"
            _copy_with_aliases(database, candidate_database, arm_candidates)
            changed_names = {str(row["name"]) for row in arm_candidates}
            dataset_results = {}
            for dataset_name, config in datasets.items():
                candidate_result = _score_dataset(
                    manifest=config["manifest"],
                    full_truth=config["full_truth"],
                    scale_path=scale_paths[dataset_name],
                    database=candidate_database,
                    taxonomy=taxonomy,
                )
                affected = _affected_groups(
                    config["manifest"], changed_names
                )
                current_affected = _affected_mae(
                    current_results[dataset_name], affected
                )
                candidate_affected = _affected_mae(
                    candidate_result, affected
                )
                dataset_results[dataset_name] = {
                    "changed_alias_groups": len(affected),
                    "current": _compact(current_results[dataset_name]),
                    "candidate": _compact(candidate_result),
                    "overall_exact_mae_delta_kcal": (
                        candidate_result["variants"][
                            "true_mass_exact_shares"
                        ]["equal_group_kcal_mae"]
                        - current_results[dataset_name]["variants"][
                            "true_mass_exact_shares"
                        ]["equal_group_kcal_mae"]
                    ),
                    "changed_groups_exact_mae_current_kcal": current_affected,
                    "changed_groups_exact_mae_candidate_kcal": (
                        candidate_affected
                    ),
                    "changed_groups_exact_mae_delta_kcal": (
                        candidate_affected - current_affected
                        if candidate_affected is not None
                        and current_affected is not None
                        else None
                    ),
                }
            n5k = dataset_results["nutrition5k_validation"]
            nv = dataset_results["nutritionverse_quality"]
            arm_results[arm_name] = {
                "aliases": sorted(changed_names),
                "datasets": dataset_results,
                "decision": {
                    "n5k_changed_groups_improve": (
                        n5k["changed_groups_exact_mae_delta_kcal"] is not None
                        and n5k["changed_groups_exact_mae_delta_kcal"] < 0
                    ),
                    "nutritionverse_direct_support": (
                        nv["changed_alias_groups"] > 0
                    ),
                    "status": (
                        "eligible_after_validation"
                        if n5k["changed_groups_exact_mae_delta_kcal"] is not None
                        and n5k["changed_groups_exact_mae_delta_kcal"] < 0
                        and nv["changed_alias_groups"] > 0
                        and nv["changed_groups_exact_mae_delta_kcal"] <= 0
                        else (
                            "provisional_no_nutritionverse_support"
                            if n5k["changed_groups_exact_mae_delta_kcal"]
                            is not None
                            and n5k["changed_groups_exact_mae_delta_kcal"] < 0
                            and nv["changed_alias_groups"] == 0
                            else "rejected"
                        )
                    ),
                },
            }

    return {
        "schema_version": 1,
        "scope": {
            "selection_order": (
                "preparation-state semantics first; Nutrition5K train density "
                "only describes/tie-breaks semantically valid profiles"
            ),
            "validation": [
                "Nutrition5K validation",
                "NutritionVerse quality",
            ],
            "frozen_test_rows_used_for_selection_or_validation": 0,
            "main_database_mutated": False,
        },
        "candidates": candidate_rows,
        "baseline": {
            name: _compact(result)
            for name, result in current_results.items()
        },
        "arms": arm_results,
        "transfer_warning": (
            "A zero changed-alias count on NutritionVerse quality is no direct "
            "cross-kitchen evidence and blocks promotion even if overall "
            "metrics are unchanged."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--n5k-manifest", type=Path, required=True)
    parser.add_argument("--n5k-full-truth", type=Path, required=True)
    parser.add_argument("--nv-quality-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_audit(
        candidates_path=args.candidates,
        database=args.database,
        taxonomy=args.taxonomy,
        n5k_manifest=args.n5k_manifest,
        n5k_full_truth=args.n5k_full_truth,
        nv_quality_manifest=args.nv_quality_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arms": {
                    name: {
                        "aliases": arm["aliases"],
                        "decision": arm["decision"],
                        "n5k_changed_groups_delta_kcal": arm["datasets"][
                            "nutrition5k_validation"
                        ]["changed_groups_exact_mae_delta_kcal"],
                        "nutritionverse_changed_groups": arm["datasets"][
                            "nutritionverse_quality"
                        ]["changed_alias_groups"],
                    }
                    for name, arm in payload["arms"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
