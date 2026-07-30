"""Build leakage-safe, multi-domain SCALE-v2 manifests.

Nutrition5K keeps its existing dish-level splits and gains deterministic
samples from all four side cameras. NutritionVerse-Real uses the authors'
scene-level Train/Val split: Val is frozen as the phone-photo test set, while
Train is emitted separately for non-commercial research and excluded from
shippable training by default. All views from a scene stay together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any


DISH_PATTERN = re.compile(r"dish_(\d+)_")
CAMERA_PATTERN = re.compile(r"camera_([A-D])frame(\d+)\.jpeg$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _nutritionverse_split_by_group(
    official_split_csv: Path,
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, str]:
    official_by_group: dict[str, set[str]] = defaultdict(set)
    with official_split_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            match = DISH_PATTERN.match(row["file_name"])
            if not match:
                raise ValueError(f"unrecognized NutritionVerse filename: {row['file_name']}")
            group_id = f"nv-real:dish_{int(match.group(1))}"
            official_by_group[group_id].add(row["category"].strip().lower())

    leaky = {group: values for group, values in official_by_group.items() if len(values) != 1}
    if leaky:
        raise ValueError(f"NutritionVerse official split leaks groups: {leaky}")

    result = {}
    for group_id, categories in official_by_group.items():
        category = next(iter(categories))
        if category == "val":
            result[group_id] = "test"
        elif category == "train":
            result[group_id] = (
                "valid"
                if _stable_fraction(group_id, seed) < validation_fraction
                else "train"
            )
        else:
            raise ValueError(f"unknown NutritionVerse split category: {category}")
    return result


def _check_no_group_leakage(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            group_id = str(row["group_id"])
            previous = seen.setdefault(group_id, split)
            if previous != split:
                raise ValueError(
                    f"group {group_id} appears in both {previous} and {split}"
                )


def _evenly_spaced(rows: list[Path], count: int) -> list[Path]:
    if count >= len(rows):
        return rows
    return [
        rows[min(len(rows) - 1, round((index + 0.5) * len(rows) / count - 0.5))]
        for index in range(count)
    ]


def _side_angle_rows(
    base_rows: list[dict[str, Any]],
    *,
    side_angle_root: Path,
    ml_root: Path,
    split: str,
    frames_per_camera: int,
) -> list[dict[str, Any]]:
    if frames_per_camera <= 0:
        return []
    mass_by_group = {
        str(row["group_id"]): float(row["total_mass_g"])
        for row in base_rows
        if row["source"] == "nutrition5k"
    }
    output = []
    for group_id, mass in sorted(mass_by_group.items()):
        directory = side_angle_root / group_id
        by_camera: dict[str, list[Path]] = defaultdict(list)
        for path in sorted(directory.glob("camera_*frame*.jpeg")):
            match = CAMERA_PATTERN.fullmatch(path.name)
            if match:
                by_camera[match.group(1)].append(path)
        for camera, paths in sorted(by_camera.items()):
            for path in _evenly_spaced(paths, frames_per_camera):
                try:
                    image_path = str(path.resolve().relative_to(ml_root.resolve()))
                except ValueError:
                    image_path = path.resolve().as_uri()
                output.append(
                    {
                        "schema_version": 1,
                        "id": f"nutrition5k:{group_id}:{path.stem}",
                        "group_id": group_id,
                        "split": split,
                        "image_path": image_path,
                        "total_mass_g": mass,
                        "source": "nutrition5k",
                    }
                )
    return output


def _raw_metadata_base_rows(
    metadata_csv: Path,
    *,
    excluded_groups: set[str],
    domain: str,
) -> list[dict[str, Any]]:
    rows = []
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for fields in csv.reader(handle):
            if len(fields) < 3:
                continue
            group_id = fields[0]
            mass = float(fields[2])
            if group_id in excluded_groups or mass <= 0:
                continue
            rows.append(
                {
                    "schema_version": 1,
                    "id": f"nutrition5k:{group_id}:cafe2-source",
                    "group_id": group_id,
                    "split": "train",
                    "image_path": "",
                    "total_mass_g": mass,
                    "source": "nutrition5k",
                    "domain": domain,
                }
            )
    return rows


def build(
    *,
    nutrition5k_dir: Path,
    nutritionverse_manifest: Path,
    official_split_csv: Path,
    output_dir: Path,
    validation_fraction: float,
    seed: int,
    side_angle_root: Path | None = None,
    ml_root: Path = Path.cwd(),
    frames_per_camera: int = 0,
    test_frames_per_camera: int = 0,
    include_noncommercial_training: bool = False,
    cafe1_metadata: Path | None = None,
    cafe2_metadata: Path | None = None,
    fpb_manifest_dir: Path | None = None,
) -> dict[str, Any]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: _read_jsonl(nutrition5k_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    test_side_rows: list[dict[str, Any]] = []
    additional_groups: dict[str, int] = {}
    if side_angle_root is not None:
        rows_by_split["train"].extend(
            _side_angle_rows(
                rows_by_split["train"],
                side_angle_root=side_angle_root,
                ml_root=ml_root,
                split="train",
                frames_per_camera=frames_per_camera,
            )
        )
        test_side_rows = _side_angle_rows(
            rows_by_split["test"],
            side_angle_root=side_angle_root,
            ml_root=ml_root,
            split="test",
            frames_per_camera=test_frames_per_camera,
        )
        for domain, metadata_path in (
            ("cafe1-extra", cafe1_metadata),
            ("cafe2", cafe2_metadata),
        ):
            if metadata_path is None:
                continue
            excluded_groups = {
                str(row["group_id"])
                for rows in rows_by_split.values()
                for row in rows
            }
            raw_base = _raw_metadata_base_rows(
                metadata_path,
                excluded_groups=excluded_groups,
                domain=domain,
            )
            raw_sides = _side_angle_rows(
                raw_base,
                side_angle_root=side_angle_root,
                ml_root=ml_root,
                split="train",
                frames_per_camera=frames_per_camera,
            )
            additional_groups[domain] = len(
                {str(row["group_id"]) for row in raw_sides}
            )
            for row in raw_sides:
                row["domain"] = domain
            rows_by_split["train"].extend(raw_sides)
    nv_split = _nutritionverse_split_by_group(
        official_split_csv,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    research_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
    }
    for source in _read_jsonl(nutritionverse_manifest):
        group_id = str(source["group_id"])
        if group_id not in nv_split:
            raise ValueError(f"NutritionVerse group missing from official split: {group_id}")
        split = nv_split[group_id]
        row = dict(source)
        row["split"] = split
        if split == "test" or include_noncommercial_training:
            rows_by_split[split].append(row)
        else:
            research_rows[split].append(row)
    fpb_records = 0
    if fpb_manifest_dir is not None:
        for split in ("train", "valid"):
            fpb_rows = _read_jsonl(fpb_manifest_dir / f"{split}.jsonl")
            for row in fpb_rows:
                row["split"] = split
            rows_by_split[split].extend(fpb_rows)
            fpb_records += len(fpb_rows)

    _check_no_group_leakage(rows_by_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: (str(row["source"]), str(row["id"])))
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        for source in sorted({str(row["source"]) for row in rows}):
            source_rows = [row for row in rows if row["source"] == source]
            slug = source.replace("-v2", "")
            (output_dir / f"{split}-{slug}.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in source_rows
                ),
                encoding="utf-8",
            )
        for source in ("nutrition5k", "nutritionverse-real"):
            path = output_dir / f"{split}-{source}.jsonl"
            if not path.exists() or not any(
                str(row["source"]).replace("-v2", "") == source for row in rows
            ):
                path.write_text("", encoding="utf-8")
    if test_side_rows:
        (output_dir / "test-nutrition5k-side.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in test_side_rows
            ),
            encoding="utf-8",
        )
    for split, rows in research_rows.items():
        (output_dir / f"research-{split}-nutritionverse-real.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    counts: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        source_counts = Counter(str(row["source"]) for row in rows)
        source_groups = {
            source: len(
                {
                    str(row["group_id"])
                    for row in rows
                    if str(row["source"]) == source
                }
            )
            for source in source_counts
        }
        counts[split] = {
            "records": len(rows),
            "groups": len({str(row["group_id"]) for row in rows}),
            "records_by_source": dict(sorted(source_counts.items())),
            "groups_by_source": dict(sorted(source_groups.items())),
        }
    metadata = {
        "schema_version": 2,
        "seed": seed,
        "nutritionverse_policy": {
            "official_val": "frozen_test",
            "official_train_validation_fraction": validation_fraction,
            "group_key": "dish_id",
            "noncommercial_training_included": include_noncommercial_training,
            "license_boundary": (
                "CC BY-NC-SA data is evaluation/research-only by default"
            ),
        },
        "food_portion_benchmark_policy": {
            "included": fpb_manifest_dir is not None,
            "records": fpb_records,
            "license": "CC BY-NC 4.0",
            "official_train_used_for_training": fpb_manifest_dir is not None,
            "official_val_used_for_validation": fpb_manifest_dir is not None,
        },
        "nutrition5k_side_angle_policy": {
            "frames_per_camera_train": frames_per_camera,
            "frames_per_camera_test": test_frames_per_camera,
            "test_side_records": len(test_side_rows),
            "additional_raw_groups": additional_groups,
        },
        "counts": counts,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nutrition5k-dir", type=Path, default=Path("datasets/scale_v1")
    )
    parser.add_argument(
        "--nutritionverse-manifest",
        type=Path,
        default=Path("datasets/nutritionverse-real/scale-eval-v2.jsonl"),
    )
    parser.add_argument(
        "--official-split-csv",
        type=Path,
        default=Path(
            "datasets/nutritionverse-real/extracted/nutritionverse-manual/"
            "nutritionverse-manual/updated-manual-dataset-splits.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets/scale_v2")
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--side-angle-root",
        type=Path,
        default=Path("Nutrition5K/imagery/side_angles"),
    )
    parser.add_argument("--frames-per-camera", type=int, default=2)
    parser.add_argument("--test-frames-per-camera", type=int, default=1)
    parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--cafe1-metadata",
        type=Path,
        default=Path("Nutrition5K/metadata_raw/dish_metadata_cafe1.csv"),
    )
    parser.add_argument(
        "--cafe2-metadata",
        type=Path,
        default=Path("Nutrition5K/metadata_raw/dish_metadata_cafe2.csv"),
    )
    parser.add_argument(
        "--include-noncommercial-training",
        action="store_true",
        help="Research-only: mix CC BY-NC-SA NutritionVerse Train into SCALE.",
    )
    parser.add_argument(
        "--fpb-manifest-dir",
        type=Path,
        help="Non-commercial SCALE manifests produced by make_fpb_scale_data.py.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                nutrition5k_dir=args.nutrition5k_dir,
                nutritionverse_manifest=args.nutritionverse_manifest,
                official_split_csv=args.official_split_csv,
                output_dir=args.output_dir,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                side_angle_root=args.side_angle_root,
                ml_root=args.ml_root,
                frames_per_camera=args.frames_per_camera,
                test_frames_per_camera=args.test_frames_per_camera,
                include_noncommercial_training=args.include_noncommercial_training,
                cafe1_metadata=args.cafe1_metadata,
                cafe2_metadata=args.cafe2_metadata,
                fpb_manifest_dir=args.fpb_manifest_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
