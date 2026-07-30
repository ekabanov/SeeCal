"""Convert Food Portion Benchmark YOLO weights into SCALE manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


CAPTURE_SUFFIX = re.compile(r"_(?:RGB|RGb|rgb|Depth)_.+$")


class IncompleteWeightError(ValueError):
    """A source label explicitly marks at least one object weight as unknown."""


def _capture_name(stem: str) -> str:
    return stem.split(".rf.", 1)[0].lower()


def _group_name(stem: str, grouping: str) -> str:
    base = stem.split(".rf.", 1)[0]
    if grouping == "capture":
        return base.lower()
    if grouping == "food-size":
        return CAPTURE_SUFFIX.sub("", base).lower()
    raise ValueError(f"unsupported FPB grouping policy: {grouping}")


def _total_weight(label_path: Path) -> float:
    total = 0.0
    for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"{label_path}:{index + 1} has fewer than six fields")
        weight = float(fields[-1])
        if not math.isfinite(weight):
            raise ValueError(f"{label_path}:{index + 1} has invalid weight {weight}")
        if weight <= 0:
            raise IncompleteWeightError(
                f"{label_path}:{index + 1} has unknown weight {weight}"
            )
        total += weight
    if total <= 0:
        raise ValueError(f"{label_path} contains no positive weights")
    return total


def build(
    *,
    dataset_root: Path,
    output_dir: Path,
    ml_root: Path,
    splits: tuple[str, ...] = ("train", "val"),
    skip_incomplete_weights: bool = False,
    grouping: str = "capture",
    exclude_test_capture_overlap: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_capture_names: set[str] = set()
    if exclude_test_capture_overlap:
        test_image_root = dataset_root / "FPB_Dataset" / "RGB" / "test" / "images"
        if not test_image_root.is_dir():
            raise FileNotFoundError(
                "--exclude-test-capture-overlap requires the FPB test images"
            )
        test_capture_names = {
            _capture_name(path.stem)
            for path in test_image_root.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
    counts = {}
    for split in splits:
        split_root = dataset_root / "FPB_Dataset" / "RGB" / split
        rows = []
        skipped_incomplete = 0
        skipped_test_overlap = 0
        image_stems = {
            path.stem
            for path in (split_root / "images").glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        label_stems = {
            path.stem
            for path in (split_root / "labels").glob("*.txt")
        }
        for image_path in sorted((split_root / "images").glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if split != "test" and _capture_name(image_path.stem) in test_capture_names:
                skipped_test_overlap += 1
                continue
            label_path = split_root / "labels" / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"missing FPB label for {image_path.name}")
            try:
                total_weight = _total_weight(label_path)
            except IncompleteWeightError:
                if not skip_incomplete_weights:
                    raise
                skipped_incomplete += 1
                continue
            try:
                relative = str(image_path.resolve().relative_to(ml_root.resolve()))
            except ValueError:
                # ScaleDataset accepts absolute filesystem paths because joining
                # an absolute Path to ml_root preserves the absolute operand.
                relative = str(image_path.resolve())
            group = _group_name(image_path.stem, grouping)
            rows.append(
                {
                    "schema_version": 1,
                    "id": f"fpb:{split}:{image_path.stem}",
                    "group_id": f"fpb:{group}",
                    "split": {"val": "valid"}.get(split, split),
                    "image_path": relative,
                    "total_mass_g": total_weight,
                    "source": "food-portion-benchmark",
                }
            )
        output_name = {"val": "valid"}.get(split, split)
        (output_dir / f"{output_name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        counts[output_name] = {
            "records": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "skipped_incomplete_weight_records": skipped_incomplete,
            "skipped_test_capture_overlap_records": skipped_test_overlap,
            "orphan_label_records": len(label_stems - image_stems),
        }
    metadata = {
        "schema_version": 1,
        "source_revision": "53fcacf4b9dbe24c1c6ffa5a2cdb9d8c502e482f",
        "license": "CC BY-NC 4.0",
        "target": "sum of per-object gram weights in YOLO labels",
        "grouping": grouping,
        "test_capture_overlap_excluded": exclude_test_capture_overlap,
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
        "--dataset-root",
        type=Path,
        default=Path("datasets/food-portion-benchmark"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/food-portion-benchmark/scale"),
    )
    parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val"),
    )
    parser.add_argument(
        "--skip-incomplete-weights",
        action="store_true",
        help="Exclude records containing non-positive/unknown object weights.",
    )
    parser.add_argument(
        "--grouping",
        choices=("capture", "food-size"),
        default="capture",
        help=(
            "Use source-capture groups for training/validation. The legacy "
            "food-size grouping is retained only to reproduce the frozen zero-shot audit."
        ),
    )
    parser.add_argument(
        "--exclude-test-capture-overlap",
        action="store_true",
        help="Exclude train/validation records derived from a frozen test capture.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                ml_root=args.ml_root,
                splits=tuple(args.splits),
                skip_incomplete_weights=args.skip_incomplete_weights,
                grouping=args.grouping,
                exclude_test_capture_overlap=args.exclude_test_capture_overlap,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
