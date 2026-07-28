"""Validate and select a deterministic FRB 2022 pilot from COCO annotations.

The selector deliberately uses only source annotations.  It does not invent
nutrition targets: numeric nutrition fields are explicitly disabled in every
output record so a later teacher-labeling step cannot accidentally train on
unlicensed or hallucinated calorie/macro values.

Run from ``ml/`` after downloading and extracting the v2.1 training archive:

    .venv/bin/python -m teacher_labeling.food_recognition_2022 \
      --annotations datasets/food_recognition_2022/train/annotations.json \
      --images-root datasets/food_recognition_2022/train/images \
      --out datasets/food_recognition_2022/manifests/pilot-5000.jsonl \
      --expected
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping


EXPECTED_TRAIN_IMAGES = 54_392
EXPECTED_TRAIN_ANNOTATIONS = 100_256
EXPECTED_CLASSES = 323
DEFAULT_SAMPLE_SIZE = 5_000
MANIFEST_SCHEMA_VERSION = 1

HARD_CLASS_TERMS = (
    "bowl",
    "curry",
    "dressing",
    "fried",
    "gravy",
    "mixed",
    "salad",
    "sauce",
    "soup",
    "stew",
)


class DatasetValidationError(RuntimeError):
    """The extracted dataset does not satisfy the source-data contract."""


@dataclass(frozen=True)
class SourceImage:
    image_id: int | str
    file_name: str
    width: int
    height: int
    category_ids: tuple[int | str, ...]
    annotation_count: int


def _id_sort_key(value: int | str) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _require_list(payload: Mapping[str, Any], name: str) -> list[Any]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise DatasetValidationError(f"COCO field {name!r} must be a list")
    return value


def load_and_validate_coco(
    annotations_path: Path,
    *,
    expected_images: int | None = None,
    expected_annotations: int | None = None,
    expected_classes: int | None = None,
) -> tuple[list[SourceImage], dict[int | str, str], dict[str, Any]]:
    """Load a COCO file and validate all image/category references."""

    try:
        raw_bytes = annotations_path.read_bytes()
        payload = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"cannot load COCO annotations: {annotations_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetValidationError("COCO root must be an object")

    raw_images = _require_list(payload, "images")
    raw_annotations = _require_list(payload, "annotations")
    raw_categories = _require_list(payload, "categories")

    expected = (
        ("images", len(raw_images), expected_images),
        ("annotations", len(raw_annotations), expected_annotations),
        ("categories", len(raw_categories), expected_classes),
    )
    for label, actual, wanted in expected:
        if wanted is not None and actual != wanted:
            raise DatasetValidationError(
                f"expected {wanted} {label}, found {actual}"
            )

    categories: dict[int | str, str] = {}
    for position, item in enumerate(raw_categories):
        if not isinstance(item, dict):
            raise DatasetValidationError(f"categories[{position}] must be an object")
        category_id = item.get("id")
        name = item.get("name")
        if category_id is None or not isinstance(name, str) or not name.strip():
            raise DatasetValidationError(
                f"categories[{position}] needs a non-empty id and name"
            )
        if category_id in categories:
            raise DatasetValidationError(f"duplicate category id: {category_id!r}")
        categories[category_id] = name.strip()

    image_rows: dict[int | str, dict[str, Any]] = {}
    for position, item in enumerate(raw_images):
        if not isinstance(item, dict):
            raise DatasetValidationError(f"images[{position}] must be an object")
        image_id = item.get("id")
        file_name = item.get("file_name")
        width = item.get("width")
        height = item.get("height")
        if image_id is None or not isinstance(file_name, str) or not file_name:
            raise DatasetValidationError(
                f"images[{position}] needs a non-empty id and file_name"
            )
        if image_id in image_rows:
            raise DatasetValidationError(f"duplicate image id: {image_id!r}")
        if not isinstance(width, int) or width <= 0:
            raise DatasetValidationError(f"image {image_id!r} has invalid width")
        if not isinstance(height, int) or height <= 0:
            raise DatasetValidationError(f"image {image_id!r} has invalid height")
        image_rows[image_id] = {
            "file_name": file_name,
            "width": width,
            "height": height,
        }

    category_sets: dict[int | str, set[int | str]] = defaultdict(set)
    annotation_counts: Counter[int | str] = Counter()
    annotation_ids: set[int | str] = set()
    for position, item in enumerate(raw_annotations):
        if not isinstance(item, dict):
            raise DatasetValidationError(
                f"annotations[{position}] must be an object"
            )
        annotation_id = item.get("id")
        image_id = item.get("image_id")
        category_id = item.get("category_id")
        if annotation_id is None:
            raise DatasetValidationError(f"annotations[{position}] needs an id")
        if annotation_id in annotation_ids:
            raise DatasetValidationError(
                f"duplicate annotation id: {annotation_id!r}"
            )
        annotation_ids.add(annotation_id)
        if image_id not in image_rows:
            raise DatasetValidationError(
                f"annotation {annotation_id!r} references unknown image {image_id!r}"
            )
        if category_id not in categories:
            raise DatasetValidationError(
                f"annotation {annotation_id!r} references unknown category "
                f"{category_id!r}"
            )
        category_sets[image_id].add(category_id)
        annotation_counts[image_id] += 1

    images = [
        SourceImage(
            image_id=image_id,
            file_name=row["file_name"],
            width=row["width"],
            height=row["height"],
            category_ids=tuple(
                sorted(category_sets[image_id], key=_id_sort_key)
            ),
            annotation_count=annotation_counts[image_id],
        )
        for image_id, row in image_rows.items()
    ]
    images.sort(key=lambda item: _id_sort_key(item.image_id))

    stats = {
        "images": len(images),
        "annotations": len(raw_annotations),
        "classes": len(categories),
        "images_without_annotations": sum(not image.category_ids for image in images),
        "multi_class_images": sum(len(image.category_ids) > 1 for image in images),
        "annotations_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    return images, categories, stats


def _class_frequencies(images: Iterable[SourceImage]) -> Counter[int | str]:
    frequencies: Counter[int | str] = Counter()
    for image in images:
        frequencies.update(image.category_ids)
    return frequencies


def _hard_class_ids(categories: Mapping[int | str, str]) -> set[int | str]:
    return {
        category_id
        for category_id, name in categories.items()
        if any(term in name.casefold() for term in HARD_CLASS_TERMS)
    }


def select_stratified_pilot(
    images: list[SourceImage],
    categories: Mapping[int | str, str],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 20_260_728,
    min_per_class: int = 5,
    excluded_image_ids: frozenset[int | str] = frozenset(),
) -> list[SourceImage]:
    """Choose a deterministic class-balanced, mixed-dish-weighted pilot.

    Each iteration serves the category with the lowest achieved/target ratio.
    Within that category it prefers images that cover rare classes, contain
    multiple source classes, or match the hard-food proxies above.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    eligible = [
        image
        for image in images
        if image.category_ids and image.image_id not in excluded_image_ids
    ]
    if sample_size > len(eligible):
        raise ValueError(
            f"sample_size {sample_size} exceeds {len(eligible)} annotated images"
        )
    if min_per_class <= 0:
        raise ValueError("min_per_class must be positive")

    frequencies = _class_frequencies(eligible)
    missing = sorted(set(categories) - set(frequencies), key=_id_sort_key)
    if missing:
        raise DatasetValidationError(
            f"{len(missing)} categories have no annotated images"
        )

    target: dict[int | str, int] = {
        category_id: min(
            count,
            max(min_per_class, round(sample_size * count / len(eligible))),
        )
        for category_id, count in frequencies.items()
    }
    hard_ids = _hard_class_ids(categories)
    rng = random.Random(seed)
    tie_break = {image.image_id: rng.random() for image in eligible}

    def candidate_score(image: SourceImage) -> tuple[float, float, str]:
        rarity = sum(1.0 / frequencies[item] for item in image.category_ids)
        mixed_bonus = math.log2(1 + len(image.category_ids))
        hard_bonus = 0.35 * sum(item in hard_ids for item in image.category_ids)
        polygon_bonus = 0.05 * min(image.annotation_count, 10)
        return (
            rarity + mixed_bonus + hard_bonus + polygon_bonus,
            tie_break[image.image_id],
            str(image.image_id),
        )

    pools: dict[int | str, list[SourceImage]] = defaultdict(list)
    for image in eligible:
        for category_id in image.category_ids:
            pools[category_id].append(image)
    for pool in pools.values():
        pool.sort(key=candidate_score, reverse=True)

    selected: list[SourceImage] = []
    selected_ids: set[int | str] = set()
    achieved: Counter[int | str] = Counter()
    cursors: Counter[int | str] = Counter()

    while len(selected) < sample_size:
        available_categories = [
            category_id
            for category_id, pool in pools.items()
            if cursors[category_id] < len(pool)
        ]
        if not available_categories:
            break
        category_id = min(
            available_categories,
            key=lambda item: (
                achieved[item] / target[item],
                achieved[item],
                _id_sort_key(item),
            ),
        )
        pool = pools[category_id]
        cursor = cursors[category_id]
        while cursor < len(pool) and pool[cursor].image_id in selected_ids:
            cursor += 1
        cursors[category_id] = cursor
        if cursor >= len(pool):
            continue

        chosen = pool[cursor]
        cursors[category_id] += 1
        selected.append(chosen)
        selected_ids.add(chosen.image_id)
        achieved.update(chosen.category_ids)

    if len(selected) != sample_size:
        raise DatasetValidationError(
            f"selector produced {len(selected)} of {sample_size} requested images"
        )
    return selected


def _selected_stats(
    selected: list[SourceImage],
    categories: Mapping[int | str, str],
) -> dict[str, Any]:
    frequencies = _class_frequencies(selected)
    return {
        "selected_images": len(selected),
        "represented_classes": len(frequencies),
        "source_classes": len(categories),
        "multi_class_images": sum(len(image.category_ids) > 1 for image in selected),
        "mean_classes_per_image": round(
            sum(len(image.category_ids) for image in selected) / len(selected),
            6,
        ),
        "minimum_class_coverage": min(frequencies.values(), default=0),
    }


def write_pilot_manifest(
    *,
    selected: list[SourceImage],
    categories: Mapping[int | str, str],
    annotations_path: Path,
    images_root: Path,
    output_path: Path,
    seed: int,
    source_stats: Mapping[str, Any],
    verify_files: bool,
    release: str = "v2.1",
    archive_sha256: str | None = None,
) -> Path:
    """Write JSONL records plus a sidecar summary, failing on missing files."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    records: list[dict[str, Any]] = []
    for image in selected:
        image_path = images_root / image.file_name
        if verify_files and not image_path.is_file():
            missing.append(image.file_name)
            continue
        records.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "source": {
                    "dataset": "food-recognition-benchmark-2022",
                    "release": release,
                    "license": "CC-BY-4.0",
                    "image_id": image.image_id,
                    "file_name": image.file_name,
                    "annotations_sha256": source_stats["annotations_sha256"],
                    **(
                        {"archive_sha256": archive_sha256}
                        if archive_sha256 is not None
                        else {}
                    ),
                },
                "image": {
                    "path": str(image_path),
                    "width": image.width,
                    "height": image.height,
                },
                "source_labels": [
                    {
                        "category_id": category_id,
                        "name": categories[category_id],
                    }
                    for category_id in image.category_ids
                ],
                "source_annotation_count": image.annotation_count,
                "loss_mask": {
                    "semantic_food_fields": True,
                    "calories": False,
                    "grams": False,
                    "macros": False,
                    "hidden_ingredients": False,
                },
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetValidationError(
            f"{len(missing)} selected image files are missing under "
            f"{images_root}; first: {preview}"
        )

    text = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        for record in records
    )
    output_path.write_text(text, encoding="utf-8")
    summary = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selector": "class-ratio-greedy-v1",
        "source_release": release,
        "source_archive_sha256": archive_sha256,
        "seed": seed,
        "annotations_path": str(annotations_path),
        "images_root": str(images_root),
        "source_stats": dict(source_stats),
        "pilot_stats": _selected_stats(selected, categories),
        "manifest_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=20_260_728)
    parser.add_argument(
        "--release",
        choices=("v2.0", "v2.1"),
        default="v2.1",
    )
    parser.add_argument(
        "--archive-sha256",
        help="SHA-256 of the original downloaded archive.",
    )
    parser.add_argument(
        "--exclude-image-id",
        action="append",
        type=int,
        default=[],
        help="Source image ID to exclude after exact/perceptual deduplication.",
    )
    parser.add_argument(
        "--expected",
        action="store_true",
        help="Require the published v2.1 train counts (54,392/100,256/323).",
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Allow annotation-only planning before the image archive is present.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    images, categories, source_stats = load_and_validate_coco(
        args.annotations,
        expected_images=EXPECTED_TRAIN_IMAGES if args.expected else None,
        expected_annotations=EXPECTED_TRAIN_ANNOTATIONS if args.expected else None,
        expected_classes=EXPECTED_CLASSES if args.expected else None,
    )
    selected = select_stratified_pilot(
        images,
        categories,
        sample_size=args.sample_size,
        seed=args.seed,
        excluded_image_ids=frozenset(args.exclude_image_id),
    )
    summary_path = write_pilot_manifest(
        selected=selected,
        categories=categories,
        annotations_path=args.annotations,
        images_root=args.images_root,
        output_path=args.out,
        seed=args.seed,
        source_stats=source_stats,
        verify_files=not args.skip_file_check,
        release=args.release,
        archive_sha256=args.archive_sha256,
    )
    print(summary_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
