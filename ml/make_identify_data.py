"""Build new-schema IDENTIFY JSONL from derived labels only.

Nutrition5K relative portion units come from measured ingredient gram
fractions. A generic
COCO segmentation source (FoodSeg103 or a compatible food dataset) contributes
class names and mask-area-derived relative units. Deterministic runtime code,
not the model, normalizes units to shares. COCO negatives keep the proven
100-record training cap. Image paths are relative to ``ml/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from PIL import Image

from cache_eval_images import cache as cache_images
from factored_pipeline.contract import (
    IDENTIFY_PROMPT,
    NOT_FOOD_COMPLETION,
    canonical_identification,
    identification_to_shares,
    normalize_name,
)
from factored_pipeline.resolver import SQLiteNutritionResolver
from factored_pipeline.visible_labels import visible_component_weights


def _record(
    image_path: str,
    completion: str,
    *,
    record_id: str | None = None,
    group_id: str | None = None,
    evaluation_ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "images": [image_path],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": IDENTIFY_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": completion}],
            },
        ],
    }
    if record_id is not None:
        record["id"] = record_id
    if group_id is not None:
        record["group_id"] = group_id
    if evaluation_ground_truth is not None:
        record["evaluation_ground_truth"] = evaluation_ground_truth
    return record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nutrition5k_records(path: Path) -> list[dict[str, Any]]:
    output = []
    for source in _read_jsonl(path):
        content = source["messages"][-1]["content"]
        text = content[0]["text"] if isinstance(content, list) else content
        payload = json.loads(text)
        weighted_names = [
            (str(item["name"]), float(item["estimated_grams"]))
            for item in payload["items"]
        ]
        weighted_names = visible_component_weights(weighted_names)
        completion = canonical_identification(
            container="tray",
            weighted_names=weighted_names,
        )
        identification = identification_to_shares(json.loads(completion))
        source_items = {
            normalize_name(str(item["name"])): item for item in payload["items"]
        }
        evaluation_items = []
        for item in identification["items"]:
            source_item = source_items[normalize_name(str(item["name"]))]
            evaluation_items.append(
                {
                    **source_item,
                    "portion_units": item["portion_units"],
                    "share_pct": item["share_pct"],
                }
            )
        evaluation_ground_truth = {
            **payload,
            "not_food": False,
            "container": "tray",
            "items": evaluation_items,
        }
        image_path = source["images"][0]
        output.append(
            _record(
                image_path,
                completion,
                record_id=Path(image_path).parent.name,
                group_id=Path(image_path).parent.name,
                evaluation_ground_truth=evaluation_ground_truth,
            )
        )
    return output


def segmentation_records(
    annotations_path: Path,
    image_root: Path,
    *,
    ml_root: Path,
    include_image_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    source = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {
        int(row["id"]): str(row["name"]).replace("-", " ")
        for row in source["categories"]
    }
    images = {int(row["id"]): row for row in source["images"]}
    labels: dict[int, dict[str, float]] = {}
    for annotation in source["annotations"]:
        image_id = int(annotation["image_id"])
        if include_image_ids is not None and image_id not in include_image_ids:
            continue
        name = categories[int(annotation["category_id"])]
        area = float(annotation.get("area") or 0)
        if area <= 0:
            continue
        labels.setdefault(image_id, {})[name] = labels.setdefault(image_id, {}).get(name, 0) + area

    output = []
    for image_id, weighted in sorted(labels.items()):
        metadata = images[image_id]
        absolute = image_root / metadata["file_name"]
        if not absolute.is_file():
            continue
        try:
            image_path = str(absolute.resolve().relative_to(ml_root.resolve()))
        except ValueError:
            image_path = absolute.resolve().as_uri()
        output.append(
            _record(
                image_path,
                canonical_identification(container="other", weighted_names=weighted.items()),
                record_id=f"segmentation:{image_id}",
                group_id=f"segmentation:{image_id}",
            )
        )
    return output


def foodseg_records(
    root: Path,
    split: str,
    *,
    ml_root: Path,
) -> list[dict[str, Any]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    categories = {
        index: name
        for index, name in enumerate(metadata["categories"])
        if index != 0
    }
    split_root = root / split
    output = []
    for mask_path in sorted((split_root / "masks").glob("*.png")):
        image_path = split_root / "images" / f"{mask_path.stem}.jpg"
        if not image_path.is_file():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"))
        class_ids, pixel_counts = np.unique(mask, return_counts=True)
        weighted = [
            (categories[int(class_id)], float(count))
            for class_id, count in zip(class_ids, pixel_counts)
            if int(class_id) in categories and count > 0
        ]
        if not weighted:
            continue
        try:
            relative_image = str(image_path.resolve().relative_to(ml_root.resolve()))
        except ValueError:
            relative_image = image_path.resolve().as_uri()
        output.append(
            _record(
                relative_image,
                canonical_identification(
                    container="other",
                    weighted_names=weighted,
                ),
                record_id=f"foodseg103:{split}:{mask_path.stem}",
                group_id=f"foodseg103:{split}:{mask_path.stem}",
            )
        )
    return output


def negative_records(path: Path, *, cap: int | None) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    rows.sort(
        key=lambda row: hashlib.sha256(str(row["images"][0]).encode()).hexdigest()
    )
    if cap is not None:
        rows = rows[:cap]
    ground_truth = json.loads(NOT_FOOD_COMPLETION)
    return [
        _record(
            row["images"][0],
            NOT_FOOD_COMPLETION,
            record_id=f"negative:{Path(row['images'][0]).stem}",
            group_id=f"negative:{Path(row['images'][0]).stem}",
            evaluation_ground_truth=ground_truth,
        )
        for row in rows
    ]


def frb_teacher_records(
    path: Path,
    *,
    split: str,
    ml_root: Path,
    validation_fraction: float,
    seed: int,
    allowed_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build visible-name supervision from the reviewed FRB teacher pilot.

    FRB has no measured mass. Every accepted visible item therefore receives
    one relative unit: these records teach recognition and schema, not portion
    ratios. A deterministic image-level split prevents pilot leakage.
    """
    if split not in {"train", "valid"}:
        return []
    if not 0 < validation_fraction < 1:
        raise ValueError("FRB validation fraction must be between 0 and 1")
    candidates = []
    for row in _read_jsonl(path):
        teacher = row.get("teacher") or {}
        enrichment = teacher.get("accepted_enrichment") or {}
        visible = enrichment.get("visible_foods") or []
        weighted = [
            (str(item.get("name", "")), 1.0)
            for item in visible
            if normalize_name(str(item.get("name", "")))
        ]
        if not weighted:
            continue
        if allowed_names is not None and any(
            normalize_name(name) not in allowed_names for name, _ in weighted
        ):
            continue
        source = row.get("source") or {}
        image_id = int(source["image_id"])
        digest = hashlib.sha256(f"{seed}:frb:{image_id}".encode()).hexdigest()
        candidates.append((digest, image_id, row, weighted, enrichment))

    candidates.sort(key=lambda item: item[0])
    validation_count = max(1, round(len(candidates) * validation_fraction))
    selected = (
        candidates[:validation_count]
        if split == "valid"
        else candidates[validation_count:]
    )
    container_map = {
        "plate": "plate",
        "bowl": "bowl",
        "cup_or_glass": "cup",
        "tray": "tray",
        "wrapper_or_package": "packaging",
        "none_or_unclear": "other",
    }
    output = []
    for _, image_id, row, weighted, enrichment in selected:
        absolute = ml_root / str(row["image"]["path"])
        if not absolute.is_file():
            raise FileNotFoundError(f"missing FRB pilot image: {absolute}")
        image_path = str(absolute.resolve().relative_to(ml_root.resolve()))
        container = container_map.get(str(enrichment.get("container")), "other")
        output.append(
            _record(
                image_path,
                canonical_identification(
                    container=container,
                    weighted_names=weighted,
                ),
                record_id=f"frb-v2.0:{image_id}",
                group_id=f"frb-v2.0:{image_id}",
            )
        )
    return output


def frb_rung12_names(path: Path, database: Path) -> set[str]:
    names = set()
    for row in _read_jsonl(path):
        enrichment = ((row.get("teacher") or {}).get("accepted_enrichment") or {})
        for item in enrichment.get("visible_foods") or []:
            normalized = normalize_name(str(item.get("name", "")))
            if normalized:
                names.add(normalized)
    resolver = SQLiteNutritionResolver(database, fuzzy_threshold=0.84)
    try:
        return {
            name
            for name in names
            if resolver.resolve(name).rung in {"exact_alias", "fuzzy"}
        }
    finally:
        resolver.close()


def _ids_from_jsonl(path: Path) -> set[int]:
    ids = set()
    for row in _read_jsonl(path):
        source = row.get("source", row)
        if "image_id" in source:
            ids.add(int(source["image_id"]))
    return ids


def write_split(path: Path, records: Iterable[dict[str, Any]]) -> int:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def deterministically_interleave(
    records: Iterable[dict[str, Any]],
    *,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows = list(records)
    random.Random(f"{seed}:{split}").shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n5k-dir", type=Path, default=Path("finetune_data_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("finetune_data_id_v1"))
    parser.add_argument("--seg-train-annotations", type=Path)
    parser.add_argument("--seg-train-images", type=Path)
    parser.add_argument("--seg-valid-annotations", type=Path)
    parser.add_argument("--seg-valid-images", type=Path)
    parser.add_argument(
        "--seg-include-ids",
        type=Path,
        help="Optional JSONL subset manifest (for a reproducible pilot/mix ablation).",
    )
    parser.add_argument(
        "--foodseg-root",
        type=Path,
        help="Materialized FoodSeg103 root from download_foodseg103.py.",
    )
    parser.add_argument(
        "--frb-teacher-manifest",
        type=Path,
        help=(
            "Optional FRB v2.0 teacher-enriched pilot. Accepted visible names "
            "receive equal relative units; FRB never supplies mass supervision."
        ),
    )
    parser.add_argument("--frb-validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--frb-require-fully-resolved",
        action="store_true",
        help=(
            "Keep only FRB records whose complete visible truth resolves at "
            "rungs 1-2. Requires --resolver-database."
        ),
    )
    parser.add_argument("--resolver-database", type=Path)
    parser.add_argument(
        "--max-image-edge",
        type=int,
        help=(
            "Materialize resized copies only for images above this edge length "
            "and rewrite the manifests. Use 1024 for the 2,048-token trainer."
        ),
    )
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        help="Defaults to <output-dir>/cached-images when --max-image-edge is set.",
    )
    parser.add_argument("--negatives-dir", type=Path, default=Path("negatives"))
    parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Deterministic tiny overfit/probe subset; omit for production data.",
    )
    parser.add_argument("--subset-seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.frb_require_fully_resolved and not args.resolver_database:
        parser.error("--frb-require-fully-resolved requires --resolver-database")
    frb_allowed_names = (
        frb_rung12_names(args.frb_teacher_manifest, args.resolver_database)
        if args.frb_teacher_manifest and args.frb_require_fully_resolved
        else None
    )
    include_ids = _ids_from_jsonl(args.seg_include_ids) if args.seg_include_ids else None
    image_cache_dir = args.image_cache_dir or (args.output_dir / "cached-images")
    counts: dict[str, int] = {}
    image_cache_results: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        records = nutrition5k_records(args.n5k_dir / f"{split}.jsonl")
        annotation_path = getattr(args, f"seg_{split}_annotations", None)
        image_root = getattr(args, f"seg_{split}_images", None)
        if annotation_path and image_root:
            records.extend(
                segmentation_records(
                    annotation_path,
                    image_root,
                    ml_root=args.ml_root,
                    include_image_ids=include_ids,
                )
            )
        if args.foodseg_root and split in {"train", "valid"}:
            records.extend(
                foodseg_records(
                    args.foodseg_root,
                    "train" if split == "train" else "validation",
                    ml_root=args.ml_root,
                )
            )
        if args.frb_teacher_manifest and split in {"train", "valid"}:
            records.extend(
                frb_teacher_records(
                    args.frb_teacher_manifest,
                    split=split,
                    ml_root=args.ml_root,
                    validation_fraction=args.frb_validation_fraction,
                    seed=args.subset_seed,
                    allowed_names=frb_allowed_names,
                )
            )
        negative_path = args.negatives_dir / f"{split}.jsonl"
        if negative_path.is_file():
            records.extend(
                negative_records(negative_path, cap=100 if split == "train" else None)
            )
        # The source arms are accumulated independently above. Shuffle every
        # split deterministically so an epoch is genuinely interleaved rather
        # than presenting all N5K records before all FoodSeg records.
        records = deterministically_interleave(
            records,
            split=split,
            seed=args.subset_seed,
        )
        if args.limit_per_split is not None:
            records = records[: args.limit_per_split]
        split_path = args.output_dir / f"{split}.jsonl"
        counts[split] = write_split(split_path, records)
        if args.max_image_edge is not None:
            image_cache_results[split] = cache_images(
                split_path,
                split_path,
                cache_dir=image_cache_dir,
                ml_root=args.ml_root,
                max_edge=args.max_image_edge,
                reuse_small_originals=True,
            )
    metadata = {
        "schema_version": 2,
        "prompt": IDENTIFY_PROMPT,
        "counts": counts,
        "portion_unit_range": [1, 20],
        "share_normalization_step": 5,
        "visible_component_policy": {
            "minimum_original_mass_fraction": 0.02,
            "maximum_items": 8,
            "hidden_recipe_components_excluded": True,
        },
        "negative_train_cap": 100,
        "container_policies": {
            "nutrition5k": {
                "label": "tray",
                "use": "trained_and_scored",
            },
            "foodseg103": {
                "label": "other",
                "use": "diagnostic_only_for_mixed_arm",
                "reason": "FoodSeg103 has no container annotations",
            },
            "frb-v2.0-teacher-pilot": {
                "label": "teacher_visible_container_mapped_to_contract",
                "use": "recognition_and_schema_only",
                "portion_policy": "equal_relative_units_no_mass_claim",
                "license": "CC BY 4.0",
            },
            "negative": {
                "label": "other",
                "use": "structural_not_food_contract",
            },
        },
        "limit_per_split": args.limit_per_split,
        "subset_seed": args.subset_seed,
        "frb_resolution_gate": {
            "require_fully_resolved_records": args.frb_require_fully_resolved,
            "allowed_rung_1_2_names": (
                len(frb_allowed_names) if frb_allowed_names is not None else None
            ),
            "resolver_database": (
                str(args.resolver_database)
                if args.resolver_database is not None
                else None
            ),
            "resolver_database_sha256": (
                hashlib.sha256(args.resolver_database.read_bytes()).hexdigest()
                if args.resolver_database is not None
                else None
            ),
        },
        "image_token_safety": {
            "max_image_edge": args.max_image_edge,
            "cache_dir": (
                str(image_cache_dir) if args.max_image_edge is not None else None
            ),
            "per_split": image_cache_results,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
