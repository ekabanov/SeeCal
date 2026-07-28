"""Build leakage-safe specialist manifests from local Nutrition5K and FRB data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .constants import COOKING_STATES, CONTAINERS, FRB_CLASS_COUNT, OCCLUSION_STATES


class ManifestError(RuntimeError):
    """A source manifest violates the specialist data contract."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assistant_payload(record: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(record["messages"][-1]["content"][0]["text"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ManifestError("invalid fine-tuning completion") from exc


def _dish_id(record: dict[str, Any]) -> str:
    try:
        return Path(record["images"][0]).parent.name
    except (KeyError, IndexError, TypeError) as exc:
        raise ManifestError("fine-tuning record has no image") from exc


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _nutrition_rows(
    ml_root: Path,
    split: str,
    *,
    include_sides: bool,
) -> Iterable[dict[str, Any]]:
    source = ml_root / "finetune_data_v2" / f"{split}.jsonl"
    for record in _read_jsonl(source):
        payload = _assistant_payload(record)
        if payload.get("not_food") is True:
            raise ManifestError(f"food split contains refusal in {source}")
        dish_id = _dish_id(record)
        items = payload.get("items", [])
        targets = {
            "mass_g": sum(float(item["estimated_grams"]) for item in items),
            "calories": float(payload["total_calories"]),
            "protein_g": float(payload["protein_g"]),
            "fat_g": float(payload["fat_g"]),
            "carbs_g": float(payload["carbs_g"]),
        }
        image_paths = [("overhead", record["images"][0])]
        if include_sides:
            for view in ("side_a", "side_c"):
                path = f"dataset_clean/{dish_id}/{view}.jpg"
                if (ml_root / path).is_file():
                    image_paths.append((view, path))
        for view, image_path in image_paths:
            yield {
                "id": f"nutrition5k:{dish_id}:{view}",
                "split": split,
                "source": "nutrition5k",
                "group_id": dish_id,
                "view": view,
                "image_path": image_path,
                "targets": {
                    "numeric": targets,
                    "food": 1.0,
                },
                "loss_mask": {
                    "numeric": True,
                    "food": True,
                    "frb_classes": False,
                    "teacher_attributes": False,
                },
            }


def _negative_rows(ml_root: Path, split: str, cap: int | None) -> list[dict[str, Any]]:
    path = ml_root / "negatives" / f"{split}.jsonl"
    if not path.is_file():
        return []
    records = _read_jsonl(path)
    if cap is not None:
        records = sorted(
            records,
            key=lambda row: _stable_key(str(row["images"][0]), 42),
        )[:cap]
    return [
        {
            "id": f"coco-negative:{Path(record['images'][0]).stem}",
            "split": split,
            "source": "negative",
            "group_id": Path(record["images"][0]).stem,
            "view": "natural",
            "image_path": record["images"][0],
            "targets": {"food": 0.0},
            "loss_mask": {
                "numeric": False,
                "food": True,
                "frb_classes": False,
                "teacher_attributes": False,
            },
        }
        for record in records
    ]


def select_frb_validation(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> set[int]:
    """Choose a deterministic validation subset covering every represented class."""
    if not 0 < count < len(records):
        raise ManifestError("FRB validation count must be between 1 and n-1")
    labels = {
        int(record["source"]["image_id"]): {
            int(label["category_id"]) for label in record["source_labels"]
        }
        for record in records
    }
    uncovered = set().union(*labels.values())
    chosen: set[int] = set()
    ordered_ids = sorted(labels, key=lambda value: _stable_key(str(value), seed))
    while uncovered:
        candidates = [value for value in ordered_ids if value not in chosen]
        best = max(candidates, key=lambda value: len(labels[value] & uncovered))
        gain = labels[best] & uncovered
        if not gain:
            raise ManifestError(f"cannot cover FRB classes: {sorted(uncovered)}")
        chosen.add(best)
        uncovered -= gain
    if len(chosen) > count:
        raise ManifestError(
            f"class coverage requires {len(chosen)} records, above requested {count}"
        )
    coverage_ids = sorted(
        chosen, key=lambda value: _stable_key(str(value), seed)
    )
    fill_ids = [value for value in ordered_ids if value not in chosen]
    return set((coverage_ids + fill_ids)[:count])


def _teacher_targets(record: dict[str, Any]) -> dict[str, Any] | None:
    enrichment = record.get("teacher", {}).get("accepted_enrichment")
    if not enrichment:
        return None
    cooking = sorted(
        {
            item["cooking_state"]
            for item in enrichment["visible_foods"]
            if item.get("cooking_state") in COOKING_STATES
        }
    )
    return {
        "container": CONTAINERS.index(enrichment["container"]),
        "cooking_states": [COOKING_STATES.index(value) for value in cooking],
        "mixed_dish": float(enrichment["mixed_dish"]),
        "occlusion": OCCLUSION_STATES.index(enrichment["occlusion"]),
    }


def _frb_rows(
    records: list[dict[str, Any]],
    *,
    validation_ids: set[int],
) -> Iterable[dict[str, Any]]:
    for record in records:
        image_id = int(record["source"]["image_id"])
        class_ids = sorted(
            {int(label["category_id"]) for label in record["source_labels"]}
        )
        if not class_ids or class_ids[-1] >= FRB_CLASS_COUNT:
            raise ManifestError(f"invalid FRB labels for image {image_id}")
        teacher = _teacher_targets(record)
        yield {
            "id": f"frb-v2.0:{image_id}",
            "split": "valid" if image_id in validation_ids else "train",
            "source": "frb",
            "group_id": f"frb:{image_id}",
            "view": "natural",
            "image_path": record["image"]["path"],
            "targets": {
                "food": 1.0,
                "frb_class_ids": class_ids,
                **({"teacher": teacher} if teacher else {}),
            },
            "loss_mask": {
                "numeric": False,
                "food": True,
                "frb_classes": True,
                "teacher_attributes": teacher is not None,
            },
            "provenance": {
                "release": record["source"]["release"],
                "license": record["source"]["license"],
                "archive_sha256": record["source"]["archive_sha256"],
                "teacher_model": record.get("teacher", {}).get("model"),
            },
        }


def build_manifests(
    *,
    ml_root: Path,
    frb_manifest: Path,
    output_dir: Path,
    frb_validation_count: int = 500,
    seed: int = 20260728,
) -> dict[str, int]:
    frb_records = _read_jsonl(frb_manifest)
    validation_ids = select_frb_validation(
        frb_records, count=frb_validation_count, seed=seed
    )
    rows = list(
        _frb_rows(frb_records, validation_ids=validation_ids)
    )
    for split in ("train", "valid", "test"):
        rows.extend(
            _nutrition_rows(
                ml_root,
                split,
                include_sides=split != "test",
            )
        )
    rows.extend(_negative_rows(ml_root, "train", cap=100))
    rows.extend(_negative_rows(ml_root, "valid", cap=None))
    rows.extend(_negative_rows(ml_root, "test", cap=None))

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_rows.sort(key=lambda row: row["id"])
        output = output_dir / f"{split}.jsonl"
        output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows),
            encoding="utf-8",
        )
        counts[split] = len(split_rows)
    summary = {
        "schema_version": 1,
        "seed": seed,
        "frb_validation_count": frb_validation_count,
        "counts": counts,
        "frb_source_manifest": str(frb_manifest),
        "frb_source_sha256": hashlib.sha256(frb_manifest.read_bytes()).hexdigest(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--frb-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frb-validation-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    counts = build_manifests(
        ml_root=args.ml_root.resolve(),
        frb_manifest=args.frb_manifest.resolve(),
        output_dir=args.out_dir.resolve(),
        frb_validation_count=args.frb_validation_count,
        seed=args.seed,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
