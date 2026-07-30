"""Build the NutritionVerse-Real OOD evaluation JSONL.

Run from ``ml/`` after ``download_nutritionverse_real.sh``. The model
completion remains the frozen IDENTIFY schema, while the top-level evaluation
ground truth preserves measured grams and source nutrition for gram-weighted
scoring and later end-to-end factored evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

from factored_pipeline.contract import (
    IDENTIFY_PROMPT,
    canonical_identification,
    identification_to_shares,
    normalize_name,
)
from factored_pipeline.visible_labels import visible_component_weights
from prepare_finetune import SYSTEM_PROMPT, USER_PROMPT

DISH_PATTERN = re.compile(r"dish_(\d+)_")


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else 0.0


def _source_items(row: dict[str, str]) -> list[dict[str, Any]]:
    items = []
    for index in range(1, 8):
        raw_name = row.get(f"food_item_type_{index}") or ""
        grams = _number(row, f"food_weight_g_{index}")
        if not raw_name or grams <= 0:
            continue
        items.append(
            {
                "name": raw_name.replace("-", " "),
                "estimated_grams": grams,
                "calories": _number(row, f"calories(kCal)_{index}"),
                "protein_g": _number(row, f"protein(g)_{index}"),
                "fat_g": _number(row, f"fat(g)_{index}"),
                "carbs_g": _number(row, f"carbohydrates(g)_{index}"),
            }
        )
    return items


def build_records(
    *,
    metadata_csv: Path,
    image_root: Path,
    ml_root: Path,
) -> list[dict[str, Any]]:
    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        metadata = {
            int(row["dish_id"]): row for row in csv.DictReader(handle)
        }
    records = []
    for image_path in sorted(image_root.glob("*.jpg")):
        match = DISH_PATTERN.match(image_path.name)
        if not match:
            continue
        dish_id = int(match.group(1))
        if dish_id not in metadata:
            raise ValueError(f"image has no NutritionVerse metadata: {image_path.name}")
        source = metadata[dish_id]
        source_items = _source_items(source)
        visible_weights = visible_component_weights(
            [
                (str(item["name"]), float(item["estimated_grams"]))
                for item in source_items
            ]
        )
        completion = canonical_identification(
            container="other",
            weighted_names=visible_weights,
        )
        identification = identification_to_shares(json.loads(completion))
        items_by_name = {
            normalize_name(str(item["name"])): item for item in source_items
        }
        evaluation_items = [
            {
                **items_by_name[normalize_name(str(item["name"]))],
                "portion_units": item["portion_units"],
                "share_pct": item["share_pct"],
            }
            for item in identification["items"]
        ]
        try:
            relative_image = str(image_path.resolve().relative_to(ml_root.resolve()))
        except ValueError:
            relative_image = image_path.resolve().as_uri()
        records.append(
            {
                "id": f"nv-real:{image_path.stem}",
                "group_id": f"nv-real:dish_{dish_id}",
                "images": [relative_image],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": relative_image},
                            {"type": "text", "text": IDENTIFY_PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": completion}],
                    },
                ],
                "evaluation_ground_truth": {
                    "not_food": False,
                    "container": "other",
                    "items": evaluation_items,
                    "total_mass_g": _number(source, "total_food_weight"),
                    "total_calories": _number(source, "total_calories"),
                    "protein_g": _number(source, "total_protein"),
                    "fat_g": _number(source, "total_fats"),
                    "carbs_g": _number(source, "total_carbohydrates"),
                },
            }
        )
    return records


def monolith_records(
    identify_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT
    for record in identify_records:
        truth = record["evaluation_ground_truth"]
        completion = {
            "total_calories": truth["total_calories"],
            "protein_g": truth["protein_g"],
            "fat_g": truth["fat_g"],
            "carbs_g": truth["carbs_g"],
            "items": [
                {
                    key: item[key]
                    for key in (
                        "name",
                        "estimated_grams",
                        "calories",
                        "protein_g",
                        "fat_g",
                        "carbs_g",
                    )
                }
                for item in truth["items"]
            ],
        }
        image = record["images"][0]
        output.append(
            {
                "id": record["id"],
                "group_id": record["group_id"],
                "images": [image],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    completion,
                                    separators=(",", ":"),
                                ),
                            }
                        ],
                    },
                ],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/nutritionverse-real/extracted"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/nutritionverse-real/eval.jsonl"),
    )
    parser.add_argument(
        "--monolith-output",
        type=Path,
        default=Path("datasets/nutritionverse-real/monolith-eval.jsonl"),
    )
    parser.add_argument(
        "--scale-output",
        type=Path,
        default=Path("datasets/nutritionverse-real/scale-eval.jsonl"),
    )
    parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    records = build_records(
        metadata_csv=args.dataset_root / "nutritionverse_dish_metadata3.csv",
        image_root=(
            args.dataset_root
            / "nutritionverse-manual"
            / "nutritionverse-manual"
            / "images"
        ),
        ml_root=args.ml_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    monolith = monolith_records(records)
    args.monolith_output.parent.mkdir(parents=True, exist_ok=True)
    args.monolith_output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in monolith),
        encoding="utf-8",
    )
    args.scale_output.parent.mkdir(parents=True, exist_ok=True)
    args.scale_output.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": record["id"],
                    "group_id": record["group_id"],
                    "split": "test",
                    "image_path": record["images"][0],
                    "total_mass_g": record["evaluation_ground_truth"][
                        "total_mass_g"
                    ],
                    "source": "nutritionverse-real-v2",
                },
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "images": len(records),
                "dishes": len(
                    {
                        DISH_PATTERN.match(Path(row["images"][0]).name).group(1)
                        for row in records
                    }
                ),
                "output": str(args.output),
                "monolith_output": str(args.monolith_output),
                "scale_output": str(args.scale_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
