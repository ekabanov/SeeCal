"""Freeze the name taxonomy used by HMR/IDR evaluation.

The output is a reviewed, versioned scoring input.  It snapshots mappings from
the pre-candidate resolver state and is never regenerated as a side effect of
runtime alias work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from factored_pipeline.eval_taxonomy import normalize_eval_name_v1
from factored_pipeline.resolver import SQLiteNutritionResolver


NAME_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("processed_meat", ("bacon", "sausage", "salami", "pepperoni", "hot dog", "ham")),
    ("poultry", ("chicken", "turkey", "duck", "goose")),
    ("fish", ("fish", "salmon", "tuna", "cod", "tilapia", "trout", "sardine")),
    ("shellfish", ("shrimp", "prawn", "crab", "lobster", "scallop", "mussel", "oyster")),
    ("red_meat", ("beef", "steak", "veal", "lamb", "mutton")),
    ("pork", ("pork",)),
    ("egg", ("egg",)),
    (
        "vegetable",
        (
            "salad", "green", "lettuce", "spinach", "kale", "arugula", "cabbage",
            "broccoli", "cauliflower", "carrot", "tomato", "cucumber", "squash",
            "zucchini", "eggplant", "pepper", "onion", "mushroom", "bean sprout",
        ),
    ),
    (
        "fruit",
        (
            "berry", "berries", "grape", "apple", "banana", "orange", "melon",
            "pineapple", "mango", "peach", "pear", "plum", "cherry", "fruit",
        ),
    ),
    ("legume", ("bean", "lentil", "chickpea", "tofu", "tempeh")),
    ("grain_starch", ("rice", "pasta", "noodle", "bread", "quinoa", "potato", "yam", "grain", "oat")),
    ("dairy", ("cheese", "milk", "yogurt", "cream")),
    ("fat", ("oil", "butter", "margarine")),
    ("condiment", ("dressing", "sauce", "vinegar", "mustard", "ketchup", "mayonnaise")),
    ("dessert", ("cake", "cookie", "ice cream", "waffle", "pancake", "candy", "chocolate")),
    ("beverage", ("juice", "coffee", "tea", "soda", "drink", "beverage")),
)

CATEGORY_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("sausages and luncheon meats", "processed_meat"),
    ("poultry products", "poultry"),
    ("finfish and shellfish products", "fish"),
    ("beef products", "red_meat"),
    ("lamb, veal, and game products", "red_meat"),
    ("pork products", "pork"),
    ("vegetables and vegetable products", "vegetable"),
    ("fruits and fruit juices", "fruit"),
    ("legumes and legume products", "legume"),
    ("cereal grains and pasta", "grain_starch"),
    ("baked products", "grain_starch"),
    ("dairy and egg products", "dairy"),
    ("fats and oils", "fat"),
    ("spices and herbs", "condiment"),
    ("sweets", "dessert"),
    ("beverages", "beverage"),
)


def coarse_family_v2(name: str, category: str | None) -> str | None:
    """Map a frozen name/category pair to a reviewed visual food family."""
    normalized = normalize_eval_name_v1(name)
    padded = f" {normalized} "
    for family, phrases in NAME_FAMILY_RULES:
        for phrase in phrases:
            normalized_phrase = normalize_eval_name_v1(phrase)
            if (
                normalized == normalized_phrase
                or f" {normalized_phrase} " in padded
            ):
                return family
    normalized_category = normalize_eval_name_v1(category or "")
    for source, family in CATEGORY_FAMILY_RULES:
        if normalize_eval_name_v1(source) == normalized_category:
            return family
    return None


def _items(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield from payload.get("items") or []


def names_from_json(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("paired_results") or payload.get("paired") or []
    names: set[str] = set()
    for row in rows:
        for side in ("ground_truth", "prediction"):
            for item in _items(row.get(side)):
                name = str(item.get("name", "")).strip()
                if name:
                    names.add(name)
    return names


def names_from_jsonl(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        payloads: list[dict[str, Any]] = []
        if isinstance(row.get("evaluation_ground_truth"), dict):
            payloads.append(row["evaluation_ground_truth"])
        if "messages" in row:
            content = row["messages"][-1]["content"]
            text = content[0]["text"] if isinstance(content, list) else content
            payloads.append(json.loads(text))
        else:
            payloads.append(row)
        for payload in payloads:
            for item in _items(payload):
                name = str(item.get("name", "")).strip()
                if name:
                    names.add(name)
    return names


def names_from_taxonomy(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("entries") or {})


def build_taxonomy(
    *,
    database: Path,
    json_paths: list[Path],
    jsonl_paths: list[Path],
    version: str,
    base_taxonomy_paths: list[Path] | None = None,
    override_paths: list[Path] | None = None,
) -> dict[str, Any]:
    names: set[str] = set()
    for path in json_paths:
        names |= names_from_json(path)
    for path in jsonl_paths:
        names |= names_from_jsonl(path)
    for path in base_taxonomy_paths or []:
        names |= names_from_taxonomy(path)

    connection = sqlite3.connect(database)
    try:
        names |= {
            row[0]
            for row in connection.execute("SELECT normalized_alias FROM aliases")
        }
        names |= {
            row[0]
            for row in connection.execute("SELECT normalized_name FROM foods")
        }
    finally:
        connection.close()

    resolver = SQLiteNutritionResolver(database)
    entries: dict[str, dict[str, Any]] = {}
    try:
        for name in sorted(names, key=normalize_eval_name_v1):
            normalized = normalize_eval_name_v1(name)
            if not normalized:
                continue
            result = resolver.resolve(name)
            category = result.profile.category if result.profile else None
            entries[normalized] = {
                "fdc_id": result.profile.fdc_id if result.profile else None,
                "category": category,
                "family": (
                    coarse_family_v2(name, category)
                    if not version.endswith("_v1")
                    else None
                ),
            }
    finally:
        resolver.close()
    override_count = 0
    for path in override_paths or []:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                normalized = normalize_eval_name_v1(row["name"])
                if not normalized:
                    raise ValueError(f"empty taxonomy override name in {path}")
                entries[normalized] = {
                    "fdc_id": int(row["fdc_id"]) if row.get("fdc_id") else None,
                    "category": row.get("category") or None,
                    "family": row.get("family") or None,
                }
                override_count += 1
    return {
        "schema_version": 1,
        "taxonomy_version": version,
        "matching": {
            "soft_threshold": 0.72,
            "hard_threshold": 0.42,
        },
        "entries": entries,
        "provenance": {
            "reviewed_override_count": override_count,
            "reviewed_overrides": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in override_paths or []
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--json", type=Path, action="append", default=[])
    parser.add_argument("--jsonl", type=Path, action="append", default=[])
    parser.add_argument("--base-taxonomy", type=Path, action="append", default=[])
    parser.add_argument("--reviewed-overrides", type=Path, action="append", default=[])
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            f"{args.output} already exists; create a new taxonomy version instead"
        )
    payload = build_taxonomy(
        database=args.database,
        json_paths=args.json,
        jsonl_paths=args.jsonl,
        base_taxonomy_paths=args.base_taxonomy,
        override_paths=args.reviewed_overrides,
        version=args.version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "taxonomy_version": args.version,
                "entries": len(payload["entries"]),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
