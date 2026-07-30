"""Build SeeCal's pruned, deterministic nutrition SQLite database.

Input directories are extracted USDA FoodData Central CSV archives. Pass the
Foundation, SR Legacy, and FNDDS directories together; duplicate supporting
tables are harmless. Run from ``ml/``:

  .venv/bin/python make_fdc_db.py \
    --source datasets/fdc/foundation \
    --source datasets/fdc/sr-legacy \
    --source datasets/fdc/fndds \
    --output datasets/fdc/seecal-nutrition.sqlite

The source energy value is retained for provenance. ``kcal_per_100g`` is
derived with 4/9/4 Atwater arithmetic from the looked-up macros, making AIR
zero by construction as required by the factored-pipeline contract.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Iterable

from factored_pipeline.contract import normalize_name

MACRO_IDS = {"protein": 1003, "fat": 1004, "carbs": 1005}
ENERGY_IDS = (1008, 2047, 2048)
ALLOWED_DATA_TYPES = {
    "foundation",
    "foundation_food",
    "sr legacy",
    "sr_legacy_food",
    "survey (fndds)",
    "survey_fndds_food",
}


@dataclass(frozen=True)
class FoodRow:
    fdc_id: int
    name: str
    data_type: str
    category_id: int | None


def _find(source: Path, filename: str) -> Path | None:
    direct = source / filename
    if direct.is_file():
        return direct
    matches = sorted(source.rglob(filename))
    return matches[0] if matches else None


def _rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    try:
        parsed = float(value) if value not in (None, "") else None
        return parsed if parsed is not None and math.isfinite(parsed) else None
    except ValueError:
        return None


def _type_allowed(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in ALLOWED_DATA_TYPES or any(
        token in normalized for token in ("foundation", "sr legacy", "survey")
    )


def _load_foods(sources: list[Path]) -> dict[int, FoodRow]:
    foods: dict[int, FoodRow] = {}
    for source in sources:
        path = _find(source, "food.csv")
        if not path:
            raise FileNotFoundError(f"food.csv not found under {source}")
        for row in _rows(path):
            data_type = row.get("data_type", "")
            if not _type_allowed(data_type):
                continue
            fdc_id = _int(row.get("fdc_id"))
            name = (row.get("description") or "").strip()
            if fdc_id is None or not normalize_name(name):
                continue
            foods[fdc_id] = FoodRow(
                fdc_id,
                name,
                data_type,
                _int(row.get("food_category_id")),
            )
    return foods


def _load_categories(
    sources: list[Path],
) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    categories: dict[int, str] = {}
    survey_categories_by_food: dict[int, str] = {}
    survey_category_names: dict[int, str] = {}
    for source in sources:
        category_path = _find(source, "food_category.csv")
        if category_path:
            for row in _rows(category_path):
                category_id = _int(row.get("id") or row.get("food_category_id"))
                description = (row.get("description") or "").strip()
                if category_id is not None and description:
                    categories[category_id] = description
        survey_path = _find(source, "wweia_food_category.csv")
        if survey_path:
            for row in _rows(survey_path):
                fdc_id = _int(row.get("fdc_id"))
                category_id = _int(row.get("wweia_food_category"))
                description = (
                    row.get("wweia_food_category_description")
                    or row.get("description")
                    or ""
                ).strip()
                if fdc_id is not None and description:
                    survey_categories_by_food[fdc_id] = description
                if category_id is not None and description:
                    survey_category_names[category_id] = description
    return categories, survey_categories_by_food, survey_category_names


def _load_nutrients(
    sources: list[Path],
    food_ids: set[int],
) -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, tuple[int, float]]] = {}
    for source in sources:
        nutrient_path = _find(source, "nutrient.csv")
        if not nutrient_path:
            continue
        # FNDDS 2021–2023 stores legacy nutrient numbers (203/204/205/208)
        # in food_nutrient.nutrient_id even though nutrient.csv also exposes
        # the modern IDs (1003/1004/1005/1008). Accept both columns by mapping
        # every source-local identifier to the canonical field first.
        source_fields: dict[int, tuple[str, int, str]] = {}
        for row in _rows(nutrient_path):
            canonical_id = _int(row.get("id"))
            nutrient_number = _int(row.get("nutrient_nbr"))
            if canonical_id in MACRO_IDS.values():
                field = next(
                    name
                    for name, expected_id in MACRO_IDS.items()
                    if canonical_id == expected_id
                )
                priority = 0
            elif canonical_id in ENERGY_IDS:
                field = "energy"
                priority = ENERGY_IDS.index(canonical_id)
            else:
                continue
            unit = (row.get("unit_name") or "").strip().lower()
            for identifier in (canonical_id, nutrient_number):
                if identifier is not None:
                    source_fields[identifier] = (field, priority, unit)
        path = _find(source, "food_nutrient.csv")
        if not path:
            raise FileNotFoundError(f"food_nutrient.csv not found under {source}")
        for row in _rows(path):
            fdc_id = _int(row.get("fdc_id"))
            nutrient_id = _int(row.get("nutrient_id"))
            amount = _float(row.get("amount"))
            mapping = source_fields.get(nutrient_id or -1)
            if fdc_id not in food_ids or mapping is None or amount is None:
                continue
            field, priority, unit = mapping
            # Energy entries must be kcal. IDs are stable, but checking the
            # supporting table catches malformed/custom fixtures.
            if field == "energy":
                if unit not in {"kcal", "kilocalorie"}:
                    continue
            prior = values.setdefault(fdc_id, {}).get(field)
            if prior is None or priority < prior[0]:
                values[fdc_id][field] = (priority, amount)
    return {
        fdc_id: {field: value for field, (_, value) in fields.items()}
        for fdc_id, fields in values.items()
    }


def _load_portions(sources: list[Path], food_ids: set[int]) -> dict[int, float]:
    """Choose one observed serving kind per food; never average unlike kinds.

    FoodData Central commonly lists both a single piece/slice and a whole
    package or dish for the same food.  A median between those rows is not an
    actual portion (for example, 112 g pizza slice and 897 g whole pizza used
    to become a fictitious 504.5 g serving).  Prefer the row backed by the most
    observations, then the source's stable sequence number.
    """
    values: dict[int, list[tuple[int, int, float]]] = {}
    for source in sources:
        path = _find(source, "food_portion.csv")
        if not path:
            continue
        for row in _rows(path):
            fdc_id = _int(row.get("fdc_id"))
            grams = _float(row.get("gram_weight"))
            if fdc_id in food_ids and grams is not None and 0 < grams <= 2000:
                data_points = _int(row.get("data_points")) or 0
                sequence = _int(row.get("seq_num")) or 2**31 - 1
                values.setdefault(fdc_id, []).append(
                    (data_points, sequence, grams)
                )
    return {
        fdc_id: sorted(
            candidates,
            key=lambda candidate: (-candidate[0], candidate[1], candidate[2]),
        )[0][2]
        for fdc_id, candidates in values.items()
    }


def _profile_is_sane(protein: float, fat: float, carbs: float, kcal: float) -> bool:
    values = (protein, fat, carbs, kcal)
    return (
        all(math.isfinite(value) and value >= 0 for value in values)
        and all(value <= 100 for value in (protein, fat, carbs))
        and kcal <= 900
    )


def _singular_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _alias_variants(name: str) -> dict[str, int]:
    normalized = normalize_name(name)
    variants = {normalized: 10}
    before_comma = normalize_name(name.split(",", 1)[0])
    if len(before_comma) >= 3:
        variants[before_comma] = 5
    without_parenthetical = normalize_name(name.split("(", 1)[0])
    if len(without_parenthetical) >= 3:
        variants[without_parenthetical] = max(variants.get(without_parenthetical, 0), 5)
    for variant, priority in list(variants.items()):
        singular = " ".join(_singular_token(token) for token in variant.split())
        if singular and singular != variant:
            variants[singular] = max(variants.get(singular, 0), priority - 1)
    return {variant: priority for variant, priority in variants.items() if variant}


SCHEMA = """
PRAGMA journal_mode = DELETE;
PRAGMA foreign_keys = ON;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE foods (
    fdc_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    category TEXT NOT NULL,
    data_type TEXT NOT NULL,
    kcal_per_100g REAL NOT NULL CHECK(kcal_per_100g BETWEEN 0 AND 900),
    source_kcal_per_100g REAL,
    protein_per_100g REAL NOT NULL CHECK(protein_per_100g BETWEEN 0 AND 100),
    fat_per_100g REAL NOT NULL CHECK(fat_per_100g BETWEEN 0 AND 100),
    carbs_per_100g REAL NOT NULL CHECK(carbs_per_100g BETWEEN 0 AND 100)
    ,typical_portion_g REAL
);
CREATE INDEX foods_normalized_name ON foods(normalized_name);
CREATE INDEX foods_category ON foods(category);
CREATE TABLE aliases (
    normalized_alias TEXT NOT NULL,
    fdc_id INTEGER NOT NULL REFERENCES foods(fdc_id),
    priority INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    PRIMARY KEY(normalized_alias, fdc_id)
);
CREATE INDEX aliases_lookup ON aliases(normalized_alias, priority DESC);
CREATE TABLE category_defaults (
    category TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    kcal_per_100g REAL NOT NULL,
    source_kcal_per_100g REAL,
    protein_per_100g REAL NOT NULL,
    fat_per_100g REAL NOT NULL,
    carbs_per_100g REAL NOT NULL,
    typical_portion_g REAL,
    data_type TEXT NOT NULL
);
"""


def build_database(sources: list[Path], output: Path) -> dict[str, object]:
    foods = _load_foods(sources)
    (
        category_names,
        survey_categories_by_food,
        survey_category_names,
    ) = _load_categories(sources)
    nutrients = _load_nutrients(sources, set(foods))
    portions = _load_portions(sources, set(foods))
    selected: list[tuple[object, ...]] = []
    skipped_missing = skipped_unsafe = 0
    for food in foods.values():
        values = nutrients.get(food.fdc_id, {})
        if not all(field in values for field in ("protein", "fat", "carbs")):
            skipped_missing += 1
            continue
        protein, fat, carbs = values["protein"], values["fat"], values["carbs"]
        canonical_kcal = 4 * protein + 9 * fat + 4 * carbs
        if not _profile_is_sane(protein, fat, carbs, canonical_kcal):
            skipped_unsafe += 1
            continue
        category = (
            survey_categories_by_food.get(food.fdc_id)
            or survey_category_names.get(food.category_id or -1)
            or category_names.get(food.category_id or -1)
            or "Uncategorized"
        )
        selected.append(
            (
                food.fdc_id,
                food.name,
                normalize_name(food.name),
                category,
                food.data_type,
                canonical_kcal,
                values.get("energy"),
                protein,
                fat,
                carbs,
                portions.get(food.fdc_id),
            )
        )

    if not selected:
        raise RuntimeError("no complete, sane food profiles were found")
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO foods VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            selected,
        )
        alias_candidates: list[tuple[str, int, int, str]] = []
        for row in selected:
            for alias, priority in _alias_variants(str(row[1])).items():
                alias_candidates.append(
                    (alias, int(row[0]), priority, "fdc_description")
                )
        # Shortened/singular aliases are only safe when they identify one FDC
        # row. Full descriptions retain their deterministic fdc_id tie-break.
        targets: dict[str, set[int]] = {}
        for alias, fdc_id, _, _ in alias_candidates:
            targets.setdefault(alias, set()).add(fdc_id)
        aliases = [
            row
            for row in alias_candidates
            if row[2] == 10 or len(targets[row[0]]) == 1
        ]
        connection.executemany(
            "INSERT OR IGNORE INTO aliases VALUES (?,?,?,?)",
            aliases,
        )
        by_category: dict[str, list[tuple[object, ...]]] = {}
        for row in selected:
            by_category.setdefault(str(row[3]), []).append(row)
        defaults = []
        for category, rows in sorted(by_category.items()):
            median_protein = statistics.median(float(row[7]) for row in rows)
            median_fat = statistics.median(float(row[8]) for row in rows)
            median_carbs = statistics.median(float(row[9]) for row in rows)
            defaults.append(
                (
                    category,
                    f"Typical {category}",
                    normalize_name(f"Typical {category}"),
                    4 * median_protein + 9 * median_fat + 4 * median_carbs,
                    statistics.median(
                        float(row[6]) for row in rows if row[6] is not None
                    )
                    if any(row[6] is not None for row in rows)
                    else None,
                    median_protein,
                    median_fat,
                    median_carbs,
                    statistics.median(
                        float(row[10]) for row in rows if row[10] is not None
                    )
                    if any(row[10] is not None for row in rows)
                    else None,
                    "category_median",
                )
            )
        connection.executemany(
            "INSERT INTO category_defaults VALUES (?,?,?,?,?,?,?,?,?,?)",
            defaults,
        )
        metadata = {
            "schema_version": "1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": json.dumps([str(source) for source in sources], sort_keys=True),
            "energy_policy": "atwater_4_9_4_from_fdc_macros",
            "foods": str(len(selected)),
            "categories": str(len(defaults)),
        }
        connection.executemany("INSERT INTO metadata VALUES (?,?)", metadata.items())
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    return {
        "output": str(output),
        "foods": len(selected),
        "categories": len(defaults),
        "aliases": len(aliases),
        "skipped_missing_macros": skipped_missing,
        "skipped_unsafe": skipped_unsafe,
        "bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_database(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
