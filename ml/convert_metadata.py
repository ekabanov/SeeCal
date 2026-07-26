#!/usr/bin/env python3
"""
convert_metadata.py
--------------------
Converts the OFFICIAL Nutrition5k raw metadata CSVs (as fetched by
download_dataset.sh into Nutrition5K/metadata_raw/) into the tidy,
fixed-schema CSVs this repo's pipeline actually reads from Nutrition5K/:

  dish_nutrition_values.csv   dish_id,calories,mass,fat,carb,protein
  dish_ingredients.csv        dish_id,ingr_id,ingr_name,grams,calories,fat,carb,protein

Raw format (dish_metadata_cafe{1,2}.csv): headerless, ragged/variable-width
CSV. Each line is

  dish_id,total_cal,total_mass,total_fat,total_carb,total_protein,
    ingr_id,ingr_name,grams,cal,fat,carb,protein,   <- repeated once per ingredient
    ingr_id,ingr_name,grams,cal,fat,carb,protein, ...

`ingr_id` already arrives pre-formatted as "ingr_XXXXXXXXXX" (10-digit,
zero-padded) inside this file — no lookup needed to build dish_ingredients.csv.

IMPORTANT — historical cleaning rule this converter reproduces (verified
2026-07-26 against the live Nutrition5k GCS bucket; see
tests/test_convert_metadata.py's real-file integration test and CLAUDE.md):
the dish_nutrition_values.csv / dish_ingredients.csv already checked into
this machine's Nutrition5K/ directory were derived from
dish_metadata_cafe1.csv ONLY.

  - cafe1 has 4768 unique dish_ids; cafe2 has 238; ZERO overlap between them.
  - The existing tidy dish_nutrition_values.csv has exactly 4768 rows, whose
    dish_id set is byte-for-byte identical to cafe1's — every one of cafe2's
    238 dishes is absent, and every cafe1 dish is present.
  - Row order in the tidy files exactly matches cafe1's raw row order (no
    sorting was applied at any point).
  - dish_ingredients.csv similarly has exactly 27225 rows: precisely the
    ingredient rows cafe1's dishes expand to (cafe2 contributes 0).

This does not appear to be documented anywhere in the official Nutrition5k
repo or in this repo's own prior history — it looks like an undocumented
choice (or oversight) made whenever the tidy CSVs were first hand-produced,
before this reorg (cafe2's 238 dishes, ~228 of which have
realsense_overhead imagery, would otherwise be perfectly usable additional
training data). To reproduce the existing tidy files exactly, this
script's DEFAULT (--cafes cafe1) matches that history: cafe1 only. Pass
--cafes all to fold cafe2's 238 dishes in too (5006 dishes total) — this
produces a strictly larger, more complete dataset that does NOT match the
historical tidy files or the published metrics (useful for a deliberate
from-scratch conversion, not for reproducing what's already there).
download_dataset.sh's automatic post-download conversion always uses the
default (--cafes cafe1), so a fresh clone reproduces the published numbers
by default; opt into --cafes all yourself if you want the larger dataset.

Numeric formatting: every numeric field is round-tripped through Python's
`str(float(x))`. Verified byte-exact against the real dataset: 0 mismatches
across 23,840 nutrition fields + 136,125 ingredient fields + 2,220
ingredients-metadata fields.

ingredients_metadata.csv conversion: the raw file
(metadata/ingredients_metadata.csv) has a header row "ingr,id,cal/g,
fat(g),carb(g),protein(g)" with a bare integer id. It converts to header
"ingr_name,ingr_id,cal/g,fat(g),carb(g),protein(g)" with ingr_id reformatted
as f"ingr_{int(id):010d}". Row order is preserved (no sorting).

Usage (run from ml/, or from anywhere with --raw-dir/--out-dir):
  python convert_metadata.py [--raw-dir Nutrition5K/metadata_raw]
                             [--cafe1 FILE] [--cafe2 FILE]
                             [--ingredients-metadata FILE]
                             [--out-dir Nutrition5K]
                             [--cafes {cafe1,all}] [--dry-run]

Normally invoked automatically by download_dataset.sh right after a
download, when the tidy CSVs are missing but the raw ones were just
fetched. Safe to re-run directly at any time; it always overwrites the
three output CSVs (regenerating the tidy files is meant to be cheap and
deterministic, not incremental).
"""

import argparse
import csv
import sys
from pathlib import Path

NUTRITION_FIELDS = ["dish_id", "calories", "mass", "fat", "carb", "protein"]
INGREDIENT_FIELDS = [
    "dish_id", "ingr_id", "ingr_name", "grams", "calories", "fat", "carb", "protein",
]
INGREDIENTS_METADATA_FIELDS = [
    "ingr_name", "ingr_id", "cal/g", "fat(g)", "carb(g)", "protein(g)",
]

# Per-ingredient group width within a dish_metadata_cafe*.csv row:
# ingr_id, ingr_name, grams, calories, fat, carb, protein
INGREDIENT_GROUP_SIZE = 7


def format_num(raw: str) -> str:
    """Reproduce the tidy CSVs' exact numeric formatting: str(float(x))."""
    return str(float(raw))


def parse_dish_metadata_file(path: Path):
    """
    Parse one dish_metadata_cafe{1,2}.csv (headerless, ragged rows).

    Returns (nutrition_rows, ingredient_rows), each a list of dicts matching
    NUTRITION_FIELDS / INGREDIENT_FIELDS, in the exact order the file lists
    dishes (and, within a dish, ingredients).
    """
    nutrition_rows = []
    ingredient_rows = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for line_no, parts in enumerate(reader, 1):
            if not parts:
                continue
            if len(parts) < 6:
                raise ValueError(
                    f"{path}:{line_no}: expected at least 6 columns "
                    f"(dish_id + 5 totals), got {len(parts)}"
                )
            dish_id, cal, mass, fat, carb, protein = parts[:6]
            nutrition_rows.append({
                "dish_id": dish_id,
                "calories": format_num(cal),
                "mass": format_num(mass),
                "fat": format_num(fat),
                "carb": format_num(carb),
                "protein": format_num(protein),
            })

            rest = parts[6:]
            if len(rest) % INGREDIENT_GROUP_SIZE != 0:
                raise ValueError(
                    f"{path}:{line_no} ({dish_id}): trailing ingredient columns "
                    f"({len(rest)}) are not a multiple of {INGREDIENT_GROUP_SIZE} "
                    "— malformed/truncated row"
                )
            for i in range(0, len(rest), INGREDIENT_GROUP_SIZE):
                ingr_id, ingr_name, grams, i_cal, i_fat, i_carb, i_protein = (
                    rest[i:i + INGREDIENT_GROUP_SIZE]
                )
                ingredient_rows.append({
                    "dish_id": dish_id,
                    "ingr_id": ingr_id,
                    "ingr_name": ingr_name,
                    "grams": format_num(grams),
                    "calories": format_num(i_cal),
                    "fat": format_num(i_fat),
                    "carb": format_num(i_carb),
                    "protein": format_num(i_protein),
                })
    return nutrition_rows, ingredient_rows


def parse_ingredients_metadata_file(path: Path):
    """
    Parse metadata/ingredients_metadata.csv (raw). Header is
    "ingr,id,cal/g,fat(g),carb(g),protein(g)" with a bare integer id;
    output header is "ingr_name,ingr_id,cal/g,fat(g),carb(g),protein(g)"
    with ingr_id reformatted as ingr_%010d. Row order preserved.
    """
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        expected = {"ingr", "id", "cal/g", "fat(g)", "carb(g)", "protein(g)"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                f"{path}: unexpected header {reader.fieldnames!r}, "
                f"expected {sorted(expected)}"
            )
        for line_no, row in enumerate(reader, 2):
            try:
                ingr_id = f"ingr_{int(row['id']):010d}"
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_no}: non-integer id {row['id']!r}"
                ) from exc
            rows.append({
                "ingr_name": row["ingr"],
                "ingr_id": ingr_id,
                "cal/g": format_num(row["cal/g"]),
                "fat(g)": format_num(row["fat(g)"]),
                "carb(g)": format_num(row["carb(g)"]),
                "protein(g)": format_num(row["protein(g)"]),
            })
    return rows


def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        # lineterminator="\n": matches the existing tidy CSVs' plain-LF line
        # endings exactly (csv.writer's own default is "\r\n").
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def convert(cafe1_path: Path, cafe2_path: Path, ingredients_metadata_path: Path,
            include_cafe2: bool):
    """
    Returns (nutrition_rows, ingredient_rows, ingredients_metadata_rows, stats)
    without touching the filesystem for output.
    """
    nutrition_rows, ingredient_rows = parse_dish_metadata_file(cafe1_path)
    cafe1_dish_count = len(nutrition_rows)
    cafe2_dish_count = 0

    if include_cafe2:
        if not cafe2_path.exists():
            raise FileNotFoundError(
                f"--cafes all was given but cafe2 file not found: {cafe2_path}"
            )
        cafe2_nutrition, cafe2_ingredients = parse_dish_metadata_file(cafe2_path)
        cafe2_dish_count = len(cafe2_nutrition)
        nutrition_rows = nutrition_rows + cafe2_nutrition
        ingredient_rows = ingredient_rows + cafe2_ingredients
    elif cafe2_path.exists():
        # Present but intentionally skipped — count it for the summary so
        # the historical-cleaning-rule note isn't silent.
        cafe2_dish_count = sum(
            1 for _ in cafe2_path.open(newline="")
        )

    ingredients_metadata_rows = parse_ingredients_metadata_file(ingredients_metadata_path)

    stats = {
        "cafe1_dishes": cafe1_dish_count,
        "cafe2_dishes_available": cafe2_dish_count,
        "cafe2_included": include_cafe2,
        "total_dishes": len(nutrition_rows),
        "total_ingredient_rows": len(ingredient_rows),
        "ingredients_metadata_rows": len(ingredients_metadata_rows),
    }
    return nutrition_rows, ingredient_rows, ingredients_metadata_rows, stats


def main():
    script_dir = Path(__file__).parent
    default_raw_dir = script_dir / "Nutrition5K" / "metadata_raw"
    default_out_dir = script_dir / "Nutrition5K"

    parser = argparse.ArgumentParser(
        description=(
            "Convert official Nutrition5k raw metadata (dish_metadata_cafe1.csv / "
            "dish_metadata_cafe2.csv / ingredients_metadata.csv) into the tidy CSVs "
            "prepare_finetune.py and select_images.py read. By default reproduces "
            "the historically-derived tidy files exactly: cafe1 dishes only (see "
            "module docstring for why cafe2 is excluded by default)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=default_raw_dir,
        help="Directory containing the raw CSVs (default matches download_dataset.sh's "
             "output location).",
    )
    parser.add_argument(
        "--cafe1", type=Path, default=None,
        help="Path to dish_metadata_cafe1.csv (default: <raw-dir>/dish_metadata_cafe1.csv).",
    )
    parser.add_argument(
        "--cafe2", type=Path, default=None,
        help="Path to dish_metadata_cafe2.csv (default: <raw-dir>/dish_metadata_cafe2.csv). "
             "Only read when --cafes all is passed.",
    )
    parser.add_argument(
        "--ingredients-metadata", type=Path, default=None,
        help="Path to raw ingredients_metadata.csv "
             "(default: <raw-dir>/ingredients_metadata.csv).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=default_out_dir,
        help="Directory to write dish_nutrition_values.csv, dish_ingredients.csv, "
             "and ingredients_metadata.csv into.",
    )
    parser.add_argument(
        "--cafes", choices=["cafe1", "all"], default="cafe1",
        help="'cafe1' (default): reproduces the historical tidy files EXACTLY "
             "(4768 dishes / 27225 ingredient rows) — this is the reproduce-exactly "
             "bar and what download_dataset.sh's automatic conversion always uses. "
             "'all': also folds in dish_metadata_cafe2.csv's 238 dishes (~228 of "
             "which have realsense_overhead imagery), recovering additional usable "
             "dishes for future training runs (5006 total) — but this does NOT "
             "match the historical tidy files/published metrics, since cafe2 was "
             "historically dropped (apparently an oversight, not a deliberate "
             "filter). Only use 'all' for a deliberate from-scratch conversion.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Parse and report stats but do not write any output files.",
    )
    args = parser.parse_args()

    cafe1_path = args.cafe1 or (args.raw_dir / "dish_metadata_cafe1.csv")
    cafe2_path = args.cafe2 or (args.raw_dir / "dish_metadata_cafe2.csv")
    ingredients_metadata_path = (
        args.ingredients_metadata or (args.raw_dir / "ingredients_metadata.csv")
    )

    for label, path in [
        ("dish_metadata_cafe1.csv", cafe1_path),
        ("ingredients_metadata.csv", ingredients_metadata_path),
    ]:
        if not path.exists():
            print(f"ERROR: required raw file not found: {path} ({label})", file=sys.stderr)
            print(
                "Run ./download_dataset.sh first (fetches metadata_raw/ from the "
                "Nutrition5k bucket).",
                file=sys.stderr,
            )
            sys.exit(1)

    include_cafe2 = args.cafes == "all"
    nutrition_rows, ingredient_rows, ingredients_metadata_rows, stats = convert(
        cafe1_path, cafe2_path, ingredients_metadata_path, include_cafe2
    )

    print(f"cafe1 dishes                 : {stats['cafe1_dishes']}")
    if stats["cafe2_dishes_available"]:
        status = "included" if stats["cafe2_included"] else "SKIPPED (historical cleaning rule)"
        print(f"cafe2 dishes available       : {stats['cafe2_dishes_available']} ({status})")
    print(f"Total dishes written          : {stats['total_dishes']}")
    print(f"Total ingredient rows written : {stats['total_ingredient_rows']}")
    print(f"Ingredients-metadata rows      : {stats['ingredients_metadata_rows']}")

    if not stats["cafe2_included"] and stats["cafe2_dishes_available"]:
        print(
            "\nNote: dish_metadata_cafe2.csv was found but not converted — this "
            "matches the existing tidy CSVs in this repo, which are cafe1-only. "
            "Pass --cafes all to fold it in (produces a different, larger dataset "
            "that will NOT match the checked-in tidy files)."
        )

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    write_csv(args.out_dir / "dish_nutrition_values.csv", NUTRITION_FIELDS, nutrition_rows)
    write_csv(args.out_dir / "dish_ingredients.csv", INGREDIENT_FIELDS, ingredient_rows)
    write_csv(
        args.out_dir / "ingredients_metadata.csv",
        INGREDIENTS_METADATA_FIELDS,
        ingredients_metadata_rows,
    )
    print(f"\nWrote tidy CSVs to {args.out_dir}/")


if __name__ == "__main__":
    main()
