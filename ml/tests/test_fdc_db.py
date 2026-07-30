import csv
from pathlib import Path
import sqlite3

from factored_pipeline.resolver import (
    SQLiteNutritionResolver,
    category_lexical_score,
    lexical_score,
)
from make_alias_table import build_aliases, reviewed_alias_names
from make_fdc_db import _load_portions, build_database


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_fdc_fixture(root: Path) -> Path:
    root.mkdir()
    _write_csv(
        root / "food.csv",
        ["fdc_id", "data_type", "description", "food_category_id"],
        [
            {
                "fdc_id": 1,
                "data_type": "Foundation",
                "description": "Cucumber, raw",
                "food_category_id": 10,
            },
            {
                "fdc_id": 2,
                "data_type": "Survey (FNDDS)",
                "description": "Cheeseburger",
                "food_category_id": "",
            },
        ],
    )
    _write_csv(
        root / "food_category.csv",
        ["id", "description"],
        [{"id": 10, "description": "Vegetables"}],
    )
    _write_csv(
        root / "wweia_food_category.csv",
        ["fdc_id", "wweia_food_category_description"],
        [{"fdc_id": 2, "wweia_food_category_description": "Burgers"}],
    )
    _write_csv(
        root / "nutrient.csv",
        ["id", "name", "unit_name"],
        [
            {"id": 1003, "name": "Protein", "unit_name": "g"},
            {"id": 1004, "name": "Total lipid (fat)", "unit_name": "g"},
            {"id": 1005, "name": "Carbohydrate", "unit_name": "g"},
            {"id": 1008, "name": "Energy", "unit_name": "kcal"},
        ],
    )
    rows = []
    for fdc_id, values in {
        1: {1003: 0.7, 1004: 0.1, 1005: 3.6, 1008: 15},
        2: {1003: 13, 1004: 14, 1005: 24, 1008: 270},
    }.items():
        rows.extend(
            {"fdc_id": fdc_id, "nutrient_id": nutrient_id, "amount": amount}
            for nutrient_id, amount in values.items()
        )
    _write_csv(root / "food_nutrient.csv", ["fdc_id", "nutrient_id", "amount"], rows)
    return root


def test_build_database_and_resolution_ladder(tmp_path):
    source = make_fdc_fixture(tmp_path / "source")
    database = tmp_path / "nutrition.sqlite"
    result = build_database([source], database)
    assert result["foods"] == 2

    resolver = SQLiteNutritionResolver(database)
    exact = resolver.resolve("cucumber")
    fuzzy = resolver.resolve("cheese burger")
    assert exact.rung == "exact_alias"
    assert exact.profile.category == "Vegetables"
    assert fuzzy.rung == "fuzzy"
    assert fuzzy.profile.fdc_id == 2
    # Canonical energy is arithmetic over looked-up macros.
    assert exact.profile.kcal_per_100g == 4 * 0.7 + 9 * 0.1 + 4 * 3.6
    resolver.close()

    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    default = connection.execute(
        """
        SELECT kcal_per_100g,protein_per_100g,fat_per_100g,carbs_per_100g
        FROM category_defaults WHERE category='Vegetables'
        """
    ).fetchone()
    assert default[0] == 4 * default[1] + 9 * default[2] + 4 * default[3]
    connection.close()


def test_portion_loader_selects_one_observed_kind_instead_of_averaging(tmp_path):
    source = make_fdc_fixture(tmp_path / "source")
    _write_csv(
        source / "food_portion.csv",
        [
            "fdc_id",
            "seq_num",
            "modifier",
            "gram_weight",
            "data_points",
        ],
        [
            {
                "fdc_id": 1,
                "seq_num": 1,
                "modifier": "slice",
                "gram_weight": 112,
                "data_points": 96,
            },
            {
                "fdc_id": 1,
                "seq_num": 2,
                "modifier": "whole",
                "gram_weight": 897,
                "data_points": 12,
            },
        ],
    )

    portions = _load_portions([source], {1})

    assert portions[1] == 112
    assert portions[1] != (112 + 897) / 2


def test_alias_builder_never_promotes_category_default(tmp_path):
    source = make_fdc_fixture(tmp_path / "source")
    database = tmp_path / "nutrition.sqlite"
    build_database([source], database)
    misses = tmp_path / "misses.tsv"
    result = build_aliases(
        database=database,
        names={"raw cucumber", "totally unrelated mystery"},
        misses_path=misses,
        accept_fuzzy=0.70,
    )
    assert result["accepted"] == 1
    assert result["misses"] == 1
    assert "totally unrelated mystery" in misses.read_text()


def test_alias_builder_applies_only_explicit_reviewed_source_profile(tmp_path):
    source = make_fdc_fixture(tmp_path / "source")
    database = tmp_path / "nutrition.sqlite"
    build_database([source], database)
    reviewed = tmp_path / "reviewed.tsv"
    reviewed.write_text(
        "name\tfdc_id\treview_note\n"
        "mystery garden vegetable\t1\thuman-reviewed fixture\n"
    )
    result = build_aliases(
        database=database,
        names={"mystery garden vegetable"},
        misses_path=tmp_path / "misses.tsv",
        accept_fuzzy=0.84,
        reviewed_aliases_path=reviewed,
    )
    assert result["rung1_pct"] == 100
    assert result["misses"] == 0
    resolver = SQLiteNutritionResolver(database)
    resolution = resolver.resolve("mystery garden vegetable")
    resolver.close()
    assert resolution.rung == "exact_alias"
    assert resolution.profile.fdc_id == 1


def test_reviewed_alias_file_can_supply_its_own_vocabulary(tmp_path):
    reviewed = tmp_path / "reviewed.tsv"
    reviewed.write_text(
        "name\tfdc_id\treview_note\n"
        "regional cucumber plate\t1\thuman-reviewed fixture\n"
    )
    assert reviewed_alias_names(reviewed) == {"regional cucumber plate"}


def test_short_query_can_match_descriptive_usda_name():
    assert lexical_score("carrot", "Carrots, raw") < 0.76
    assert category_lexical_score("carrot", "Carrots, raw") > 0.76
    assert lexical_score("chicken breast", "chicken sausage") < 0.76


def test_build_database_accepts_fndds_legacy_nutrient_numbers(tmp_path):
    source = make_fdc_fixture(tmp_path / "source")
    _write_csv(
        source / "food.csv",
        ["fdc_id", "data_type", "description", "food_category_id"],
        [
            {
                "fdc_id": 1,
                "data_type": "Foundation",
                "description": "Cucumber, raw",
                "food_category_id": 10,
            },
            {
                "fdc_id": 2,
                "data_type": "survey_fndds_food",
                "description": "Cheeseburger",
                "food_category_id": 1002,
            },
        ],
    )
    _write_csv(
        source / "wweia_food_category.csv",
        ["wweia_food_category", "wweia_food_category_description"],
        [{"wweia_food_category": 1002, "wweia_food_category_description": "Burgers"}],
    )
    _write_csv(
        source / "nutrient.csv",
        ["id", "name", "unit_name", "nutrient_nbr"],
        [
            {
                "id": 1003,
                "name": "Protein",
                "unit_name": "g",
                "nutrient_nbr": 203,
            },
            {
                "id": 1004,
                "name": "Total lipid (fat)",
                "unit_name": "g",
                "nutrient_nbr": 204,
            },
            {
                "id": 1005,
                "name": "Carbohydrate",
                "unit_name": "g",
                "nutrient_nbr": 205,
            },
            {
                "id": 1008,
                "name": "Energy",
                "unit_name": "kcal",
                "nutrient_nbr": 208,
            },
        ],
    )
    rows = []
    modern_to_legacy = {1003: 203, 1004: 204, 1005: 205, 1008: 208}
    for fdc_id, values in {
        1: {1003: 0.7, 1004: 0.1, 1005: 3.6, 1008: 15},
        2: {1003: 13, 1004: 14, 1005: 24, 1008: 270},
    }.items():
        rows.extend(
            {
                "fdc_id": fdc_id,
                "nutrient_id": modern_to_legacy[nutrient_id],
                "amount": amount,
            }
            for nutrient_id, amount in values.items()
        )
    _write_csv(
        source / "food_nutrient.csv",
        ["fdc_id", "nutrient_id", "amount"],
        rows,
    )

    database = tmp_path / "nutrition.sqlite"
    result = build_database([source], database)

    assert result["foods"] == 2
    resolver = SQLiteNutritionResolver(database)
    assert resolver.resolve("cucumber").profile.protein_per_100g == 0.7
    assert resolver.resolve("cheeseburger").profile.category == "Burgers"
    resolver.close()
