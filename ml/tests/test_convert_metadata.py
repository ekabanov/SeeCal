"""
Tests for convert_metadata.py — converting the official Nutrition5k raw
metadata CSVs into the tidy dish_nutrition_values.csv / dish_ingredients.csv
/ ingredients_metadata.csv this repo's pipeline reads.

Run with:  .venv/bin/python -m pytest tests/test_convert_metadata.py
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "convert_metadata", REPO_ROOT / "convert_metadata.py"
)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


# ---------------------------------------------------------------------------
# Small inline fixture: hand-built raw dish_metadata_cafe1.csv content
# ---------------------------------------------------------------------------

CAFE1_FIXTURE = (
    # dish with 2 ingredients
    "dish_0000000001,300.794281,193.000000,12.387489,28.218290,18.633970,"
    "ingr_0000000508,soy sauce,3.398568,1.80124104,0.020391408,0.166529832,0.275284008,"
    "ingr_0000000026,white rice,8.496420,11.045346,0.02548926,2.3789976,0.22940334\n"
    # dish with 0 ingredients (totals only)
    "dish_0000000002,20.590000,103.000000,0.148000,4.625000,0.956000\n"
)

CAFE2_FIXTURE = (
    "dish_0000000099,50.000000,10.000000,1.000000,2.000000,3.000000,"
    "ingr_0000000001,cottage cheese,10.000000,9.8,0.43,0.34,1.1\n"
)

INGREDIENTS_METADATA_FIXTURE = (
    "ingr,id,cal/g,fat(g),carb(g),protein(g)\n"
    "cottage cheese,1,0.980,0.043,0.034,0.110\n"
    "soy sauce,508,0.530,0.006,0.049,0.081\n"
)


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "metadata_raw"
    d.mkdir()
    (d / "dish_metadata_cafe1.csv").write_text(CAFE1_FIXTURE)
    (d / "dish_metadata_cafe2.csv").write_text(CAFE2_FIXTURE)
    (d / "ingredients_metadata.csv").write_text(INGREDIENTS_METADATA_FIXTURE)
    return d


# ---------------------------------------------------------------------------
# parse_dish_metadata_file
# ---------------------------------------------------------------------------

def test_parses_nutrition_rows_in_order(raw_dir):
    nutrition_rows, _ = cm.parse_dish_metadata_file(raw_dir / "dish_metadata_cafe1.csv")
    assert [r["dish_id"] for r in nutrition_rows] == ["dish_0000000001", "dish_0000000002"]
    assert nutrition_rows[0] == {
        "dish_id": "dish_0000000001",
        "calories": "300.794281",
        "mass": "193.0",
        "fat": "12.387489",
        "carb": "28.21829",
        "protein": "18.63397",
    }
    assert nutrition_rows[1] == {
        "dish_id": "dish_0000000002",
        "calories": "20.59",
        "mass": "103.0",
        "fat": "0.148",
        "carb": "4.625",
        "protein": "0.956",
    }


def test_dish_with_zero_ingredients_contributes_no_ingredient_rows(raw_dir):
    _, ingredient_rows = cm.parse_dish_metadata_file(raw_dir / "dish_metadata_cafe1.csv")
    assert [r["dish_id"] for r in ingredient_rows] == ["dish_0000000001", "dish_0000000001"]


def test_ingredient_row_fields_and_order(raw_dir):
    _, ingredient_rows = cm.parse_dish_metadata_file(raw_dir / "dish_metadata_cafe1.csv")
    assert ingredient_rows[0] == {
        "dish_id": "dish_0000000001",
        "ingr_id": "ingr_0000000508",
        "ingr_name": "soy sauce",
        "grams": "3.398568",
        "calories": "1.80124104",
        "fat": "0.020391408",
        "carb": "0.166529832",
        "protein": "0.275284008",
    }
    assert ingredient_rows[1]["ingr_name"] == "white rice"


def test_malformed_row_raises_clear_error(tmp_path):
    bad = tmp_path / "bad_cafe1.csv"
    # 4 trailing columns is not a multiple of 7 -> malformed ingredient group
    bad.write_text("dish_x,1.0,1.0,1.0,1.0,1.0,ingr_1,name,1.0,1.0\n")
    with pytest.raises(ValueError, match="not a multiple of 7"):
        cm.parse_dish_metadata_file(bad)


# ---------------------------------------------------------------------------
# parse_ingredients_metadata_file
# ---------------------------------------------------------------------------

def test_ingredients_metadata_header_rename_and_id_format(raw_dir):
    rows = cm.parse_ingredients_metadata_file(raw_dir / "ingredients_metadata.csv")
    assert rows[0] == {
        "ingr_name": "cottage cheese",
        "ingr_id": "ingr_0000000001",
        "cal/g": "0.98",
        "fat(g)": "0.043",
        "carb(g)": "0.034",
        "protein(g)": "0.11",
    }
    assert rows[1]["ingr_id"] == "ingr_0000000508"


def test_ingredients_metadata_wrong_header_raises(tmp_path):
    bad = tmp_path / "bad_meta.csv"
    bad.write_text("name,id,cal,fat,carb,protein\nx,1,1,1,1,1\n")
    with pytest.raises(ValueError, match="unexpected header"):
        cm.parse_ingredients_metadata_file(bad)


# ---------------------------------------------------------------------------
# convert(): cafe2 exclusion is the default (historical cleaning rule)
# ---------------------------------------------------------------------------

def test_cafe2_excluded_by_default(raw_dir):
    nutrition_rows, ingredient_rows, _, stats = cm.convert(
        raw_dir / "dish_metadata_cafe1.csv",
        raw_dir / "dish_metadata_cafe2.csv",
        raw_dir / "ingredients_metadata.csv",
        include_cafe2=False,
    )
    dish_ids = {r["dish_id"] for r in nutrition_rows}
    assert "dish_0000000099" not in dish_ids
    assert dish_ids == {"dish_0000000001", "dish_0000000002"}
    assert stats["cafe2_included"] is False
    assert stats["cafe2_dishes_available"] == 1


def test_include_cafe2_flag_folds_it_in(raw_dir):
    nutrition_rows, ingredient_rows, _, stats = cm.convert(
        raw_dir / "dish_metadata_cafe1.csv",
        raw_dir / "dish_metadata_cafe2.csv",
        raw_dir / "ingredients_metadata.csv",
        include_cafe2=True,
    )
    dish_ids = {r["dish_id"] for r in nutrition_rows}
    assert "dish_0000000099" in dish_ids
    assert stats["cafe2_included"] is True
    assert any(r["dish_id"] == "dish_0000000099" for r in ingredient_rows)


def test_include_cafe2_missing_file_raises(raw_dir, tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        cm.convert(
            raw_dir / "dish_metadata_cafe1.csv",
            missing,
            raw_dir / "ingredients_metadata.csv",
            include_cafe2=True,
        )


# ---------------------------------------------------------------------------
# Full-file integration check: reproduce the real, existing tidy CSVs
# exactly. Skipped when the raw files (or the tidy files to diff against)
# aren't present locally — the raw metadata is fetched by download_dataset.sh
# and both live under the gitignored Nutrition5K/ directory.
# ---------------------------------------------------------------------------

NUTRITION5K_DIR = REPO_ROOT / "Nutrition5K"
RAW_DIR = NUTRITION5K_DIR / "metadata_raw"

_raw_files_present = all(
    (RAW_DIR / name).exists()
    for name in ("dish_metadata_cafe1.csv", "ingredients_metadata.csv")
)
_tidy_files_present = all(
    (NUTRITION5K_DIR / name).exists()
    for name in ("dish_nutrition_values.csv", "dish_ingredients.csv", "ingredients_metadata.csv")
)


@pytest.mark.skipif(
    not (_raw_files_present and _tidy_files_present),
    reason="real Nutrition5k raw/tidy metadata files not present locally "
           "(run ./download_dataset.sh to fetch them)",
)
def test_real_dataset_reproduces_existing_tidy_csvs_exactly(tmp_path):
    nutrition_rows, ingredient_rows, ingredients_metadata_rows, stats = cm.convert(
        RAW_DIR / "dish_metadata_cafe1.csv",
        RAW_DIR / "dish_metadata_cafe2.csv",
        RAW_DIR / "ingredients_metadata.csv",
        include_cafe2=False,
    )

    assert stats["total_dishes"] == 4768
    assert stats["total_ingredient_rows"] == 27225

    out_dir = tmp_path / "converted"
    cm.write_csv(out_dir / "dish_nutrition_values.csv", cm.NUTRITION_FIELDS, nutrition_rows)
    cm.write_csv(out_dir / "dish_ingredients.csv", cm.INGREDIENT_FIELDS, ingredient_rows)
    cm.write_csv(
        out_dir / "ingredients_metadata.csv",
        cm.INGREDIENTS_METADATA_FIELDS,
        ingredients_metadata_rows,
    )

    for name in ("dish_nutrition_values.csv", "dish_ingredients.csv", "ingredients_metadata.csv"):
        converted = (out_dir / name).read_text()
        existing = (NUTRITION5K_DIR / name).read_text()
        assert converted == existing, f"{name} does not match the existing tidy file byte-for-byte"


@pytest.mark.skipif(
    not _raw_files_present,
    reason="real Nutrition5k raw metadata files not present locally "
           "(run ./download_dataset.sh to fetch them)",
)
def test_real_dataset_cafes_all_recovers_cafe2_dishes():
    """--cafes all (include_cafe2=True) folds in cafe2's 238 dishes, for a
    total of 5006 — this does NOT match the published/historical tidy
    files (that's the cafe1-only default's job), but should be exactly
    cafe1's 4768 + cafe2's 238 with zero overlap or loss."""
    nutrition_rows, ingredient_rows, _, stats = cm.convert(
        RAW_DIR / "dish_metadata_cafe1.csv",
        RAW_DIR / "dish_metadata_cafe2.csv",
        RAW_DIR / "ingredients_metadata.csv",
        include_cafe2=True,
    )
    assert stats["cafe1_dishes"] == 4768
    assert stats["cafe2_dishes_available"] == 238
    assert stats["cafe2_included"] is True
    assert stats["total_dishes"] == 5006
    assert len({r["dish_id"] for r in nutrition_rows}) == 5006
