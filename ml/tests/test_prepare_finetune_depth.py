"""
Tests for the depth-variant support in prepare_finetune.py (Task D2,
docs/design/2026-07-26-depth-design-brief.md section (f)).

Run with:  .venv/bin/python -m pytest tests/test_prepare_finetune_depth.py
"""

import hashlib
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "prepare_finetune", REPO_ROOT / "prepare_finetune.py"
)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


# ---------------------------------------------------------------------------
# Feature dropout (brief item 5): md5(dish_id) % 100 < 12
# ---------------------------------------------------------------------------

def test_dropout_matches_md5_rule_exactly():
    ids = [f"dish_{i}" for i in range(500)] + [
        "dish_1556572657", "dish_1564159636", "dish_1556573514"
    ]
    for dish_id in ids:
        expected = int(hashlib.md5(dish_id.encode()).hexdigest(), 16) % 100 < 12
        assert pf.depth_dropout(dish_id) == expected, dish_id


def test_dropout_is_deterministic():
    for dish_id in ("dish_1556573514", "dish_1560368996"):
        assert pf.depth_dropout(dish_id) == pf.depth_dropout(dish_id)


def test_dropout_rate_near_12_percent():
    # Large synthetic population: rate should be close to 12%.
    ids = [f"dish_{i}" for i in range(20000)]
    rate = sum(pf.depth_dropout(d) for d in ids) / len(ids)
    assert 0.10 < rate < 0.14, rate


# ---------------------------------------------------------------------------
# Variant B line (brief item 6): verbatim format string, :.0f rounding
# ---------------------------------------------------------------------------

def test_variant_b_line_exact():
    stats = {"volume_ml": 329.6, "max_height_mm": 42.4}
    assert pf.format_depth_line(stats) == (
        "\n\nEstimated food volume from depth sensor: "
        "~330 ml (max height 42 mm)."
    )


def test_variant_b_line_accepts_csv_strings():
    # depth_stats.csv rows carry stringified floats.
    stats = {"volume_ml": "329.6000", "max_height_mm": "42.4000"}
    assert pf.format_depth_line(stats) == (
        "\n\nEstimated food volume from depth sensor: "
        "~330 ml (max height 42 mm)."
    )


def test_variant_b_line_whole_number_rounding_is_python_0f():
    # :.0f is round-half-even: 99.5 -> 100, 100.5 -> 100.
    stats = {"volume_ml": 99.5, "max_height_mm": 100.5}
    assert pf.format_depth_line(stats) == (
        "\n\nEstimated food volume from depth sensor: "
        "~100 ml (max height 100 mm)."
    )


# ---------------------------------------------------------------------------
# build_record: v5 byte-identity default + variant shapes
# ---------------------------------------------------------------------------

BASE_TEXT = pf.SYSTEM_PROMPT + "\n\n" + pf.USER_PROMPT


def test_build_record_default_is_v5_format():
    rec = pf.build_record("dataset_clean/dish_x/overhead.jpg", '{"total_calories": 1.0}')
    assert rec["images"] == ["dataset_clean/dish_x/overhead.jpg"]
    user = rec["messages"][0]
    assert user["role"] == "user"
    assert len(user["content"]) == 2
    assert user["content"][0] == {
        "type": "image", "image": "dataset_clean/dish_x/overhead.jpg"
    }
    assert user["content"][1] == {"type": "text", "text": BASE_TEXT}
    assert rec["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": '{"total_calories": 1.0}'}],
    }


def test_build_record_variant_b_appends_line_only():
    line = pf.format_depth_line({"volume_ml": 250.0, "max_height_mm": 31.0})
    rec = pf.build_record(
        "dataset_clean/dish_x/overhead.jpg", "{}", depth_line=line
    )
    assert rec["images"] == ["dataset_clean/dish_x/overhead.jpg"]  # still 1 image
    text = rec["messages"][0]["content"][-1]["text"]
    assert text == BASE_TEXT + line
    assert text.startswith(BASE_TEXT)  # base prompt untouched
    assert text.endswith("~250 ml (max height 31 mm).")


def test_build_record_variant_a_two_images_text_unchanged():
    rec = pf.build_record(
        "dataset_clean/dish_x/overhead.jpg",
        "{}",
        height_image_path="dataset_clean/dish_x/height.png",
    )
    assert rec["images"] == [
        "dataset_clean/dish_x/overhead.jpg",
        "dataset_clean/dish_x/height.png",
    ]
    content = rec["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert content[1] == {"type": "image", "image": "dataset_clean/dish_x/height.png"}
    assert content[2]["text"] == BASE_TEXT  # variant A: text unchanged


# ---------------------------------------------------------------------------
# Depth stats rows: status classification for missing/corrupt files
# ---------------------------------------------------------------------------

def test_stats_row_missing_depth(tmp_path):
    (tmp_path / "dish_a").mkdir()
    row = pf.compute_depth_stats_row(tmp_path, "dish_a")
    assert row["status"] == "missing"
    assert row["volume_ml"] == ""


def test_stats_row_corrupt_zero_byte_depth(tmp_path):
    dish = tmp_path / "dish_b"
    dish.mkdir()
    (dish / "depth_raw.png").write_bytes(b"")  # the dish_1564159636 case
    row = pf.compute_depth_stats_row(tmp_path, "dish_b")
    assert row["status"] == "corrupt"


def test_stats_csv_roundtrip(tmp_path):
    rows = {
        "dish_a": {"dish_id": "dish_a", "status": "ok", "volume_ml": "329.6000",
                   "max_height_mm": "42.4000", "coverage_cm2": "150.0000"},
        "dish_b": {"dish_id": "dish_b", "status": "corrupt", "volume_ml": "",
                   "max_height_mm": "", "coverage_cm2": ""},
    }
    csv_path = tmp_path / "depth_stats.csv"
    pf.write_depth_stats_csv(csv_path, rows)
    loaded = pf.load_depth_stats_csv(csv_path)
    assert loaded == rows
