import json
from pathlib import Path

import pytest

from exclusion_residual_audit import (
    _summary_from_errors,
    residual_rows,
    run_audit,
)


def _write_full(path: Path, record_id: str, items: list[dict]) -> None:
    completion = {
        "total_calories": sum(item["calories"] for item in items),
        "items": items,
    }
    row = {
        "images": [f"dataset_clean/{record_id}/overhead.jpg"],
        "messages": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": json.dumps(completion)},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _write_visible(path: Path, record_id: str, items: list[dict]) -> None:
    row = {
        "id": record_id,
        "group_id": record_id,
        "evaluation_ground_truth": {
            "total_calories": sum(item["calories"] for item in items),
            "items": items,
        },
        "messages": [],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_residual_redistributes_full_mass_over_visible_density(tmp_path):
    visible_path = tmp_path / "visible.jsonl"
    full_path = tmp_path / "full.jsonl"
    visible_item = {
        "name": "lettuce",
        "estimated_grams": 100,
        "calories": 20,
    }
    hidden_item = {
        "name": "olive oil",
        "estimated_grams": 10,
        "calories": 90,
    }
    _write_visible(visible_path, "dish_1", [visible_item])
    _write_full(full_path, "dish_1", [visible_item, hidden_item])

    rows, accounting = residual_rows(
        visible_path,
        full_path,
        split="validation",
    )

    assert accounting["scored_rows"] == 1
    assert rows[0]["truth_density_oracle_kcal"] == pytest.approx(22)
    assert rows[0]["signed_residual_kcal"] == pytest.approx(-88)
    assert rows[0]["excluded_condiment_or_fat"] is True
    assert rows[0]["visible_sauce_dressing_prone"] is True


def test_bias_variance_decomposition_telescopes():
    summary = _summary_from_errors([-2, 0, 4])
    decomposition = summary["mse_bias_variance_decomposition"]
    assert decomposition["mse_kcal_squared"] == pytest.approx(
        decomposition["bias_squared_kcal_squared"]
        + decomposition["variance_kcal_squared"]
    )
    assert (
        decomposition["bias_squared_fraction"]
        + decomposition["variance_fraction"]
    ) == pytest.approx(1)


def test_run_audit_derives_correction_from_train_only(tmp_path):
    train_visible = tmp_path / "train-visible.jsonl"
    train_full = tmp_path / "train-full.jsonl"
    val_visible = tmp_path / "val-visible.jsonl"
    val_full = tmp_path / "val-full.jsonl"
    visible_item = {
        "name": "lettuce",
        "estimated_grams": 100,
        "calories": 20,
    }
    hidden_item = {
        "name": "olive oil",
        "estimated_grams": 10,
        "calories": 90,
    }
    _write_visible(train_visible, "dish_train", [visible_item])
    _write_full(train_full, "dish_train", [visible_item, hidden_item])
    _write_visible(val_visible, "dish_val", [visible_item])
    _write_full(val_full, "dish_val", [visible_item, hidden_item])

    payload = run_audit(
        identify_train=train_visible,
        full_train=train_full,
        identify_validation=val_visible,
        full_validation=val_full,
    )

    correction = payload["train_derived_additive_debiasing"]
    assert correction["correction_kcal_added_to_assembly"] == pytest.approx(88)
    assert correction["validation_after_correction"]["mean_signed_kcal"] == 0
    assert correction["acceptance_guard"]["passes_all"] is True
    assert payload["floor_derived_item_count_p90_intervals"][
        "validation_overall_coverage"
    ] == 1


def test_frozen_test_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="frozen test"):
        residual_rows(
            tmp_path / "unused-visible.jsonl",
            tmp_path / "unused-full.jsonl",
            split="test",
        )
