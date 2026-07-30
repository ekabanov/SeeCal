import json

import pytest

from factored_pipeline.contract import (
    ContractError,
    IDENTIFY_PROMPT,
    canonical_identification,
    identification_to_shares,
    normalize_legacy_share_identification,
    normalize_name,
    portion_units_from_weights,
    shares_from_weights,
    validate_identification,
    validate_identification_v1,
    validate_identification_with_repair,
)


def test_frozen_prompt_excludes_nutrition_outputs():
    assert "without estimating grams, calories, nutrients" in IDENTIFY_PROMPT
    assert "portion_units" in IDENTIFY_PROMPT
    assert "do not need to sum" in IDENTIFY_PROMPT


def test_share_quantization_is_sorted_bucketed_and_exact():
    items = shares_from_weights([("rice", 61), ("beans", 26), ("salsa", 13)])
    assert sum(item["share_pct"] for item in items) == 100
    assert all(item["share_pct"] % 5 == 0 for item in items)
    assert [item["share_pct"] for item in items] == sorted(
        [item["share_pct"] for item in items], reverse=True
    )
    assert [item["name"] for item in items] == ["rice", "beans", "salsa"]


def test_share_quantization_caps_tiny_tail_without_zero_items():
    items = shares_from_weights((f"item {index}", index + 1) for index in range(34))
    assert len(items) == 20
    assert all(item["share_pct"] >= 5 for item in items)
    assert sum(item["share_pct"] for item in items) == 100


def test_contract_rejects_extra_keys_and_bad_share_sum():
    with pytest.raises(ContractError):
        validate_identification_v1(
            {
                "not_food": False,
                "container": "plate",
                "items": [{"name": "rice", "share_pct": 95}],
                "calories": 100,
            }
        )
    with pytest.raises(ContractError):
        validate_identification_v1(
            {
                "not_food": False,
                "container": "plate",
                "items": [{"name": "rice", "share_pct": 95}],
            }
        )


def test_evaluator_repairs_only_bounded_share_sum_slips():
    repaired, changed, original_sum = validate_identification_with_repair(
        {
            "not_food": False,
            "container": "plate",
            "items": [
                {"name": "rice", "share_pct": 55},
                {"name": "beans", "share_pct": 40},
            ],
        }
    )
    assert changed
    assert original_sum == 95
    assert sum(item["share_pct"] for item in repaired["items"]) == 100
    with pytest.raises(ContractError):
        validate_identification_with_repair(
            {
                "not_food": False,
                "container": "plate",
                "items": [{"name": "rice", "share_pct": 85}],
            }
        )


def test_canonical_identification_round_trips():
    payload = json.loads(
        canonical_identification(
            container="tray",
            weighted_names=[("Chicken", 100), ("Rice", 100)],
        )
    )
    assert validate_identification(payload) == payload
    assert payload["items"] == [
        {"name": "Chicken", "portion_units": 10},
        {"name": "Rice", "portion_units": 10},
    ]
    assert normalize_name("Crème & Rice") == "creme and rice"


def test_portion_units_are_independent_and_normalized_by_code():
    payload = {
        "not_food": False,
        "container": "plate",
        "items": [
            {"name": "rice", "portion_units": 5},
            {"name": "beans", "portion_units": 3},
            {"name": "greens", "portion_units": 2},
        ],
    }
    normalized = identification_to_shares(payload)
    assert [item["share_pct"] for item in normalized["items"]] == [50, 30, 20]
    assert sum(item["share_pct"] for item in normalized["items"]) == 100


def test_legacy_shares_normalize_arbitrary_positive_weights():
    normalized = normalize_legacy_share_identification(
        {
            "not_food": False,
            "container": "plate",
            "items": [
                {"name": "rice", "share_pct": 12},
                {"name": "chicken", "share_pct": 7.5},
                {"name": "Rice", "share_pct": 3},
            ],
        }
    )
    assert sum(item["share_pct"] for item in normalized["items"]) == 100
    assert [item["name"] for item in normalized["items"]] == ["rice", "chicken"]


def test_portion_unit_validation_merges_duplicates_and_sorts():
    result = validate_identification(
        {
            "not_food": False,
            "container": "plate",
            "items": [
                {"name": "rice", "portion_units": 2},
                {"name": "beans", "portion_units": 4},
                {"name": "Rice", "portion_units": 3},
            ],
        }
    )
    assert result["items"] == [
        {"name": "rice", "portion_units": 5},
        {"name": "beans", "portion_units": 4},
    ]


def test_portion_units_derived_without_sum_constraint():
    items = portion_units_from_weights(
        [("rice", 61), ("beans", 26), ("salsa", 13)]
    )
    assert items == [
        {"name": "rice", "portion_units": 12},
        {"name": "beans", "portion_units": 5},
        {"name": "salsa", "portion_units": 3},
    ]
