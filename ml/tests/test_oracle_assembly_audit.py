import pytest

from oracle_assembly_audit import floor_attribution, regret_decomposition


def test_regret_decomposition_applies_preagreed_half_total_rule():
    identify_bound = regret_decomposition(
        total_kcal_mae=60,
        true_mass_floor_kcal_mae=40,
    )
    scale_bound = regret_decomposition(
        total_kcal_mae=60,
        true_mass_floor_kcal_mae=20,
    )

    assert identify_bound == {
        "total_c1_kcal_mae": 60,
        "true_mass_floor_kcal_mae": 40,
        "mass_attributable_excess_kcal_mae": 20,
        "non_mass_fraction_of_total": pytest.approx(2 / 3),
        "binding_constraint_by_half_total_rule": "identify_resolve",
    }
    assert scale_bound["binding_constraint_by_half_total_rule"] == "scale"


def test_floor_attribution_waterfall_telescopes_to_bucketed_floor():
    truth_by_id = {
        "dish": {
            "total_calories": 100,
            "items": [
                {"name": "first", "estimated_grams": 50, "calories": 40},
                {"name": "second", "estimated_grams": 50, "calories": 60},
            ],
        }
    }
    exact = {
        "paired": [
            {
                "id": "dish",
                "group_id": "dish",
                "complete": True,
                "resolution_rungs": ["exact_alias", "category_default"],
                "prediction": {
                    "total_calories": 130,
                    "items": [
                        {"estimated_grams": 50, "calories": 50},
                        {"estimated_grams": 50, "calories": 80},
                    ],
                },
            }
        ]
    }
    bucketed = {
        "paired": [
            {
                "id": "dish",
                "group_id": "dish",
                "complete": True,
                "prediction": {"total_calories": 140},
            }
        ]
    }

    result = floor_attribution(
        exact_assembly=exact,
        bucketed_assembly=bucketed,
        truth_by_id=truth_by_id,
    )

    assert result["waterfall_kcal_mae"] == {
        "visible_label_and_exclusion_residual_kcal_mae": 0,
        "rung_1_2_density_mismatch_kcal_mae": 10,
        "rung_3_plus_density_mismatch_kcal_mae": 20,
        "share_bucketing_kcal_mae": 10,
    }
    assert result["waterfall_sum_kcal_mae"] == 40
    assert result["measured_true_mass_bucketed_floor_kcal_mae"] == 40
