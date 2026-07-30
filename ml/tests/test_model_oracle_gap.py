from model_oracle_gap import compare_model_to_oracle


def test_model_oracle_gap_reports_conditional_gap_and_incomplete_coverage():
    oracle = {
        "paired": {
            "p50_mass_bucketed_shares": [
                {
                    "id": "a",
                    "group_id": "a",
                    "complete": True,
                    "calorie_absolute_error": 10,
                },
                {
                    "id": "b",
                    "group_id": "b",
                    "complete": True,
                    "calorie_absolute_error": 20,
                },
            ]
        }
    }
    model = {
        "paired": [
            {
                "id": "a",
                "group_id": "a",
                "complete": True,
                "calorie_absolute_error": 30,
            },
            {
                "id": "b",
                "group_id": "b",
                "complete": False,
                "calorie_absolute_error": 0,
            },
        ]
    }

    result = compare_model_to_oracle(
        model_assembly=model,
        oracle_audit=oracle,
    )

    assert result["model_completion_rate_on_oracle_scope"] == 0.5
    assert result["model_kcal_mae_shared_complete"] == 30
    assert result["oracle_kcal_mae_shared_complete"] == 10
    assert result["model_minus_oracle_kcal_mae"] == 20
