from qwen_mass_audit import audit, compare_scale


def test_qwen_mass_audit_equal_weights_groups_and_rejects_invalid_rows():
    paired = [
        {
            "id": "a1",
            "group_id": "a",
            "status": "ok",
            "ground_truth": {"total_mass_g": 100},
            "prediction": {"items": [{"estimated_grams": 80}]},
        },
        {
            "id": "a2",
            "group_id": "a",
            "status": "ok",
            "ground_truth": {"total_mass_g": 100},
            "prediction": {"items": [{"estimated_grams": 100}]},
        },
        {
            "id": "b1",
            "group_id": "b",
            "status": "ok",
            "ground_truth": {"items": [{"estimated_grams": 200}]},
            "prediction": {"items": [{"estimated_grams": 240}]},
        },
        {
            "id": "bad",
            "group_id": "bad",
            "status": "parse_error",
            "ground_truth": {"total_mass_g": 1},
            "prediction": {},
        },
    ]

    result = audit(paired)

    assert result["records"] == 3
    assert result["groups"] == 2
    assert result["rejected_records"] == 1
    assert result["equal_group_mass_mae_g"] == 25

    comparison = compare_scale(
        result,
        [
            {"id": "a1", "p50_g": 90},
            {"id": "a2", "p50_g": 90},
            {"id": "b1", "p50_g": 210},
        ],
    )
    assert comparison["scale_equal_group_mass_mae_g"] == 10
    assert comparison["paired_scale_minus_qwen_mae_g"] == -15
