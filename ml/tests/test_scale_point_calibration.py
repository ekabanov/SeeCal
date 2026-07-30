import json

from scale_point_calibration import (
    apply_calibration,
    fit_calibration,
    transform_value,
)


def _rows(source, groups, *, multiplier):
    return [
        {
            "id": f"{source}:{group}",
            "group_id": f"{source}:{group}",
            "source": source,
            "target_mass_g": float((group + 1) * 100),
            "p10_g": float((group + 1) * 70 / multiplier),
            "p50_g": float((group + 1) * 100 / multiplier),
            "p90_g": float((group + 1) * 130 / multiplier),
            "absolute_error_g": 0.0,
            "absolute_percentage_error": 0.0,
            "covered": True,
        }
        for group in range(groups)
    ]


def test_log_affine_transform_is_nonnegative_and_monotone():
    parameters = {"slope": 1.2, "intercept": -0.3}
    values = [
        transform_value("log_affine", parameters, value)
        for value in (0, 10, 100)
    ]
    assert values == sorted(values)
    assert values[0] >= 0


def test_fit_and_apply_selects_material_source_calibration(tmp_path):
    rows = _rows("nutrition5k", 10, multiplier=1.0)
    rows += _rows("nutritionverse-real-v2", 10, multiplier=2.0)
    evaluation = tmp_path / "valid.json"
    evaluation.write_text(json.dumps({"paired": rows}))

    result = fit_calibration(
        evaluation,
        folds=5,
        seed="fixture",
        minimum_relative_improvement=0.02,
        target_coverage=0.8,
        runtime_default_source="nutritionverse-real-v2",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(result))

    assert result["sources"]["nutrition5k"]["selected"] == "identity"
    assert result["sources"]["nutritionverse-real-v2"]["selected"] != "identity"
    applied = apply_calibration(evaluation, calibration)
    assert applied["metrics"]["equal_group_mass_mae_g"] < 1e-6
    assert applied["metrics"]["equal_group_p10_p90_coverage"] >= 0.8
