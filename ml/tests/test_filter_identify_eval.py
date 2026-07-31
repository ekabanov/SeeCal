import pytest

from filter_identify_eval import filter_evaluation


def test_filter_identify_eval_recomputes_failure_counts():
    payload = {
        "samples": 3,
        "evaluation_seconds": 10,
        "paired_results": [
            {"id": "a", "status": "ok"},
            {"id": "b", "status": "parse_error"},
            {"id": "c", "status": "schema_error"},
        ],
    }

    result = filter_evaluation(
        payload,
        [{"id": "a"}, {"id": "c"}],
    )

    assert result["samples"] == 2
    assert result["parse_failures"] == 0
    assert result["schema_failures"] == 1
    assert result["post_repair_rejection_rate"] == 0.5
    assert result["filtered_from_samples"] == 3


def test_filter_identify_eval_rejects_missing_manifest_ids():
    with pytest.raises(ValueError, match="absent"):
        filter_evaluation(
            {"samples": 1, "paired_results": [{"id": "a", "status": "ok"}]},
            [{"id": "missing"}],
        )
