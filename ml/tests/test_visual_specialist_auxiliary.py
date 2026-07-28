import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_specialist.auxiliary import (
    build_conditioned_qwen,
    conformal_margin,
    perturb_payload,
    record_id_from_qwen,
    render_auxiliary_block,
)
from visual_specialist.constants import NUMERIC_FIELDS


def _prediction(record_id="nutrition5k:dish_1:overhead"):
    return {
        "id": record_id,
        "numeric": {
            field: {"p10": 10.0, "p50": 20.0, "p90": 30.0}
            for field in NUMERIC_FIELDS
        },
    }


def _calibration(margin=2.0):
    return {
        "fields": {
            field: {"additive_margin": margin}
            for field in NUMERIC_FIELDS
        }
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_conformal_margin_uses_finite_sample_higher_quantile():
    intervals = [
        (10.0, 20.0, 15.0),
        (10.0, 20.0, 21.0),
        (10.0, 20.0, 22.0),
        (10.0, 20.0, 23.0),
    ]
    assert conformal_margin(intervals, target_coverage=0.5) == 2.0


def test_canonical_render_is_compact_ordered_and_calibrated():
    block = render_auxiliary_block(_prediction(), _calibration())
    expected_payload = {
        "available": True,
        **{
            field: {"estimate": 20.0, "low": 8.0, "high": 32.0}
            for field in NUMERIC_FIELDS
        },
    }
    expected_json = json.dumps(expected_payload, separators=(",", ":"))
    assert block.endswith(expected_json)
    assert "\n" in block
    assert "food_probability" not in block


def test_record_id_from_qwen_requires_supported_overhead_path():
    row = {"images": ["dataset_clean/dish_1/overhead.jpg"]}
    assert record_id_from_qwen(row) == "nutrition5k:dish_1:overhead"
    with pytest.raises(ValueError):
        record_id_from_qwen({"images": ["other.jpg"]})
    assert record_id_from_qwen(
        {"images": ["negatives/000000123456.jpg"]}
    ) == "coco-negative:000000123456"


def test_perturbation_is_deterministic_ordered_and_target_free():
    payload = {
        "available": True,
        **{
            field: {"estimate": 20.0, "low": 10.0, "high": 30.0}
            for field in NUMERIC_FIELDS
        },
    }
    first = perturb_payload(payload, record_id="dish", seed=7)
    second = perturb_payload(payload, record_id="dish", seed=7)
    assert first == second
    assert first != payload
    for field in NUMERIC_FIELDS:
        assert first[field]["low"] <= first[field]["estimate"]
        assert first[field]["estimate"] <= first[field]["high"]


def test_builder_only_appends_user_text(tmp_path):
    source = {
        "images": ["dataset_clean/dish_1/overhead.jpg"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dataset_clean/dish_1/overhead.jpg"},
                    {"type": "text", "text": "original prompt"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        ],
    }
    qwen_path = tmp_path / "qwen.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    calibration_path = tmp_path / "calibration.json"
    output = tmp_path / "conditioned.jsonl"
    _write_jsonl(qwen_path, [source])
    _write_jsonl(prediction_path, [_prediction()])
    calibration_path.write_text(
        json.dumps(_calibration()),
        encoding="utf-8",
    )

    build_conditioned_qwen(
        qwen_manifest_path=qwen_path,
        predictions_path=prediction_path,
        calibration_path=calibration_path,
        output=output,
    )
    built = json.loads(output.read_text(encoding="utf-8"))
    assert built["images"] == source["images"]
    assert built["messages"][1] == source["messages"][1]
    assert built["messages"][0]["content"][0] == source["messages"][0]["content"][0]
    assert built["messages"][0]["content"][1]["text"].startswith(
        "original prompt\n\nAuxiliary visual measurement"
    )


def test_negative_can_be_unavailable_without_a_prediction(tmp_path):
    source = {
        "images": ["negatives/000000123456.jpg"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "negatives/000000123456.jpg"},
                    {"type": "text", "text": "original prompt"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "{\"not_food\":true}"}],
            },
        ],
    }
    qwen_path = tmp_path / "qwen.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    calibration_path = tmp_path / "calibration.json"
    output = tmp_path / "conditioned.jsonl"
    _write_jsonl(qwen_path, [source])
    _write_jsonl(prediction_path, [])
    calibration_path.write_text(
        json.dumps(_calibration()),
        encoding="utf-8",
    )

    metadata = build_conditioned_qwen(
        qwen_manifest_path=qwen_path,
        predictions_path=prediction_path,
        calibration_path=calibration_path,
        output=output,
        negatives_unavailable=True,
    )
    built = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["robustness"]["variant_counts"]["unavailable"] == 1
    assert '{"available":false}' in built["messages"][0]["content"][1]["text"]
