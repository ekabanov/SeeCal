import json

from nutritionverse_quality_slice import build_quality_slice


def _record(record_id, group_id, name, *, protein, fat, carbs, calories):
    return {
        "id": record_id,
        "group_id": group_id,
        "evaluation_ground_truth": {
            "items": [
                {
                    "name": name,
                    "estimated_grams": 100,
                    "protein_g": protein,
                    "fat_g": fat,
                    "carbs_g": carbs,
                    "calories": calories,
                }
            ]
        },
    }


def test_quality_slice_uses_training_anomaly_and_drops_whole_eval_group(tmp_path):
    training = tmp_path / "train.jsonl"
    training.write_text(
        "".join(
            json.dumps(
                _record(
                    f"train-{index}",
                    f"train-{index}",
                    "whole chicken",
                    protein=9,
                    fat=20,
                    carbs=60,
                    calories=456,
                )
            )
            + "\n"
            for index in range(5)
        )
    )
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text(
        json.dumps(
            _record(
                "bad-a",
                "bad",
                "whole chicken",
                protein=30,
                fat=10,
                carbs=0,
                calories=210,
            )
        )
        + "\n"
        + json.dumps(
            _record(
                "bad-b",
                "bad",
                "rice",
                protein=3,
                fat=1,
                carbs=30,
                calories=141,
            )
        )
        + "\n"
        + json.dumps(
            _record(
                "good",
                "good",
                "rice",
                protein=3,
                fat=1,
                carbs=30,
                calories=141,
            )
        )
        + "\n"
    )
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps(
            {
                "entries": {
                    "whole chicken": {"family": "poultry"},
                    "rice": {"family": "grain_starch"},
                }
            }
        )
    )
    output = tmp_path / "quality.jsonl"

    result = build_quality_slice(training, evaluation, taxonomy, output)

    assert set(result["excluded_labels"]) == {"whole chicken"}
    assert result["excluded_groups"] == 1
    assert result["output_groups"] == 1
    assert json.loads(output.read_text())["id"] == "good"
