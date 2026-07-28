import json

from teacher_labeling.gemini_batch import _request_key
from teacher_labeling.merge_teacher_labels import merge_teacher_labels


def test_merge_accepts_source_compatible_names_and_masks_guesses(tmp_path):
    image = tmp_path / "image.jpg"
    record = {
        "source": {
            "dataset": "food-recognition-benchmark-2022",
            "release": "v2.0",
            "image_id": 1,
        },
        "source_labels": [{"category_id": 1, "name": "tomato-raw"}],
        "loss_mask": {
            "semantic_food_fields": True,
            "calories": False,
            "grams": False,
            "macros": False,
            "hidden_ingredients": False,
        },
        "image": {"path": str(image), "width": 10, "height": 10},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = {
        "key": _request_key(record),
        "model_version": "test",
        "semantic": {
            "visible_foods": [
                {
                    "name": "tomato slices",
                    "cooking_state": "raw",
                    "visibility": "clear",
                },
                {
                    "name": "chicken",
                    "cooking_state": "grilled",
                    "visibility": "clear",
                },
            ],
            "container": "plate",
            "mixed_dish": False,
            "occlusion": "none",
            "abstain": False,
            "is_food": True,
            "ambiguity_reason": "",
        },
    }
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    output = tmp_path / "merged.jsonl"

    summary_path = merge_teacher_labels(
        manifest_path=manifest,
        result_paths=[result_path],
        output_path=output,
        model="test-model",
    )

    merged = json.loads(output.read_text(encoding="utf-8"))
    accepted = merged["teacher"]["accepted_enrichment"]["visible_foods"]
    rejected = merged["teacher"]["rejected_visible_foods"]
    assert [item["name"] for item in accepted] == ["tomato slices"]
    assert [item["name"] for item in rejected] == ["chicken"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["numeric_fields_added"] == 0
    assert summary["hidden_ingredients_added"] == 0
