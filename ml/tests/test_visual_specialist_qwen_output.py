import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_specialist.evaluate_qwen_output import evaluate_output, normalize_name


def test_name_normalization_removes_case_punctuation_and_parentheticals():
    assert normalize_name("White-rice (cooked)") == "white rice"


def test_output_evaluation_pools_names_and_checks_item_sums(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "paired_results": [
                    {
                        "status": "ok",
                        "prediction": {
                            "total_calories": 100,
                            "protein_g": 10,
                            "fat_g": 5,
                            "carbs_g": 20,
                            "items": [
                                {
                                    "name": "White Rice",
                                    "calories": 90,
                                    "protein_g": 9,
                                    "fat_g": 5,
                                    "carbs_g": 20,
                                }
                            ],
                        },
                        "ground_truth": {
                            "total_calories": 105,
                            "protein_g": 10,
                            "fat_g": 5,
                            "carbs_g": 20,
                            "items": [
                                {"name": "white rice (cooked)"},
                                {"name": "chicken"},
                            ]
                        },
                    },
                    {"status": "parse_error"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_output(path)
    assert result["valid_predictions"] == 1
    assert result["normalized_exact_name_micro"]["true_positive"] == 1
    assert result["normalized_exact_name_micro"]["false_negative"] == 1
    assert result["declared_vs_item_sum_mae"]["total_calories"] == 10
    assert result["item_sum_vs_ground_truth_mae"]["total_calories"] == 15
