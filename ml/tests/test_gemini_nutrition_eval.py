import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teacher_labeling.gemini_collect import parse_batch_results
from teacher_labeling.gemini_nutrition_eval import _percentile


def _result(value):
    return json.dumps(
        {
            "key": "nutrition5k-dish_1",
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(value)}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                },
            },
        }
    ).encode()


def test_numeric_batch_result_parser():
    rows, usage = parse_batch_results(
        _result(
            {
                "total_calories": 321,
                "protein_g": 12.5,
                "fat_g": 9,
                "carbs_g": 44,
            }
        ),
        expected_records=1,
        response_kind="nutrition",
    )
    assert rows[0]["nutrition"]["total_calories"] == 321
    assert usage["successful_records"] == 1


def test_percentile_interpolates():
    assert _percentile([0, 10, 20, 30, 40], 0.75) == 30
