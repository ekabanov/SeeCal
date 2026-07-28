import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_specialist.compare import compare, load_infer_log, load_predictions


def test_paired_comparison_preserves_direction_and_bins():
    baseline = {
        "a": (50, 100),
        "b": (200, 300),
        "c": (600, 900),
    }
    candidate = {
        "a": (50, 60),
        "b": (200, 220),
        "c": (600, 650),
    }
    result = compare(
        baseline, candidate, bootstrap_samples=1000, seed=1
    )
    assert result["mean_delta_candidate_minus_baseline"] < 0
    assert result["candidate_win_rate"] == 1
    assert result["by_calorie_bin"]["500+"]["n"] == 1


def test_loads_failure_aware_qwen_eval(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "paired_results": [
                    {
                        "id": "dish_1",
                        "status": "ok",
                        "ground_truth": {"total_calories": 100},
                        "prediction": {"total_calories": 110},
                    },
                    {
                        "id": "dish_2",
                        "status": "parse_error",
                        "ground_truth": {"total_calories": 200},
                        "prediction": {"parse_error": True},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_predictions(path) == {"dish_1": (100.0, 110.0)}


def test_loads_only_first_inference_block_from_combined_log(tmp_path):
    path = tmp_path / "eval.log"
    path.write_text(
        "Evaluating on 2 test samples...\n"
        "[1/2] dish_1/overhead.jpg — cal err: 10.0 kcal  |  pred: 90  gt: 100\n"
        "[2/2] dish_2/overhead.jpg — PARSE ERROR\n"
        "Results on 2 samples (1 parse failures, 0 schema failures):\n"
        "Evaluating on 1 test samples...\n"
        "[1/1] dish_1/overhead.jpg — cal err: 50.0 kcal  |  pred: 50  gt: 100\n",
        encoding="utf-8",
    )
    assert load_infer_log(path) == {"dish_1": (100.0, 110.0)}
