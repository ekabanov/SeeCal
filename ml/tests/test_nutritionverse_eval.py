import csv
import json

from PIL import Image

from factored_pipeline.contract import IDENTIFY_PROMPT, validate_identification
from make_nutritionverse_eval import build_records, monolith_records
from prepare_finetune import SYSTEM_PROMPT, USER_PROMPT


def test_nutritionverse_metadata_preserves_measured_ground_truth(tmp_path):
    metadata = tmp_path / "metadata.csv"
    fieldnames = [
        "dish_id",
        "total_food_weight",
        "total_calories",
        "total_fats",
        "total_carbohydrates",
        "total_protein",
        "food_item_type_1",
        "food_weight_g_1",
        "calories(kCal)_1",
        "fat(g)_1",
        "carbohydrates(g)_1",
        "protein(g)_1",
        "food_item_type_2",
        "food_weight_g_2",
        "calories(kCal)_2",
        "fat(g)_2",
        "carbohydrates(g)_2",
        "protein(g)_2",
    ]
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dish_id": 7,
                "total_food_weight": 100,
                "total_calories": 150,
                "total_fats": 2,
                "total_carbohydrates": 25,
                "total_protein": 8,
                "food_item_type_1": "white-rice",
                "food_weight_g_1": 75,
                "calories(kCal)_1": 100,
                "fat(g)_1": 1,
                "carbohydrates(g)_1": 20,
                "protein(g)_1": 4,
                "food_item_type_2": "green-beans",
                "food_weight_g_2": 25,
                "calories(kCal)_2": 50,
                "fat(g)_2": 1,
                "carbohydrates(g)_2": 5,
                "protein(g)_2": 4,
            }
        )
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 10), "white").save(images / "dish_7_IMG_1.jpg")
    records = build_records(
        metadata_csv=metadata,
        image_root=images,
        ml_root=tmp_path,
    )
    assert len(records) == 1
    assert records[0]["group_id"] == "nv-real:dish_7"
    assert records[0]["messages"][0]["content"][1]["text"] == IDENTIFY_PROMPT
    completion = json.loads(records[0]["messages"][1]["content"][0]["text"])
    validate_identification(completion)
    assert completion["items"] == [
        {"name": "white rice", "portion_units": 15},
        {"name": "green beans", "portion_units": 5},
    ]
    ground_truth = records[0]["evaluation_ground_truth"]
    assert ground_truth["total_mass_g"] == 100
    assert ground_truth["items"][0]["estimated_grams"] == 75
    monolith = monolith_records(records)
    assert monolith[0]["messages"][0]["content"][1]["text"] == (
        SYSTEM_PROMPT + "\n\n" + USER_PROMPT
    )
    monolith_truth = json.loads(monolith[0]["messages"][1]["content"][0]["text"])
    assert monolith_truth["total_calories"] == 150
    assert monolith_truth["items"][1]["name"] == "green beans"
