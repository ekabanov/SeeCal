import json
from pathlib import Path

from factored_pipeline.contract import IDENTIFY_PROMPT, validate_identification
from make_identify_data import (
    deterministically_interleave,
    frb_teacher_records,
    nutrition5k_records,
    segmentation_records,
)


def _training_record(image: str, payload: dict) -> dict:
    return {
        "images": [image],
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": image}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(payload)}],
            },
        ],
    }


def test_nutrition5k_measured_grams_become_portion_units(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            _training_record(
                "dataset_clean/dish_1/overhead.jpg",
                {
                    "items": [
                        {"name": "rice", "estimated_grams": 75},
                        {"name": "beans", "estimated_grams": 25},
                    ]
                },
            )
        )
        + "\n"
    )
    records = nutrition5k_records(source)
    prompt = records[0]["messages"][0]["content"][1]["text"]
    payload = json.loads(records[0]["messages"][1]["content"][0]["text"])
    evaluation = records[0]["evaluation_ground_truth"]
    assert records[0]["id"] == "dish_1"
    assert records[0]["group_id"] == "dish_1"
    assert prompt == IDENTIFY_PROMPT
    assert payload["container"] == "tray"
    assert payload["items"] == [
        {"name": "rice", "portion_units": 15},
        {"name": "beans", "portion_units": 5},
    ]
    assert evaluation["items"] == [
        {
            "name": "rice",
            "estimated_grams": 75,
            "portion_units": 15,
            "share_pct": 75,
        },
        {
            "name": "beans",
            "estimated_grams": 25,
            "portion_units": 5,
            "share_pct": 25,
        },
    ]


def test_segmentation_area_becomes_weak_portion_label(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "one.jpg").write_bytes(b"fixture")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 1, "name": "rice"},
                    {"id": 2, "name": "green-beans"},
                ],
                "images": [{"id": 10, "file_name": "one.jpg"}],
                "annotations": [
                    {"image_id": 10, "category_id": 1, "area": 80},
                    {"image_id": 10, "category_id": 2, "area": 20},
                ],
            }
        )
    )
    records = segmentation_records(annotations, image_root, ml_root=tmp_path)
    assert records[0]["id"] == "segmentation:10"
    payload = json.loads(records[0]["messages"][1]["content"][0]["text"])
    validate_identification(payload)
    assert payload["items"] == [
        {"name": "rice", "portion_units": 16},
        {"name": "green beans", "portion_units": 4},
    ]


def test_source_arms_are_deterministically_interleaved():
    records = [
        {"source": source, "index": index}
        for source in ("n5k", "foodseg", "negative")
        for index in range(10)
    ]
    first = deterministically_interleave(records, split="train", seed=20260729)
    second = deterministically_interleave(records, split="train", seed=20260729)
    assert first == second
    assert first != records
    assert {row["source"] for row in first[:7]} == {"n5k", "foodseg", "negative"}


def test_frb_teacher_pilot_teaches_visible_names_not_mass(tmp_path):
    rows = []
    for image_id in range(10):
        image_path = tmp_path / f"{image_id}.jpg"
        image_path.write_bytes(b"fixture")
        rows.append(
            {
                "image": {"path": image_path.name},
                "source": {"image_id": image_id, "release": "v2.0"},
                "teacher": {
                    "accepted_enrichment": {
                        "container": "cup_or_glass",
                        "visible_foods": [
                            {"name": "rice"},
                            {"name": "beans"},
                        ],
                    }
                },
            }
        )
    manifest = tmp_path / "frb.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    train = frb_teacher_records(
        manifest,
        split="train",
        ml_root=tmp_path,
        validation_fraction=0.2,
        seed=7,
    )
    valid = frb_teacher_records(
        manifest,
        split="valid",
        ml_root=tmp_path,
        validation_fraction=0.2,
        seed=7,
    )

    assert len(train) == 8
    assert len(valid) == 2
    assert {row["id"] for row in train}.isdisjoint(
        {row["id"] for row in valid}
    )
    payload = json.loads(train[0]["messages"][1]["content"][0]["text"])
    assert payload == {
        "not_food": False,
        "container": "cup",
        "items": [
            {"name": "beans", "portion_units": 10},
            {"name": "rice", "portion_units": 10},
        ],
    }


def test_frb_resolution_gate_keeps_only_fully_groundable_records(tmp_path):
    rows = []
    for image_id, names in enumerate(
        [
            ["rice", "beans"],
            ["rice", "mystery garnish"],
            ["beans"],
        ]
    ):
        image_path = tmp_path / f"{image_id}.jpg"
        image_path.write_bytes(b"fixture")
        rows.append(
            {
                "image": {"path": image_path.name},
                "source": {"image_id": image_id, "release": "v2.0"},
                "teacher": {
                    "accepted_enrichment": {
                        "container": "plate",
                        "visible_foods": [{"name": name} for name in names],
                    }
                },
            }
        )
    manifest = tmp_path / "frb.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    train = frb_teacher_records(
        manifest,
        split="train",
        ml_root=tmp_path,
        validation_fraction=0.5,
        seed=7,
        allowed_names={"rice", "beans"},
    )
    valid = frb_teacher_records(
        manifest,
        split="valid",
        ml_root=tmp_path,
        validation_fraction=0.5,
        seed=7,
        allowed_names={"rice", "beans"},
    )

    assert len(train) + len(valid) == 2
    assert {row["id"] for row in train + valid} == {
        "frb-v2.0:0",
        "frb-v2.0:2",
    }
