import json
from pathlib import Path

import pytest

from teacher_labeling.food_recognition_2022 import (
    DatasetValidationError,
    load_and_validate_coco,
    select_stratified_pilot,
    write_pilot_manifest,
)


def _write_coco(path: Path) -> None:
    categories = [
        {"id": 1, "name": "common rice"},
        {"id": 2, "name": "rare curry"},
        {"id": 3, "name": "fried sauce"},
    ]
    images = [
        {
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "width": 100,
            "height": 80,
        }
        for image_id in range(1, 13)
    ]
    annotations = []
    annotation_id = 1
    for image_id in range(1, 13):
        category_ids = [1]
        if image_id in {1, 2, 3}:
            category_ids.append(2)
        if image_id in {1, 4, 5, 6}:
            category_ids.append(3)
        for category_id in category_ids:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [],
                }
            )
            annotation_id += 1
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
        ),
        encoding="utf-8",
    )


def test_load_validate_and_select_is_deterministic_and_covers_rare_classes(tmp_path):
    annotations = tmp_path / "annotations.json"
    _write_coco(annotations)
    images, categories, stats = load_and_validate_coco(
        annotations,
        expected_images=12,
        expected_annotations=19,
        expected_classes=3,
    )

    first = select_stratified_pilot(
        images,
        categories,
        sample_size=6,
        seed=42,
        min_per_class=2,
    )
    second = select_stratified_pilot(
        images,
        categories,
        sample_size=6,
        seed=42,
        min_per_class=2,
    )

    assert [item.image_id for item in first] == [item.image_id for item in second]
    assert {category for item in first for category in item.category_ids} == {1, 2, 3}
    assert stats["multi_class_images"] == 6
    assert len(stats["annotations_sha256"]) == 64


def test_invalid_annotation_reference_fails_closed(tmp_path):
    annotations = tmp_path / "annotations.json"
    _write_coco(annotations)
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    payload["annotations"][0]["image_id"] = 999
    annotations.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="unknown image"):
        load_and_validate_coco(annotations)


def test_selector_excludes_known_duplicate_ids(tmp_path):
    annotations = tmp_path / "annotations.json"
    _write_coco(annotations)
    images, categories, _stats = load_and_validate_coco(annotations)

    selected = select_stratified_pilot(
        images,
        categories,
        sample_size=6,
        seed=42,
        min_per_class=2,
        excluded_image_ids=frozenset({1}),
    )

    assert 1 not in {item.image_id for item in selected}


def test_manifest_masks_numeric_targets_and_checks_files(tmp_path):
    annotations = tmp_path / "annotations.json"
    _write_coco(annotations)
    images, categories, stats = load_and_validate_coco(annotations)
    selected = select_stratified_pilot(
        images,
        categories,
        sample_size=3,
        seed=7,
        min_per_class=1,
    )
    images_root = tmp_path / "images"
    images_root.mkdir()
    for image in selected:
        (images_root / image.file_name).write_bytes(b"test-image")
    manifest = tmp_path / "pilot.jsonl"

    summary_path = write_pilot_manifest(
        selected=selected,
        categories=categories,
        annotations_path=annotations,
        images_root=images_root,
        output_path=manifest,
        seed=7,
        source_stats=stats,
        verify_files=True,
        release="v2.0",
        archive_sha256="f" * 64,
    )

    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert records[0]["loss_mask"] == {
        "semantic_food_fields": True,
        "calories": False,
        "grams": False,
        "macros": False,
        "hidden_ingredients": False,
    }
    assert records[0]["source"]["release"] == "v2.0"
    assert records[0]["source"]["archive_sha256"] == "f" * 64
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["pilot_stats"]["represented_classes"] == 3
    assert len(summary["manifest_sha256"]) == 64


def test_manifest_refuses_missing_selected_files(tmp_path):
    annotations = tmp_path / "annotations.json"
    _write_coco(annotations)
    images, categories, stats = load_and_validate_coco(annotations)
    selected = select_stratified_pilot(
        images,
        categories,
        sample_size=2,
        min_per_class=1,
    )

    with pytest.raises(DatasetValidationError, match="files are missing"):
        write_pilot_manifest(
            selected=selected,
            categories=categories,
            annotations_path=annotations,
            images_root=tmp_path / "missing",
            output_path=tmp_path / "pilot.jsonl",
            seed=1,
            source_stats=stats,
            verify_files=True,
        )
