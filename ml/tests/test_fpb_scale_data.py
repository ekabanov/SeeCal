import json

from PIL import Image

from make_fpb_scale_data import build


def test_fpb_converter_sums_object_weights_and_groups_views(tmp_path):
    for split in ("train", "val"):
        root = tmp_path / "FPB_Dataset" / "RGB" / split
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        for view in (1, 2):
            stem = f"rice_average_RGB_IMG_1_jpg.rf.hash{view}"
            Image.new("RGB", (10, 10)).save(root / "images" / f"{stem}.jpg")
            (root / "labels" / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.2 0.2 80\n1 0.5 0.5 0.2 0.2 20\n"
            )
    output = tmp_path / "scale"
    metadata = build(dataset_root=tmp_path, output_dir=output, ml_root=tmp_path)
    rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert {row["group_id"] for row in rows} == {
        "fpb:rice_average_rgb_img_1_jpg"
    }
    assert {row["total_mass_g"] for row in rows} == {100}
    assert metadata["counts"]["train"] == {
        "records": 2,
        "groups": 1,
        "skipped_incomplete_weight_records": 0,
        "skipped_test_capture_overlap_records": 0,
        "orphan_label_records": 0,
    }


def test_fpb_converter_can_build_clean_test_subset(tmp_path):
    root = tmp_path / "FPB_Dataset" / "RGB" / "test"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for stem, weight in (("rice_small_RGB_1", "80"), ("rice_big_RGB_2", "-1")):
        Image.new("RGB", (10, 10)).save(root / "images" / f"{stem}.jpg")
        (root / "labels" / f"{stem}.txt").write_text(
            f"0 0.5 0.5 0.2 0.2 {weight}\n"
        )
    (root / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2 50\n")

    output = tmp_path / "scale"
    metadata = build(
        dataset_root=tmp_path,
        output_dir=output,
        ml_root=tmp_path,
        splits=("test",),
        skip_incomplete_weights=True,
    )

    rows = [
        json.loads(line)
        for line in (output / "test.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["split"] == "test"
    assert rows[0]["total_mass_g"] == 80
    assert metadata["counts"]["test"] == {
        "records": 1,
        "groups": 1,
        "skipped_incomplete_weight_records": 1,
        "skipped_test_capture_overlap_records": 0,
        "orphan_label_records": 1,
    }


def test_fpb_converter_excludes_frozen_test_capture_overlap(tmp_path):
    for split, suffix in (("train", "trainhash"), ("test", "testhash")):
        root = tmp_path / "FPB_Dataset" / "RGB" / split
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        stem = f"rice_average_RGB_IMG_1_jpg.rf.{suffix}"
        Image.new("RGB", (10, 10)).save(root / "images" / f"{stem}.jpg")
        (root / "labels" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2 80\n"
        )

    output = tmp_path / "scale"
    metadata = build(
        dataset_root=tmp_path,
        output_dir=output,
        ml_root=tmp_path,
        splits=("train",),
        exclude_test_capture_overlap=True,
    )

    assert (output / "train.jsonl").read_text() == ""
    assert metadata["counts"]["train"]["skipped_test_capture_overlap_records"] == 1
