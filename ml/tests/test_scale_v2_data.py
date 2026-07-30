import csv
import json

from make_scale_v2_data import build


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_scale_v2_preserves_official_phone_holdout_and_groups(tmp_path):
    n5k = tmp_path / "n5k"
    n5k.mkdir()
    for split in ("train", "valid", "test"):
        _write_jsonl(
            n5k / f"{split}.jsonl",
            [
                {
                    "id": f"nutrition5k:{split}:overhead",
                    "group_id": f"n5k-{split}",
                    "image_path": "unused.jpg",
                    "total_mass_g": 100,
                    "source": "nutrition5k",
                    "split": split,
                }
            ],
        )

    nv = tmp_path / "nv.jsonl"
    _write_jsonl(
        nv,
        [
            {
                "id": f"nv-real:dish_{dish}_view_{view}",
                "group_id": f"nv-real:dish_{dish}",
                "image_path": "unused.jpg",
                "total_mass_g": 200,
                "source": "nutritionverse-real-v2",
                "split": "test",
            }
            for dish in (1, 2, 3)
            for view in (1, 2)
        ],
    )
    split_csv = tmp_path / "splits.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file_name", "category"))
        writer.writeheader()
        for dish, category in ((1, "Train"), (2, "Train"), (3, "Val")):
            for view in (1, 2):
                writer.writerow(
                    {
                        "file_name": f"dish_{dish}_IMG_{view}.jpg",
                        "category": category,
                    }
                )

    output = tmp_path / "output"
    metadata = build(
        nutrition5k_dir=n5k,
        nutritionverse_manifest=nv,
        official_split_csv=split_csv,
        output_dir=output,
        validation_fraction=0.5,
        seed=9,
    )

    split_by_group = {}
    for split in ("train", "valid", "test"):
        for line in (output / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            previous = split_by_group.setdefault(row["group_id"], split)
            assert previous == split
    assert split_by_group["nv-real:dish_3"] == "test"
    assert "nv-real:dish_1" not in split_by_group
    assert "nv-real:dish_2" not in split_by_group
    assert metadata["counts"]["test"]["groups_by_source"][
        "nutritionverse-real-v2"
    ] == 1
    assert metadata["nutritionverse_policy"]["noncommercial_training_included"] is False
    research = [
        json.loads(line)
        for split in ("train", "valid")
        for line in (
            output / f"research-{split}-nutritionverse-real.jsonl"
        ).read_text().splitlines()
    ]
    assert {row["group_id"] for row in research} == {
        "nv-real:dish_1",
        "nv-real:dish_2",
    }


def test_scale_v2_keeps_fpb_official_validation_out_of_training(tmp_path):
    n5k = tmp_path / "n5k"
    n5k.mkdir()
    for split in ("train", "valid", "test"):
        _write_jsonl(
            n5k / f"{split}.jsonl",
            [
                {
                    "id": f"nutrition5k:{split}:overhead",
                    "group_id": f"n5k-{split}",
                    "image_path": "unused.jpg",
                    "total_mass_g": 100,
                    "source": "nutrition5k",
                    "split": split,
                }
            ],
        )

    nv = tmp_path / "nv.jsonl"
    _write_jsonl(
        nv,
        [
            {
                "id": "nv-real:dish_1_view_1",
                "group_id": "nv-real:dish_1",
                "image_path": "unused.jpg",
                "total_mass_g": 200,
                "source": "nutritionverse-real-v2",
                "split": "test",
            }
        ],
    )
    split_csv = tmp_path / "splits.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file_name", "category"))
        writer.writeheader()
        writer.writerow({"file_name": "dish_1_IMG_1.jpg", "category": "Val"})

    fpb = tmp_path / "fpb"
    fpb.mkdir()
    for split in ("train", "valid"):
        _write_jsonl(
            fpb / f"{split}.jsonl",
            [
                {
                    "id": f"fpb:{split}:capture",
                    "group_id": f"fpb:{split}:capture",
                    "image_path": "unused.jpg",
                    "total_mass_g": 150,
                    "source": "food-portion-benchmark",
                    "split": split,
                }
            ],
        )

    output = tmp_path / "output"
    metadata = build(
        nutrition5k_dir=n5k,
        nutritionverse_manifest=nv,
        official_split_csv=split_csv,
        output_dir=output,
        validation_fraction=0.5,
        seed=9,
        fpb_manifest_dir=fpb,
    )

    train_ids = {
        json.loads(line)["id"]
        for line in (output / "train.jsonl").read_text().splitlines()
    }
    valid_ids = {
        json.loads(line)["id"]
        for line in (output / "valid.jsonl").read_text().splitlines()
    }
    assert "fpb:train:capture" in train_ids
    assert "fpb:valid:capture" not in train_ids
    assert "fpb:valid:capture" in valid_ids
    assert metadata["food_portion_benchmark_policy"][
        "official_val_used_for_validation"
    ]
