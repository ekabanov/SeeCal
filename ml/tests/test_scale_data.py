import json

from make_scale_data import build


def test_scale_manifest_contains_only_authoritative_mass(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    record = {
        "images": ["dataset_clean/dish_1/overhead.jpg"],
        "messages": [
            {"role": "user", "content": []},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "items": [
                                    {"estimated_grams": 80},
                                    {"estimated_grams": 20},
                                ]
                            }
                        ),
                    }
                ],
            },
        ],
    }
    for split in ("train", "valid", "test"):
        (input_dir / f"{split}.jsonl").write_text(json.dumps(record) + "\n")
    output_dir = tmp_path / "output"
    counts = build(input_dir, output_dir, include_sides=False)
    assert counts == {"train": 1, "valid": 1, "test": 1}
    row = json.loads((output_dir / "train.jsonl").read_text())
    assert row["total_mass_g"] == 100
    assert "calories" not in row
