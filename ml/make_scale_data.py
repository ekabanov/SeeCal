"""Derive SCALE manifests from measured Nutrition5K mass labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(input_dir: Path, output_dir: Path, *, include_sides: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        rows = []
        with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                source = json.loads(line)
                content = source["messages"][-1]["content"]
                text = content[0]["text"] if isinstance(content, list) else content
                payload = json.loads(text)
                mass = sum(float(item["estimated_grams"]) for item in payload["items"])
                overhead = Path(source["images"][0])
                dish_id = overhead.parent.name
                images = [("overhead", str(overhead))]
                if include_sides and split != "test":
                    for view in ("side_a", "side_c"):
                        candidate = overhead.parent / f"{view}.jpg"
                        if candidate.is_file():
                            images.append((view, str(candidate)))
                for view, image_path in images:
                    rows.append(
                        {
                            "schema_version": 1,
                            "id": f"nutrition5k:{dish_id}:{view}",
                            "group_id": dish_id,
                            "split": split,
                            "image_path": image_path,
                            "total_mass_g": mass,
                            "source": "nutrition5k",
                        }
                    )
        rows.sort(key=lambda row: row["id"])
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        counts[split] = len(rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "total_mass_g",
                "quantiles": [0.1, 0.5, 0.9],
                "counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("finetune_data_v2"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets/scale_v1")
    )
    parser.add_argument("--include-sides", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.input_dir, args.output_dir, include_sides=args.include_sides)))


if __name__ == "__main__":
    main()
