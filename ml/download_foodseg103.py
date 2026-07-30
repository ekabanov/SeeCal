"""Download and materialize the pinned FoodSeg103 mirror used by IDENTIFY.

The authors' archive endpoint was unavailable during implementation. This
uses the Apache-2.0 Hugging Face parquet mirror pinned to the verified commit
whose LFS object hashes are recorded below. Output is gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPOSITORY = "EduardoPacheco/FoodSeg103"
REVISION = "34e1208e14bc3595d544fc8c3f3c6673253fd9ef"
LFS_SHA256 = {
    "train-00000-of-00003.parquet": "7288a62861c06f1c0f677674fd9564aae277a727a146bf31688cf86befa452b7",
    "train-00001-of-00003.parquet": "814ef4e25036a9073a4017e8d05bad80b213df74cc9cd60e972136b5ae2e0ab9",
    "train-00002-of-00003.parquet": "7b16b566c273e049a10d651caf6d28951f6cdf730fba6596e832b451230d6ac1",
    "validation-00000-of-00001.parquet": "73d4d49abc2ef135dfccbce86f395d8e39215cf656d2954b516a88572f24b0e6",
}
FOODSEG103_CATEGORIES = (
    "background", "candy", "egg tart", "french fries", "chocolate",
    "biscuit", "popcorn", "pudding", "ice cream", "cheese butter", "cake",
    "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond",
    "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut",
    "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry",
    "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon",
    "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange",
    "watermelon", "steak", "pork", "chicken duck", "sausage", "fried meat",
    "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread",
    "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings",
    "pasta", "noodles", "rice", "pie", "tofu", "eggplant", "potato",
    "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion",
    "rape", "ginger", "okra", "lettuce", "pumpkin", "cucumber",
    "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli",
    "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts",
    "onion", "pepper", "green beans", "French beans",
    "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom",
    "white button mushroom", "salad", "other ingredients",
)


def verify_file(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"FoodSeg103 source checksum mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )


def verified_source_paths() -> dict[str, Path]:
    paths = {}
    for filename, expected_sha256 in LFS_SHA256.items():
        path = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=f"data/{filename}",
                repo_type="dataset",
                revision=REVISION,
            )
        )
        verify_file(path, expected_sha256)
        paths[filename] = path
    return paths


def materialize(output: Path) -> dict[str, object]:
    verified_source_paths()
    dataset = load_dataset(
        REPOSITORY,
        revision=REVISION,
        cache_dir=str(output / ".cache"),
    )
    counts = {}
    for source_split, destination_split in (
        ("train", "train"),
        ("validation", "validation"),
    ):
        images = output / destination_split / "images"
        masks = output / destination_split / "masks"
        images.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(dataset[source_split]):
            image_id = int(row.get("id", index))
            stem = f"{image_id:08d}"
            image_path = images / f"{stem}.jpg"
            mask_path = masks / f"{stem}.png"
            if not image_path.is_file():
                row["image"].convert("RGB").save(
                    image_path, format="JPEG", quality=95, optimize=True
                )
            if not mask_path.is_file():
                row["label"].convert("L").save(mask_path, format="PNG", optimize=True)
        counts[destination_split] = len(dataset[source_split])
    metadata = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "Apache-2.0",
        "lfs_sha256": LFS_SHA256,
        "categories": list(FOODSEG103_CATEGORIES),
        "counts": counts,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/foodseg103"),
    )
    args = parser.parse_args()
    materialize(args.output)


if __name__ == "__main__":
    main()
