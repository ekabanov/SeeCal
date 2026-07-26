"""
02_prepare_finetune.py
----------------------
Generates train / validation / test JSONL files for fine-tuning Qwen3.5
with mlx-lm, using the cleaned image directory produced by 01_select_images.py.

Each JSONL record uses the mlx-vlm multimodal chat format:
  {
    "images": ["relative/path/to/image.jpg"],
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "relative/path/to/image.jpg"},
          {"type": "text",  "text": "<prompt>"}
        ]
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "<structured nutrition response>"}]
      }
    ]
  }

The top-level "images" field is what mlx-vlm 0.6.7 actually reads for pixel
loading; the content-embedded image entry is kept too for readability/older
tooling but is ignored by the 0.6.7 loader.

Image paths are stored relative to the project root (the directory containing
this script, i.e. SeeCal/), which is also the CWD from which training is
launched.  Example: "dataset_clean/dish_xxx/overhead.jpg".  Pass absolute
paths via --image-root to override (produces file:// URIs).

One record is emitted per image (not per dish), so a dish with 3 images
contributes 3 training examples — each asking about the same dish from a
different angle. This trains the model to be robust to viewpoint.

Splits (stratified by dish, not image):
  train : 80%
  valid : 10%
  test  : 10%

Usage:
  python 02_prepare_finetune.py [--clean-dir DATASET_CLEAN] [--nutrition CSV]
                                [--out-dir FINETUNE_DATA] [--seed 42]
                                [--image-root /abs/path/override]
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt & response templates
# ---------------------------------------------------------------------------

USER_PROMPT = (
    "Look at this meal and identify its ingredients and nutritional content. "
    "Provide your answer as a JSON object with the following keys: "
    "total_calories (kcal), protein_g, fat_g, carbs_g, and items "
    "(a list of objects with name, estimated_grams, calories, protein_g, fat_g, carbs_g, "
    "sorted by weight descending)."
)

SYSTEM_PROMPT = (
    "You are a nutrition expert. When shown a photo of a meal, "
    "you identify the ingredients with their weights and estimate the total "
    "nutritional content with high accuracy. Always respond with a valid JSON object."
)


MIN_ITEM_GRAMS = 0.05  # ingredient items below this round to 0.0g and violate
                        # the iOS app's grams > 0 validation; drop them from targets.


def format_assistant_response(
    nutrition_row: dict, ingredients: list[dict]
) -> tuple[str, int]:
    """
    Build the target assistant JSON string combining dish-level nutrition and
    per-ingredient breakdown.

    Ingredients are sorted by grams descending (dominant items first) so the
    model learns to prioritise visually prominent components.
    Values are rounded to 1 decimal — the model learns scale, not spurious precision.

    Items with grams < MIN_ITEM_GRAMS are dropped entirely (they round to 0.0g
    and would violate the iOS app's grams > 0 validation). Dish-level totals
    are NOT recomputed from the remaining items — they come straight from the
    nutrition CSV and stay as-is.

    Schema matches the iOS FoodScanResult / ScanItem Codable structs.

    Returns (json_str, num_items_dropped).
    """
    kept_ingredients = [r for r in ingredients if float(r["grams"]) >= MIN_ITEM_GRAMS]
    num_dropped = len(ingredients) - len(kept_ingredients)
    sorted_ingredients = sorted(
        kept_ingredients, key=lambda r: float(r["grams"]), reverse=True
    )
    text = json.dumps(
        {
            "total_calories": round(float(nutrition_row["calories"]), 1),
            "protein_g":      round(float(nutrition_row["protein"]),  1),
            "fat_g":          round(float(nutrition_row["fat"]),      1),
            "carbs_g":        round(float(nutrition_row["carb"]),     1),
            "items": [
                {
                    "name":            r["ingr_name"],
                    "estimated_grams": round(float(r["grams"]),    1),
                    "calories":        round(float(r["calories"]), 1),
                    "protein_g":       round(float(r["protein"]),  1),
                    "fat_g":           round(float(r["fat"]),      1),
                    "carbs_g":         round(float(r["carb"]),     1),
                }
                for r in sorted_ingredients
            ],
        },
        separators=(", ", ": "),
    )
    return text, num_dropped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nutrition(csv_path: Path) -> dict[str, dict]:
    """Returns {dish_id: {calories, protein, fat, carb, mass}} from the CSV."""
    nutrition = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            nutrition[row["dish_id"]] = row
    return nutrition


def load_ingredients(csv_path: Path) -> dict[str, list[dict]]:
    """Returns {dish_id: [ingr_row, ...]} from dish_ingredients.csv."""
    ingredients: dict[str, list[dict]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            ingredients.setdefault(row["dish_id"], []).append(row)
    return ingredients


def build_record(image_path: str, assistant_text: str) -> dict:
    """Build one JSONL record using Qwen3.5-VL's native array content format.

    No system message — the system instruction is prepended to the user text.
    This resolves two mutually exclusive constraints:
      1. PyArrow (used by HuggingFace datasets inside mlx-vlm) requires ALL
         content fields across the messages array to have the same type.
      2. Qwen3.5's Jinja2 chat template raises "System message cannot contain
         images." for ANY list/array content in the system role — even a
         plain text-only array like [{"type": "text", "text": "..."}].

    Combining both constraints means system must be a string AND user must be
    an array, which is impossible with a uniform pyarrow schema.  The fix:
    drop the system message entirely and fold its text into the user message,
    exactly as mlx-vlm's own Qwen auto-formatter does (lora.py lines 94-104).

    The {"type": "image"} entry causes the Qwen3.5 processor to insert real
    vision tokens (<|vision_start|><|image_pad|><|vision_end|>) during both
    training and inference, giving the LoRA adapters genuine visual grounding.

    A top-level "images" field is also included, duplicating the same path.
    mlx-vlm 0.6.7 reads images ONLY from this top-level field for pixel
    loading — content-embedded image entries are ignored by the 0.6.7 loader
    (they were required by the older 0.4.0 stack). Keeping the content entry
    too makes the record self-describing and harmless under both stacks.
    """
    user_text = SYSTEM_PROMPT + "\n\n" + USER_PROMPT
    return {
        "images": [image_path],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ],
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(records):>5} records → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build mlx-lm fine-tuning JSONL from Nutrition5K."
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path(__file__).parent / "dataset_clean",
        help="Output of 01_select_images.py (default: ./dataset_clean).",
    )
    parser.add_argument(
        "--nutrition",
        type=Path,
        default=Path(__file__).parent / "Nutrition5K" / "dish_nutrition_values.csv",
        help="dish_nutrition_values.csv from Nutrition5K.",
    )
    parser.add_argument(
        "--ingredients",
        type=Path,
        default=Path(__file__).parent / "Nutrition5K" / "dish_ingredients.csv",
        help="dish_ingredients.csv from Nutrition5K.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "finetune_data_v2",
        help="Directory to write train/valid/test JSONL (default: ./finetune_data_v2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible splits."
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.80, help="Fraction for training set."
    )
    parser.add_argument(
        "--valid-frac", type=float, default=0.10, help="Fraction for validation set."
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help=(
            "Optional absolute path prefix for image URIs. "
            "If set, images are stored as 'file://<image-root>/dish_id/img.jpg'. "
            "If omitted (default), images are stored as paths relative to the "
            "project root (the directory containing this script), e.g. "
            "'dataset_clean/dish_id/img.jpg' — NOT relative to --out-dir."
        ),
    )
    parser.add_argument(
        "--only-overhead",
        action="store_true",
        default=False,
        help=(
            "Include only overhead.jpg images (640×480, ~300 image tokens). "
            "Excludes side_a.jpg and side_c.jpg (1920×1080, ~2040 tokens each). "
            "Use this to avoid GPU OOM during training. Matches deployment scenario "
            "(iPhone takes a top-down photo)."
        ),
    )
    args = parser.parse_args()

    if not args.clean_dir.exists():
        raise FileNotFoundError(
            f"Clean image directory not found: {args.clean_dir}\n"
            "Run 01_select_images.py first."
        )

    nutrition = load_nutrition(args.nutrition)
    print(f"Loaded nutrition data for {len(nutrition)} dishes.")
    ingredients = load_ingredients(args.ingredients)
    print(f"Loaded ingredients for   {len(ingredients)} dishes.")

    # ------------------------------------------------------------------
    # Collect dish IDs that exist in BOTH the clean dir AND the CSV
    # ------------------------------------------------------------------
    clean_dishes = sorted(p.name for p in args.clean_dir.iterdir() if p.is_dir())
    matched = [d for d in clean_dishes if d in nutrition]
    missing_nutrition = len(clean_dishes) - len(matched)

    print(f"Clean dishes found  : {len(clean_dishes)}")
    print(f"Matched to CSV      : {len(matched)}")
    if missing_nutrition:
        print(f"Skipped (no CSV row): {missing_nutrition}")

    # ------------------------------------------------------------------
    # Split by dish (stratified shuffle), then expand to per-image records
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    rng.shuffle(matched)

    n = len(matched)
    n_train = int(n * args.train_frac)
    n_valid = int(n * args.valid_frac)

    dish_splits = {
        "train": matched[:n_train],
        "valid": matched[n_train : n_train + n_valid],
        "test":  matched[n_train + n_valid :],
    }

    print(f"\nDish split  →  train: {len(dish_splits['train'])}  "
          f"valid: {len(dish_splits['valid'])}  test: {len(dish_splits['test'])}")

    # ------------------------------------------------------------------
    # Resolve how image paths will be represented in the JSONL
    # ------------------------------------------------------------------
    out_dir_resolved = args.out_dir.resolve()
    clean_dir_resolved = args.clean_dir.resolve()

    def image_str(img_path: Path) -> str:
        """
        Return the string to embed in the JSONL for this image.

        - With --image-root: absolute file:// URI under that root.
          e.g. file:///Users/jevgeni/SeeCal/dataset_clean/dish_xxx/overhead.jpg
        - Default: path relative to the project root (directory containing this
          script), which is also the CWD where training is launched.
          e.g. dataset_clean/dish_xxx/overhead.jpg
        """
        if args.image_root:
            # Reconstruct path under the user-supplied root
            relative_to_clean = img_path.resolve().relative_to(clean_dir_resolved)
            return Path(args.image_root).joinpath(relative_to_clean).as_uri()
        else:
            # Relative to project root (same dir as this script = SeeCal/)
            # so paths like "dataset_clean/dish_xxx/overhead.jpg" resolve
            # correctly when training is run from that directory.
            project_root = Path(__file__).parent.resolve()
            return os.path.relpath(img_path.resolve(), project_root)

    # ------------------------------------------------------------------
    # Build records per image
    # ------------------------------------------------------------------
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    split_records: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    records_with_dropped_items = 0
    total_items_dropped = 0

    for split_name, dish_ids in dish_splits.items():
        for dish_id in dish_ids:
            dish_dir = args.clean_dir / dish_id
            images = sorted(
                p for p in dish_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
                and (not args.only_overhead or p.name == "overhead.jpg")
            )
            if not images:
                continue

            assistant_text, num_dropped = format_assistant_response(
                nutrition[dish_id],
                ingredients.get(dish_id, []),
            )
            if num_dropped:
                records_with_dropped_items += len(images)
                total_items_dropped += num_dropped * len(images)

            for img_path in images:
                record = build_record(image_str(img_path), assistant_text)
                split_records[split_name].append(record)

    print(f"Image records  →  train: {len(split_records['train'])}  "
          f"valid: {len(split_records['valid'])}  test: {len(split_records['test'])}\n")
    print(
        f"Records with items dropped (grams < {MIN_ITEM_GRAMS}): "
        f"{records_with_dropped_items} ({total_items_dropped} item instances total)\n"
    )

    # ------------------------------------------------------------------
    # Write JSONL files
    # ------------------------------------------------------------------
    write_jsonl(split_records["train"], args.out_dir / "train.jsonl")
    write_jsonl(split_records["valid"], args.out_dir / "valid.jsonl")
    write_jsonl(split_records["test"],  args.out_dir / "test.jsonl")

    # ------------------------------------------------------------------
    # Print a sample record for inspection
    # ------------------------------------------------------------------
    if split_records["train"]:
        print("\nSample training record:")
        sample = split_records["train"][0]
        print(json.dumps(sample, indent=2))

    print("\nDone. Fine-tuning data ready in:", args.out_dir)
    print("\nTo start fine-tuning with mlx-vlm:")
    print(
        f"  python -m mlx_vlm.lora \\\n"
        f"    --model-path /Users/jevgenikabanov/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit \\\n"
        f"    --dataset {args.out_dir} \\\n"
        f"    --train-mode sft \\\n"
        f"    --train-on-completions \\\n"
        f"    --lora-rank 16 \\\n"
        f"    --lora-alpha 32 \\\n"
        f"    --batch-size 4 \\\n"
        f"    --iters 1000"
    )


if __name__ == "__main__":
    main()
