"""
prepare_finetune.py
--------------------
Generates train / validation / test JSONL files for fine-tuning Qwen3.5
with mlx-lm, using the cleaned image directory produced by select_images.py.

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

Image paths are stored relative to the pipeline root (the directory containing
this script, i.e. SeeCal/ml/), which is also the CWD from which training is
launched.  Example: "dataset_clean/dish_xxx/overhead.jpg".  Pass absolute
paths via --image-root to override (produces file:// URIs).

One record is emitted per image (not per dish), so a dish with 3 images
contributes 3 training examples — each asking about the same dish from a
different angle. This trains the model to be robust to viewpoint.

Splits (stratified by dish, not image):
  train : 80%
  valid : 10%
  test  : 10%

Usage (run from ml/):
  python prepare_finetune.py [--clean-dir DATASET_CLEAN] [--nutrition CSV]
                             [--out-dir FINETUNE_DATA] [--seed 42]
                             [--image-root /abs/path/override]
"""

import argparse
import csv
import hashlib
import json
import math
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


# ---------------------------------------------------------------------------
# Depth-variant support (docs/design/2026-07-26-depth-design-brief.md, sec. (f))
# ---------------------------------------------------------------------------

# Variant B depth line, appended to the base user text (which stays byte-identical
# to v5). Verbatim format string pinned by the design brief item 6 — whole-number
# rounding via :.0f, no coverage_cm2 (keep the iOS parity surface minimal).
DEPTH_LINE_FMT = (
    "\n\nEstimated food volume from depth sensor: "
    "~{volume_ml:.0f} ml (max height {max_height_mm:.0f} mm)."
)

# Feature dropout (train split only): drop the depth feature for this % of
# depth-ok train dishes so the no-depth iPhone fallback stays in-distribution.
DEPTH_DROPOUT_PCT = 12

DEPTH_STATS_FILENAME = "depth_stats.csv"
DEPTH_STATS_COLUMNS = ["dish_id", "status", "volume_ml", "max_height_mm", "coverage_cm2"]
DEPTH_STATUSES = {"ok", "missing", "corrupt", "fit_failed"}

# Out-dir defaults per depth mode (brief items 2/6/7).
DEPTH_MODE_OUT_DIRS = {
    "none": "finetune_data_v2",
    "text": "finetune_data_v2d_txt",
    "image": "finetune_data_v2d_img",
}


def depth_dropout(dish_id: str) -> bool:
    """
    Deterministic per-dish feature dropout (brief item 5): drop depth for 12% of
    depth-ok TRAIN dishes, selected from the dish_id alone —
    md5(dish_id) % 100 < 12 — so the choice is stable across runs and identical
    for both variants (A and B drop the same dishes).
    """
    digest = hashlib.md5(dish_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100 < DEPTH_DROPOUT_PCT


def format_depth_line(stats: dict) -> str:
    """Render the exact variant-B depth line from a depth-stats dict."""
    return DEPTH_LINE_FMT.format(
        volume_ml=float(stats["volume_ml"]),
        max_height_mm=float(stats["max_height_mm"]),
    )


def compute_depth_stats_row(clean_dir: Path, dish_id: str) -> dict:
    """
    Compute one depth_stats.csv row for a dish via depth_features.

    status:
      missing    — no depth_raw.png in dataset_clean/<dish>/
      corrupt    — load_depth raised (e.g. zero-byte/unreadable PNG)
      fit_failed — plane_fit/food_stats raised, or any stat is NaN/inf
      ok         — stats computed and finite
    Non-ok dishes get plain (v5-format) records in both variants.
    """
    import depth_features  # deferred: needs numpy+PIL, only for --depth-mode != none

    row = {
        "dish_id": dish_id,
        "status": "",
        "volume_ml": "",
        "max_height_mm": "",
        "coverage_cm2": "",
    }
    depth_path = clean_dir / dish_id / "depth_raw.png"
    if not depth_path.exists():
        row["status"] = "missing"
        return row
    try:
        depth = depth_features.load_depth(depth_path)
    except Exception:
        row["status"] = "corrupt"
        return row
    try:
        plane = depth_features.plane_fit(depth)
        stats = depth_features.food_stats(depth, plane)
    except Exception:
        row["status"] = "fit_failed"
        return row
    values = [stats["volume_ml"], stats["max_height_mm"], stats["coverage_cm2"]]
    if not all(math.isfinite(float(v)) for v in values):
        row["status"] = "fit_failed"
        return row
    row["status"] = "ok"
    row["volume_ml"] = f"{float(stats['volume_ml']):.4f}"
    row["max_height_mm"] = f"{float(stats['max_height_mm']):.4f}"
    row["coverage_cm2"] = f"{float(stats['coverage_cm2']):.4f}"
    return row


def load_depth_stats_csv(path: Path) -> dict[str, dict]:
    stats = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            stats[row["dish_id"]] = row
    return stats


def write_depth_stats_csv(path: Path, rows_by_dish: dict[str, dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEPTH_STATS_COLUMNS)
        writer.writeheader()
        for dish_id in sorted(rows_by_dish):
            writer.writerow(rows_by_dish[dish_id])


def ensure_depth_stats(clean_dir: Path, dish_ids: list[str], recompute: bool) -> dict[str, dict]:
    """
    Load the depth-stats cache (dataset_clean/depth_stats.csv) if present, compute
    any missing dishes, and persist. --recompute-depth-stats forces a full recompute.
    """
    csv_path = clean_dir / DEPTH_STATS_FILENAME
    cached: dict[str, dict] = {}
    if csv_path.exists() and not recompute:
        cached = load_depth_stats_csv(csv_path)
        print(f"Loaded depth stats cache: {len(cached)} dishes from {csv_path}")
    todo = [d for d in dish_ids if d not in cached]
    if todo:
        print(f"Computing depth stats for {len(todo)} dishes...")
        for i, dish_id in enumerate(todo, 1):
            cached[dish_id] = compute_depth_stats_row(clean_dir, dish_id)
            if i % 200 == 0:
                print(f"  depth stats: {i}/{len(todo)} dishes...")
        write_depth_stats_csv(csv_path, cached)
        print(f"Wrote depth stats cache: {len(cached)} dishes → {csv_path}")
    return cached


def ensure_height_image(clean_dir: Path, dish_id: str, recompute: bool) -> Path:
    """
    Render (or reuse) the variant-A height image for a depth-ok dish:
    dataset_clean/<dish>/height.png (grayscale L, 320x240, 0 = plane,
    255 = 120 mm — see depth_features.depth_to_height_image).
    """
    import depth_features  # deferred: needs numpy+PIL, only for --depth-mode image

    out_path = clean_dir / dish_id / "height.png"
    if out_path.exists() and not recompute:
        return out_path
    depth = depth_features.load_depth(clean_dir / dish_id / "depth_raw.png")
    plane = depth_features.plane_fit(depth)
    depth_features.depth_to_height_image(depth, plane).save(out_path)
    return out_path


def dish_ids_per_split_from_jsonl(v2_dir: Path) -> dict[str, set]:
    """Dish-id sets per split extracted from existing v2 JSONL image paths."""
    out: dict[str, set] = {}
    for split in ("train", "valid", "test"):
        ids = set()
        with (v2_dir / f"{split}.jsonl").open() as f:
            for line in f:
                rec = json.loads(line)
                ids.add(Path(rec["images"][0]).parent.name)
        out[split] = ids
    return out


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


def build_record(
    image_path: str,
    assistant_text: str,
    depth_line: str | None = None,
    height_image_path: str | None = None,
) -> dict:
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

    Depth variants (docs/design/2026-07-26-depth-design-brief.md, sec. (f)):
      - depth_line (variant B): appended verbatim to the base user text; the
        base text itself stays byte-identical to v5.
      - height_image_path (variant A): added as a second top-level image and a
        second content image entry ([image rgb, image height, text]); the user
        text is unchanged.
    With both left as None (the default), the record is byte-identical to the
    v5 format.
    """
    user_text = SYSTEM_PROMPT + "\n\n" + USER_PROMPT
    if depth_line is not None:
        user_text = user_text + depth_line
    images = [image_path]
    user_content: list[dict] = [{"type": "image", "image": image_path}]
    if height_image_path is not None:
        images.append(height_image_path)
        user_content.append({"type": "image", "image": height_image_path})
    user_content.append({"type": "text", "text": user_text})
    return {
        "images": images,
        "messages": [
            {
                "role": "user",
                "content": user_content,
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
        help="Output of select_images.py (default: ./dataset_clean).",
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
        default=None,
        help="Directory to write train/valid/test JSONL. Default depends on "
             "--depth-mode: none → ./finetune_data_v2, text → ./finetune_data_v2d_txt, "
             "image → ./finetune_data_v2d_img.",
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
    parser.add_argument(
        "--depth-mode",
        choices=["none", "image", "text"],
        default="none",
        help=(
            "Depth-variant generation (design brief sec. (f)). "
            "'none' (default): byte-identical to the existing v5/v2 output. "
            "'text' (variant B): append the depth-sensor volume line to the user text. "
            "'image' (variant A): add the rendered height image as a second image."
        ),
    )
    parser.add_argument(
        "--recompute-depth-stats",
        action="store_true",
        default=False,
        help="Force recomputation of dataset_clean/depth_stats.csv (and re-render "
             "height images in --depth-mode image) instead of reusing the cache.",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(__file__).parent / DEPTH_MODE_OUT_DIRS[args.depth_mode]

    if not args.clean_dir.exists():
        raise FileNotFoundError(
            f"Clean image directory not found: {args.clean_dir}\n"
            "Run select_images.py first."
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
    # Depth stats (only for depth modes): compute/load the per-dish cache
    # ------------------------------------------------------------------
    depth_stats: dict[str, dict] = {}
    if args.depth_mode != "none":
        depth_stats = ensure_depth_stats(
            args.clean_dir, sorted(matched), args.recompute_depth_stats
        )
        status_counts: dict[str, int] = {}
        for dish_id in matched:
            row = depth_stats.get(dish_id)
            status = row["status"] if row else "missing"
            status_counts[status] = status_counts.get(status, 0) + 1
        n_ok = status_counts.get("ok", 0)
        print("Depth stats over matched dishes: "
              + "  ".join(f"{s}: {status_counts.get(s, 0)}"
                          for s in ("ok", "missing", "corrupt", "fit_failed")))
        print(f"Depth-ok coverage: {n_ok}/{len(matched)} ({100.0 * n_ok / len(matched):.1f}%)")

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
    DEPTH_ARTIFACTS = {"depth_raw.png", "height.png"}  # never training RGB inputs
    split_records: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    emitted_dish_ids: dict[str, set] = {"train": set(), "valid": set(), "test": set()}
    records_with_dropped_items = 0
    total_items_dropped = 0
    depth_records = {"train": 0, "valid": 0, "test": 0}
    train_depth_ok_dishes = 0
    train_dropout_dishes = 0
    dishes_processed = 0

    for split_name, dish_ids in dish_splits.items():
        for dish_id in dish_ids:
            dish_dir = args.clean_dir / dish_id
            images = sorted(
                p for p in dish_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
                and p.name not in DEPTH_ARTIFACTS
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

            # --- Depth feature for this dish (depth modes only) ---
            depth_line = None
            height_image = None
            if args.depth_mode != "none":
                row = depth_stats.get(dish_id)
                depth_ok = row is not None and row["status"] == "ok"
                if split_name == "train" and depth_ok:
                    train_depth_ok_dishes += 1
                # Feature dropout: train split only; valid/test always keep
                # depth when status == ok (brief item 5).
                dropped = (
                    split_name == "train" and depth_ok and depth_dropout(dish_id)
                )
                if dropped:
                    train_dropout_dishes += 1
                if depth_ok and not dropped:
                    if args.depth_mode == "text":
                        depth_line = format_depth_line(row)
                    else:  # image
                        height_path = ensure_height_image(
                            args.clean_dir, dish_id, args.recompute_depth_stats
                        )
                        height_image = image_str(height_path)

            for img_path in images:
                record = build_record(
                    image_str(img_path),
                    assistant_text,
                    depth_line=depth_line,
                    height_image_path=height_image,
                )
                split_records[split_name].append(record)
                if depth_line is not None or height_image is not None:
                    depth_records[split_name] += 1
            emitted_dish_ids[split_name].add(dish_id)

            dishes_processed += 1
            if args.depth_mode != "none" and dishes_processed % 200 == 0:
                print(f"  records: {dishes_processed} dishes processed...")

    print(f"Image records  →  train: {len(split_records['train'])}  "
          f"valid: {len(split_records['valid'])}  test: {len(split_records['test'])}\n")
    print(
        f"Records with items dropped (grams < {MIN_ITEM_GRAMS}): "
        f"{records_with_dropped_items} ({total_items_dropped} item instances total)\n"
    )

    if args.depth_mode != "none":
        print(f"Records with depth feature  →  "
              f"train: {depth_records['train']}  valid: {depth_records['valid']}  "
              f"test: {depth_records['test']}")
        pct = (100.0 * train_dropout_dishes / train_depth_ok_dishes
               if train_depth_ok_dishes else 0.0)
        print(f"Train feature dropout: {train_dropout_dishes}/{train_depth_ok_dishes} "
              f"depth-ok train dishes ({pct:.1f}%)\n")

        # --------------------------------------------------------------
        # Split identity (brief item 4): dish→split assignment must equal
        # finetune_data_v2's. Compare dish-id sets per split.
        # --------------------------------------------------------------
        v2_dir = Path(__file__).parent / "finetune_data_v2"
        v2_splits = dish_ids_per_split_from_jsonl(v2_dir)
        for split_name in ("train", "valid", "test"):
            ours, theirs = emitted_dish_ids[split_name], v2_splits[split_name]
            assert ours == theirs, (
                f"Split identity violated for '{split_name}': "
                f"{len(ours - theirs)} dishes not in v2, "
                f"{len(theirs - ours)} v2 dishes missing "
                f"(e.g. {sorted(ours ^ theirs)[:5]})"
            )
        print("Split identity check vs finetune_data_v2: OK "
              "(dish-id sets identical for train/valid/test)\n")

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
        f"    --model-path ~/models/mlx-community/Qwen3.5-4B-MLX-4bit \\\n"
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
