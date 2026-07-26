"""
01_select_images.py
-------------------
Selects up to 3 images per dish from the Nutrition5K dataset and copies them
to a clean output directory structure.

Selection strategy:
  1. realsense_overhead/rgb.png  — always included (best single-image candidate:
                                   consistent top-down view, closest to real-world usage)
  2. side_angles/camera_Aframe005.jpeg — mid-sequence frame, camera A
  3. side_angles/camera_Cframe005.jpeg — mid-sequence frame, camera C (opposite angle)

Depth images (depth_color.png, depth_raw.png) are excluded — not useful for
a deployed system where users photograph food with a phone.

Output structure:
  dataset_clean/
    dish_<id>/
      overhead.jpg          ← renamed from rgb.png
      side_a.jpg            ← renamed from camera_Aframe005.jpeg  (if exists)
      side_c.jpg            ← renamed from camera_Cframe005.jpeg  (if exists)

Usage:
  python 01_select_images.py [--src NUTRITION5K_DIR] [--dst OUTPUT_DIR] [--max-images 3]
"""

import argparse
import shutil
from pathlib import Path


SIDE_ANGLE_CANDIDATES = [
    ("camera_Aframe005.jpeg", "side_a.jpg"),
    ("camera_Cframe005.jpeg", "side_c.jpg"),
]


def select_images_for_dish(
    dish_id: str,
    overhead_dir: Path,
    side_dir: Path,
    out_dir: Path,
    max_images: int,
) -> dict:
    """
    Copies selected images for one dish into out_dir/dish_id/.
    Returns a summary dict with the dish_id and list of copied image paths.
    """
    dish_out = out_dir / dish_id
    dish_out.mkdir(parents=True, exist_ok=True)

    copied = []
    budget = max_images

    # --- 1. Overhead RGB (highest priority) ---
    overhead_src = overhead_dir / dish_id / "rgb.png"
    if overhead_src.exists() and budget > 0:
        dst = dish_out / "overhead.jpg"
        shutil.copy2(overhead_src, dst)
        copied.append(str(dst))
        budget -= 1

    # --- 2. Side angle frames (fill remaining budget) ---
    for src_name, dst_name in SIDE_ANGLE_CANDIDATES:
        if budget <= 0:
            break
        side_src = side_dir / dish_id / src_name
        if side_src.exists():
            dst = dish_out / dst_name
            shutil.copy2(side_src, dst)
            copied.append(str(dst))
            budget -= 1

    return {"dish_id": dish_id, "images": copied}


def main():
    parser = argparse.ArgumentParser(description="Select and copy Nutrition5K images.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).parent / "Nutrition5K",
        help="Path to the Nutrition5K root directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).parent / "dataset_clean",
        help="Output directory for selected images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Maximum images to keep per dish (default: 3). "
             "Use 1 to keep only the overhead shot — best for simulating real-world usage.",
    )
    args = parser.parse_args()

    overhead_dir = args.src / "imagery" / "realsense_overhead"
    side_dir = args.src / "imagery" / "side_angles"

    if not overhead_dir.exists() or not side_dir.exists():
        raise FileNotFoundError(
            f"Expected imagery subdirs not found under {args.src}. "
            "Check that --src points to the Nutrition5K root."
        )

    # Collect all dish IDs from overhead dir (ground truth of what has annotations)
    dish_ids = sorted(p.name for p in overhead_dir.iterdir() if p.is_dir())
    print(f"Found {len(dish_ids)} dishes in {overhead_dir}")
    print(f"Max images per dish: {args.max_images}")
    print(f"Output directory:    {args.dst}\n")

    args.dst.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = 0
    for i, dish_id in enumerate(dish_ids, 1):
        result = select_images_for_dish(
            dish_id=dish_id,
            overhead_dir=overhead_dir,
            side_dir=side_dir,
            out_dir=args.dst,
            max_images=args.max_images,
        )
        if result["images"]:
            results.append(result)
        else:
            skipped += 1

        if i % 500 == 0:
            print(f"  Processed {i}/{len(dish_ids)} dishes...")

    total_images = sum(len(r["images"]) for r in results)
    print(f"\nDone.")
    print(f"  Dishes processed : {len(results)}")
    print(f"  Dishes skipped   : {skipped} (no images found)")
    print(f"  Total images kept: {total_images}")
    print(f"  Average per dish : {total_images / max(len(results), 1):.1f}")
    print(f"  Output written to: {args.dst}")


if __name__ == "__main__":
    main()
