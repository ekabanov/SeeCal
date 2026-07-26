"""
patch_fused_vision.py
---------------------
The fused model produced by mlx_lm.fuse is missing vision tower weights
(mlx-lm only handles the language model). This script copies the vision
tower weights from the original model into the fused model directory and
updates the safetensors index file.

Usage:
  python patch_fused_vision.py \
    --original-model ~/models/Qwen3.5-4B-MLX-bf16 \
    --fused-model ./fused-model

After running, the fused-model directory will be a complete multimodal model
that mlx-vlm can load directly (no adapter needed).
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

import mlx.core as mx


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            return json.load(f)
    # Single-shard model — build a fake index using mx.load
    shard = model_dir / "model.safetensors"
    if shard.exists():
        keys = list(mx.load(str(shard)).keys())
        return {"weight_map": {k: "model.safetensors" for k in keys}}
    raise FileNotFoundError(f"No safetensors index or model.safetensors found in {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Patch fused model with vision tower weights.")
    parser.add_argument(
        "--original-model", type=Path,
        default=Path("~/models/Qwen3.5-4B-MLX-bf16"),
        help="Path to the original (unfused) base model.",
    )
    parser.add_argument(
        "--fused-model", type=Path, default=Path("./fused-model"),
        help="Path to the fused model directory (output of mlx_lm.fuse).",
    )
    args = parser.parse_args()

    orig_dir = args.original_model.expanduser()
    fused_dir = args.fused_model.expanduser()

    print(f"Original model: {orig_dir}")
    print(f"Fused model:    {fused_dir}")

    # --- Load indexes ---
    print("\nLoading weight indexes...")
    orig_index = load_index(orig_dir)
    fused_index = load_index(fused_dir)

    orig_weight_map = orig_index["weight_map"]
    fused_weight_map = fused_index["weight_map"]

    # --- Find weights present in original but missing from fused model ---
    fused_keys = set(fused_weight_map.keys())
    orig_keys = set(orig_weight_map.keys())
    missing_keys = orig_keys - fused_keys

    print(f"\nOriginal model weights: {len(orig_keys)}")
    print(f"Fused model weights:    {len(fused_keys)}")
    print(f"Missing from fused:     {len(missing_keys)}")

    if not missing_keys:
        print("\nNo missing weights — fused model is already complete!")
        return

    # Show categories of missing keys
    categories = defaultdict(int)
    for k in missing_keys:
        prefix = k.split(".")[0]
        categories[prefix] += 1
    print("\nMissing weight categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} weights")

    # --- Group missing keys by which shard they come from ---
    missing_by_shard = defaultdict(list)
    for k in missing_keys:
        shard = orig_weight_map[k]
        missing_by_shard[shard].append(k)

    print(f"\nNeed to extract from {len(missing_by_shard)} original shard(s):")
    for shard, keys in missing_by_shard.items():
        print(f"  {shard}: {len(keys)} weights")

    # --- Extract missing weights from original shards using mlx.load (handles bfloat16) ---
    extracted = {}
    for shard_name, keys in missing_by_shard.items():
        shard_path = orig_dir / shard_name
        print(f"\nLoading {shard_name}...")
        # mx.load natively handles bfloat16 (safe_open with numpy/mlx framework does not)
        shard_data = dict(mx.load(str(shard_path)))
        for k in keys:
            if k in shard_data:
                extracted[k] = shard_data[k]
            else:
                print(f"  WARNING: key '{k}' not found in shard!")

    print(f"\nExtracted {len(extracted)} weights from original model.")

    # --- Save extracted weights as a new shard in the fused model ---
    vision_shard_name = "vision_tower.safetensors"
    vision_shard_path = fused_dir / vision_shard_name
    print(f"Saving vision weights to {vision_shard_path}...")
    mx.save_safetensors(str(vision_shard_path), extracted)

    # --- Update the fused model's index ---
    for k in extracted:
        fused_weight_map[k] = vision_shard_name

    fused_index["weight_map"] = fused_weight_map
    fused_index_path = fused_dir / "model.safetensors.index.json"
    with fused_index_path.open("w") as f:
        json.dump(fused_index, f, indent=2)
    print(f"Updated index: {fused_index_path}")
    print(f"  Total weights now: {len(fused_weight_map)}")

    # --- Also copy tokenizer/processor config files if present ---
    config_files_to_copy = [
        "preprocessor_config.json",
        "chat_template.jinja",
    ]
    for fname in config_files_to_copy:
        src = orig_dir / fname
        dst = fused_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {fname}")

    print("\nDone! The fused model now includes vision tower weights.")
    print("Test with:")
    print("  python 03_infer.py --model-path ./fused-model --image path/to/food.jpg")


if __name__ == "__main__":
    main()
