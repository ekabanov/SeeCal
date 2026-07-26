"""
fix_adapter_for_fuse.py
-----------------------
Patches the mlx-vlm LoRA adapter output so it can be fused with mlx-lm.

mlx-vlm saves adapters with:
  1. Minimal config (only rank, alpha, dropout — missing num_layers)
  2. Weight keys prefixed with "language_model.model.layers..."

mlx-lm fuse expects:
  1. Config with num_layers field
  2. Weight keys prefixed with "model.layers..."

This script:
  - Adds the missing num_layers to adapter_config.json
  - Renames weight keys by stripping the "language_model." prefix
  - Saves to a new directory ready for mlx-lm fuse

Usage:
  python fix_adapter_for_fuse.py [--adapter-path adapters] [--output-path adapters_fixed]

Then fuse with:
  python -m mlx_lm.fuse \
    --model ~/models/Qwen3.5-4B-MLX-bf16 \
    --adapter-path adapters_fixed \
    --save-path ./fused-model
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description="Fix mlx-vlm adapters for mlx-lm fuse.")
    parser.add_argument(
        "--adapter-path", type=Path, default=Path("adapters"),
        help="Input adapter directory from mlx-vlm training.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("adapters_fixed"),
        help="Output directory with fixed adapter files.",
    )
    args = parser.parse_args()

    src = args.adapter_path
    dst = args.output_path
    dst.mkdir(parents=True, exist_ok=True)

    # --- 1. Fix adapter_config.json ---
    config_path = src / "adapter_config.json"
    with config_path.open() as f:
        config = json.load(f)

    print(f"Original mlx-vlm config: {config}")

    # Count how many layers are in the adapter weights
    weights = load_file(str(src / "adapters.safetensors"))
    import re
    layer_indices = set()
    for k in weights:
        m = re.search(r'layers\.(\d+)\.', k)
        if m:
            layer_indices.add(int(m.group(1)))

    num_layers = len(layer_indices)

    # Determine which LoRA modules were used based on weight key patterns
    lora_modules = set()
    for k in weights:
        # Extract the module path between "layers.N." and ".A" or ".B"
        m = re.search(r'layers\.\d+\.(.+)\.[AB]$', k)
        if m:
            lora_modules.add(m.group(1))

    # Convert mlx-vlm config → mlx-lm config format
    # mlx-vlm uses: {rank, alpha, dropout}
    # mlx-lm uses:  {num_layers, lora_parameters: {rank, scale, dropout, keys}}
    # where scale = alpha / rank
    rank = config["rank"]
    alpha = config.get("alpha", rank * 2)  # default alpha = 2*rank
    dropout = config.get("dropout", 0.0)
    scale = alpha / rank

    new_config = {
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": dropout,
            "keys": sorted(lora_modules),
        },
    }

    out_config = dst / "adapter_config.json"
    with out_config.open("w") as f:
        json.dump(new_config, f, indent=2)
    print(f"Fixed mlx-lm config: {json.dumps(new_config, indent=2)}")
    print(f"  → {out_config}")

    # --- 2. Rename weight keys (strip "language_model." prefix) ---
    new_weights = {}
    prefix = "language_model."
    renamed = 0
    for k, v in weights.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            renamed += 1
        else:
            new_key = k
        new_weights[new_key] = v

    out_weights = dst / "adapters.safetensors"
    save_file(new_weights, str(out_weights))
    print(f"\nRenamed {renamed}/{len(weights)} weight keys (stripped '{prefix}' prefix)")
    print(f"  → {out_weights}")

    # Show a few example key mappings
    for old_k in sorted(weights.keys())[:3]:
        new_k = old_k[len(prefix):] if old_k.startswith(prefix) else old_k
        print(f"    {old_k}  →  {new_k}")

    print(f"\nDone! Now fuse with:")
    print(f"  python -m mlx_lm.fuse \\")
    print(f"    --model ~/models/Qwen3.5-4B-MLX-bf16 \\")
    print(f"    --adapter-path {dst} \\")
    print(f"    --save-path ./fused-model")


if __name__ == "__main__":
    main()
