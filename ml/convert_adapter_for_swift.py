#!/usr/bin/env python3
"""Convert an mlx-vlm LoRA adapter to the format mlx-swift-lm's LoRAContainer loads.

mlx-swift-lm (Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift) expects:
  - adapters.safetensors with keys ending `.lora_a` / `.lora_b`
    (shapes: lora_a [in_dims, rank], lora_b [rank, out_dims] — same layout as
    mlx-vlm's `.A` / `.B`, so no transpose is needed)
  - adapter_config.json in mlx-lm style:
      {"fine_tune_type": "lora", "num_layers": N,
       "lora_parameters": {"rank": R, "scale": S, "keys": [...]}}

Scale semantics (IMPORTANT):
  - Legacy mlx-vlm (<0.5.0, e.g. the patched 0.4.0 venv that trained adapters_v4)
    computes `y + alpha * (x @ A @ B)` — effective scale = alpha (32.0 here),
    NOT alpha/rank.
  - mlx-vlm >= 0.5.0 uses the standard convention: scale = alpha / rank.
  - mlx-swift-lm's LoRALinear/QLoRALinear compute `y + scale * (x @ lora_a @ lora_b)`,
    so we emit the *effective* scale directly.

Format detection: keys ending `.A`/`.B` → legacy; keys ending `.lora_a`/`.lora_b`
→ new format (weights pass through, only the config may need conversion).

The emitted `lora_parameters.keys` lists the module key paths (relative to each
decoder layer) that actually carry adapter weights, so the Swift side only wraps
those Linear layers instead of every Linear in the layer.

Usage:
    python convert_adapter_for_swift.py [--adapter-path adapters_v4] \
        [--output-path adapters_v4_swift]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
# module path relative to the decoder layer, e.g. "self_attn.q_proj"
RELATIVE_KEY_RE = re.compile(r"\.layers\.\d+\.(.+)\.(?:A|B|lora_a|lora_b)$")


def detect_format(keys):
    legacy = sum(1 for k in keys if k.endswith(".A") or k.endswith(".B"))
    new = sum(1 for k in keys if k.endswith(".lora_a") or k.endswith(".lora_b"))
    if legacy and new:
        sys.exit(f"ERROR: mixed key styles ({legacy} legacy, {new} new) — refusing to convert")
    if legacy == len(keys):
        return "legacy"
    if new == len(keys):
        return "new"
    unknown = [k for k in keys
               if not k.endswith((".A", ".B", ".lora_a", ".lora_b"))]
    sys.exit(f"ERROR: unrecognised adapter key style, e.g. {unknown[:3]}")


def convert_key(key):
    if key.endswith(".A"):
        return key[:-2] + ".lora_a"
    if key.endswith(".B"):
        return key[:-2] + ".lora_b"
    return key


def effective_scale(config, fmt):
    """Return (rank, effective_scale) for the Swift-side y + scale*(x@A@B)."""
    if "lora_parameters" in config:
        # Already mlx-lm style — scale is already effective.
        params = config["lora_parameters"]
        return int(params["rank"]), float(params["scale"])
    rank = int(config["rank"])
    alpha = float(config["alpha"])
    if fmt == "legacy":
        # Legacy mlx-vlm applied y + alpha*(x@A@B): effective scale == alpha.
        return rank, alpha
    # mlx-vlm >= 0.5.0: standard scale = alpha / rank.
    return rank, alpha / rank


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter-path", default="adapters_v4",
                        help="mlx-vlm adapter directory (adapter_config.json + adapters.safetensors)")
    parser.add_argument("--output-path", default="adapters_v4_swift",
                        help="output directory for the Swift-loadable adapter")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_path)
    out_dir = Path(args.output_path)

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    weights = mx.load(str(adapter_dir / "adapters.safetensors"))

    keys = sorted(weights.keys())
    fmt = detect_format(keys)
    rank, scale = effective_scale(config, fmt)

    converted = {}
    renamed = 0
    for key in keys:
        new_key = convert_key(key)
        if new_key != key:
            renamed += 1
        converted[new_key] = weights[key]  # shapes/dtypes preserved, no transpose

    # Sanity checks: pair completeness and rank consistency.
    bases_a = {k[:-len(".lora_a")] for k in converted if k.endswith(".lora_a")}
    bases_b = {k[:-len(".lora_b")] for k in converted if k.endswith(".lora_b")}
    if bases_a != bases_b:
        sys.exit(f"ERROR: unpaired lora_a/lora_b keys: {sorted(bases_a ^ bases_b)[:5]}")
    for base in sorted(bases_a):
        a, b = converted[base + ".lora_a"], converted[base + ".lora_b"]
        if a.shape[1] != rank or b.shape[0] != rank:
            sys.exit(f"ERROR: {base}: A{a.shape} / B{b.shape} inconsistent with rank {rank}")

    layer_indices = sorted({int(m.group(1)) for k in converted
                            if (m := LAYER_RE.search(k))})
    num_layers = layer_indices[-1] + 1 if layer_indices else 0
    if layer_indices != list(range(num_layers)):
        print(f"WARNING: non-contiguous layer coverage: {layer_indices}")

    relative_keys = sorted({m.group(1) for k in keys
                            if (m := RELATIVE_KEY_RE.search(k))})

    swift_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "keys": relative_keys,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), converted)
    (out_dir / "adapter_config.json").write_text(json.dumps(swift_config, indent=2) + "\n")

    # Summary
    dtypes = sorted({str(v.dtype) for v in converted.values()})
    prefixes = sorted({LAYER_RE.sub(".layers.N.", k) for k in converted})
    print(f"source format:   {fmt}")
    print(f"total keys:      {len(converted)} ({renamed} renamed)")
    print(f"adapted modules: {len(bases_a)} across layers 0..{num_layers - 1}")
    print(f"rank={rank} effective_scale={scale} dtypes={dtypes}")
    print("module key paths (per layer):")
    for k in relative_keys:
        print(f"  {k}")
    print("key patterns:")
    for p in prefixes:
        print(f"  {p}")
    print(f"wrote {out_dir / 'adapters.safetensors'}")
    print(f"wrote {out_dir / 'adapter_config.json'}:")
    print(json.dumps(swift_config, indent=2))


if __name__ == "__main__":
    main()
