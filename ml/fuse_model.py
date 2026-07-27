#!/usr/bin/env python
"""Fuse a trained LoRA adapter into the base model, offline.

Produces a standalone MLX model directory that needs NO adapter at load time —
the LoRA delta is merged into the base weights once, here, instead of every app
launch. This is the exact operation the iOS app did at load (`adapter.fuse`),
lifted out to a one-time offline step so distribution can ship a single model
and the app skips the fuse-time memory transient.

Quantization: by default the fused weights are RE-QUANTIZED to the base model's
own per-layer format (the 4-bit MLX model → ~2.3 GB out, on-device ready). This
mirrors what fuse-at-load produced. Pass --dequantize for a full-precision
(fp16) fused model instead (~8 GB; not for on-device use).

Run from ml/ like every other pipeline command.

Examples:
  ./.venv/bin/python fuse_model.py \
      --base ~/models/mlx-community/Qwen3.5-4B-MLX-4bit \
      --adapter-path adapters_v5 --out-path fused_v5

  # also publish to a (free, public) Hugging Face repo:
  ./.venv/bin/python fuse_model.py --base ... --adapter-path adapters_v5 \
      --out-path fused_v5 --upload-repo <user>/seecal-qwen3.5-4b-v5
"""
import argparse
import glob
import shutil
from pathlib import Path

import mlx.nn as nn  # noqa: F401  (imported for parity with mlx_vlm expectations)
from mlx.utils import tree_unflatten

from mlx_vlm.trainer.utils import apply_lora_layers
from mlx_vlm.utils import (
    create_model_card,
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    upload_to_hub,
)


def fuse_adapter(
    base: str,
    adapter_path: str,
    out_path: str,
    dequantize: bool = False,
    upload_repo: str | None = None,
) -> None:
    adapter_dir = Path(adapter_path)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(f"no adapter_config.json in {adapter_dir}")
    if not (adapter_dir / "adapters.safetensors").is_file():
        raise SystemExit(f"no adapters.safetensors in {adapter_dir}")

    print(f"[fuse] loading base model: {base}")
    model_path = get_model_path(base)
    model, config, processor = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=True
    )

    print(f"[fuse] applying adapter: {adapter_dir}")
    model = apply_lora_layers(model, str(adapter_dir))

    # Merge every LoRA-wrapped linear back into a plain (or re-quantized) linear.
    fused = [
        (name, module.fuse(dequantize=dequantize))
        for name, module in model.named_modules()
        if hasattr(module, "fuse")
    ]
    if not fused:
        raise SystemExit(
            "no fusable LoRA layers found — did the adapter apply to the model?"
        )
    print(f"[fuse] fusing {len(fused)} LoRA layers (dequantize={dequantize})")
    model.update_modules(tree_unflatten(fused))

    # Text-only fallbacks wrap the real LM under language_model._model; the VLM
    # (our case) is the model itself. Matches mlx_vlm.convert's handling.
    target = (
        model.language_model._model
        if getattr(model, "_is_text_model", False)
        else model
    )

    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[fuse] writing weights to {out}")
    save_weights(out, target, donate_weights=True)

    # Copy the non-weight model files (config/tokenizer/preprocessor/py) so the
    # fused directory is a complete, self-contained model. Skip the base weight
    # index — save_weights wrote the correct one for the fused shards.
    for pattern in ("*.py", "*.json"):
        for f in glob.glob(str(model_path / pattern)):
            if Path(f).name == "model.safetensors.index.json":
                continue
            shutil.copy(f, out)
    for item in model_path.iterdir():
        if item.is_dir():
            dest = out / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
    processor.save_pretrained(out)

    # Provenance: record which adapter was fused in (the iOS ModelInfoResolver
    # already tolerates unknown keys). Keeps the base quantization block intact.
    config["seecal_fused_adapter"] = adapter_dir.name
    save_config(config, config_path=out / "config.json")

    create_model_card(out, None)
    print(f"[fuse] done -> {out}")

    if upload_repo:
        print(f"[fuse] uploading to Hugging Face: {upload_repo}")
        upload_to_hub(out, upload_repo)
        print(f"[fuse] uploaded: https://huggingface.co/{upload_repo}")


def main() -> None:
    default_base = str(Path("~/models/mlx-community/Qwen3.5-4B-MLX-4bit").expanduser())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=default_base, help=f"Base MLX model dir. Default: {default_base}")
    ap.add_argument("--adapter-path", required=True, help="mlx-vlm LoRA adapter dir (e.g. adapters_v5)")
    ap.add_argument("--out-path", required=True, help="Output directory for the fused model")
    ap.add_argument("--dequantize", action="store_true", help="Fuse to fp16 instead of re-quantizing to the base's format")
    ap.add_argument("--upload-repo", default=None, help="Optional Hugging Face repo id to publish the fused model to")
    args = ap.parse_args()
    fuse_adapter(args.base, args.adapter_path, args.out_path, args.dequantize, args.upload_repo)


if __name__ == "__main__":
    main()
