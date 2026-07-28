"""Audit Qwen training lengths without redundantly decoding every image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mlx_vlm.prompt_utils import apply_chat_template
from transformers import AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument(
        "--image-tokens",
        type=int,
        default=300,
        help="Verified Qwen tokens for a Nutrition5K 640x480 overhead image.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.model_dir / "config.json").read_text())
    processor = AutoProcessor.from_pretrained(str(args.model_dir))
    tokenizer = processor.tokenizer
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lengths = []
    completion_lengths = []
    over_limit = []
    first_records = []
    for index, row in enumerate(rows):
        conversation = row["messages"]
        full = apply_chat_template(
            processor,
            config,
            conversation,
            add_generation_prompt=False,
            num_images=1,
        )
        prefix = apply_chat_template(
            processor,
            config,
            conversation[:-1],
            add_generation_prompt=True,
            num_images=1,
        )
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        placeholder_count = full_ids.count(image_pad_id)
        if placeholder_count != 1:
            raise ValueError(
                f"row {index}: expected one image placeholder, "
                f"found {placeholder_count}"
            )
        negative = row["images"][0].startswith("negatives/")
        # Negative completions are only {"not_food":true}; their varying COCO
        # grids cannot approach the limit. Food images all use the verified
        # 30x40 grid = 300 post-merge image tokens.
        added_image_tokens = 0 if negative else args.image_tokens - 1
        length = len(full_ids) + added_image_tokens
        completion_length = len(full_ids) - len(prefix_ids)
        lengths.append(length)
        completion_lengths.append(completion_length)
        record = {
            "index": index,
            "image": row["images"][0],
            "tokens": length,
            "completion_tokens": completion_length,
            "negative_grid_omitted": negative,
        }
        if index < 8:
            first_records.append(record)
        if not negative and length > args.limit:
            over_limit.append(record)

    result = {
        "schema_version": 2,
        "method": "template_tokens_plus_verified_nutrition5k_image_grid",
        "data": str(args.data),
        "model_dir": str(args.model_dir),
        "records": len(lengths),
        "limit": args.limit,
        "nutrition_image_tokens": args.image_tokens,
        "over_limit": len(over_limit),
        "over_limit_rate": len(over_limit) / len(lengths),
        "sequence_tokens": {
            "median": float(np.quantile(lengths, 0.5)),
            "p90": float(np.quantile(lengths, 0.9)),
            "p95": float(np.quantile(lengths, 0.95)),
            "p99": float(np.quantile(lengths, 0.99)),
            "max": max(lengths),
        },
        "completion_tokens": {
            "median": float(np.quantile(completion_lengths, 0.5)),
            "p90": float(np.quantile(completion_lengths, 0.9)),
            "p95": float(np.quantile(completion_lengths, 0.95)),
            "p99": float(np.quantile(completion_lengths, 0.99)),
            "max": max(completion_lengths),
        },
        "first_records": first_records,
        "longest_over_limit": sorted(
            over_limit, key=lambda row: row["tokens"], reverse=True
        )[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
