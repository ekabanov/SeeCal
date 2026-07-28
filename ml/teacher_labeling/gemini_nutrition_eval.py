"""Prepare and score an evaluation-only Gemini nutrition benchmark."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from .budget import BudgetPolicy, PricingCatalog, load_secret_env
from .cli import DEFAULT_CONFIG, DEFAULT_PRICING, DEFAULT_SECRET
from .gemini_batch import _chunked, _encoded_image


PROMPT = """\
Estimate the nutritional totals for all food visible in this photograph.
Return only the requested JSON. Estimate the complete pictured serving, not a
generic serving size. Use numbers without units. Do not use external metadata,
and do not explain the answer.
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "total_calories": {"type": "number", "minimum": 0},
        "protein_g": {"type": "number", "minimum": 0},
        "fat_g": {"type": "number", "minimum": 0},
        "carbs_g": {"type": "number", "minimum": 0},
    },
    "required": ["total_calories", "protein_g", "fat_g", "carbs_g"],
    "additionalProperties": False,
}


def _records(test_manifest: Path) -> list[dict[str, Any]]:
    rows = []
    for line in test_manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        truth = json.loads(record["messages"][-1]["content"][0]["text"])
        image_path = record["images"][0]
        dish_id = Path(image_path).parent.name
        rows.append(
            {
                "key": f"nutrition5k-{dish_id}",
                "dish_id": dish_id,
                "image_path": image_path,
                "truth": {
                    field: float(truth[field])
                    for field in (
                        "total_calories",
                        "protein_g",
                        "fat_g",
                        "carbs_g",
                    )
                },
            }
        )
    return rows


def prepare(
    *,
    test_manifest: Path,
    ml_root: Path,
    output_dir: Path,
    policy: BudgetPolicy,
    pricing: PricingCatalog,
    model: str,
    max_records_per_batch: int = 500,
) -> Path:
    rows = _records(test_manifest)
    prepared = []
    for row in rows:
        mime_type, encoded, image_metadata = _encoded_image(
            ml_root / row["image_path"], max_edge=1024, jpeg_quality=88
        )
        request = {
            "key": row["key"],
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": encoded,
                                }
                            },
                            {"text": PROMPT},
                        ],
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 256,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": SCHEMA,
                    "thinkingConfig": {"thinkingLevel": "MINIMAL"},
                },
            },
        }
        prepared.append((request, {**row, "image": image_metadata}))

    output_dir.mkdir(parents=True, exist_ok=False)
    batches = []
    for index, chunk in enumerate(
        _chunked(prepared, max_records_per_batch), start=1
    ):
        request_path = output_dir / f"requests-{index:03d}.jsonl"
        metadata_path = output_dir / f"metadata-{index:03d}.jsonl"
        request_text = "".join(
            json.dumps(request, sort_keys=True) + "\n"
            for request, _ in chunk
        )
        metadata_text = "".join(
            json.dumps(metadata, sort_keys=True) + "\n"
            for _, metadata in chunk
        )
        request_path.write_text(request_text, encoding="utf-8")
        metadata_path.write_text(metadata_text, encoding="utf-8")
        batches.append(
            {
                "batch_index": index,
                "records": len(chunk),
                "request_file": request_path.name,
                "request_sha256": hashlib.sha256(
                    request_text.encode()
                ).hexdigest(),
                "request_bytes": len(request_text.encode()),
                "metadata_file": metadata_path.name,
                "metadata_sha256": hashlib.sha256(
                    metadata_text.encode()
                ).hexdigest(),
            }
        )
    input_tokens = len(rows) * 1500
    output_tokens = len(rows) * 150
    estimate = pricing.cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_buffer=policy.usage_buffer,
    )
    if estimate > policy.automatic_total_cap:
        raise RuntimeError("nutrition evaluation estimate exceeds automatic cap")
    plan = {
        "schema_version": 1,
        "run_id": "gemini-nutrition5k-eval",
        "response_kind": "nutrition",
        "evaluation_only": True,
        "manifest_path": str(test_manifest),
        "manifest_sha256": hashlib.sha256(test_manifest.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(
            json.dumps(SCHEMA, sort_keys=True).encode()
        ).hexdigest(),
        "batches": batches,
        "models": {
            model: {
                "provider": "google",
                "records": len(rows),
                "input_tokens_reserved": input_tokens,
                "output_tokens_reserved": output_tokens,
                "buffered_estimate_usd": str(estimate),
            }
        },
        "combined_buffered_estimate_usd": str(estimate),
        "automatic_cap_usd": str(policy.automatic_total_cap),
        "pricing_snapshot_hash": pricing.snapshot_hash,
        "pricing_source": pricing.source_url,
        "paid_calls_submitted": False,
    }
    path = output_dir / "batch-plan.json"
    path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def evaluate(output_dir: Path) -> Path:
    metadata = {}
    for path in sorted(output_dir.glob("metadata-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            metadata[row["key"]] = row
    results = {}
    for path in sorted((output_dir / "jobs").glob("*-results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            results[row["key"]] = row
    if set(metadata) != set(results):
        raise RuntimeError("metadata/result key coverage mismatch")
    paired = []
    for key in sorted(metadata):
        result = results[key]
        if "nutrition" not in result:
            continue
        paired.append(
            {
                "key": key,
                "dish_id": metadata[key]["dish_id"],
                "truth": metadata[key]["truth"],
                "prediction": result["nutrition"],
            }
        )
    calorie_errors = [
        row["prediction"]["total_calories"] - row["truth"]["total_calories"]
        for row in paired
    ]
    absolute = [abs(value) for value in calorie_errors]
    summary = {
        "schema_version": 1,
        "evaluation_only": True,
        "records": len(metadata),
        "successful_records": len(paired),
        "failed_records": len(metadata) - len(paired),
        "calories_mae": mean(absolute),
        "calories_median_ae": median(absolute),
        "calories_bias": mean(calorie_errors),
        "calories_ae_p75": _percentile(absolute, 0.75),
        "calories_ae_p90": _percentile(absolute, 0.90),
        "calories_ae_p95": _percentile(absolute, 0.95),
        "paired": paired,
    }
    path = output_dir / "evaluation.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--test-manifest", type=Path, required=True)
    prepare_parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    prepare_parser.add_argument("--out-dir", type=Path, required=True)
    prepare_parser.add_argument("--model", default="gemini-3.5-flash-lite")
    prepare_parser.add_argument("--secret", type=Path, default=DEFAULT_SECRET)
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        policy = BudgetPolicy.load(
            secret_env=load_secret_env(args.secret),
            run_config_path=args.config,
        )
        path = prepare(
            test_manifest=args.test_manifest,
            ml_root=args.ml_root,
            output_dir=args.out_dir,
            policy=policy,
            pricing=PricingCatalog.load(args.pricing),
            model=args.model,
        )
    else:
        path = evaluate(args.out_dir)
    print(path)


if __name__ == "__main__":
    main()
