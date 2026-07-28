"""Prepare Gemini Batch API JSONL without making or reserving paid calls.

This module is intentionally split from submission.  ``prepare`` is offline
and reviewable: it resizes each image, embeds it in a request, pins the exact
prompt/schema, and emits a cost plan.  A later submission command must reserve
the corresponding batch in ``BudgetLedger`` *before* any non-idempotent API
call.
"""

from __future__ import annotations

import argparse
import base64
from decimal import Decimal
import hashlib
from io import BytesIO
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps

from .budget import (
    BudgetPolicy,
    ConfigurationError,
    PricingCatalog,
    load_secret_env,
)
from .cli import DEFAULT_CONFIG, DEFAULT_PRICING, DEFAULT_SECRET


BATCH_SCHEMA_VERSION = 1
DEFAULT_MAX_EDGE = 1_024
DEFAULT_JPEG_QUALITY = 88
DEFAULT_INPUT_TOKENS_PER_RECORD = 1_500
DEFAULT_OUTPUT_TOKENS_PER_RECORD = 300
SUPPORTED_MODELS = ("gemini-3.5-flash", "gemini-3.5-flash-lite")

SEMANTIC_PROMPT = """\
Inspect only what is visibly supported by this food photograph.

Return the requested JSON. Follow these rules:
- Name visible foods or ingredients at a useful culinary level.
- Distinguish an ingredient that is visibly separate from one merely plausible
  inside a mixed dish. Never state a hidden recipe ingredient as visible.
- Describe cooking state only when visually supported.
- Set abstain=true when blur, occlusion, framing, or ambiguity prevents a
  dependable semantic description, and explain why briefly.
- Do not estimate calories, nutrition, macros, weight, volume, serving size,
  or any other numeric amount.
- Do not infer brands, identity, health claims, or a recipe from context.
- Use lowercase concise names and do not duplicate synonyms.
"""

SEMANTIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_food": {"type": "boolean"},
        "visible_foods": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "cooking_state": {
                        "type": "string",
                        "enum": [
                            "raw",
                            "boiled",
                            "baked",
                            "grilled",
                            "fried",
                            "steamed",
                            "roasted",
                            "mixed_or_unclear",
                        ],
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["clear", "partial", "ambiguous"],
                    },
                },
                "required": ["name", "cooking_state", "visibility"],
                "additionalProperties": False,
            },
        },
        "container": {
            "type": "string",
            "enum": [
                "plate",
                "bowl",
                "tray",
                "cup_or_glass",
                "wrapper_or_package",
                "none_or_unclear",
            ],
        },
        "mixed_dish": {"type": "boolean"},
        "occlusion": {
            "type": "string",
            "enum": ["none", "partial", "severe"],
        },
        "abstain": {"type": "boolean"},
        "ambiguity_reason": {"type": "string"},
    },
    "required": [
        "is_food",
        "visible_foods",
        "container",
        "mixed_dish",
        "occlusion",
        "abstain",
        "ambiguity_reason",
    ],
    "additionalProperties": False,
}


class BatchPreparationError(RuntimeError):
    """Input records or images cannot be converted into a safe batch."""


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BatchPreparationError(f"cannot read manifest: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BatchPreparationError(f"blank manifest line: {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BatchPreparationError(
                f"invalid manifest JSON at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise BatchPreparationError(
                f"manifest line {line_number} must be an object"
            )
        yield record


def _request_key(record: Mapping[str, Any]) -> str:
    try:
        source = record["source"]
        dataset = str(source["dataset"])
        release = str(source["release"])
        image_id = str(source["image_id"])
    except (KeyError, TypeError) as exc:
        raise BatchPreparationError(
            "manifest record lacks source dataset/release/image_id"
        ) from exc
    digest = hashlib.sha256(
        f"{dataset}\0{release}\0{image_id}".encode("utf-8")
    ).hexdigest()[:20]
    return f"frb-{digest}"


def _encoded_image(
    path: Path,
    *,
    max_edge: int,
    jpeg_quality: int,
) -> tuple[str, str, dict[str, Any]]:
    if max_edge <= 0:
        raise ValueError("max_edge must be positive")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be in 1..95")
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            source_size = image.size
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "L"}:
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image)
                image = background
            elif image.mode == "L":
                image = image.convert("RGB")
            output = BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
            )
            normalized_size = image.size
    except (OSError, ValueError) as exc:
        raise BatchPreparationError(f"cannot decode image: {path}") from exc

    data = output.getvalue()
    return (
        "image/jpeg",
        base64.b64encode(data).decode("ascii"),
        {
            "source_width": source_size[0],
            "source_height": source_size[1],
            "normalized_width": normalized_size[0],
            "normalized_height": normalized_size[1],
            "normalized_bytes": len(data),
            "normalized_sha256": hashlib.sha256(data).hexdigest(),
        },
    )


def build_batch_request(
    record: Mapping[str, Any],
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one source-independent, structured-output Gemini request."""

    try:
        image_path = Path(record["image"]["path"])
        loss_mask = record["loss_mask"]
    except (KeyError, TypeError) as exc:
        raise BatchPreparationError(
            "manifest record lacks image.path or loss_mask"
        ) from exc
    required_false = ("calories", "grams", "macros", "hidden_ingredients")
    if not loss_mask.get("semantic_food_fields") or any(
        loss_mask.get(field) is not False for field in required_false
    ):
        raise BatchPreparationError(
            "teacher record must enable semantics and mask every numeric/hidden field"
        )

    mime_type, encoded, image_metadata = _encoded_image(
        image_path,
        max_edge=max_edge,
        jpeg_quality=jpeg_quality,
    )
    request = {
        "key": _request_key(record),
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
                        {"text": SEMANTIC_PROMPT},
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 768,
                "responseMimeType": "application/json",
                "responseJsonSchema": SEMANTIC_SCHEMA,
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            },
        },
    }
    metadata = {
        "key": request["key"],
        "source": record["source"],
        "image": image_metadata,
    }
    return request, metadata


def _chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def prepare_batch_files(
    *,
    manifest_path: Path,
    output_dir: Path,
    policy: BudgetPolicy,
    pricing: PricingCatalog,
    models: tuple[str, ...] = SUPPORTED_MODELS,
    max_records_per_batch: int = 500,
    max_edge: int = DEFAULT_MAX_EDGE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    input_tokens_per_record: int = DEFAULT_INPUT_TOKENS_PER_RECORD,
    output_tokens_per_record: int = DEFAULT_OUTPUT_TOKENS_PER_RECORD,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> Path:
    """Prepare reviewable JSONL chunks and a buffered two-model cost plan."""

    if max_records_per_batch <= 0:
        raise ValueError("max_records_per_batch must be positive")
    if not models:
        raise ValueError("at least one model is required")
    unknown = set(models) - set(pricing.prices)
    if unknown:
        raise ConfigurationError(f"unpriced models: {sorted(unknown)}")

    manifest_records = list(_jsonl(manifest_path))
    if record_offset < 0:
        raise ValueError("record_offset must be non-negative")
    manifest_records = manifest_records[record_offset:]
    if record_limit is not None:
        if record_limit <= 0:
            raise ValueError("record_limit must be positive")
        manifest_records = manifest_records[:record_limit]
    if not manifest_records:
        raise BatchPreparationError("manifest contains no records")
    prepared = [
        build_batch_request(
            record,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
        )
        for record in manifest_records
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    batch_rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(
        _chunked(prepared, max_records_per_batch),
        start=1,
    ):
        request_path = output_dir / f"requests-{index:03d}.jsonl"
        metadata_path = output_dir / f"metadata-{index:03d}.jsonl"
        request_text = "".join(
            json.dumps(request, sort_keys=True) + "\n"
            for request, _metadata in chunk
        )
        metadata_text = "".join(
            json.dumps(metadata, sort_keys=True) + "\n"
            for _request, metadata in chunk
        )
        request_path.write_text(request_text, encoding="utf-8")
        metadata_path.write_text(metadata_text, encoding="utf-8")
        batch_rows.append(
            {
                "batch_index": index,
                "records": len(chunk),
                "request_file": request_path.name,
                "request_sha256": hashlib.sha256(
                    request_text.encode("utf-8")
                ).hexdigest(),
                "request_bytes": len(request_text.encode("utf-8")),
                "metadata_file": metadata_path.name,
                "metadata_sha256": hashlib.sha256(
                    metadata_text.encode("utf-8")
                ).hexdigest(),
            }
        )

    record_count = len(prepared)
    model_plans: dict[str, Any] = {}
    combined = Decimal("0")
    for model in models:
        estimated = pricing.cost(
            model=model,
            input_tokens=record_count * input_tokens_per_record,
            output_tokens=record_count * output_tokens_per_record,
            usage_buffer=policy.usage_buffer,
        )
        combined += estimated
        model_plans[model] = {
            "provider": pricing.prices[model].provider,
            "records": record_count,
            "input_tokens_reserved": record_count * input_tokens_per_record,
            "output_tokens_reserved": record_count * output_tokens_per_record,
            "buffered_estimate_usd": str(estimated),
        }
    if combined > policy.automatic_total_cap:
        raise BatchPreparationError(
            "combined buffered estimate exceeds automatic cap: "
            f"{combined} > {policy.automatic_total_cap}"
        )

    plan = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "run_id": policy.run_id,
        "manifest_path": str(manifest_path),
        "record_limit": record_limit,
        "record_offset": record_offset,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            SEMANTIC_PROMPT.encode("utf-8")
        ).hexdigest(),
        "schema_sha256": hashlib.sha256(
            json.dumps(SEMANTIC_SCHEMA, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "normalization": {
            "max_edge": max_edge,
            "jpeg_quality": jpeg_quality,
            "format": "JPEG",
        },
        "planning_usage_per_record": {
            "input_tokens": input_tokens_per_record,
            "output_tokens": output_tokens_per_record,
            "reservation_buffer": str(policy.usage_buffer),
        },
        "batches": batch_rows,
        "models": model_plans,
        "combined_buffered_estimate_usd": str(combined),
        "automatic_cap_usd": str(policy.automatic_total_cap),
        "pricing_snapshot_hash": pricing.snapshot_hash,
        "pricing_source": pricing.source_url,
        "paid_calls_submitted": False,
    }
    plan_path = output_dir / "batch-plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--secret", type=Path, default=DEFAULT_SECRET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--max-records-per-batch", type=int, default=500)
    parser.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument(
        "--record-limit",
        type=int,
        help="Prepare only the first N deterministic manifest records.",
    )
    parser.add_argument("--record-offset", type=int, default=0)
    parser.add_argument(
        "--model",
        action="append",
        choices=SUPPORTED_MODELS,
        help="Model to price/enable in this plan; repeat for multiple models.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    secret_env = load_secret_env(args.secret)
    policy = BudgetPolicy.load(
        secret_env=secret_env,
        run_config_path=args.config,
    )
    pricing = PricingCatalog.load(args.pricing)
    plan_path = prepare_batch_files(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        policy=policy,
        pricing=pricing,
        max_records_per_batch=args.max_records_per_batch,
        max_edge=args.max_edge,
        jpeg_quality=args.jpeg_quality,
        record_limit=args.record_limit,
        record_offset=args.record_offset,
        models=tuple(args.model) if args.model else SUPPORTED_MODELS,
    )
    print(plan_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
