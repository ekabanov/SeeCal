"""Calibrate specialist intervals and build canonical conditioned Qwen data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .constants import NUMERIC_FIELDS


SCHEMA_VERSION = 1
PROMPT_PREFIX = (
    "Auxiliary visual measurement (fallible; use as evidence, not ground truth):"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_id_from_qwen(record: dict[str, Any]) -> str:
    images = record.get("images") or []
    if len(images) != 1:
        raise ValueError(f"expected exactly one image, found {len(images)}")
    path = Path(images[0])
    if path.parent.name == "negatives" and path.suffix.lower() == ".jpg":
        return f"coco-negative:{path.stem}"
    if path.name != "overhead.jpg" or not path.parent.name.startswith("dish_"):
        raise ValueError(f"unsupported Qwen image path: {images[0]}")
    return f"nutrition5k:{path.parent.name}:overhead"


def conformal_margin(
    intervals: list[tuple[float, float, float]],
    *,
    target_coverage: float,
) -> float:
    """Return the finite-sample split-conformal widening for an interval."""
    if not intervals:
        raise ValueError("cannot calibrate an empty interval set")
    if not 0 < target_coverage < 1:
        raise ValueError("target coverage must be between zero and one")
    scores = sorted(max(low - target, target - high, 0.0)
                    for low, high, target in intervals)
    rank = min(len(scores), math.ceil((len(scores) + 1) * target_coverage))
    return float(scores[rank - 1])


def _coverage(
    intervals: list[tuple[float, float, float]],
    *,
    margin: float,
) -> float:
    covered = sum(
        low - margin <= target <= high + margin
        for low, high, target in intervals
    )
    return covered / len(intervals)


def fit_calibration(
    *,
    predictions_path: Path,
    specialist_manifest_path: Path,
    output: Path,
    target_coverage: float = 0.8,
) -> dict[str, Any]:
    predictions = {
        row["id"]: row for row in _read_jsonl(predictions_path)
    }
    targets = {
        row["id"]: row["targets"]["numeric"]
        for row in _read_jsonl(specialist_manifest_path)
        if row["source"] == "nutrition5k"
        and row["view"] == "overhead"
        and row["loss_mask"]["numeric"]
    }
    missing = sorted(set(targets) - set(predictions))
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} targets")

    fields: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        intervals = [
            (
                float(predictions[record_id]["numeric"][field]["p10"]),
                float(predictions[record_id]["numeric"][field]["p90"]),
                float(target[field]),
            )
            for record_id, target in targets.items()
        ]
        margin = conformal_margin(
            intervals,
            target_coverage=target_coverage,
        )
        fields[field] = {
            "additive_margin": margin,
            "records": len(intervals),
            "raw_coverage": _coverage(intervals, margin=0.0),
            "calibrated_coverage": _coverage(intervals, margin=margin),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "method": "split_conformal_additive_interval_widening",
        "target_coverage": target_coverage,
        "predictions_sha256": _sha256(predictions_path),
        "manifest_sha256": _sha256(specialist_manifest_path),
        "fields": fields,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def calibrated_payload(
    prediction: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": True,
    }
    for field in NUMERIC_FIELDS:
        raw = prediction["numeric"][field]
        margin = float(calibration["fields"][field]["additive_margin"])
        payload[field] = {
            "estimate": round(float(raw["p50"]), 1),
            "low": round(max(0.0, float(raw["p10"]) - margin), 1),
            "high": round(float(raw["p90"]) + margin, 1),
        }
    return payload


def render_auxiliary_block(
    prediction: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
) -> str:
    if prediction is None:
        payload: dict[str, Any] = {"available": False}
    else:
        if calibration is None:
            raise ValueError("calibration is required for available predictions")
        payload = calibrated_payload(prediction, calibration)
    return render_auxiliary_payload(payload)


def render_auxiliary_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"{PROMPT_PREFIX}\n{encoded}"


def _unit_interval(*parts: object) -> float:
    digest = hashlib.sha256(
        ":".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def perturb_payload(
    payload: dict[str, Any],
    *,
    record_id: str,
    seed: int,
) -> dict[str, Any]:
    """Deterministically corrupt a measurement without consulting its target."""
    result = copy.deepcopy(payload)
    for field in NUMERIC_FIELDS:
        values = result[field]
        factor = 0.80 + 0.40 * _unit_interval(seed, record_id, field)
        estimate = float(values["estimate"])
        low = float(values["low"])
        high = float(values["high"])
        shifted = estimate * factor
        extra = abs(shifted - estimate)
        values["estimate"] = round(shifted, 1)
        values["low"] = round(max(0.0, low * factor - extra / 2), 1)
        values["high"] = round(high * factor + extra / 2, 1)
    return result


def merge_training_predictions(
    *,
    oof_path: Path,
    deployment_path: Path,
    output: Path,
) -> dict[str, Any]:
    oof = _read_jsonl(oof_path)
    deployment_negatives = [
        row for row in _read_jsonl(deployment_path)
        if row["id"].startswith("coco-negative:")
    ]
    rows = oof + deployment_negatives
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs in merged training predictions")
    rows.sort(key=lambda row: row["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "oof_records": len(oof),
        "negative_records": len(deployment_negatives),
        "records": len(rows),
        "unique_ids": len(set(ids)),
        "oof_sha256": _sha256(oof_path),
        "deployment_sha256": _sha256(deployment_path),
        "output_sha256": _sha256(output),
        "output": str(output),
    }


def build_conditioned_qwen(
    *,
    qwen_manifest_path: Path,
    predictions_path: Path,
    calibration_path: Path,
    output: Path,
    metadata_output: Path | None = None,
    unavailable_rate: float = 0.0,
    perturb_rate: float = 0.0,
    robustness_seed: int = 20260728,
    negatives_unavailable: bool = False,
) -> dict[str, Any]:
    if unavailable_rate < 0 or perturb_rate < 0:
        raise ValueError("robustness rates cannot be negative")
    if unavailable_rate + perturb_rate > 1:
        raise ValueError("robustness rates cannot sum above one")
    qwen_rows = _read_jsonl(qwen_manifest_path)
    predictions = {
        row["id"]: row for row in _read_jsonl(predictions_path)
    }
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    conditioned = []
    missing = []
    variant_counts = {"clean": 0, "perturbed": 0, "unavailable": 0}
    for source_row in qwen_rows:
        row = json.loads(json.dumps(source_row))
        record_id = record_id_from_qwen(row)
        prediction = predictions.get(record_id)
        negative = record_id.startswith("coco-negative:")
        if prediction is None and not (negatives_unavailable and negative):
            missing.append(record_id)
            continue
        selector = _unit_interval(robustness_seed, record_id, "variant")
        if negatives_unavailable and negative:
            variant = "unavailable"
        elif selector < unavailable_rate:
            variant = "unavailable"
        elif selector < unavailable_rate + perturb_rate:
            variant = "perturbed"
        else:
            variant = "clean"
        if variant == "unavailable":
            block = render_auxiliary_block(None, None)
        else:
            payload = calibrated_payload(prediction, calibration)
            if variant == "perturbed":
                payload = perturb_payload(
                    payload,
                    record_id=record_id,
                    seed=robustness_seed,
                )
            block = render_auxiliary_payload(payload)
        variant_counts[variant] += 1

        content = row["messages"][0]["content"]
        text_items = [item for item in content if item["type"] == "text"]
        if len(text_items) != 1:
            raise ValueError(
                f"{record_id}: expected exactly one user text item"
            )
        text_items[0]["text"] = (
            text_items[0]["text"]
            + "\n\n"
            + block
        )
        conditioned.append(row)
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} Qwen rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in conditioned),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "records": len(conditioned),
        "source_qwen_sha256": _sha256(qwen_manifest_path),
        "predictions_sha256": _sha256(predictions_path),
        "calibration_sha256": _sha256(calibration_path),
        "output_sha256": _sha256(output),
        "prompt_prefix": PROMPT_PREFIX,
        "robustness": {
            "unavailable_rate": unavailable_rate,
            "perturb_rate": perturb_rate,
            "seed": robustness_seed,
            "negatives_unavailable": negatives_unavailable,
            "variant_counts": variant_counts,
        },
        "output": str(output),
    }
    metadata_output = metadata_output or output.with_suffix(".metadata.json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--predictions", type=Path, required=True)
    fit.add_argument("--specialist-manifest", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--target-coverage", type=float, default=0.8)

    build = subparsers.add_parser("build")
    build.add_argument("--qwen-manifest", type=Path, required=True)
    build.add_argument("--predictions", type=Path, required=True)
    build.add_argument("--calibration", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--metadata-output",
        type=Path,
        help=(
            "Optional provenance JSON path. Keep this outside a Hugging Face "
            "dataset directory so its split loader sees only JSONL files."
        ),
    )
    build.add_argument("--unavailable-rate", type=float, default=0.0)
    build.add_argument("--perturb-rate", type=float, default=0.0)
    build.add_argument("--robustness-seed", type=int, default=20260728)
    build.add_argument("--negatives-unavailable", action="store_true")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--oof", type=Path, required=True)
    merge.add_argument("--deployment", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "fit":
        result = fit_calibration(
            predictions_path=args.predictions,
            specialist_manifest_path=args.specialist_manifest,
            output=args.output,
            target_coverage=args.target_coverage,
        )
    elif args.command == "build":
        result = build_conditioned_qwen(
            qwen_manifest_path=args.qwen_manifest,
            predictions_path=args.predictions,
            calibration_path=args.calibration,
            output=args.output,
            metadata_output=args.metadata_output,
            unavailable_rate=args.unavailable_rate,
            perturb_rate=args.perturb_rate,
            robustness_seed=args.robustness_seed,
            negatives_unavailable=args.negatives_unavailable,
        )
    else:
        result = merge_training_predictions(
            oof_path=args.oof,
            deployment_path=args.deployment,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
