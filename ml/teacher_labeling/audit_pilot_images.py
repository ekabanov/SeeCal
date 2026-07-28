"""Audit selected pilot images for integrity and duplicate leakage."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


class PilotAuditError(RuntimeError):
    """The selected pilot contains invalid or duplicate images."""


def _difference_hash(image: Image.Image) -> int:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.get_flattened_data())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (
                pixels[y * 9 + x] > pixels[y * 9 + x + 1]
            )
    return value


def audit_pilot(
    *,
    manifest_path: Path,
    output_path: Path,
    perceptual_distance: int = 2,
) -> dict[str, Any]:
    if perceptual_distance < 0:
        raise ValueError("perceptual_distance must be non-negative")
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exact_groups: dict[str, list[str]] = defaultdict(list)
    hashes: list[tuple[str, int]] = []
    invalid: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["image"]["path"])
        name = record["source"]["file_name"]
        try:
            data = path.read_bytes()
            exact_groups[hashlib.sha256(data).hexdigest()].append(name)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                expected_size = (
                    record["image"]["width"],
                    record["image"]["height"],
                )
                if image.size != expected_size:
                    invalid.append(
                        {
                            "file_name": name,
                            "error": "dimension_mismatch",
                            "expected": expected_size,
                            "actual": image.size,
                        }
                    )
                hashes.append((name, _difference_hash(image)))
        except (OSError, ValueError) as exc:
            invalid.append(
                {
                    "file_name": name,
                    "error": type(exc).__name__,
                    "detail": str(exc),
                }
            )

    exact = [names for names in exact_groups.values() if len(names) > 1]
    perceptual: list[dict[str, Any]] = []
    for index, (first_name, first_hash) in enumerate(hashes):
        for second_name, second_hash in hashes[index + 1 :]:
            distance = (first_hash ^ second_hash).bit_count()
            if distance <= perceptual_distance:
                perceptual.append(
                    {
                        "first": first_name,
                        "second": second_name,
                        "distance": distance,
                    }
                )

    report = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "records": len(records),
        "valid_images": len(records) - len(invalid),
        "invalid": invalid,
        "exact_duplicate_groups": exact,
        "perceptual_distance_threshold": perceptual_distance,
        "perceptual_duplicate_candidates": perceptual,
        "passed": not invalid and not exact and not perceptual,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--perceptual-distance", type=int, default=2)
    args = parser.parse_args()
    report = audit_pilot(
        manifest_path=args.manifest,
        output_path=args.out,
        perceptual_distance=args.perceptual_distance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise PilotAuditError("pilot image audit failed; inspect the report")


if __name__ == "__main__":
    main()
