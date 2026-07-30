"""Cache resized copies of VLM or SCALE JSONL images for fast iteration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def _replace_message_image(record: dict[str, Any], image_path: str) -> None:
    for message in record.get("messages", []):
        for content in message.get("content", []):
            if content.get("type") == "image":
                content["image"] = image_path


def cache(
    input_path: Path,
    output_path: Path,
    *,
    cache_dir: Path,
    ml_root: Path,
    max_edge: int,
    reuse_small_originals: bool = False,
) -> dict[str, int]:
    if max_edge <= 0:
        raise ValueError("max_edge must be positive")
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = []
    resized = 0
    reused = 0
    for source_record in rows:
        record = json.loads(json.dumps(source_record))
        if "images" in record:
            source_image_path = record["images"][0]
        elif "image_path" in record:
            source_image_path = record["image_path"]
        else:
            raise ValueError("record must contain images[0] or image_path")
        original = (ml_root / source_image_path).resolve()
        digest = hashlib.sha256(str(original).encode()).hexdigest()[:16]
        destination = cache_dir / f"{digest}.jpg"
        with Image.open(original) as opened:
            image = ImageOps.exif_transpose(opened)
            needs_resize = max(image.size) > max_edge
            if reuse_small_originals and not needs_resize:
                destination = original
                reused += 1
            elif not destination.is_file():
                image = image.convert("RGB")
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                image.save(destination, format="JPEG", quality=90, optimize=True)
                resized += 1
        try:
            relative = str(destination.resolve().relative_to(ml_root.resolve()))
        except ValueError:
            relative = destination.resolve().as_uri()
        if "images" in record:
            record["images"] = [relative]
            _replace_message_image(record, relative)
        else:
            record["image_path"] = relative
        output.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in output),
        encoding="utf-8",
    )
    return {
        "records": len(output),
        "images_created": resized,
        "images_reused": reused,
        "max_edge": max_edge,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument(
        "--reuse-small-originals",
        action="store_true",
        help="Keep original paths for images already within --max-edge.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            cache(
                args.input,
                args.output,
                cache_dir=args.cache_dir,
                ml_root=args.ml_root,
                max_edge=args.max_edge,
                reuse_small_originals=args.reuse_small_originals,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
