"""Convert the FRB v2.0 Deep Lake mirror into reviewable local artifacts.

The downloaded mirror stores the original JPEG bytes and annotations in Deep
Lake 2.x chunks.  This converter uses a separate legacy-reader environment and
does two explicit operations:

1. export source metadata to a COCO-compatible JSON file without decoding all
   images;
2. export only images selected by the pilot manifest, preserving their exact
   compressed JPEG bytes.

Run from the repository root with ``ml/.venv-frb/bin/python``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


EXPECTED_V20_TRAIN_IMAGES = 39_962
EXPECTED_V20_TRAIN_ANNOTATIONS = 76_491
EXPECTED_V20_CLASSES = 498


class DeepLakeExportError(RuntimeError):
    """The mirror is inconsistent with the declared source release."""


def _load_dataset(path: Path):
    os.environ.setdefault("DEEPLAKE_DISABLE_ANALYTICS", "1")
    os.environ.setdefault("BUGGER_OFF", "True")
    try:
        import deeplake
    except ImportError as exc:
        raise DeepLakeExportError(
            "deeplake<4 is required; use ml/.venv-frb/bin/python"
        ) from exc
    return deeplake.load(str(path), read_only=True, verbose=False)


def _sample_value(sample: Any) -> Any:
    data = sample.data()
    if not isinstance(data, dict) or "value" not in data:
        raise DeepLakeExportError("unexpected Deep Lake JSON sample")
    return data["value"]


def export_coco_metadata(
    *,
    dataset_path: Path,
    output_path: Path,
    archive_sha256: str,
    expected_images: int = EXPECTED_V20_TRAIN_IMAGES,
    expected_annotations: int = EXPECTED_V20_TRAIN_ANNOTATIONS,
    expected_classes: int = EXPECTED_V20_CLASSES,
) -> Path:
    """Export image/category records and retain the Deep Lake index."""

    dataset = _load_dataset(dataset_path)
    if len(dataset) != expected_images:
        raise DeepLakeExportError(
            f"expected {expected_images} images, found {len(dataset)}"
        )
    class_names = list(dataset.categories.info.class_names)
    if len(class_names) != expected_classes:
        raise DeepLakeExportError(
            f"expected {expected_classes} classes, found {len(class_names)}"
        )

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen_image_ids: set[Any] = set()
    annotation_id = 1
    for index in range(len(dataset)):
        metadata = _sample_value(dataset.images_meta[index])
        image_id = metadata.get("id")
        if image_id is None or image_id in seen_image_ids:
            raise DeepLakeExportError(
                f"missing or duplicate image id at Deep Lake index {index}"
            )
        seen_image_ids.add(image_id)
        image_row = {
            "id": image_id,
            "file_name": metadata["file_name"],
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "deeplake_index": index,
        }
        images.append(image_row)
        category_ids = dataset.categories[index].numpy().reshape(-1).tolist()
        for category_id in category_ids:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(category_id),
                }
            )
            annotation_id += 1

    if len(annotations) != expected_annotations:
        raise DeepLakeExportError(
            f"expected {expected_annotations} annotations, found "
            f"{len(annotations)}"
        )
    payload = {
        "info": {
            "dataset": "food-recognition-benchmark-2022",
            "release": "v2.0",
            "license": "CC-BY-4.0",
            "source_format": "deeplake-2.2.2",
            "source_archive_sha256": archive_sha256,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in enumerate(class_names)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    output_path.write_text(encoded, encoding="utf-8")
    summary = {
        "images": len(images),
        "annotations": len(annotations),
        "classes": len(class_names),
        "archive_sha256": archive_sha256,
        "metadata_sha256": hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest(),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _manifest_records(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise DeepLakeExportError(f"blank manifest line {line_number}")
        record = json.loads(line)
        if record["source"]["release"] != "v2.0":
            raise DeepLakeExportError(
                f"manifest line {line_number} is not FRB v2.0"
            )
        yield record


def export_selected_images(
    *,
    dataset_path: Path,
    manifest_path: Path,
    output_root: Path,
) -> Path:
    """Write selected JPEGs byte-for-byte and emit a checksum inventory."""

    dataset = _load_dataset(dataset_path)
    index_by_id: dict[Any, int] = {}
    for index in range(len(dataset)):
        metadata = _sample_value(dataset.images_meta[index])
        image_id = metadata.get("id")
        if image_id in index_by_id:
            raise DeepLakeExportError(f"duplicate image id: {image_id!r}")
        index_by_id[image_id] = index

    output_root_resolved = output_root.resolve()
    inventory: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for record in _manifest_records(manifest_path):
        image_id = record["source"]["image_id"]
        file_name = record["source"]["file_name"]
        if file_name in seen_names:
            raise DeepLakeExportError(f"duplicate output file name: {file_name}")
        seen_names.add(file_name)
        try:
            index = index_by_id[image_id]
        except KeyError as exc:
            raise DeepLakeExportError(
                f"selected image id is absent: {image_id!r}"
            ) from exc

        output_path = (output_root / file_name).resolve()
        if output_root_resolved not in output_path.parents:
            raise DeepLakeExportError(f"unsafe output file name: {file_name!r}")
        expected_path = Path(record["image"]["path"]).resolve()
        if output_path != expected_path:
            raise DeepLakeExportError(
                f"manifest/output path mismatch for {image_id!r}"
            )
        data = dataset.images[index].tobytes()
        if not data.startswith(b"\xff\xd8"):
            raise DeepLakeExportError(f"image {image_id!r} is not a JPEG")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        inventory.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "deeplake_index": index,
            }
        )

    inventory_path = manifest_path.with_suffix(
        manifest_path.suffix + ".images.json"
    )
    inventory_text = json.dumps(
        {
            "schema_version": 1,
            "images": inventory,
            "count": len(inventory),
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    inventory_path.write_text(inventory_text, encoding="utf-8")
    return inventory_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("--dataset", type=Path, required=True)
    metadata.add_argument("--out", type=Path, required=True)
    metadata.add_argument("--archive-sha256", required=True)

    images = subparsers.add_parser("images")
    images.add_argument("--dataset", type=Path, required=True)
    images.add_argument("--manifest", type=Path, required=True)
    images.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "metadata":
        output = export_coco_metadata(
            dataset_path=args.dataset,
            output_path=args.out,
            archive_sha256=args.archive_sha256,
        )
    else:
        output = export_selected_images(
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            output_root=args.output_root,
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if args.command == "metadata":
        summary = payload
    else:
        summary = {
            "inventory": str(output),
            "count": payload["count"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
