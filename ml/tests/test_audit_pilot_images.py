import json
from pathlib import Path

from PIL import Image

from teacher_labeling.audit_pilot_images import audit_pilot


def _record(path: Path, image_id: int) -> dict:
    return {
        "source": {"file_name": path.name, "image_id": image_id},
        "image": {"path": str(path), "width": 20, "height": 10},
    }


def test_audit_passes_distinct_images(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (20, 10), "black").save(first)
    image = Image.new("RGB", (20, 10), "black")
    for x in range(10):
        for y in range(10):
            image.putpixel((x, y), (255, 255, 255))
    image.save(second)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(_record(first, 1))
        + "\n"
        + json.dumps(_record(second, 2))
        + "\n",
        encoding="utf-8",
    )

    report = audit_pilot(
        manifest_path=manifest,
        output_path=tmp_path / "audit.json",
        perceptual_distance=0,
    )

    assert report["passed"] is True
    assert report["records"] == 2


def test_audit_reports_exact_duplicates(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (20, 10), "orange").save(first)
    second.write_bytes(first.read_bytes())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(_record(first, 1))
        + "\n"
        + json.dumps(_record(second, 2))
        + "\n",
        encoding="utf-8",
    )

    report = audit_pilot(
        manifest_path=manifest,
        output_path=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["exact_duplicate_groups"] == [["first.jpg", "second.jpg"]]
