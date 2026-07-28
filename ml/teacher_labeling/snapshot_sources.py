"""Preserve timestamped source/license pages with hashes before downloading data.

Run from ``ml/``:

    .venv/bin/python -m teacher_labeling.snapshot_sources

Snapshots are local-only under ``ml/datasets``.  The committed source config
documents what should be fetched; the generated manifest records redirects,
headers, byte counts, and SHA-256 hashes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


PACKAGE_DIR = Path(__file__).resolve().parent
ML_DIR = PACKAGE_DIR.parent
DEFAULT_SOURCE = PACKAGE_DIR / "sources" / "food_recognition_2022.json"
DEFAULT_OUTPUT_ROOT = (
    ML_DIR / "datasets" / "food_recognition_2022" / "provenance"
)
SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SnapshotError(RuntimeError):
    """A source page could not be preserved safely."""


def _load_source(path: Path) -> dict:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot load source config: {path}") from exc
    if source.get("schema_version") != 1:
        raise SnapshotError("unsupported source schema_version")
    urls = source.get("snapshot_urls")
    if not isinstance(urls, list) or not urls:
        raise SnapshotError("snapshot_urls must be a non-empty list")
    for item in urls:
        if (
            not isinstance(item, dict)
            or not SAFE_NAME.fullmatch(str(item.get("name", "")))
            or not str(item.get("url", "")).startswith("https://")
        ):
            raise SnapshotError("invalid snapshot URL entry")
    return source


def snapshot_sources(
    *,
    source_path: Path,
    output_root: Path,
    now: datetime | None = None,
) -> Path:
    source = _load_source(source_path)
    timestamp = now or datetime.now(timezone.utc)
    run_name = timestamp.strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_name
    if output_dir.exists():
        raise SnapshotError(f"snapshot directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)

    records = []
    for item in source["snapshot_urls"]:
        request = urllib.request.Request(
            item["url"],
            headers={
                "User-Agent": "SeeCal-research-provenance/1.0",
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                final_url = response.geturl()
                status = getattr(response, "status", 200)
                content_type = response.headers.get("Content-Type", "")
                last_modified = response.headers.get("Last-Modified")
                etag = response.headers.get("ETag")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SnapshotError(f"failed to snapshot {item['name']}") from exc

        if status != 200 or not body:
            raise SnapshotError(
                f"unexpected response for {item['name']}: status={status}"
            )
        suffix = ".html" if "html" in content_type.lower() else ".bin"
        filename = f"{item['name']}{suffix}"
        target = output_dir / filename
        target.write_bytes(body)
        target.chmod(0o600)
        records.append(
            {
                "name": item["name"],
                "requested_url": item["url"],
                "final_url": final_url,
                "status": status,
                "content_type": content_type,
                "last_modified": last_modified,
                "etag": etag,
                "filename": filename,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "source_config": str(source_path),
        "source_config_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_id": source["source_id"],
        "captured_at_utc": timestamp.isoformat(),
        "expected_dataset": source["expected_dataset"],
        "records": records,
    }
    manifest_path = output_dir / "snapshot-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = snapshot_sources(
        source_path=args.source,
        output_root=args.output_root,
    )
    print(manifest)


if __name__ == "__main__":
    main()
