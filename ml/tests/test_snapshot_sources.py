from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from teacher_labeling.snapshot_sources import SnapshotError, snapshot_sources


class _Response:
    def __init__(self, body: bytes, final_url: str) -> None:
        self._body = body
        self._final_url = final_url
        self.status = 200
        self.headers = {
            "Content-Type": "text/html; charset=utf-8",
            "ETag": '"test-etag"',
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body

    def geturl(self):
        return self._final_url


def _write_source(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id": "test-source",
                "expected_dataset": {"images": 1},
                "snapshot_urls": [
                    {
                        "name": "rules",
                        "url": "https://example.test/rules",
                    },
                    {
                        "name": "dataset_files",
                        "url": "https://example.test/files",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_snapshot_records_hashes_redirects_and_manifest(tmp_path):
    source = tmp_path / "source.json"
    _write_source(source)
    responses = iter(
        [
            _Response(b"<html>rules</html>", "https://example.test/rules"),
            _Response(
                b"<html>login</html>",
                "https://example.test/sign-in",
            ),
        ]
    )

    with patch(
        "teacher_labeling.snapshot_sources.urllib.request.urlopen",
        side_effect=lambda *_args, **_kwargs: next(responses),
    ):
        manifest_path = snapshot_sources(
            source_path=source,
            output_root=tmp_path / "output",
            now=datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_id"] == "test-source"
    assert manifest["captured_at_utc"] == "2026-07-28T12:00:00+00:00"
    assert len(manifest["records"]) == 2
    assert (
        manifest["records"][1]["final_url"]
        == "https://example.test/sign-in"
    )
    assert (manifest_path.parent / "rules.html").read_bytes() == b"<html>rules</html>"


def test_snapshot_refuses_to_overwrite_existing_run(tmp_path):
    source = tmp_path / "source.json"
    _write_source(source)
    timestamp = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    existing = tmp_path / "output" / "20260728T120000Z"
    existing.mkdir(parents=True)

    with pytest.raises(SnapshotError, match="already exists"):
        snapshot_sources(
            source_path=source,
            output_root=tmp_path / "output",
            now=timestamp,
        )
