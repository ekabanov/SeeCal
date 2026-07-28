"""Merge source-authoritative FRB labels with filtered Gemini enrichment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .evaluate_bakeoff import labels_match
from .gemini_batch import _request_key


class TeacherMergeError(RuntimeError):
    """Teacher result coverage or identity does not match the pilot."""


def _result_rows(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            if not isinstance(key, str) or key in results:
                raise TeacherMergeError(f"missing/duplicate result key: {key!r}")
            results[key] = row
    return results


def merge_teacher_labels(
    *,
    manifest_path: Path,
    result_paths: list[Path],
    output_path: Path,
    model: str,
) -> Path:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = _result_rows(result_paths)
    expected_keys = {_request_key(record) for record in records}
    if set(results) != expected_keys:
        missing = expected_keys - set(results)
        unexpected = set(results) - expected_keys
        raise TeacherMergeError(
            f"result coverage mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    merged: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for record in records:
        key = _request_key(record)
        result = results[key]
        if "error" in result:
            counters["teacher_errors"] += 1
            teacher = {
                "model": model,
                "key": key,
                "error": result["error"],
                "accepted_enrichment": None,
            }
        else:
            semantic = result["semantic"]
            source_names = [item["name"] for item in record["source_labels"]]
            accepted = [
                item
                for item in semantic["visible_foods"]
                if any(
                    labels_match(item["name"], source)
                    for source in source_names
                )
            ]
            rejected = [
                item
                for item in semantic["visible_foods"]
                if item not in accepted
            ]
            counters["teacher_visible_foods"] += len(semantic["visible_foods"])
            counters["accepted_visible_foods"] += len(accepted)
            counters["rejected_visible_foods"] += len(rejected)
            counters["abstentions"] += bool(semantic["abstain"])
            teacher = {
                "model": model,
                "model_version": result.get("model_version"),
                "key": key,
                "raw_semantic": semantic,
                "accepted_enrichment": (
                    None
                    if semantic["abstain"]
                    else {
                        "visible_foods": accepted,
                        "container": semantic["container"],
                        "mixed_dish": semantic["mixed_dish"],
                        "occlusion": semantic["occlusion"],
                    }
                ),
                "rejected_visible_foods": rejected,
                "acceptance_basis": (
                    "visible names require lexical compatibility with at least "
                    "one source annotation; numeric/hidden fields remain masked"
                ),
            }
        merged.append({**record, "teacher": teacher})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        for record in merged
    )
    output_path.write_text(text, encoding="utf-8")
    total_visible = counters["teacher_visible_foods"]
    summary = {
        "schema_version": 1,
        "records": len(merged),
        "model": model,
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "merged_manifest_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        **dict(counters),
        "visible_food_acceptance_rate": (
            counters["accepted_visible_foods"] / total_visible
            if total_visible
            else 0.0
        ),
        "numeric_fields_added": 0,
        "hidden_ingredients_added": 0,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Add every *-results.jsonl file in this directory.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    result_paths = list(args.result)
    if args.result_dir is not None:
        result_paths.extend(sorted(args.result_dir.glob("*-results.jsonl")))
    if not result_paths:
        raise TeacherMergeError("at least one result file is required")
    summary_path = merge_teacher_labels(
        manifest_path=args.manifest,
        result_paths=result_paths,
        output_path=args.out,
        model=args.model,
    )
    print(summary_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
