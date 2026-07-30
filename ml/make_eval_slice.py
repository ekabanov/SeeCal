"""Create a deterministic group-aware JSONL slice for fast model iteration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def one_record_per_group(rows: list[dict], *, seed: str) -> list[dict]:
    """Select one deterministic record from every retained group."""
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        group = str(row.get("group_id") or row.get("id"))
        by_group.setdefault(group, []).append(row)
    return [
        min(
            group_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row.get('id')}".encode()
            ).hexdigest(),
        )
        for _, group_rows in sorted(by_group.items())
    ]


def build_slice(
    input_path: Path,
    output_path: Path,
    *,
    groups: int,
    seed: str,
    one_per_group: bool = False,
) -> dict[str, int]:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        group = str(row.get("group_id") or row.get("id"))
        by_group.setdefault(group, []).append(row)
    ranked = sorted(
        by_group,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    selected_groups = set(ranked[:groups])
    selected = [row for row in rows if str(row.get("group_id") or row.get("id")) in selected_groups]
    if one_per_group:
        selected = one_record_per_group(selected, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    return {
        "groups": len(selected_groups),
        "records": len(selected),
    }


def filter_by_group_manifests(
    input_path: Path,
    output_path: Path,
    *,
    group_manifest_paths: list[Path],
    group_tail: bool = False,
    one_per_group: bool = False,
    seed: str = "20260729-fast-v2",
) -> dict[str, int]:
    def key(row: dict) -> str:
        value = str(row.get("group_id") or row.get("id"))
        return value.rsplit(":", 1)[-1] if group_tail else value

    allowed = {
        key(json.loads(line))
        for path in group_manifest_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if key(row) in allowed]
    selected_groups = {key(row) for row in selected}
    missing = allowed - selected_groups
    if missing:
        raise ValueError(
            f"{len(missing)} requested groups are absent from {input_path}: "
            f"{sorted(missing)[:5]}"
        )
    if one_per_group:
        selected = one_record_per_group(selected, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    return {
        "groups": len(selected_groups),
        "records": len(selected),
        "source_group_manifests": len(group_manifest_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--groups", type=int)
    selector.add_argument(
        "--groups-from",
        type=Path,
        action="append",
        help="Keep exactly the group IDs present in one or more JSONL manifests.",
    )
    parser.add_argument(
        "--group-tail",
        action="store_true",
        help="Join by the final colon-delimited group component across datasets.",
    )
    parser.add_argument(
        "--one-per-group",
        action="store_true",
        help="Keep one deterministic view from every selected group.",
    )
    parser.add_argument("--seed", default="20260729-fast-v2")
    args = parser.parse_args()
    print(
        json.dumps(
            (
                filter_by_group_manifests(
                    args.input,
                    args.output,
                    group_manifest_paths=args.groups_from,
                    group_tail=args.group_tail,
                    one_per_group=args.one_per_group,
                    seed=args.seed,
                )
                if args.groups_from
                else build_slice(
                    args.input,
                    args.output,
                    groups=args.groups,
                    seed=args.seed,
                    one_per_group=args.one_per_group,
                )
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
