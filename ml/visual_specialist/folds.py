"""Create five dish-grouped manifests for leakage-safe specialist predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def fold_for_group(group_id: str, *, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    return int(digest, 16) % folds


def build_folds(
    *,
    manifest_dir: Path,
    output_dir: Path,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("at least two folds are required")
    train = _read(manifest_dir / "train.jsonl")
    valid = _read(manifest_dir / "valid.jsonl")
    measured = [
        row for row in train if row["source"] == "nutrition5k"
    ]
    shared = [
        row for row in train if row["source"] != "nutrition5k"
    ]
    assignments = {
        row["group_id"]: fold_for_group(
            row["group_id"], folds=folds, seed=seed
        )
        for row in measured
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "folds": folds,
        "seed": seed,
        "source_manifest_sha256": hashlib.sha256(
            (manifest_dir / "train.jsonl").read_bytes()
        ).hexdigest(),
        "fold": {},
    }
    for fold in range(folds):
        root = output_dir / f"fold-{fold}"
        heldout_groups = {
            group for group, assigned in assignments.items()
            if assigned == fold
        }
        fold_train = [
            row for row in measured if row["group_id"] not in heldout_groups
        ] + shared
        heldout = [
            row
            for row in measured
            if row["group_id"] in heldout_groups and row["view"] == "overhead"
        ]
        fold_train.sort(key=lambda row: row["id"])
        heldout.sort(key=lambda row: row["id"])
        _write(root / "train.jsonl", fold_train)
        _write(root / "valid.jsonl", valid)
        _write(root / "heldout.jsonl", heldout)
        summary["fold"][str(fold)] = {
            "heldout_groups": len(heldout_groups),
            "heldout_overhead_records": len(heldout),
            "train_records": len(fold_train),
            "valid_records": len(valid),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    print(
        json.dumps(
            build_folds(
                manifest_dir=args.manifest_dir,
                output_dir=args.out_dir,
                folds=args.folds,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
