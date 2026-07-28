"""Validate and merge leakage-safe specialist out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def merge_oof(
    *,
    folds_dir: Path,
    predictions_dir: Path,
    output: Path,
    fold_count: int,
) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    fold_summaries = []
    for fold in range(fold_count):
        fold_dir = folds_dir / f"fold-{fold}"
        heldout = _read_jsonl(fold_dir / "heldout.jsonl")
        train = _read_jsonl(fold_dir / "train.jsonl")
        predicted = _read_jsonl(predictions_dir / f"fold-{fold}.jsonl")
        heldout_ids = {row["id"] for row in heldout}
        train_ids = {row["id"] for row in train}
        predicted_ids = {row["id"] for row in predicted}
        heldout_groups = {
            row["group_id"] for row in heldout if row.get("group_id")
        }
        train_groups = {
            row["group_id"] for row in train if row.get("group_id")
        }
        if len(heldout_ids) != len(heldout):
            raise ValueError(f"fold {fold}: duplicate held-out IDs")
        if len(predicted_ids) != len(predicted):
            raise ValueError(f"fold {fold}: duplicate prediction IDs")
        if heldout_ids != predicted_ids:
            missing = sorted(heldout_ids - predicted_ids)[:5]
            unexpected = sorted(predicted_ids - heldout_ids)[:5]
            raise ValueError(
                f"fold {fold}: prediction mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        leaked = heldout_ids & train_ids
        if leaked:
            raise ValueError(
                f"fold {fold}: {len(leaked)} held-out IDs occur in training"
            )
        leaked_groups = heldout_groups & train_groups
        if leaked_groups:
            raise ValueError(
                f"fold {fold}: {len(leaked_groups)} held-out dish groups "
                "occur in training"
            )
        duplicate_across_folds = heldout_ids & all_ids
        if duplicate_across_folds:
            raise ValueError(
                f"fold {fold}: held-out IDs repeated across folds: "
                f"{sorted(duplicate_across_folds)[:5]}"
            )
        hashes = {row["checkpoint_sha256"] for row in predicted}
        if len(hashes) != 1:
            raise ValueError(
                f"fold {fold}: expected one checkpoint hash, found {len(hashes)}"
            )
        all_ids.update(heldout_ids)
        merged.extend({**row, "fold": fold} for row in predicted)
        fold_summaries.append(
            {
                "fold": fold,
                "heldout_records": len(heldout),
                "heldout_groups": len(heldout_groups),
                "training_records": len(train),
                "checkpoint_sha256": next(iter(hashes)),
            }
        )

    merged.sort(key=lambda row: row["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "fold_count": fold_count,
        "records": len(merged),
        "unique_ids": len(all_ids),
        "folds": fold_summaries,
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args()
    summary = merge_oof(
        folds_dir=args.folds_dir,
        predictions_dir=args.predictions_dir,
        output=args.output,
        fold_count=args.fold_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
