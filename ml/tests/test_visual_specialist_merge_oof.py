import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_specialist.merge_oof import merge_oof


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction(record_id, checkpoint):
    return {
        "id": record_id,
        "checkpoint_sha256": checkpoint,
        "numeric": {},
        "food_probability": 1.0,
    }


def test_merge_oof_requires_exact_disjoint_prediction_sets(tmp_path):
    folds = tmp_path / "folds"
    predictions = tmp_path / "predictions"
    _write(folds / "fold-0" / "heldout.jsonl", [{"id": "a"}])
    _write(folds / "fold-0" / "train.jsonl", [{"id": "b"}])
    _write(predictions / "fold-0.jsonl", [_prediction("a", "hash-0")])
    _write(folds / "fold-1" / "heldout.jsonl", [{"id": "b"}])
    _write(folds / "fold-1" / "train.jsonl", [{"id": "a"}])
    _write(predictions / "fold-1.jsonl", [_prediction("b", "hash-1")])

    output = tmp_path / "merged.jsonl"
    summary = merge_oof(
        folds_dir=folds,
        predictions_dir=predictions,
        output=output,
        fold_count=2,
    )

    assert summary["records"] == 2
    assert summary["unique_ids"] == 2
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["id"], row["fold"]) for row in rows] == [("a", 0), ("b", 1)]


def test_merge_oof_rejects_train_heldout_leakage(tmp_path):
    folds = tmp_path / "folds"
    predictions = tmp_path / "predictions"
    _write(folds / "fold-0" / "heldout.jsonl", [{"id": "a"}])
    _write(folds / "fold-0" / "train.jsonl", [{"id": "a"}])
    _write(predictions / "fold-0.jsonl", [_prediction("a", "hash")])

    with pytest.raises(ValueError, match="occur in training"):
        merge_oof(
            folds_dir=folds,
            predictions_dir=predictions,
            output=tmp_path / "merged.jsonl",
            fold_count=1,
        )


def test_merge_oof_rejects_side_view_group_leakage(tmp_path):
    folds = tmp_path / "folds"
    predictions = tmp_path / "predictions"
    _write(
        folds / "fold-0" / "heldout.jsonl",
        [{"id": "dish:overhead", "group_id": "dish"}],
    )
    _write(
        folds / "fold-0" / "train.jsonl",
        [{"id": "dish:side-a", "group_id": "dish"}],
    )
    _write(
        predictions / "fold-0.jsonl",
        [_prediction("dish:overhead", "hash")],
    )

    with pytest.raises(ValueError, match="dish groups"):
        merge_oof(
            folds_dir=folds,
            predictions_dir=predictions,
            output=tmp_path / "merged.jsonl",
            fold_count=1,
        )
