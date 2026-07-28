"""Evaluate a specialist checkpoint on the untouched overhead/negative test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .constants import NUMERIC_FIELDS
from .data import SpecialistDataset
from .model import SpecialistConfig, VisualSpecialist
from .train import EXPERIMENTS, _move, _numeric_metrics, choose_device


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> Path:
    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VisualSpecialist(SpecialistConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model = model.to(device).eval()
    dataset = SpecialistDataset(
        args.manifest_dir / "test.jsonl",
        ml_root=args.ml_root,
        train=False,
        include_sides=False,
        include_frb=EXPERIMENTS[args.experiment]["include_frb"],
        include_teacher=EXPERIMENTS[args.experiment]["include_teacher"],
        include_negatives=True,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    predictions = []
    targets = []
    paired = []
    food_true_positive = food_false_positive = 0
    food_true_negative = food_false_negative = 0
    started = time.monotonic()
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = model(batch["image"])
        numeric = torch.expm1(
            output["numeric_log1p"].clamp(0, 10)
        ).clamp_min(0)
        food_probability = torch.sigmoid(output["food_logit"])
        numeric_mask = batch["numeric_mask"]
        if numeric_mask.any():
            predictions.append(numeric[numeric_mask, :, 1].cpu())
            targets.append(batch["numeric"][numeric_mask].cpu())
        for index, record_id in enumerate(raw_batch["id"]):
            predicted_food = bool(food_probability[index] >= 0.5)
            actual_food = bool(batch["food"][index] >= 0.5)
            if predicted_food and actual_food:
                food_true_positive += 1
            elif predicted_food:
                food_false_positive += 1
            elif actual_food:
                food_false_negative += 1
            else:
                food_true_negative += 1
            row = {
                "id": record_id,
                "source": raw_batch["source"][index],
                "food_probability": float(food_probability[index].cpu()),
                "food_target": actual_food,
            }
            if bool(numeric_mask[index]):
                row["target"] = {
                    field: float(batch["numeric"][index, field_index].cpu())
                    for field_index, field in enumerate(NUMERIC_FIELDS)
                }
                row["prediction"] = {
                    field: {
                        "p10": float(numeric[index, field_index, 0].cpu()),
                        "p50": float(numeric[index, field_index, 1].cpu()),
                        "p90": float(numeric[index, field_index, 2].cpu()),
                    }
                    for field_index, field in enumerate(NUMERIC_FIELDS)
                }
            paired.append(row)
    elapsed = time.monotonic() - started
    metrics = _numeric_metrics(predictions, targets)
    metrics.update(
        {
            "food_true_positive": food_true_positive,
            "food_false_positive": food_false_positive,
            "food_true_negative": food_true_negative,
            "food_false_negative": food_false_negative,
            "food_accuracy": (
                (food_true_positive + food_true_negative) / len(dataset)
            ),
        }
    )
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_validation": payload["metrics"],
        "test_records": len(dataset),
        "elapsed_seconds": elapsed,
        "milliseconds_per_image": elapsed / len(dataset) * 1000,
        "metrics": metrics,
        "paired": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(evaluate_checkpoint(args))


if __name__ == "__main__":
    main()
