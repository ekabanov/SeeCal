"""Train a reproducible MobileNetV3 visual-specialist experiment."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import NUMERIC_FIELDS
from .data import SpecialistDataset, balanced_sampler
from .losses import specialist_loss
from .model import SpecialistConfig, VisualSpecialist


EXPERIMENTS = {
    "s0-overhead": dict(
        include_sides=False, include_frb=False, include_teacher=False,
        include_nutrition=True,
    ),
    "s1-sideviews": dict(
        include_sides=True, include_frb=False, include_teacher=False,
        include_nutrition=True,
    ),
    "s2-frb-raw": dict(
        include_sides=True, include_frb=True, include_teacher=False,
        include_nutrition=True,
    ),
    "s3-frb-teacher": dict(
        include_sides=True, include_frb=True, include_teacher=True,
        include_nutrition=True,
    ),
    "c0-frb-pretrain": dict(
        include_sides=False, include_frb=True, include_teacher=False,
        include_nutrition=False, include_negatives=False,
    ),
    "s2b-frb-staged": dict(
        include_sides=True, include_frb=True, include_teacher=False,
        include_nutrition=True,
    ),
    "s3b-frb-teacher-staged": dict(
        include_sides=True, include_frb=True, include_teacher=True,
        include_nutrition=True,
    ),
}


def _write_status(output_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically publish machine-readable progress for observers."""
    path = output_dir / "status.json"
    temporary = output_dir / "status.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _overfit_subset(
    dataset: SpecialistDataset, count: int
) -> list[dict[str, Any]]:
    """Keep all active source buckets represented in tiny memorization runs."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(dataset.rows):
        buckets.setdefault(dataset.source_bucket(index), []).append(row)
    selected: list[dict[str, Any]] = []
    bucket_names = sorted(buckets)
    cursor = 0
    while len(selected) < min(count, len(dataset.rows)):
        bucket = bucket_names[cursor % len(bucket_names)]
        rows = buckets[bucket]
        row_index = cursor // len(bucket_names)
        if row_index < len(rows):
            selected.append(rows[row_index])
        elif all(row_index >= len(buckets[name]) for name in bucket_names):
            break
        cursor += 1
    return selected


def _numeric_metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict[str, float]:
    if not predictions:
        return {}
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    absolute = (prediction - target).abs()
    metrics: dict[str, float] = {}
    for index, field in enumerate(NUMERIC_FIELDS):
        metrics[f"{field}_mae"] = float(absolute[:, index].mean())
        metrics[f"{field}_bias"] = float((prediction[:, index] - target[:, index]).mean())
    calorie_error = absolute[:, NUMERIC_FIELDS.index("calories")]
    for percentile in (50, 75, 90, 95):
        metrics[f"calories_ae_p{percentile}"] = float(
            torch.quantile(calorie_error, percentile / 100)
        )
    return metrics


def _numeric_log_medians(dataset: SpecialistDataset) -> torch.Tensor | None:
    values = [
        [float(row["targets"]["numeric"][field]) for field in NUMERIC_FIELDS]
        for row in dataset.rows
        if row["loss_mask"]["numeric"]
    ]
    if not values:
        return None
    return torch.log1p(torch.tensor(values)).median(dim=0).values


def _average_precision(scores: torch.Tensor, target: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    ranked = target[order].float()
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    precision = ranked.cumsum(0) / torch.arange(
        1, len(ranked) + 1, dtype=torch.float32
    )
    return float((precision * ranked).sum() / positives)


def _frb_metrics(
    logits_parts: list[torch.Tensor],
    target_parts: list[torch.Tensor],
) -> dict[str, float]:
    if not logits_parts:
        return {
            "frb_micro_f1_at_0_5": 0.0,
            "frb_best_micro_f1": 0.0,
            "frb_best_threshold": 0.5,
            "frb_micro_average_precision": 0.0,
            "frb_macro_average_precision": 0.0,
            "frb_top3_f1": 0.0,
        }
    logits = torch.cat(logits_parts)
    target = torch.cat(target_parts).bool()
    probability = torch.sigmoid(logits)

    def micro_f1(guess: torch.Tensor) -> float:
        true_positive = int((guess & target).sum())
        predicted_positive = int(guess.sum())
        actual_positive = int(target.sum())
        return (
            2 * true_positive
            / max(predicted_positive + actual_positive, 1)
        )

    threshold_results = [
        (float(threshold), micro_f1(probability >= threshold))
        for threshold in torch.linspace(0.05, 0.95, 19)
    ]
    best_threshold, best_f1 = max(
        threshold_results, key=lambda item: item[1]
    )
    flat_ap = _average_precision(probability.flatten(), target.flatten())
    class_aps = [
        _average_precision(probability[:, class_id], target[:, class_id])
        for class_id in range(target.shape[1])
        if target[:, class_id].any()
    ]
    top3 = torch.zeros_like(target)
    top3.scatter_(1, probability.topk(3, dim=1).indices, True)
    return {
        "frb_micro_f1_at_0_5": micro_f1(probability >= 0.5),
        "frb_best_micro_f1": best_f1,
        "frb_best_threshold": best_threshold,
        "frb_micro_average_precision": flat_ap,
        "frb_macro_average_precision": sum(class_aps) / len(class_aps),
        "frb_top3_f1": micro_f1(top3),
    }


@torch.no_grad()
def evaluate(
    model: VisualSpecialist,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    examples = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    food_correct = 0
    food_count = 0
    frb_logits: list[torch.Tensor] = []
    frb_targets: list[torch.Tensor] = []
    cooking_logits: list[torch.Tensor] = []
    cooking_targets: list[torch.Tensor] = []
    teacher_count = 0
    container_correct = mixed_correct = occlusion_correct = 0
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = model(batch["image"])
        loss, _ = specialist_loss(output, batch)
        batch_size = batch["image"].shape[0]
        loss_sum += float(loss) * batch_size
        examples += batch_size
        numeric_mask = batch["numeric_mask"]
        if numeric_mask.any():
            median = torch.expm1(
                output["numeric_log1p"][:, :, 1].clamp(0, 10)
            ).clamp_min(0)
            predictions.append(median[numeric_mask].cpu())
            targets.append(batch["numeric"][numeric_mask].cpu())
        food_mask = batch["food_mask"]
        if food_mask.any():
            guess = output["food_logit"][food_mask] >= 0
            truth = batch["food"][food_mask] >= 0.5
            food_correct += int((guess == truth).sum())
            food_count += int(food_mask.sum())
        frb_mask = batch["frb_mask"]
        if frb_mask.any():
            frb_logits.append(output["frb_logits"][frb_mask].cpu())
            frb_targets.append(batch["frb"][frb_mask].cpu())
        teacher_mask = batch["teacher_mask"]
        if teacher_mask.any():
            count = int(teacher_mask.sum())
            teacher_count += count
            container_correct += int(
                (
                    output["container_logits"][teacher_mask].argmax(dim=1)
                    == batch["container"][teacher_mask]
                ).sum()
            )
            mixed_correct += int(
                (
                    (output["mixed_logit"][teacher_mask] >= 0)
                    == (batch["mixed"][teacher_mask] >= 0.5)
                ).sum()
            )
            occlusion_correct += int(
                (
                    output["occlusion_logits"][teacher_mask].argmax(dim=1)
                    == batch["occlusion"][teacher_mask]
                ).sum()
            )
            cooking_logits.append(output["cooking_logits"][teacher_mask].cpu())
            cooking_targets.append(batch["cooking"][teacher_mask].cpu())
    cooking_metrics = {
        key.replace("frb_", "cooking_"): value
        for key, value in _frb_metrics(
            cooking_logits, cooking_targets
        ).items()
    }
    return {
        "loss": loss_sum / max(examples, 1),
        "food_accuracy": food_correct / max(food_count, 1),
        "teacher_records": teacher_count,
        "container_accuracy": container_correct / max(teacher_count, 1),
        "mixed_accuracy": mixed_correct / max(teacher_count, 1),
        "occlusion_accuracy": occlusion_correct / max(teacher_count, 1),
        **cooking_metrics,
        **_frb_metrics(frb_logits, frb_targets),
        **_numeric_metrics(predictions, targets),
    }


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    settings = EXPERIMENTS[args.experiment]
    ml_root = args.ml_root.resolve()
    train_dataset = SpecialistDataset(
        args.manifest_dir / "train.jsonl",
        ml_root=ml_root,
        train=True,
        image_size=args.image_size,
        **settings,
    )
    valid_dataset = SpecialistDataset(
        args.manifest_dir / "valid.jsonl",
        ml_root=ml_root,
        train=False,
        image_size=args.image_size,
        # Checkpoint selection must use the same measured view for every
        # experiment. FRB remains present for semantic validation, but measured
        # side frames are training augmentation rather than extra validation
        # votes.
        **{**settings, "include_sides": False},
    )
    if args.overfit_records:
        train_dataset.rows = _overfit_subset(
            train_dataset, args.overfit_records
        )
        valid_dataset.rows = list(train_dataset.rows)
    sampler = balanced_sampler(
        train_dataset,
        seed=args.seed,
        samples_per_epoch=args.samples_per_epoch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    if args.initial_checkpoint:
        initial = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=True
        )
        config = SpecialistConfig(**initial["model_config"])
        model = VisualSpecialist(config)
        model.load_state_dict(initial["model_state"])
    else:
        config = SpecialistConfig(pretrained=not args.no_pretrained)
        model = VisualSpecialist(config)
    numeric_medians = _numeric_log_medians(train_dataset)
    if numeric_medians is not None:
        model.initialize_numeric_bias(numeric_medians)
    model = model.to(device)
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith(("features.", "embedding.")):
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.learning_rate * args.backbone_lr_multiplier,
            },
            {"params": head_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "settings": settings,
        "device": str(device),
        "torch_version": torch.__version__,
        "model": config.to_dict(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_records": len(train_dataset),
        "valid_records": len(valid_dataset),
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_status(
        output_dir,
        {
            "schema_version": 1,
            "state": "running",
            "experiment": args.experiment,
            "epoch": 0,
            "epochs": args.epochs,
            "progress": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "best_calories_mae": None,
            "latest": None,
        },
    )
    history_path = output_dir / "metrics.jsonl"
    best_selection = math.inf
    selection_metric = (
        "calories_mae"
        if numeric_medians is not None
        else "negative_frb_macro_average_precision"
    )
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        epoch_start = time.monotonic()
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["image"])
            loss, _ = specialist_loss(output, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = batch["image"].shape[0]
            total_loss += float(loss.detach()) * batch_size
            examples += batch_size
        scheduler.step()
        metrics = evaluate(model, valid_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(examples, 1),
            "epoch_seconds": time.monotonic() - epoch_start,
            "elapsed_seconds": time.monotonic() - start,
            "learning_rate": scheduler.get_last_lr()[0],
            "valid": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        selection_value = (
            metrics["calories_mae"]
            if numeric_medians is not None
            else -metrics["frb_macro_average_precision"]
        )
        if selection_value < best_selection:
            best_selection = selection_value
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(config),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                output_dir / "best.pt",
            )
        elapsed = time.monotonic() - start
        _write_status(
            output_dir,
            {
                "schema_version": 1,
                "state": "running",
                "experiment": args.experiment,
                "epoch": epoch,
                "epochs": args.epochs,
                "progress": epoch / args.epochs,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / epoch * (args.epochs - epoch),
                "best_calories_mae": (
                    best_selection
                    if numeric_medians is not None
                    and math.isfinite(best_selection)
                    else None
                ),
                "selection_metric": selection_metric,
                "best_selection_value": (
                    best_selection if math.isfinite(best_selection) else None
                ),
                "latest": row,
            },
        )
    completed = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    completed["state"] = "completed"
    completed["eta_seconds"] = 0.0
    _write_status(output_dir, completed)
    return output_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    result.add_argument("--ml-root", type=Path, default=Path(__file__).parents[1])
    result.add_argument("--manifest-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--samples-per-epoch", type=int)
    result.add_argument("--overfit-records", type=int)
    result.add_argument("--seed", type=int, default=20260728)
    result.add_argument("--device", default="auto")
    result.add_argument("--no-pretrained", action="store_true")
    result.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Initialize from a prior specialist stage before resetting the optimizer.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        output = train(args)
    except BaseException as exc:
        output = args.output_dir.resolve()
        status_path = output / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["state"] = "failed"
            status["error_type"] = type(exc).__name__
            status["error"] = str(exc)
            _write_status(output, status)
        raise
    print(f"finished: {output}")


if __name__ == "__main__":
    main()
