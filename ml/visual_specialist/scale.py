"""Train, evaluate, and export the total-mass-only SCALE specialist.

SCALE is initialized from S1's shared representation and mass-quantile rows,
then optimized only for total plate mass. The output contract is
``mass_log1p: [batch, 3]`` for P10/P50/P90.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.transforms import v2

from .auxiliary import conformal_margin
from .train import choose_device

QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ScaleConfig:
    pretrained: bool = True
    dropout: float = 0.2
    ordered_quantiles: bool = False


class ScaleRegressor(nn.Module):
    def __init__(self, config: ScaleConfig = ScaleConfig()) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if config.pretrained else None
        base = mobilenet_v3_large(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.embedding = nn.Sequential(*list(base.classifier.children())[:-1])
        self.mass = nn.Linear(base.classifier[-1].in_features, len(QUANTILES))
        self.config = config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.avgpool(self.features(images))
        embedding = self.embedding(torch.flatten(features, 1))
        prediction = self.mass(embedding)
        if not self.config.ordered_quantiles:
            return prediction
        center = prediction[:, 0]
        lower_gap = torch.nn.functional.softplus(prediction[:, 1])
        upper_gap = torch.nn.functional.softplus(prediction[:, 2])
        return torch.stack((center - lower_gap, center, center + upper_gap), dim=-1)

    @torch.no_grad()
    def initialize_mass_bias(self, median_mass: float) -> None:
        center = math.log1p(median_mass)
        self.mass.weight.zero_()
        if self.config.ordered_quantiles:
            raw_gap = math.log(math.expm1(0.35))
            values = (center, raw_gap, raw_gap)
        else:
            values = (center - 0.35, center, center + 0.35)
        self.mass.bias.copy_(self.mass.bias.new_tensor(values))

    @torch.no_grad()
    def initialize_from_specialist(
        self, checkpoint: Path, *, median_mass: float | None = None
    ) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source = payload["model_state"]
        target = self.state_dict()
        for key in target:
            if key.startswith(("features.", "embedding.")) and key in source:
                target[key].copy_(source[key])
        if self.config.ordered_quantiles:
            if median_mass is None:
                raise ValueError("median_mass is required for ordered quantiles")
            self.initialize_mass_bias(median_mass)
        else:
            target["mass.weight"].copy_(source["numeric.weight"][:3])
            target["mass.bias"].copy_(source["numeric.bias"][:3])


class Letterbox:
    """Fit the full image inside a square with ImageNet-mean padding."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = self.image_size / max(image.size)
        resized = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.BILINEAR,
        )
        output = Image.new(
            "RGB",
            (self.image_size, self.image_size),
            color=(124, 116, 104),
        )
        output.paste(
            resized,
            (
                (self.image_size - resized.width) // 2,
                (self.image_size - resized.height) // 2,
            ),
        )
        return output


def scale_transforms(
    train: bool,
    image_size: int = 224,
    *,
    legacy_crop_augmentation: bool = False,
    geometry: str = "center-crop",
) -> v2.Compose:
    """Preserve whole-frame scale cues while augmenting appearance.

    RandomResizedCrop teaches a mass regressor that zooming the same plate does
    not change its target, destroying exactly the cue SCALE needs. The v2
    policy matches runtime resize/center-crop before mild appearance changes.
    """

    if geometry not in {"center-crop", "letterbox"}:
        raise ValueError(f"unknown SCALE input geometry: {geometry}")
    if legacy_crop_augmentation and geometry != "center-crop":
        raise ValueError("legacy crop augmentation cannot be combined with letterbox")
    resize_short_side = round(image_size * 232 / 224)
    geometry_transforms: list[Any] = (
        [Letterbox(image_size)]
        if geometry == "letterbox"
        else [
            v2.Resize(resize_short_side, antialias=True),
            v2.CenterCrop(image_size),
        ]
    )
    if train:
        if legacy_crop_augmentation:
            geometry_transforms = [
                v2.RandomResizedCrop(
                    image_size,
                    scale=(0.72, 1.0),
                    ratio=(0.85, 1.18),
                    antialias=True,
                )
            ]
        geometry_transforms.extend(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(5),
                v2.ColorJitter(0.18, 0.18, 0.12, 0.04),
            ]
        )
    return v2.Compose(
        [
            *geometry_transforms,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class ScaleDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        *,
        ml_root: Path,
        train: bool,
        overhead_only: bool = False,
        image_size: int = 224,
        legacy_crop_augmentation: bool = False,
        geometry: str = "center-crop",
    ) -> None:
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rows = [
            row
            for row in rows
            if not overhead_only or row["id"].endswith(":overhead")
        ]
        self.ml_root = ml_root
        self.transform = scale_transforms(
            train,
            image_size,
            legacy_crop_augmentation=legacy_crop_augmentation,
            geometry=geometry,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = Image.open(self.ml_root / row["image_path"]).convert("RGB")
        return {
            "id": row["id"],
            "group_id": row.get("group_id") or row["id"],
            "source": row.get("source", "unknown"),
            "image": self.transform(image),
            "mass": torch.tensor(float(row["total_mass_g"]), dtype=torch.float32),
        }


def mass_quantile_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_log = torch.log1p(target).unsqueeze(-1)
    error = target_log - prediction
    quantiles = prediction.new_tensor(QUANTILES).reshape(1, -1)
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()


def _decode(prediction: torch.Tensor) -> torch.Tensor:
    return torch.expm1(prediction.clamp(0, 10)).clamp_min(0)


def binomial_wilson_interval(
    rate: float,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> list[float]:
    """Approximate binomial interval using independent groups as trials."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between zero and one")
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


@torch.no_grad()
def evaluate_model(
    model: ScaleRegressor,
    loader: DataLoader,
    device: torch.device,
    *,
    margin: float | dict[str, float] = 0,
    margin_mode: str = "additive",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if margin_mode not in {"additive", "width-normalized"}:
        raise ValueError(f"unknown SCALE calibration margin mode: {margin_mode}")
    model.eval()
    rows = []
    for batch in loader:
        decoded = _decode(model(batch["image"].to(device))).cpu()
        for index, record_id in enumerate(batch["id"]):
            target = float(batch["mass"][index])
            source = str(batch["source"][index])
            record_margin = (
                float(margin[source])
                if isinstance(margin, dict)
                else float(margin)
            )
            raw_p10 = float(decoded[index, 0])
            raw_p90 = float(decoded[index, 2])
            adjustment = record_margin
            if margin_mode == "width-normalized":
                adjustment *= max(raw_p90 - raw_p10, 1e-6)
            p10 = max(0, raw_p10 - adjustment)
            p50 = float(decoded[index, 1])
            p90 = raw_p90 + adjustment
            rows.append(
                {
                    "id": record_id,
                    "group_id": batch["group_id"][index],
                    "source": source,
                    "target_mass_g": target,
                    "p10_g": p10,
                    "p50_g": p50,
                    "p90_g": p90,
                    "calibration_adjustment_g": adjustment,
                    "absolute_error_g": abs(p50 - target),
                    "absolute_percentage_error": (
                        abs(p50 - target) / target if target else 0
                    ),
                    "covered": p10 <= target <= p90,
                }
            )
    if not rows:
        raise ValueError(
            "SCALE evaluation selected zero records; use --include-all-views "
            "for manifests whose IDs do not end in :overhead"
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)
    equal_group_coverage = statistics.fmean(
        statistics.fmean(row["covered"] for row in group)
        for group in groups.values()
    )
    metrics = {
        "records": len(rows),
        "groups": len(groups),
        "mass_mae_g": statistics.fmean(row["absolute_error_g"] for row in rows),
        "mass_mape": statistics.fmean(
            row["absolute_percentage_error"] for row in rows
        ),
        "p10_p90_coverage": statistics.fmean(row["covered"] for row in rows),
        "median_absolute_error_g": statistics.median(
            row["absolute_error_g"] for row in rows
        ),
        "equal_group_mass_mae_g": statistics.fmean(
            statistics.fmean(row["absolute_error_g"] for row in group)
            for group in groups.values()
        ),
        "equal_group_mass_mape": statistics.fmean(
            statistics.fmean(row["absolute_percentage_error"] for row in group)
            for group in groups.values()
        ),
        "equal_group_p10_p90_coverage": equal_group_coverage,
        "equal_group_p10_p90_coverage_wilson_95_ci": binomial_wilson_interval(
            equal_group_coverage,
            len(groups),
        ),
    }
    return metrics, rows


def per_source_equal_group_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    output = {}
    for source in sorted({str(row["source"]) for row in rows}):
        selected = [row for row in rows if str(row["source"]) == source]
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            groups.setdefault(str(row["group_id"]), []).append(row)
        output[source] = {
            "records": len(selected),
            "groups": len(groups),
            "equal_group_mass_mae_g": statistics.fmean(
                statistics.fmean(float(row["absolute_error_g"]) for row in group)
                for group in groups.values()
            ),
            "equal_group_mass_mape": statistics.fmean(
                statistics.fmean(
                    float(row["absolute_percentage_error"]) for row in group
                )
                for group in groups.values()
            ),
        }
    return output


def select_minimax_checkpoint(
    epochs: list[dict[str, Any]],
    *,
    pareto_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Select worst-source MAPE, rejecting >tolerance per-source regret."""
    if not math.isfinite(pareto_tolerance) or pareto_tolerance < 0:
        raise ValueError("Pareto tolerance must be finite and non-negative")
    if not epochs:
        raise ValueError("cannot select from an empty epoch history")
    sources = set(epochs[0]["valid_by_source"])
    if not sources or any(set(row["valid_by_source"]) != sources for row in epochs):
        raise ValueError("every epoch must report the same non-empty source set")
    best_by_source = {
        source: min(
            float(row["valid_by_source"][source]["equal_group_mass_mape"])
            for row in epochs
        )
        for source in sources
    }
    eligible = []
    rejected = []
    for row in epochs:
        mapes = {
            source: float(
                row["valid_by_source"][source]["equal_group_mass_mape"]
            )
            for source in sources
        }
        regressions = {
            source: (
                mapes[source] / best_by_source[source] - 1
                if best_by_source[source] > 0
                else (0.0 if mapes[source] == 0 else math.inf)
            )
            for source in sources
        }
        candidate = {
            "epoch": int(row["epoch"]),
            "worst_source_equal_group_mape": max(mapes.values()),
            "mean_source_equal_group_mape": statistics.fmean(mapes.values()),
            "source_equal_group_mape": dict(sorted(mapes.items())),
            "source_relative_regret": dict(sorted(regressions.items())),
        }
        if all(value <= pareto_tolerance + 1e-12 for value in regressions.values()):
            eligible.append(candidate)
        else:
            rejected.append(candidate)
    if not eligible:
        raise ValueError(
            "no checkpoint satisfies the per-source Pareto tolerance; "
            f"best_by_source={best_by_source}"
        )
    selected = min(
        eligible,
        key=lambda row: (
            row["worst_source_equal_group_mape"],
            row["mean_source_equal_group_mape"],
            row["epoch"],
        ),
    )
    return {
        **selected,
        "pareto_tolerance": pareto_tolerance,
        "best_seen_by_source": dict(sorted(best_by_source.items())),
        "eligible_epochs": [row["epoch"] for row in eligible],
        "rejected_epochs": [row["epoch"] for row in rejected],
    }


def group_balanced_source_margins(
    rows: list[dict[str, Any]],
    *,
    target_coverage: float,
) -> dict[str, float]:
    """Calibrate each source with equal group weight and finite-group rank.

    Multi-view scenes are correlated. Each scene therefore contributes total
    weight one regardless of its view count, and the finite-sample conformal
    rank uses the number of independent groups rather than the number of
    images.
    """
    if not rows:
        raise ValueError("cannot calibrate empty SCALE rows")
    if not 0 < target_coverage < 1:
        raise ValueError("target coverage must be between zero and one")
    output = {}
    for source in sorted({str(row["source"]) for row in rows}):
        selected = [row for row in rows if str(row["source"]) == source]
        view_counts: dict[str, int] = {}
        for row in selected:
            group = str(row["group_id"])
            view_counts[group] = view_counts.get(group, 0) + 1
        group_count = len(view_counts)
        level = min(
            1.0,
            math.ceil((group_count + 1) * target_coverage) / group_count,
        )
        weighted_scores = sorted(
            (
                max(
                    float(row["p10_g"]) - float(row["target_mass_g"]),
                    float(row["target_mass_g"]) - float(row["p90_g"]),
                    0.0,
                ),
                1.0 / view_counts[str(row["group_id"])],
            )
            for row in selected
        )
        threshold = group_count * level
        cumulative = 0.0
        margin = weighted_scores[-1][0]
        for score, weight in weighted_scores:
            cumulative += weight
            if cumulative + 1e-12 >= threshold:
                margin = score
                break
        output[source] = float(margin)
    return output


def group_balanced_width_multiplier(
    rows: list[dict[str, Any]],
    *,
    target_coverage: float,
) -> float:
    """Fit one width-normalized conformal expansion with equal group weight."""
    if not rows:
        raise ValueError("cannot calibrate empty SCALE rows")
    if not 0 < target_coverage < 1:
        raise ValueError("target coverage must be between zero and one")
    view_counts: dict[str, int] = {}
    for row in rows:
        group = str(row["group_id"])
        view_counts[group] = view_counts.get(group, 0) + 1
    group_count = len(view_counts)
    level = min(
        1.0,
        math.ceil((group_count + 1) * target_coverage) / group_count,
    )
    weighted_scores = []
    for row in rows:
        p10 = float(row["p10_g"])
        p90 = float(row["p90_g"])
        target = float(row["target_mass_g"])
        width = max(p90 - p10, 1e-6)
        score = max(p10 - target, target - p90, 0.0) / width
        weighted_scores.append(
            (score, 1.0 / view_counts[str(row["group_id"])])
        )
    weighted_scores.sort()
    threshold = group_count * level
    cumulative = 0.0
    multiplier = weighted_scores[-1][0]
    for score, weight in weighted_scores:
        cumulative += weight
        if cumulative + 1e-12 >= threshold:
            multiplier = score
            break
    return float(multiplier)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def source_group_balanced_weights(
    rows: list[dict[str, Any]],
    source_weights: dict[str, float] | None = None,
) -> list[float]:
    """Give every source equal mass, then every dish equal mass within source."""

    groups_by_source: dict[str, set[str]] = {}
    views_by_group: dict[tuple[str, str], int] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        group = str(row.get("group_id") or row["id"])
        groups_by_source.setdefault(source, set()).add(group)
        key = (source, group)
        views_by_group[key] = views_by_group.get(key, 0) + 1
    if source_weights is None:
        source_weights = {
            source: 1 / len(groups_by_source) for source in groups_by_source
        }
    elif set(source_weights) != set(groups_by_source):
        raise ValueError(
            "source weights must name every manifest source exactly; "
            f"expected {sorted(groups_by_source)}, got {sorted(source_weights)}"
        )
    if any(not math.isfinite(value) or value <= 0 for value in source_weights.values()):
        raise ValueError("source weights must be finite and positive")
    weight_total = sum(source_weights.values())
    source_weights = {
        source: value / weight_total for source, value in source_weights.items()
    }
    return [
        source_weights[str(row.get("source", "unknown"))]
        / (
            len(groups_by_source[str(row.get("source", "unknown"))])
            * views_by_group[
                (
                    str(row.get("source", "unknown")),
                    str(row.get("group_id") or row["id"]),
                )
            ]
        )
        for row in rows
    ]


def _parse_source_weights(values: list[str]) -> dict[str, float] | None:
    if not values:
        return None
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source weight must be NAME=WEIGHT, got {value!r}")
        name, raw_weight = value.rsplit("=", 1)
        result[name] = float(raw_weight)
    return result


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    ml_root = args.ml_root.resolve()
    train_data = ScaleDataset(
        args.manifest_dir / "train.jsonl",
        ml_root=ml_root,
        train=True,
        image_size=args.image_size,
        legacy_crop_augmentation=args.legacy_crop_augmentation,
        geometry=args.geometry,
    )
    calibration_data = ScaleDataset(
        args.manifest_dir / "valid.jsonl",
        ml_root=ml_root,
        train=False,
        overhead_only=not args.calibration_include_all_views,
        image_size=args.image_size,
        geometry=args.geometry,
    )
    model = ScaleRegressor(
        ScaleConfig(
            pretrained=not args.no_pretrained,
            ordered_quantiles=args.ordered_quantiles,
        )
    )
    median_mass = statistics.median(
        float(row["total_mass_g"]) for row in train_data.rows
    )
    if args.initial_specialist:
        model.initialize_from_specialist(
            args.initial_specialist,
            median_mass=median_mass,
        )
    else:
        model.initialize_mass_bias(median_mass)
    model = model.to(device)
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith(("features.", "embedding.")) else head).append(
            parameter
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.learning_rate * args.backbone_lr_multiplier},
            {"params": head, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    sampler = None
    if args.sampling == "source-group-balanced":
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            source_group_balanced_weights(
                train_data.rows,
                _parse_source_weights(args.source_weight),
            ),
            num_samples=args.samples_per_epoch or len(train_data),
            replacement=True,
            generator=generator,
        )
    train_loader = _loader(
        train_data,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=sampler is None,
        sampler=sampler,
    )
    calibration_loader = _loader(
        calibration_data,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {
        "train": args.manifest_dir / "train.jsonl",
        "valid": args.manifest_dir / "valid.jsonl",
    }
    run_config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_config": asdict(model.config),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "manifests": {
            split: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for split, path in manifest_paths.items()
        },
        "train_records": len(train_data),
        "train_groups": len(
            {str(row.get("group_id") or row["id"]) for row in train_data.rows}
        ),
        "train_sources": {
            source: sum(1 for row in train_data.rows if row.get("source") == source)
            for source in sorted(
                {str(row.get("source", "unknown")) for row in train_data.rows}
            )
        },
        "calibration_records": len(calibration_data),
        "checkpoint_selection": {
            "direction": "minimize",
            "metric": (
                "valid.mass_mae_g"
                if args.checkpoint_selection == "pooled-mae"
                else "worst_source.equal_group_mass_mape"
            ),
            "policy": args.checkpoint_selection,
            "pareto_tolerance": args.pareto_tolerance,
            "split": "valid.jsonl",
            "official_test_sets_excluded": True,
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    best_mae = math.inf
    epoch_history: list[dict[str, Any]] = []
    history = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["mass"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mass_quantile_loss(model(image), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_sum += float(loss.detach()) * image.shape[0]
            examples += image.shape[0]
        scheduler.step()
        metrics, validation_rows = evaluate_model(model, calibration_loader, device)
        valid_by_source = per_source_equal_group_metrics(validation_rows)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(examples, 1),
            "valid": metrics,
            "valid_by_source": valid_by_source,
            "elapsed_seconds": time.monotonic() - started,
        }
        epoch_history.append(row)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        checkpoint_payload = {
            "schema_version": 1,
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "input_geometry": args.geometry,
            "image_size": args.image_size,
            "epoch": epoch,
            "metrics": metrics,
            "metrics_by_source": valid_by_source,
        }
        if args.checkpoint_selection == "minimax-source-mape":
            torch.save(
                checkpoint_payload,
                args.output_dir / f"epoch-{epoch:04d}.pt",
            )
        elif metrics["mass_mae_g"] < best_mae:
            best_mae = metrics["mass_mae_g"]
            torch.save(checkpoint_payload, args.output_dir / "best.pt")

    minimax_selection = None
    if args.checkpoint_selection == "minimax-source-mape":
        minimax_selection = select_minimax_checkpoint(
            epoch_history,
            pareto_tolerance=args.pareto_tolerance,
        )
        selected_path = (
            args.output_dir / f"epoch-{minimax_selection['epoch']:04d}.pt"
        )
        selected_payload = torch.load(
            selected_path, map_location="cpu", weights_only=True
        )
        torch.save(selected_payload, args.output_dir / "best.pt")
    payload = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
    excluded_tests = {}
    for name in ("test-nutrition5k.jsonl", "test-nutritionverse-real.jsonl"):
        path = args.manifest_dir / name
        if path.is_file():
            excluded_tests[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    selection: dict[str, Any] = {
        "schema_version": 1,
        "selected_epoch": int(payload["epoch"]),
        "selection_policy": args.checkpoint_selection,
        "selection_metric": (
            "valid.mass_mae_g"
            if args.checkpoint_selection == "pooled-mae"
            else "worst_source.equal_group_mass_mape"
        ),
        "selection_direction": "minimize",
        "selection_value": (
            float(payload["metrics"]["mass_mae_g"])
            if minimax_selection is None
            else float(minimax_selection["worst_source_equal_group_mape"])
        ),
        "selection_manifest": run_config["manifests"]["valid"],
        "official_test_sets_excluded": excluded_tests,
    }
    if minimax_selection is not None:
        selection["minimax"] = minimax_selection
    (args.output_dir / "checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    best = ScaleRegressor(ScaleConfig(**payload["model_config"]))
    best.load_state_dict(payload["model_state"])
    best = best.to(device)
    _, calibration_rows = evaluate_model(best, calibration_loader, device)
    intervals = [
        (row["p10_g"], row["p90_g"], row["target_mass_g"])
        for row in calibration_rows
    ]
    source_margins = {}
    if args.calibration_policy in {"max-source", "per-source"}:
        source_margins = group_balanced_source_margins(
            calibration_rows,
            target_coverage=args.target_coverage,
        )
        margin = max(source_margins.values())
        evaluation_margin = (
            source_margins
            if args.calibration_policy == "per-source"
            else margin
        )
    else:
        margin = conformal_margin(intervals, target_coverage=args.target_coverage)
        evaluation_margin = margin
    calibrated_metrics, _ = evaluate_model(
        best, calibration_loader, device, margin=evaluation_margin
    )
    calibration = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_coverage": args.target_coverage,
        "additive_margin_g": margin,
        "policy": args.calibration_policy,
        "source_margins_g": source_margins,
        "method": (
            "equal-group-weight-finite-group-rank"
            if source_margins
            else "pooled-record-split-conformal"
        ),
        "records": len(calibration_rows),
        "metrics": calibrated_metrics,
    }
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return args.output_dir


def load_checkpoint(path: Path) -> ScaleRegressor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ScaleRegressor(ScaleConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model


def checkpoint_input_geometry(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return str(payload.get("input_geometry", "center-crop"))


def evaluate_command(args: argparse.Namespace) -> Path:
    device = choose_device(args.device)
    model = load_checkpoint(args.checkpoint).to(device)
    geometry = args.geometry or checkpoint_input_geometry(args.checkpoint)
    dataset = ScaleDataset(
        args.manifest,
        ml_root=args.ml_root.resolve(),
        train=False,
        overhead_only=not args.include_all_views,
        image_size=args.image_size,
        geometry=geometry,
    )
    loader = _loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
    )
    margin: float | dict[str, float] = 0.0
    margin_mode = "additive"
    if args.calibration:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        if "width_multiplier" in calibration:
            margin = float(calibration["width_multiplier"])
            margin_mode = "width-normalized"
        elif calibration.get("policy") == "per-source":
            margin = {
                str(source): float(value)
                for source, value in calibration["source_margins_g"].items()
            }
        else:
            margin = float(calibration["additive_margin_g"])
    metrics, rows = evaluate_model(
        model,
        loader,
        device,
        margin=margin,
        margin_mode=margin_mode,
    )
    output = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "input_geometry": geometry,
        "calibration_mode": margin_mode,
        "calibration_value": margin,
        "metrics": metrics,
        "paired": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return args.output


def calibrate_command(args: argparse.Namespace) -> Path:
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    rows = evaluation["paired"]
    if args.method in {
        "width-normalized-union",
        "width-normalized-union-max-source",
    }:
        available_sources = {str(row["source"]) for row in rows}
        requested_sources = set(args.source) if args.source else available_sources
        missing = requested_sources - available_sources
        if missing:
            raise ValueError(
                f"calibration sources missing from evaluation: {sorted(missing)}"
            )
        selected = [
            row for row in rows if str(row["source"]) in requested_sources
        ]
        source_multipliers = {}
        if args.method == "width-normalized-union-max-source":
            source_multipliers = {
                source: group_balanced_width_multiplier(
                    [
                        row
                        for row in selected
                        if str(row["source"]) == source
                    ],
                    target_coverage=args.target_coverage,
                )
                for source in sorted(requested_sources)
            }
            multiplier = max(source_multipliers.values())
        else:
            multiplier = group_balanced_width_multiplier(
                selected,
                target_coverage=args.target_coverage,
            )
        calibration = {
            "schema_version": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_coverage": args.target_coverage,
            "policy": "union",
            "calibration_sources": sorted(requested_sources),
            "width_multiplier": multiplier,
            "source_width_multipliers": source_multipliers,
            "method": (
                "max-source-equal-group-finite-rank-width-normalized"
                if source_multipliers
                else "equal-group-weight-finite-group-rank-width-normalized"
            ),
            "records": len(selected),
            "groups": len({str(row["group_id"]) for row in selected}),
            "source_evaluation": str(args.evaluation),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(calibration, indent=2, sort_keys=True))
        return args.output
    source_margins = group_balanced_source_margins(
        rows,
        target_coverage=args.target_coverage,
    )
    calibration = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_coverage": args.target_coverage,
        "additive_margin_g": max(source_margins.values()),
        "policy": "per-source",
        "source_margins_g": source_margins,
        "method": "equal-group-weight-finite-group-rank",
        "records": len(rows),
        "groups_by_source": {
            source: len(
                {
                    str(row["group_id"])
                    for row in rows
                    if str(row["source"]) == source
                }
            )
            for source in source_margins
        },
        "source_evaluation": str(args.evaluation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(calibration, indent=2, sort_keys=True))
    return args.output


class ExportWrapper(nn.Module):
    def __init__(self, model: ScaleRegressor) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def export_command(args: argparse.Namespace) -> Path:
    model = ExportWrapper(load_checkpoint(args.checkpoint).eval())
    example = torch.zeros((1, 3, args.image_size, args.image_size))
    traced = torch.jit.trace(model, example, strict=True)
    if args.trace_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(args.output))
        return args.output
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools is required unless --trace-only is used") from exc
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="image", shape=example.shape, dtype=float)],
        outputs=[ct.TensorType(name="mass_log1p")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    return args.output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--manifest-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    train_parser.add_argument("--initial-specialist", type=Path)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument(
        "--geometry",
        choices=("center-crop", "letterbox"),
        default="center-crop",
    )
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--target-coverage", type=float, default=0.8)
    train_parser.add_argument("--seed", type=int, default=20260729)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--ordered-quantiles", action="store_true")
    train_parser.add_argument(
        "--sampling",
        choices=("records", "source-group-balanced"),
        default="records",
    )
    train_parser.add_argument("--samples-per-epoch", type=int)
    train_parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Desired sampling mix; repeat once for every manifest source.",
    )
    train_parser.add_argument(
        "--legacy-crop-augmentation",
        action="store_true",
        help="Use the v1 zoom-changing crop augmentation for an ablation.",
    )
    train_parser.add_argument(
        "--calibration-include-all-views",
        action="store_true",
    )
    train_parser.add_argument(
        "--calibration-policy",
        choices=("pooled", "max-source", "per-source"),
        default="pooled",
    )
    train_parser.add_argument(
        "--checkpoint-selection",
        choices=("pooled-mae", "minimax-source-mape"),
        default="pooled-mae",
    )
    train_parser.add_argument(
        "--pareto-tolerance",
        type=float,
        default=0.10,
        help="Maximum relative per-source MAPE regret for minimax selection.",
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--calibration", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser.add_argument("--workers", type=int, default=4)
    evaluate_parser.add_argument("--image-size", type=int, default=224)
    evaluate_parser.add_argument(
        "--geometry",
        choices=("center-crop", "letterbox"),
        help="Override checkpoint geometry; omitted uses checkpoint metadata.",
    )
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument(
        "--include-all-views",
        action="store_true",
        help="Evaluate every manifest record instead of only IDs ending in :overhead.",
    )

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--evaluation", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument("--target-coverage", type=float, default=0.8)
    calibrate_parser.add_argument(
        "--method",
        choices=(
            "additive-per-source",
            "width-normalized-union",
            "width-normalized-union-max-source",
        ),
        default="additive-per-source",
    )
    calibrate_parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source to include in union calibration; repeat as needed.",
    )

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--image-size", type=int, default=224)
    export_parser.add_argument("--trace-only", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        print(train(args))
    elif args.command == "evaluate":
        print(evaluate_command(args))
    elif args.command == "calibrate":
        print(calibrate_command(args))
    else:
        print(export_command(args))


if __name__ == "__main__":
    main()
