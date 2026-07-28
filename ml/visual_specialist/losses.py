"""Masked losses for mixed measured, public-semantic, and negative batches."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .constants import QUANTILES


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.sum() * 0.0


def quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    target_log = torch.log1p(target).unsqueeze(-1)
    error = target_log - prediction
    quantiles = prediction.new_tensor(QUANTILES).reshape(1, 1, -1)
    pinball = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    per_record = pinball.mean(dim=(1, 2))
    return _masked_mean(per_record, mask)


def balanced_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Give present and absent labels equal per-record influence."""
    elementwise = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    positive_count = targets.sum(dim=1).clamp_min(1.0)
    negative_count = (1.0 - targets).sum(dim=1).clamp_min(1.0)
    positive = (elementwise * targets).sum(dim=1) / positive_count
    negative = (elementwise * (1.0 - targets)).sum(dim=1) / negative_count
    return _masked_mean(0.5 * (positive + negative), mask)


def specialist_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    numeric = quantile_loss(
        output["numeric_log1p"], batch["numeric"], batch["numeric_mask"]
    )
    food = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output["food_logit"], batch["food"], reduction="none"
        ),
        batch["food_mask"],
    )
    frb = balanced_multilabel_loss(
        output["frb_logits"], batch["frb"], batch["frb_mask"]
    )
    container = _masked_mean(
        F.cross_entropy(
            output["container_logits"], batch["container"], reduction="none"
        ),
        batch["teacher_mask"],
    )
    cooking = balanced_multilabel_loss(
        output["cooking_logits"], batch["cooking"], batch["teacher_mask"]
    )
    mixed = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output["mixed_logit"], batch["mixed"], reduction="none"
        ),
        batch["teacher_mask"],
    )
    occlusion = _masked_mean(
        F.cross_entropy(
            output["occlusion_logits"], batch["occlusion"], reduction="none"
        ),
        batch["teacher_mask"],
    )
    parts = {
        "numeric": numeric,
        "food": food,
        "frb": frb,
        "container": container,
        "cooking": cooking,
        "mixed": mixed,
        "occlusion": occlusion,
    }
    total = (
        numeric
        + 0.25 * food
        + 0.50 * frb
        + 0.10 * container
        + 0.10 * cooking
        + 0.10 * mixed
        + 0.05 * occlusion
    )
    return total, parts
