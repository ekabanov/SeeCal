import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_specialist.data import SpecialistDataset, balanced_sampler
from visual_specialist.losses import (
    balanced_multilabel_loss,
    quantile_loss,
    specialist_loss,
)
from visual_specialist.manifest import select_frb_validation
from visual_specialist.model import SpecialistConfig, VisualSpecialist
from visual_specialist.train import _frb_metrics


def _frb(image_id, labels):
    return {
        "source": {"image_id": image_id},
        "source_labels": [
            {"category_id": label, "name": f"class-{label}"} for label in labels
        ],
    }


def test_frb_validation_is_deterministic_and_covers_every_class():
    records = [
        _frb(1, [0, 1]),
        _frb(2, [1, 2]),
        _frb(3, [3]),
        _frb(4, [0]),
        _frb(5, [2]),
        _frb(6, [3]),
    ]
    selected = select_frb_validation(records, count=3, seed=7)
    assert selected == select_frb_validation(records, count=3, seed=7)
    covered = set().union(
        *[
            {
                label["category_id"]
                for label in row["source_labels"]
            }
            for row in records
            if row["source"]["image_id"] in selected
        ]
    )
    assert covered == {0, 1, 2, 3}


def test_quantile_loss_ignores_unmasked_rows():
    predictions = torch.zeros((2, 5, 3))
    targets = torch.tensor([[1.0] * 5, [10_000.0] * 5])
    masked = quantile_loss(predictions, targets, torch.tensor([True, False]))
    expected = quantile_loss(
        predictions[:1], targets[:1], torch.tensor([True])
    )
    assert masked == pytest.approx(float(expected))


def test_model_output_contract_without_downloading_weights():
    model = VisualSpecialist(SpecialistConfig(pretrained=False))
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros((2, 3, 64, 64)))
    assert output["numeric_log1p"].shape == (2, 5, 3)
    assert output["frb_logits"].shape == (2, 498)
    assert output["container_logits"].shape == (2, 6)
    assert output["cooking_logits"].shape == (2, 8)


def test_numeric_bias_initialization_sets_ordered_quantiles():
    model = VisualSpecialist(SpecialistConfig(pretrained=False))
    medians = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    model.initialize_numeric_bias(medians)
    output = model(torch.zeros((1, 3, 64, 64)))["numeric_log1p"][0]
    assert torch.all(output[:, 0] < output[:, 1])
    assert torch.all(output[:, 1] < output[:, 2])
    assert torch.allclose(output[:, 1], medians)


def test_balanced_multilabel_loss_does_not_reward_all_negative_collapse():
    target = torch.zeros((1, 498))
    target[0, 12] = 1
    mask = torch.tensor([True])
    all_negative = balanced_multilabel_loss(
        torch.full_like(target, -8.0), target, mask
    )
    correct = torch.full_like(target, -8.0)
    correct[0, 12] = 8.0
    assert balanced_multilabel_loss(correct, target, mask) < all_negative


def test_frb_ranking_metrics_reward_correct_order():
    target = torch.zeros((2, 4))
    target[0, 0] = 1
    target[1, 3] = 1
    correct = torch.tensor([[8.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 8.0]])
    reversed_logits = -correct
    good = _frb_metrics([correct], [target])
    bad = _frb_metrics([reversed_logits], [target])
    assert good["frb_micro_average_precision"] > bad["frb_micro_average_precision"]
    assert good["frb_macro_average_precision"] > bad["frb_macro_average_precision"]


def test_all_masked_optional_losses_remain_finite():
    model = VisualSpecialist(SpecialistConfig(pretrained=False))
    output = model(torch.zeros((2, 3, 64, 64)))
    batch = {
        "numeric": torch.zeros((2, 5)),
        "food": torch.ones(2),
        "frb": torch.zeros((2, 498)),
        "container": torch.zeros(2, dtype=torch.long),
        "cooking": torch.zeros((2, 8)),
        "mixed": torch.zeros(2),
        "occlusion": torch.zeros(2, dtype=torch.long),
        "numeric_mask": torch.tensor([False, False]),
        "food_mask": torch.tensor([True, True]),
        "frb_mask": torch.tensor([False, False]),
        "teacher_mask": torch.tensor([False, False]),
    }
    loss, parts = specialist_loss(output, batch)
    assert torch.isfinite(loss)
    assert parts["numeric"] == 0
    assert parts["frb"] == 0
