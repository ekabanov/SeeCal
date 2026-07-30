import torch
import json
from PIL import Image

from visual_specialist.scale import (
    Letterbox,
    ScaleConfig,
    ScaleDataset,
    ScaleRegressor,
    binomial_wilson_interval,
    group_balanced_source_margins,
    group_balanced_width_multiplier,
    mass_quantile_loss,
    select_minimax_checkpoint,
    source_group_balanced_weights,
)


def test_scale_specialist_has_only_three_mass_quantiles():
    model = ScaleRegressor(ScaleConfig(pretrained=False))
    output = model(torch.zeros((2, 3, 224, 224)))
    assert output.shape == (2, 3)


def test_mass_quantile_loss_prefers_correct_log_target():
    target = torch.tensor([99.0])
    correct = torch.log1p(target).repeat(3).reshape(1, 3)
    wrong = correct + 1
    assert mass_quantile_loss(correct, target) == 0
    assert mass_quantile_loss(wrong, target) > 0


def test_ordered_scale_head_cannot_cross_quantiles():
    model = ScaleRegressor(
        ScaleConfig(pretrained=False, ordered_quantiles=True)
    )
    output = model(torch.zeros((2, 3, 224, 224)))
    assert torch.all(output[:, 0] <= output[:, 1])
    assert torch.all(output[:, 1] <= output[:, 2])


def test_source_group_balancing_equalizes_sources_and_groups():
    rows = [
        {"id": "a1", "group_id": "a", "source": "n5k"},
        {"id": "a2", "group_id": "a", "source": "n5k"},
        {"id": "b1", "group_id": "b", "source": "n5k"},
        {"id": "c1", "group_id": "c", "source": "phone"},
    ]
    weights = source_group_balanced_weights(rows)
    assert sum(weights[:3]) == sum(weights[3:])
    assert weights[0] + weights[1] == weights[2]


def test_source_group_balancing_honors_explicit_mix():
    rows = [
        {"id": "a", "group_id": "a", "source": "n5k"},
        {"id": "b", "group_id": "b", "source": "phone"},
    ]
    weights = source_group_balanced_weights(rows, {"n5k": 0.8, "phone": 0.2})
    assert weights == [0.8, 0.2]


def test_scale_dataset_can_explicitly_include_non_overhead_views(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": record_id,
                    "group_id": "dish_1",
                    "image_path": "unused.jpg",
                    "total_mass_g": 100,
                }
            )
            for record_id in ("nutrition5k:dish_1:overhead", "nv-real:dish_1:view_2")
        )
    )
    overhead = ScaleDataset(
        manifest,
        ml_root=tmp_path,
        train=False,
        overhead_only=True,
    )
    all_views = ScaleDataset(
        manifest,
        ml_root=tmp_path,
        train=False,
        overhead_only=False,
    )
    assert len(overhead) == 1
    assert len(all_views) == 2


def test_group_balanced_calibration_uses_independent_group_count():
    rows = []
    for source, offset in (("rig", 0), ("phone", 10)):
        for group, score in enumerate((0, 1, 2, 3)):
            copies = 100 if source == "rig" and group == 0 else 1
            rows.extend(
                {
                    "source": source,
                    "group_id": f"{source}:{group}",
                    "target_mass_g": 100,
                    "p10_g": 0,
                    "p90_g": 100 - offset - score,
                }
                for _ in range(copies)
            )

    margins = group_balanced_source_margins(rows, target_coverage=0.5)

    assert margins == {"phone": 12.0, "rig": 2.0}


def test_width_normalized_calibration_scales_with_predicted_interval():
    rows = [
        {
            "source": "phone",
            "group_id": "phone:1",
            "target_mass_g": 115,
            "p10_g": 90,
            "p90_g": 110,
        },
        {
            "source": "phone",
            "group_id": "phone:2",
            "target_mass_g": 230,
            "p10_g": 180,
            "p90_g": 220,
        },
    ]

    multiplier = group_balanced_width_multiplier(rows, target_coverage=0.5)

    assert multiplier == 0.25


def test_group_coverage_interval_uses_group_count():
    narrow = binomial_wilson_interval(0.8, 325)
    wide = binomial_wilson_interval(0.8, 32)
    assert narrow[0] > wide[0]
    assert narrow[1] < wide[1]


def test_letterbox_preserves_full_aspect_ratio_with_neutral_padding():
    source = Image.new("RGB", (100, 200), color=(255, 0, 0))
    output = Letterbox(224)(source)
    assert output.size == (224, 224)
    assert output.getpixel((0, 112)) == (124, 116, 104)
    assert output.getpixel((112, 112)) == (255, 0, 0)


def test_minimax_selection_applies_final_pareto_guard():
    def epoch(number, rig, phone):
        return {
            "epoch": number,
            "valid_by_source": {
                "rig": {"equal_group_mass_mape": rig},
                "phone": {"equal_group_mass_mape": phone},
            },
        }

    selection = select_minimax_checkpoint(
        [
            epoch(1, 0.10, 0.30),
            epoch(2, 0.11, 0.20),
            epoch(3, 0.20, 0.19),
        ],
        pareto_tolerance=0.10,
    )

    assert selection["epoch"] == 2
    assert selection["eligible_epochs"] == [2]
    assert selection["rejected_epochs"] == [1, 3]
