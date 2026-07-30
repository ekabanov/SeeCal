"""Prospective id-v2 policy for labels inferable from a single RGB image."""

from __future__ import annotations

import math
from typing import Iterable

from .contract import normalize_name
from .contract import shares_from_weights


# Recipe metadata frequently contains incorporated cooking aids at tiny masses.
# Training IDENTIFY to emit them teaches plausible hallucination, not vision.
HIDDEN_RECIPE_COMPONENTS = frozenset(
    {
        "basil",
        "chive",
        "cilantro",
        "garlic",
        "lemon juice",
        "lime",
        "lime juice",
        "mustard",
        "olive oil",
        "orange juice",
        "oregano",
        "parsley",
        "pepper",
        "rosemary",
        "salt",
        "soy sauce",
        "thyme",
        "vinegar",
        "wine",
    }
)


def visible_component_weights(
    weighted_names: Iterable[tuple[str, float]],
    *,
    minimum_fraction: float = 0.02,
    maximum_items: int = 8,
) -> list[tuple[str, float]]:
    """Keep material, visually distinguishable components for supervision.

    Fractions are measured against the original total before hidden ingredients
    are removed. At least the largest source component is retained.
    """
    if not 0 <= minimum_fraction < 1:
        raise ValueError("minimum_fraction must be in [0, 1)")
    if maximum_items <= 0:
        raise ValueError("maximum_items must be positive")
    merged: dict[str, tuple[str, float]] = {}
    for raw_name, raw_weight in weighted_names:
        name = str(raw_name).strip()
        key = normalize_name(name)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if not key or not math.isfinite(weight) or weight <= 0:
            continue
        prior = merged.get(key)
        merged[key] = (
            prior[0] if prior else name,
            (prior[1] if prior else 0.0) + weight,
        )
    if not merged:
        return []
    ranked = sorted(
        merged.items(),
        key=lambda row: (-row[1][1], row[0]),
    )
    total = sum(value[1] for _, value in ranked)
    visible = [
        value
        for key, value in ranked
        if key not in HIDDEN_RECIPE_COMPONENTS
        and value[1] / total >= minimum_fraction
    ][:maximum_items]
    if visible:
        return visible
    return [ranked[0][1]]


def filter_visible_prediction(payload: dict) -> dict:
    """Apply the same visible-component policy to normalized model shares."""

    if payload.get("not_food"):
        return dict(payload)
    visible = visible_component_weights(
        [
            (str(item["name"]), float(item["share_pct"]))
            for item in payload.get("items", [])
        ]
    )
    if not visible:
        return dict(payload)
    return {
        "not_food": False,
        "container": payload["container"],
        "items": shares_from_weights(visible),
    }
