"""Frozen IDENTIFY prompt and completion-schema helpers.

This module is deliberately dependency-free. Data preparation, Python
inference, parity checks, and tests import the same prompt constant.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from copy import deepcopy
from typing import Iterable

CONTAINERS = ("plate", "bowl", "cup", "tray", "packaging", "other")
SHARE_STEP = 5
MAX_ITEMS = 100 // SHARE_STEP
PORTION_UNIT_MAX = 20

# Historical percentage contract used by the failed id-v1 diagnostic arm.
IDENTIFY_PROMPT_V1 = (
    "Identify the visible food without estimating grams, calories, or nutrients. "
    "Return exactly one JSON object with keys not_food, container, and items. "
    "container must be plate, bowl, cup, tray, packaging, or other. "
    "items must contain objects with name and share_pct only, sorted by share_pct "
    "descending. share_pct values must be multiples of 5 and sum to 100. "
    "For a non-food image return not_food true, container other, and an empty items list."
)

# id-v2 makes relative composition valid by construction. The model selects
# bounded relative units; deterministic code handles merging, sorting, and
# normalization instead of requiring a 4B language model to do arithmetic.
IDENTIFY_PROMPT = (
    "Identify only visually distinguishable food components without estimating "
    "grams, calories, nutrients, seasonings, or hidden recipe ingredients. "
    "Return exactly one JSON object with keys not_food, container, and items. "
    "container must be plate, bowl, cup, tray, packaging, or other. "
    "items must contain objects with name and portion_units only. "
    "portion_units must be an integer from 1 to 20 expressing relative visible "
    "portion size; the values do not need to sum to any particular number. "
    "For a non-food image return not_food true, container other, and an empty items list."
)


class ContractError(ValueError):
    """An IDENTIFY completion violates the frozen schema."""


def normalize_name(value: str) -> str:
    """Canonical lexical form shared by DB building, resolution, and scoring."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def shares_from_weights(
    weighted_names: Iterable[tuple[str, float]],
    *,
    step: int = SHARE_STEP,
) -> list[dict[str, object]]:
    """Quantize positive weights with largest remainders and an exact 100 sum.

    At most ``100 / step`` positive buckets can exist. When a source contains
    more tiny ingredients, the lowest-weight tails are omitted before
    quantization instead of emitting zero-share items.
    """
    if step <= 0 or 100 % step:
        raise ContractError("share step must be a positive divisor of 100")
    merged: dict[str, tuple[str, float]] = {}
    for raw_name, raw_weight in weighted_names:
        name = raw_name.strip()
        weight = float(raw_weight)
        key = normalize_name(name)
        if not key or not math.isfinite(weight) or weight <= 0:
            continue
        prior = merged.get(key)
        merged[key] = (prior[0] if prior else name, (prior[1] if prior else 0) + weight)
    if not merged:
        raise ContractError("food identification requires at least one positive item")

    units = 100 // step
    ranked = sorted(merged.values(), key=lambda pair: (-pair[1], normalize_name(pair[0])))
    ranked = ranked[:units]
    total = sum(weight for _, weight in ranked)
    exact_units = [weight / total * units for _, weight in ranked]

    # Give every retained item one bucket, then distribute remaining buckets by
    # largest-remainder apportionment against the original proportions.
    allocated = [1] * len(ranked)
    remaining = units - len(ranked)
    if remaining:
        desired_extra = [max(0.0, value - 1.0) for value in exact_units]
        extra_total = sum(desired_extra)
        if extra_total == 0:
            desired_extra = exact_units
            extra_total = sum(desired_extra)
        scaled = [value / extra_total * remaining for value in desired_extra]
        floors = [math.floor(value) for value in scaled]
        allocated = [base + extra for base, extra in zip(allocated, floors)]
        leftovers = remaining - sum(floors)
        order = sorted(
            range(len(ranked)),
            key=lambda index: (-(scaled[index] - floors[index]), index),
        )
        for index in order[:leftovers]:
            allocated[index] += 1

    items = [
        {"name": name, "share_pct": bucket_count * step}
        for (name, _), bucket_count in zip(ranked, allocated)
    ]
    items.sort(key=lambda item: (-int(item["share_pct"]), normalize_name(str(item["name"]))))
    return items


def portion_units_from_weights(
    weighted_names: Iterable[tuple[str, float]],
    *,
    maximum: int = PORTION_UNIT_MAX,
) -> list[dict[str, object]]:
    """Convert positive weights to independent bounded relative units."""
    if maximum <= 0:
        raise ContractError("portion-unit maximum must be positive")
    merged: dict[str, tuple[str, float]] = {}
    for raw_name, raw_weight in weighted_names:
        name = raw_name.strip()
        weight = float(raw_weight)
        key = normalize_name(name)
        if not key or not math.isfinite(weight) or weight <= 0:
            continue
        prior = merged.get(key)
        merged[key] = (
            prior[0] if prior else name,
            (prior[1] if prior else 0.0) + weight,
        )
    if not merged:
        raise ContractError("food identification requires at least one positive item")
    ranked = sorted(
        merged.values(),
        key=lambda pair: (-pair[1], normalize_name(pair[0])),
    )[:MAX_ITEMS]
    total = sum(weight for _, weight in ranked)
    items = [
        {
            "name": name,
            "portion_units": max(1, min(maximum, round(weight / total * maximum))),
        }
        for name, weight in ranked
    ]
    items.sort(
        key=lambda item: (
            -int(item["portion_units"]),
            normalize_name(str(item["name"])),
        )
    )
    return items


def validate_identification_v1(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ContractError("completion must be a JSON object")
    if set(payload) != {"not_food", "container", "items"}:
        raise ContractError("completion keys must be exactly not_food, container, items")
    if not isinstance(payload["not_food"], bool):
        raise ContractError("not_food must be a boolean")
    if payload["container"] not in CONTAINERS:
        raise ContractError("invalid container")
    if not isinstance(payload["items"], list):
        raise ContractError("items must be a list")

    if payload["not_food"]:
        if payload["container"] != "other" or payload["items"]:
            raise ContractError("not-food completions require container other and no items")
        return payload
    if not payload["items"]:
        raise ContractError("food completions require at least one item")

    prior_share = 101
    share_sum = 0
    seen: set[str] = set()
    for index, item in enumerate(payload["items"]):
        if not isinstance(item, dict) or set(item) != {"name", "share_pct"}:
            raise ContractError(f"items[{index}] keys must be exactly name, share_pct")
        name = item["name"]
        share = item["share_pct"]
        if not isinstance(name, str) or not normalize_name(name):
            raise ContractError(f"items[{index}].name must be non-empty")
        if type(share) is not int or share <= 0 or share % SHARE_STEP:
            raise ContractError(f"items[{index}].share_pct must be a positive multiple of 5")
        if share > prior_share:
            raise ContractError("items must be sorted by share_pct descending")
        key = normalize_name(name)
        if key in seen:
            raise ContractError(f"duplicate item name: {name}")
        seen.add(key)
        prior_share = share
        share_sum += share
    if share_sum != 100:
        raise ContractError(f"share_pct values must sum to 100, got {share_sum}")
    return payload


def validate_identification_with_repair(
    payload: object,
    *,
    minimum_sum: int = 90,
    maximum_sum: int = 110,
) -> tuple[dict[str, object], bool, int | None]:
    """Validate IDENTIFY output, repairing only a bounded share-sum slip.

    Structural violations remain hard failures.  If every field is otherwise
    valid and positive 5%-bucket shares total 90–110, largest-remainder
    apportionment deterministically restores an exact 100 sum.  The caller
    receives an explicit repaired flag and the original sum for auditability.
    """

    try:
        return validate_identification_v1(payload), False, None
    except ContractError as original_error:
        if not isinstance(payload, dict) or set(payload) != {
            "not_food",
            "container",
            "items",
        }:
            raise original_error
        if payload.get("not_food") is not False:
            raise original_error
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise original_error
        raw_sum = 0
        prior_share = 101
        normalized_names: set[str] = set()
        weighted_names: list[tuple[str, float]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict) or set(item) != {"name", "share_pct"}:
                raise original_error
            name = item["name"]
            share = item["share_pct"]
            normalized = normalize_name(name) if isinstance(name, str) else ""
            if not normalized or normalized in normalized_names:
                raise original_error
            if type(share) is not int or share <= 0 or share % SHARE_STEP:
                raise original_error
            if share > prior_share:
                raise original_error
            normalized_names.add(normalized)
            weighted_names.append((name.strip(), float(share)))
            raw_sum += share
            prior_share = share
        if not minimum_sum <= raw_sum <= maximum_sum:
            raise original_error
        repaired = deepcopy(payload)
        repaired["items"] = shares_from_weights(weighted_names)
        return validate_identification_v1(repaired), True, raw_sum


def normalize_legacy_share_identification(payload: object) -> dict[str, object]:
    """Normalize arbitrary positive legacy percentage weights to exactly 100."""

    if not isinstance(payload, dict) or set(payload) != {
        "not_food",
        "container",
        "items",
    }:
        raise ContractError(
            "completion keys must be exactly not_food, container, items"
        )
    if not isinstance(payload["not_food"], bool):
        raise ContractError("not_food must be a boolean")
    if payload["container"] not in CONTAINERS:
        raise ContractError("invalid container")
    if not isinstance(payload["items"], list):
        raise ContractError("items must be a list")
    if payload["not_food"]:
        if payload["container"] != "other" or payload["items"]:
            raise ContractError("not-food completions require container other and no items")
        return dict(payload)
    if not payload["items"]:
        raise ContractError("food completions require at least one item")

    weighted_names: list[tuple[str, float]] = []
    for index, item in enumerate(payload["items"]):
        if not isinstance(item, dict) or set(item) != {"name", "share_pct"}:
            raise ContractError(
                f"items[{index}] keys must be exactly name, share_pct"
            )
        name = item["name"]
        share = item["share_pct"]
        if not isinstance(name, str) or not normalize_name(name):
            raise ContractError(f"items[{index}].name must be non-empty")
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not math.isfinite(float(share))
            or float(share) <= 0
        ):
            raise ContractError(f"items[{index}].share_pct must be positive")
        weighted_names.append((name.strip(), float(share)))
    return {
        "not_food": False,
        "container": payload["container"],
        "items": shares_from_weights(weighted_names),
    }


def validate_identification(payload: object) -> dict[str, object]:
    """Validate and canonicalize an id-v2 relative-portion completion."""
    if not isinstance(payload, dict):
        raise ContractError("completion must be a JSON object")
    if set(payload) != {"not_food", "container", "items"}:
        raise ContractError("completion keys must be exactly not_food, container, items")
    if not isinstance(payload["not_food"], bool):
        raise ContractError("not_food must be a boolean")
    if payload["container"] not in CONTAINERS:
        raise ContractError("invalid container")
    if not isinstance(payload["items"], list):
        raise ContractError("items must be a list")
    if payload["not_food"]:
        if payload["container"] != "other" or payload["items"]:
            raise ContractError("not-food completions require container other and no items")
        return dict(payload)
    if not payload["items"]:
        raise ContractError("food completions require at least one item")

    merged: dict[str, tuple[str, int]] = {}
    for index, item in enumerate(payload["items"]):
        if not isinstance(item, dict) or set(item) != {"name", "portion_units"}:
            raise ContractError(
                f"items[{index}] keys must be exactly name, portion_units"
            )
        name = item["name"]
        units = item["portion_units"]
        if not isinstance(name, str) or not normalize_name(name):
            raise ContractError(f"items[{index}].name must be non-empty")
        if type(units) is not int or not 1 <= units <= PORTION_UNIT_MAX:
            raise ContractError(
                f"items[{index}].portion_units must be an integer from 1 to "
                f"{PORTION_UNIT_MAX}"
            )
        key = normalize_name(name)
        prior = merged.get(key)
        merged[key] = (
            prior[0] if prior else name.strip(),
            (prior[1] if prior else 0) + units,
        )
    items = [
        {"name": name, "portion_units": units}
        for name, units in merged.values()
    ]
    items.sort(
        key=lambda item: (
            -int(item["portion_units"]),
            normalize_name(str(item["name"])),
        )
    )
    return {
        "not_food": False,
        "container": payload["container"],
        "items": items,
    }


def identification_to_shares(payload: object) -> dict[str, object]:
    """Normalize validated portion units into exact 5-point share buckets."""
    identification = validate_identification(payload)
    if identification["not_food"]:
        return identification
    units = [
        (str(item["name"]), float(item["portion_units"]))
        for item in identification["items"]
    ]
    shares = shares_from_weights(units)
    unit_by_name = {
        normalize_name(str(item["name"])): int(item["portion_units"])
        for item in identification["items"]
    }
    return {
        "not_food": False,
        "container": identification["container"],
        "items": [
            {
                **item,
                "portion_units": unit_by_name[normalize_name(str(item["name"]))],
            }
            for item in shares
        ],
    }


def canonical_identification(
    *,
    container: str,
    weighted_names: Iterable[tuple[str, float]],
) -> str:
    payload = {
        "not_food": False,
        "container": container,
        "items": portion_units_from_weights(weighted_names),
    }
    validate_identification(payload)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


NOT_FOOD_COMPLETION = json.dumps(
    {"not_food": True, "container": "other", "items": []},
    separators=(",", ":"),
)
