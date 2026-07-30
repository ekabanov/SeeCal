"""Shared contracts and evaluation helpers for SeeCal's factored pipeline."""

from .contract import (
    CONTAINERS,
    IDENTIFY_PROMPT,
    IDENTIFY_PROMPT_V1,
    ContractError,
    canonical_identification,
    identification_to_shares,
    normalize_name,
    portion_units_from_weights,
    shares_from_weights,
    validate_identification,
    validate_identification_v1,
    validate_identification_with_repair,
)
from .resolver import NutritionProfile, Resolution, SQLiteNutritionResolver

__all__ = [
    "CONTAINERS",
    "IDENTIFY_PROMPT",
    "IDENTIFY_PROMPT_V1",
    "ContractError",
    "NutritionProfile",
    "Resolution",
    "SQLiteNutritionResolver",
    "canonical_identification",
    "identification_to_shares",
    "normalize_name",
    "portion_units_from_weights",
    "shares_from_weights",
    "validate_identification",
    "validate_identification_v1",
    "validate_identification_with_repair",
]
