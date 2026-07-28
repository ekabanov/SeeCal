"""Shared output vocabularies and numeric-field order."""

NUMERIC_FIELDS = ("mass_g", "calories", "protein_g", "fat_g", "carbs_g")
QUANTILES = (0.1, 0.5, 0.9)
CONTAINERS = (
    "plate",
    "bowl",
    "tray",
    "cup_or_glass",
    "wrapper_or_package",
    "none_or_unclear",
)
COOKING_STATES = (
    "raw",
    "boiled",
    "baked",
    "grilled",
    "fried",
    "steamed",
    "roasted",
    "mixed_or_unclear",
)
OCCLUSION_STATES = ("none", "partial", "severe")
FRB_CLASS_COUNT = 498
