"""Deterministic nutrition-name resolution against the pruned SQLite DB."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from pathlib import Path
import sqlite3
from typing import Protocol

from .contract import normalize_name


@dataclass(frozen=True)
class NutritionProfile:
    fdc_id: int | None
    name: str
    category: str
    kcal_per_100g: float
    protein_per_100g: float
    fat_per_100g: float
    carbs_per_100g: float
    source_kcal_per_100g: float | None
    data_type: str
    typical_portion_g: float | None = None

    def is_sane(self) -> bool:
        values = (
            self.kcal_per_100g,
            self.protein_per_100g,
            self.fat_per_100g,
            self.carbs_per_100g,
        )
        if not all(math.isfinite(value) and value >= 0 for value in values):
            return False
        return (
            self.kcal_per_100g <= 900
            and self.protein_per_100g <= 100
            and self.fat_per_100g <= 100
            and self.carbs_per_100g <= 100
        )


@dataclass(frozen=True)
class Resolution:
    query: str
    rung: str
    score: float
    profile: NutritionProfile | None
    estimated: bool


class NutritionHypothesisProvider(Protocol):
    def hypothesis(self, name: str, category: str | None) -> NutritionProfile | None:
        """Return an explicit hypothesis or None; never fabricate zero values."""


def lexical_score(lhs: str, rhs: str) -> float:
    left = normalize_name(lhs)
    right = normalize_name(rhs)
    if not left or not right:
        return 0.0
    left_tokens = {_singular_token(token) for token in left.split()}
    right_tokens = {_singular_token(token) for token in right.split()}
    if left_tokens == right_tokens:
        return 0.98
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    left_compact, right_compact = left.replace(" ", ""), right.replace(" ", "")
    if left_compact == right_compact:
        return 1.0
    edit = max(
        SequenceMatcher(None, left, right).ratio(),
        SequenceMatcher(None, left_compact, right_compact).ratio(),
    )
    containment = (
        1.0
        if left in right
        or right in left
        or left_compact in right_compact
        or right_compact in left_compact
        else 0.0
    )
    return 0.55 * overlap + 0.35 * edit + 0.10 * containment


def category_lexical_score(lhs: str, rhs: str) -> float:
    """Asymmetric query coverage used only to choose a safe category rung.

    A short generic label should not become a specific nutrient profile just
    because it appears inside a long USDA description. It can, however, vote
    for that row's category when choosing an explicitly estimated median.
    """
    left = normalize_name(lhs)
    right = normalize_name(rhs)
    if not left or not right:
        return 0.0
    left_tokens = {_singular_token(token) for token in left.split()}
    right_tokens = {_singular_token(token) for token in right.split()}
    intersection = len(left_tokens & right_tokens)
    coverage = intersection / len(left_tokens)
    overlap = intersection / len(left_tokens | right_tokens)
    edit = SequenceMatcher(None, left, right).ratio()
    return 0.70 * coverage + 0.20 * overlap + 0.10 * edit


def best_category(
    query: str,
    candidates: list[sqlite3.Row],
) -> tuple[str | None, float]:
    """Choose a category by the strongest small cluster, not one odd row."""
    by_category: dict[str, list[float]] = {}
    for row in candidates:
        by_category.setdefault(row["category"], []).append(
            category_lexical_score(query, row["normalized_name"])
        )
    if not by_category:
        return None, 0.0
    category, scores = max(
        by_category.items(),
        key=lambda pair: (
            sum(sorted(pair[1], reverse=True)[:5]),
            max(pair[1]),
            pair[0],
        ),
    )
    return category, max(scores)


def _singular_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


class SQLiteNutritionResolver:
    """Read-only implementation of the RESOLVE ladder.

    Rungs 1–3 are exact alias, lexical fuzzy match, and category median.
    Rung 4 is an injected hypothesis provider. A miss remains unresolved.
    """

    def __init__(
        self,
        database: Path | str,
        *,
        fuzzy_threshold: float = 0.76,
        category_threshold: float = 0.42,
        hypothesis_provider: NutritionHypothesisProvider | None = None,
    ) -> None:
        self.database = Path(database)
        self.fuzzy_threshold = fuzzy_threshold
        self.category_threshold = category_threshold
        self.hypothesis_provider = hypothesis_provider
        if not self.database.is_file():
            raise FileNotFoundError(self.database)
        uri = f"file:{self.database.resolve()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self._foods = self.connection.execute(
            "SELECT * FROM foods ORDER BY normalized_name, fdc_id"
        ).fetchall()
        self._token_index: dict[str, list[sqlite3.Row]] = {}
        self._prefix_index: dict[str, list[sqlite3.Row]] = {}
        self._compact_index: dict[str, list[sqlite3.Row]] = {}
        for row in self._foods:
            normalized_name = row["normalized_name"]
            compact = normalized_name.replace(" ", "")
            self._compact_index.setdefault(compact, []).append(row)
            self._prefix_index.setdefault(compact[:2], []).append(row)
            for token in {_singular_token(value) for value in normalized_name.split()}:
                if len(token) >= 2:
                    self._token_index.setdefault(token, []).append(row)

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _profile(row: sqlite3.Row) -> NutritionProfile:
        profile = NutritionProfile(
            fdc_id=row["fdc_id"] if "fdc_id" in row.keys() else None,
            name=row["name"],
            category=row["category"],
            kcal_per_100g=row["kcal_per_100g"],
            protein_per_100g=row["protein_per_100g"],
            fat_per_100g=row["fat_per_100g"],
            carbs_per_100g=row["carbs_per_100g"],
            source_kcal_per_100g=(
                row["source_kcal_per_100g"]
                if "source_kcal_per_100g" in row.keys()
                else None
            ),
            data_type=row["data_type"] if "data_type" in row.keys() else "category_default",
            typical_portion_g=(
                row["typical_portion_g"]
                if "typical_portion_g" in row.keys()
                else None
            ),
        )
        if not profile.is_sane():
            raise ValueError(f"database contains an unsafe nutrition profile: {profile.name}")
        return profile

    @functools.lru_cache(maxsize=4096)
    def resolve(self, name: str) -> Resolution:
        normalized = normalize_name(name)
        if not normalized:
            return Resolution(name, "unresolved", 0.0, None, False)
        exact = self.connection.execute(
            """
            SELECT foods.*
            FROM aliases JOIN foods USING (fdc_id)
            WHERE aliases.normalized_alias = ?
            ORDER BY aliases.priority DESC, foods.fdc_id
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
        if exact:
            return Resolution(name, "exact_alias", 1.0, self._profile(exact), False)

        compact = normalized.replace(" ", "")
        compact_matches = self._compact_index.get(compact, [])
        if compact_matches:
            best_compact = compact_matches[0]
            return Resolution(
                name,
                "fuzzy",
                1.0,
                self._profile(best_compact),
                False,
            )
        candidate_by_id: dict[int, sqlite3.Row] = {}
        for token in {_singular_token(value) for value in normalized.split()}:
            for row in self._token_index.get(token, []):
                candidate_by_id[row["fdc_id"]] = row
        candidates = list(candidate_by_id.values())
        if not candidates:
            candidates = self._prefix_index.get(compact[:2], self._foods)
        best = max(
            candidates,
            key=lambda row: lexical_score(normalized, row["normalized_name"]),
            default=None,
        )
        score = lexical_score(normalized, best["normalized_name"]) if best else 0.0
        if best is not None and score >= self.fuzzy_threshold:
            return Resolution(name, "fuzzy", score, self._profile(best), False)

        category, category_score = best_category(normalized, candidates)
        if category_score < self.category_threshold:
            category = None
        if category:
            default = self.connection.execute(
                "SELECT * FROM category_defaults WHERE category = ?",
                (category,),
            ).fetchone()
            if default:
                return Resolution(
                    name,
                    "category_default",
                    category_score,
                    self._profile(default),
                    True,
                )

        if self.hypothesis_provider:
            hypothesis = self.hypothesis_provider.hypothesis(name, category)
            if hypothesis is not None:
                if not hypothesis.is_sane():
                    raise ValueError("hypothesis provider returned an unsafe profile")
                return Resolution(name, "hypothesis", max(score, category_score), hypothesis, True)
        return Resolution(name, "unresolved", max(score, category_score), None, True)
