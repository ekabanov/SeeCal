"""Frozen, resolver-independent name taxonomy for IDENTIFY evaluation.

Runtime RESOLVE is a component under test and must not define HMR/IDR.  This
module intentionally owns a versioned copy of the lexical rules used by the
evaluation harness.  Changing runtime resolver thresholds or candidate
selection therefore cannot change a fixed prediction set's Tier-1 score.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata


def normalize_eval_name_v1(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _singular_token_v1(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def lexical_score_v1(lhs: str, rhs: str) -> float:
    left = normalize_eval_name_v1(lhs)
    right = normalize_eval_name_v1(rhs)
    if not left or not right:
        return 0.0
    left_tokens = {_singular_token_v1(token) for token in left.split()}
    right_tokens = {_singular_token_v1(token) for token in right.split()}
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


@dataclass(frozen=True)
class TaxonomyResolution:
    fdc_id: int | None
    category: str | None
    family: str | None = None


class EvaluationTaxonomy:
    """Read a committed taxonomy snapshot; never consult runtime RESOLVE."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported evaluation taxonomy: {self.path}")
        self.version = str(payload["taxonomy_version"])
        self.soft_threshold = float(payload["matching"]["soft_threshold"])
        self.hard_threshold = float(payload["matching"]["hard_threshold"])
        self.entries = {
            str(name): TaxonomyResolution(
                fdc_id=(
                    int(entry["fdc_id"])
                    if entry.get("fdc_id") is not None
                    else None
                ),
                category=(
                    str(entry["category"])
                    if entry.get("category") is not None
                    else None
                ),
                family=(
                    str(entry["family"])
                    if entry.get("family") is not None
                    else None
                ),
            )
            for name, entry in payload["entries"].items()
        }

    def resolve(self, name: str) -> TaxonomyResolution:
        return self.entries.get(
            normalize_eval_name_v1(name),
            TaxonomyResolution(None, None, None),
        )

    def match_kind(self, truth_name: str, prediction_name: str) -> tuple[str, float]:
        if normalize_eval_name_v1(truth_name) == normalize_eval_name_v1(
            prediction_name
        ):
            return "exact", 1.0
        truth = self.resolve(truth_name)
        prediction = self.resolve(prediction_name)
        if (
            truth.fdc_id is not None
            and prediction.fdc_id is not None
            and truth.fdc_id == prediction.fdc_id
        ):
            return "exact", 1.0
        if truth.family and prediction.family and truth.family == prediction.family:
            return "soft", 1.0
        if (
            not truth.family
            and not prediction.family
            and truth.category
            and prediction.category
            and truth.category == prediction.category
        ):
            return "soft", 1.0
        score = lexical_score_v1(truth_name, prediction_name)
        if score >= self.soft_threshold:
            return "exact", score
        if score >= self.hard_threshold:
            return "soft", score
        return "hard", score
