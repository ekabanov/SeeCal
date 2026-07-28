"""Score teacher outputs against FRB source labels and each other.

FRB labels are a fine Swiss-food taxonomy while teacher names are open text.
The lexical score is therefore a screening proxy, not a claim of semantic
ground truth.  It normalizes spelling/preparation variants, allows containment
(``bread`` vs ``bread wholemeal``), and reports the assumptions in the output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from typing import Any, Iterable
import unicodedata

from .gemini_batch import _request_key


STOP_WORDS = {
    "addition",
    "and",
    "at",
    "baked",
    "boiled",
    "cooked",
    "cooking",
    "fidm",
    "fried",
    "full",
    "made",
    "natural",
    "of",
    "only",
    "or",
    "prepared",
    "raw",
    "roasted",
    "salt",
    "small",
    "steamed",
    "stewed",
    "the",
    "with",
    "without",
}
GENERIC_WORDS = {
    "bread",
    "cheese",
    "drink",
    "food",
    "meat",
    "salad",
    "sauce",
    "soup",
    "spread",
    "vegetable",
}
ALIASES = {
    "spatzle": "spaetzle",
    "spätzle": "spaetzle",
    "yaourt": "yogurt",
    "yahourt": "yogurt",
    "yoghourt": "yogurt",
    "yoghurt": "yogurt",
    "capsicum": "pepper",
    "fries": "chip",
}


def _tokenize(value: str) -> tuple[str, ...]:
    ascii_value = unicodedata.normalize("NFKD", value.casefold()).encode(
        "ascii",
        "ignore",
    ).decode("ascii")
    words = re.findall(r"[a-z0-9]+", ascii_value)
    normalized: list[str] = []
    for word in words:
        word = ALIASES.get(word, word)
        if word in STOP_WORDS:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        normalized.append(word)
    return tuple(sorted(set(normalized)))


def labels_match(first: str, second: str) -> bool:
    left = set(_tokenize(first))
    right = set(_tokenize(second))
    if not left or not right:
        return False
    if left == right:
        return True
    overlap = left & right
    if not overlap:
        joined_left = " ".join(sorted(left))
        joined_right = " ".join(sorted(right))
        return SequenceMatcher(None, joined_left, joined_right).ratio() >= 0.84
    non_generic = overlap - GENERIC_WORDS
    if left <= right or right <= left:
        return bool(non_generic) or min(len(left), len(right)) == 1
    union = left | right
    return bool(non_generic) and len(overlap) / len(union) >= 0.34


def _matched_counts(
    predictions: Iterable[str],
    targets: Iterable[str],
) -> tuple[int, int, int]:
    predicted = list(predictions)
    target = list(targets)
    matched_predictions = sum(
        any(labels_match(item, source) for source in target)
        for item in predicted
    )
    matched_targets = sum(
        any(labels_match(item, source) for item in predicted)
        for source in target
    )
    return matched_predictions, matched_targets, len(predicted)


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {row["key"]: row for row in rows}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_model(
    manifest_records: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prediction_matches = target_matches = predictions = targets = 0
    per_image_f1: list[float] = []
    output_food_counts: Counter[str] = Counter()
    abstentions = not_food = errors = 0
    for record in manifest_records:
        key = _request_key(record)
        result = results.get(key)
        if result is None or "error" in result:
            errors += 1
            continue
        semantic = result["semantic"]
        abstentions += bool(semantic["abstain"])
        not_food += not semantic["is_food"]
        predicted = [
            item["name"].strip().casefold()
            for item in semantic["visible_foods"]
            if item["name"].strip()
        ]
        source = [item["name"] for item in record["source_labels"]]
        output_food_counts.update(predicted)
        matched_pred, matched_target, predicted_count = _matched_counts(
            predicted,
            source,
        )
        prediction_matches += matched_pred
        target_matches += matched_target
        predictions += predicted_count
        targets += len(source)
        precision = _ratio(matched_pred, predicted_count)
        recall = _ratio(matched_target, len(source))
        per_image_f1.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    precision = _ratio(prediction_matches, predictions)
    recall = _ratio(target_matches, targets)
    return {
        "records": len(manifest_records),
        "errors": errors,
        "abstentions": abstentions,
        "not_food_outputs": not_food,
        "predicted_food_labels": predictions,
        "source_food_labels": targets,
        "matched_predictions": prediction_matches,
        "matched_source_labels": target_matches,
        "lexical_precision": round(precision, 6),
        "lexical_recall": round(recall, 6),
        "lexical_f1": round(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            6,
        ),
        "mean_per_image_f1": round(
            sum(per_image_f1) / len(per_image_f1),
            6,
        ),
        "lexical_hallucination_rate": round(1 - precision, 6),
        "unique_predicted_names": len(output_food_counts),
    }


def cross_model_agreement(
    manifest_records: list[dict[str, Any]],
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first_matches = second_matches = first_total = second_total = 0
    mixed_agreement = container_agreement = 0
    for record in manifest_records:
        key = _request_key(record)
        left = first[key]["semantic"]
        right = second[key]["semantic"]
        left_names = [item["name"] for item in left["visible_foods"]]
        right_names = [item["name"] for item in right["visible_foods"]]
        matched_left, matched_right, left_count = _matched_counts(
            left_names,
            right_names,
        )
        first_matches += matched_left
        second_matches += matched_right
        first_total += left_count
        second_total += len(right_names)
        mixed_agreement += left["mixed_dish"] == right["mixed_dish"]
        container_agreement += left["container"] == right["container"]
    left_precision = _ratio(first_matches, first_total)
    right_recall = _ratio(second_matches, second_total)
    return {
        "symmetric_label_f1": round(
            2
            * left_precision
            * right_recall
            / (left_precision + right_recall)
            if left_precision + right_recall
            else 0.0,
            6,
        ),
        "mixed_dish_agreement": round(
            mixed_agreement / len(manifest_records),
            6,
        ),
        "container_agreement": round(
            container_agreement / len(manifest_records),
            6,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--lite", type=Path, required=True)
    parser.add_argument("--flash", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    lite = _load_results(args.lite)
    flash = _load_results(args.flash)
    report = {
        "schema_version": 1,
        "records": len(manifest),
        "metric_warning": (
            "Lexical source-label agreement is a screening proxy across a "
            "fine taxonomy and open text; use contact-sheet audit before "
            "accepting enrichments."
        ),
        "flash_lite": evaluate_model(manifest, lite),
        "flash": evaluate_model(manifest, flash),
        "cross_model": cross_model_agreement(manifest, lite, flash),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
