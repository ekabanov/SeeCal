"""Generate a reviewable source-backed mapping for training vocabulary misses.

This is a curation aid, not a runtime resolver.  Candidate ranking combines
lexical relevance with Nutrition5K's measured per-100g macro profile when one
exists.  The resulting TSV must be reviewed and versioned before passing it to
``make_alias_table.py --reviewed-aliases``.
"""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

from factored_pipeline.contract import normalize_name
from factored_pipeline.resolver import SQLiteNutritionResolver


QUERY_REWRITES = {
    "artic char": "arctic char fish",
    "french beans": "green beans",
    "hamburg": "hamburger",
    "hanamaki baozi": "steamed filled bun",
    "milkshake": "milk shake",
    "rape": "rapeseed",
    "shellfish": "shrimp shellfish",
    "tatsoi": "mustard greens",
    "vinaigrette": "vinaigrette salad dressing",
    "wonton dumplings": "wonton dumpling",
}
SKIP_NAMES = {
    "cheese butter",
    "chicken duck",
    "deprecated",
    "fried meat",
    "garden salad",
    "greek salad",
    "juice",
    "other ingredients",
    "pasta salad",
    "plate only",
    "salad",
    "sauce",
    "soup",
}
MANUAL_FDC_OVERRIDES = {
    "apple": 168204,  # Apples, raw, gala, with skin.
    "banana with peel": 173944,  # Edible profile: bananas, raw.
    "bamboo shoots": 169210,  # Bamboo shoots, raw.
    "bean sprouts": 169957,  # Mung beans, sprouted, raw.
    "berries": 171711,  # Blueberries, raw; generic berry representative.
    "bread": 325871,  # Bread, white, commercially prepared.
    "brussels sprouts": 170383,  # Brussels sprouts, raw.
    "cake": 172710,  # Cake, yellow, recipe, unfrosted.
    "cherry tomatoes": 170457,  # Tomatoes, red, ripe, raw.
    "chicken": 171477,  # Chicken breast, meat only, roasted.
    "chilaquiles": 170292,  # Nachos with beans, beef, cheese, tomato.
    "egg tart": 172783,  # Pie, egg custard.
    "grilled chicken": 171534,  # Chicken breast, skinless, grilled.
    "hanamaki baozi": 172793,  # Plain prepared dinner roll.
    "ice cream": 167575,  # Vanilla ice cream.
    "mustard greens": 169256,  # Mustard greens, raw.
    "pie": 175012,  # Apple pie, prepared from recipe.
    "pork": 167905,  # Pork tenderloin, lean and fat, roasted.
    "potato": 170440,  # Potato, boiled without skin or salt.
    "rape": 172336,  # FoodSeg's rapeseed class: canola oil.
    "rice": 168878,  # White long-grain rice, cooked.
    "rice noodles": 168914,  # Rice noodles, cooked.
    "sausage": 174579,  # Pork and beef sausage, cooked.
    "shellfish": 175180,  # Shrimp, cooked; generic shellfish representative.
    "wheat berry": 168889,  # Wheat, hard red spring.
    "wild rice": 168897,  # Wild rice, cooked.
}


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: str) -> set[str]:
    return {_singular(token) for token in normalize_name(value).split()}


def _lexical(query: str, candidate: str) -> float:
    left, right = normalize_name(query), normalize_name(candidate)
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    coverage = intersection / len(left_tokens)
    overlap = intersection / len(left_tokens | right_tokens)
    edit = SequenceMatcher(None, left, right).ratio()
    return 0.70 * coverage + 0.20 * overlap + 0.10 * edit


def _nutrition5k_profiles(path: Path) -> dict[str, tuple[float, float, float, float]]:
    profiles = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            profiles[normalize_name(row["ingr_name"])] = (
                100 * float(row["cal/g"]),
                100 * float(row["protein(g)"]),
                100 * float(row["fat(g)"]),
                100 * float(row["carb(g)"]),
            )
    return profiles


def _vocabulary(paths: list[Path]) -> dict[str, dict[str, Any]]:
    vocabulary: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            content = row["messages"][-1]["content"]
            text = content[0]["text"] if isinstance(content, list) else content
            payload = json.loads(text)
            exact_weights = {
                normalize_name(str(item["name"])): float(
                    item.get("estimated_grams", item.get("share_pct", 0))
                )
                for item in (row.get("evaluation_ground_truth") or {}).get("items", [])
            }
            for item in payload.get("items", []):
                name = str(item["name"]).strip()
                normalized = normalize_name(name)
                entry = vocabulary.setdefault(
                    normalized,
                    {"name": name, "occurrences": 0, "weight": 0.0},
                )
                entry["occurrences"] += 1
                entry["weight"] += exact_weights.get(
                    normalized,
                    float(item.get("share_pct", item.get("portion_units", 0))),
                )
    return vocabulary


def _nutrient_similarity(
    target: tuple[float, float, float, float] | None,
    candidate: sqlite3.Row,
) -> float:
    if target is None:
        return 0.0
    observed = (
        float(candidate["kcal_per_100g"]),
        float(candidate["protein_per_100g"]),
        float(candidate["fat_per_100g"]),
        float(candidate["carbs_per_100g"]),
    )
    scales = (220.0, 24.0, 24.0, 35.0)
    distance = sum(
        abs(left - right) / scale
        for left, right, scale in zip(target, observed, scales)
    ) / len(scales)
    return math.exp(-distance)


def build_review(
    *,
    database: Path,
    jsonl_paths: list[Path],
    nutrition5k: Path,
) -> list[dict[str, Any]]:
    vocabulary = _vocabulary(jsonl_paths)
    profiles = _nutrition5k_profiles(nutrition5k)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    foods = connection.execute("SELECT * FROM foods").fetchall()
    connection.close()
    # Match make_alias_table.py's conservative promotion threshold so every
    # sub-threshold training name appears in the reviewed artifact.
    resolver = SQLiteNutritionResolver(database, fuzzy_threshold=0.84)
    rows = []
    try:
        for normalized, observation in sorted(
            vocabulary.items(),
            key=lambda pair: (-pair[1]["weight"], pair[0]),
        ):
            result = resolver.resolve(observation["name"])
            if result.rung in {"exact_alias", "fuzzy"} or normalized in SKIP_NAMES:
                continue
            override_id = MANUAL_FDC_OVERRIDES.get(normalized)
            if override_id is not None:
                best = next(food for food in foods if int(food["fdc_id"]) == override_id)
                rows.append(
                    {
                        "name": observation["name"],
                        "fdc_id": int(best["fdc_id"]),
                        "fdc_name": best["name"],
                        "category": best["category"],
                        "gram_weight": round(observation["weight"], 3),
                        "occurrences": observation["occurrences"],
                        "score": 1.0,
                        "lexical_score": round(
                            _lexical(observation["name"], best["normalized_name"]),
                            6,
                        ),
                        "nutrition_score": round(
                            _nutrient_similarity(profiles.get(normalized), best),
                            6,
                        ),
                        "review_note": "manual_source_profile_override",
                    }
                )
                continue
            query = QUERY_REWRITES.get(normalized, observation["name"])
            target = profiles.get(normalized)
            ranked = []
            for food in foods:
                lexical = _lexical(query, food["normalized_name"])
                if lexical < 0.38:
                    continue
                nutrition = _nutrient_similarity(target, food)
                simplicity = 1 / (1 + max(0, len(_tokens(food["normalized_name"])) - len(_tokens(query))))
                score = (
                    0.55 * lexical + 0.40 * nutrition + 0.05 * simplicity
                    if target is not None
                    else 0.90 * lexical + 0.10 * simplicity
                )
                ranked.append((score, lexical, nutrition, food))
            if not ranked:
                continue
            score, lexical, nutrition, best = max(
                ranked,
                key=lambda value: (value[0], value[1], -len(value[3]["name"])),
            )
            rows.append(
                {
                    "name": observation["name"],
                    "fdc_id": int(best["fdc_id"]),
                    "fdc_name": best["name"],
                    "category": best["category"],
                    "gram_weight": round(observation["weight"], 3),
                    "occurrences": observation["occurrences"],
                    "score": round(score, 6),
                    "lexical_score": round(lexical, 6),
                    "nutrition_score": round(nutrition, 6),
                    "review_note": (
                        "query_rewrite:" + query
                        if query != observation["name"]
                        else "lexical+measured_profile"
                        if target is not None
                        else "lexical"
                    ),
                }
            )
    finally:
        resolver.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, action="append", required=True)
    parser.add_argument("--nutrition5k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_review(
        database=args.database,
        jsonl_paths=args.jsonl,
        nutrition5k=args.nutrition5k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["name", "fdc_id"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"review_candidates": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
