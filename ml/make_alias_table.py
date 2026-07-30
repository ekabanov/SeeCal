"""Map derived training vocabularies into the pruned nutrition database.

Exact/fuzzy hits are inserted into the SQLite ``aliases`` table. Misses are
written to a review TSV and remain visible; no category or zero-valued profile
is silently promoted to an exact alias.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3

from factored_pipeline.contract import normalize_name
from factored_pipeline.resolver import SQLiteNutritionResolver


def nutrition5k_names(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["ingr_name"].strip()
            for row in csv.DictReader(handle)
            if row.get("ingr_name", "").strip()
        }


def jsonl_names(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            payload = row
            if "messages" in row:
                content = row["messages"][-1]["content"]
                text = content[0]["text"] if isinstance(content, list) else content
                payload = json.loads(text)
            for item in payload.get("items", []):
                name = str(item.get("name", "")).strip()
                if name:
                    names.add(name)
            for label in row.get("source_labels", []):
                name = str(label.get("name", "")).replace("-", " ").strip()
                if name:
                    names.add(name)
    return names


def reviewed_alias_names(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["name"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("name", "").strip()
        }


def build_aliases(
    *,
    database: Path,
    names: set[str],
    misses_path: Path,
    accept_fuzzy: float,
    reviewed_aliases_path: Path | None = None,
    reviewed_source: str = "reviewed_training_v1",
) -> dict[str, int]:
    resolver = SQLiteNutritionResolver(database, fuzzy_threshold=accept_fuzzy)
    accepted: list[tuple[str, int, int, str]] = []
    try:
        for name in sorted(names, key=normalize_name):
            result = resolver.resolve(name)
            if result.profile and result.rung == "fuzzy":
                accepted.append(
                    (
                        normalize_name(name),
                        int(result.profile.fdc_id),
                        100,
                        "derived_training_vocabulary",
                    )
                )
    finally:
        resolver.close()

    reviewed: list[tuple[str, int, int, str]] = []
    if reviewed_aliases_path is not None:
        with reviewed_aliases_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                reviewed.append(
                    (
                        normalize_name(row["name"]),
                        int(row["fdc_id"]),
                        200,
                        reviewed_source,
                    )
                )

    connection = sqlite3.connect(database)
    try:
        known_fdc_ids = {
            int(row[0]) for row in connection.execute("SELECT fdc_id FROM foods")
        }
        unknown = sorted(
            {fdc_id for _, fdc_id, _, _ in reviewed} - known_fdc_ids
        )
        if unknown:
            raise ValueError(f"reviewed aliases reference unknown FDC IDs: {unknown}")
        connection.executemany(
            "INSERT OR REPLACE INTO aliases VALUES (?,?,?,?)",
            accepted + reviewed,
        )
        connection.executemany(
            "INSERT OR REPLACE INTO metadata VALUES (?,?)",
            [
                ("derived_training_aliases", str(len(accepted))),
                (f"reviewed_aliases:{reviewed_source}", str(len(reviewed))),
            ],
        )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    resolver = SQLiteNutritionResolver(database, fuzzy_threshold=accept_fuzzy)
    try:
        resolved = {name: resolver.resolve(name) for name in names}
    finally:
        resolver.close()
    rung1 = sum(result.rung == "exact_alias" for result in resolved.values())
    rung12 = sum(
        result.rung in {"exact_alias", "fuzzy"} for result in resolved.values()
    )
    misses = [
        (
            name,
            result.profile.name if result.profile else "",
            result.score,
            result.rung,
        )
        for name, result in sorted(resolved.items(), key=lambda pair: normalize_name(pair[0]))
        if result.rung not in {"exact_alias", "fuzzy"}
    ]
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?,?)",
            ("derived_training_alias_misses", str(len(misses))),
        )
        connection.commit()
    finally:
        connection.close()
    misses_path.parent.mkdir(parents=True, exist_ok=True)
    with misses_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("name", "candidate", "score", "rung"))
        writer.writerows(misses)
    return {
        "names": len(names),
        "accepted": len(accepted),
        "reviewed": len(reviewed),
        "misses": len(misses),
        "rung1": rung1,
        "rung12": rung12,
        "rung1_pct": round(100 * rung1 / len(names), 3),
        "rung12_pct": round(100 * rung12 / len(names), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--nutrition5k", type=Path, action="append", default=[])
    parser.add_argument("--jsonl", type=Path, action="append", default=[])
    parser.add_argument("--misses", type=Path, required=True)
    parser.add_argument(
        "--reviewed-aliases",
        type=Path,
        help="Reviewed TSV with name and fdc_id columns; training vocabulary only.",
    )
    parser.add_argument(
        "--reviewed-source",
        default="reviewed_training_v1",
        help="Provenance label stored beside reviewed aliases.",
    )
    parser.add_argument("--accept-fuzzy", type=float, default=0.84)
    args = parser.parse_args()
    names: set[str] = set()
    for path in args.nutrition5k:
        names |= nutrition5k_names(path)
    for path in args.jsonl:
        names |= jsonl_names(path)
    if not names and args.reviewed_aliases:
        names = reviewed_alias_names(args.reviewed_aliases)
    if not names:
        parser.error("at least one vocabulary source is required")
    result = build_aliases(
        database=args.database,
        names=names,
        misses_path=args.misses,
        accept_fuzzy=args.accept_fuzzy,
        reviewed_aliases_path=args.reviewed_aliases,
        reviewed_source=args.reviewed_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
