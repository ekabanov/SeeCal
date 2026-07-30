"""Apply deterministic visible-component filtering to saved IDENTIFY outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factored_pipeline.visible_labels import filter_visible_prediction


def postprocess(input_path: Path, output_path: Path) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    changed = 0
    removed = 0
    for row in payload["paired_results"]:
        prediction = row["prediction"]
        if row["status"] not in {"ok", "refusal_correct"}:
            continue
        filtered = filter_visible_prediction(prediction)
        before_names = {item["name"] for item in prediction.get("items", [])}
        after_names = {item["name"] for item in filtered.get("items", [])}
        if filtered != prediction:
            changed += 1
            removed += len(before_names - after_names)
            row["prediction"] = filtered
            row["visible_filter"] = {
                "removed_names": sorted(before_names - after_names),
            }
    payload["visible_filter"] = {
        "policy": "visible_components_v2",
        "changed_predictions": changed,
        "removed_items": removed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload["visible_filter"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(postprocess(args.input, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
