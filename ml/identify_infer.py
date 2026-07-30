"""Run/evaluate the frozen new-schema IDENTIFY adapter.

This intentionally does not share the monolith's nutrition-metric path. Output
is paired by dish/image ID for ``score_harness.py audit``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from factored_pipeline.contract import (
    IDENTIFY_PROMPT,
    IDENTIFY_PROMPT_V1,
    ContractError,
    identification_to_shares,
    normalize_legacy_share_identification,
    validate_identification,
)


def _content_text(content) -> str:
    return content[0]["text"] if isinstance(content, list) else content


def _generate(model, processor, config, image: Path, *, contract: str) -> dict:
    prompt_text = IDENTIFY_PROMPT_V1 if contract == "legacy-shares" else IDENTIFY_PROMPT
    prompt = apply_chat_template(processor, config, prompt_text, num_images=1)
    if prompt.endswith("<think>\n"):
        prompt = prompt[: -len("<think>\n")]
    output = generate(
        model,
        processor,
        prompt=prompt,
        image=str(image),
        max_tokens=512,
        temperature=0.1,
        repetition_penalty=1.1,
        verbose=False,
    )
    text = output.text if hasattr(output, "text") else str(output)
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {"raw_output": text, "parse_error": True}
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {"raw_output": text, "parse_error": True}
    try:
        if contract == "legacy-shares":
            return normalize_legacy_share_identification(payload)
        return identification_to_shares(validate_identification(payload))
    except ContractError as error:
        return {
            "raw_output": text,
            "schema_error": str(error),
        }


def evaluate(args: argparse.Namespace) -> dict:
    records = [
        json.loads(line)
        for line in args.test_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    load_kwargs = {}
    if args.adapter_path is not None:
        load_kwargs["adapter_path"] = str(args.adapter_path)
    model, processor = load(
        str(args.model_path.expanduser()),
        **load_kwargs,
    )
    config = load_config(str(args.model_path.expanduser()))
    paired = []
    parse_failures = schema_failures = repaired_predictions = 0
    refusal_correct = refusal_missed = false_refusals = 0
    started = time.perf_counter()
    for index, record in enumerate(records):
        completion_ground_truth = json.loads(
            _content_text(record["messages"][-1]["content"])
        )
        ground_truth = record.get(
            "evaluation_ground_truth",
            completion_ground_truth,
        )
        image = (Path(__file__).parent / record["images"][0]).resolve()
        prediction = _generate(
            model,
            processor,
            config,
            image,
            contract=args.contract,
        )
        record_id = record.get("id") or (
            image.parent.name if image.parent.name.startswith("dish_") else image.stem
        )
        row = {
            "index": index,
            "id": record_id,
            "group_id": record.get("group_id") or record_id,
            "image": str(image),
            "ground_truth": ground_truth,
            "prediction": prediction,
        }
        if prediction.get("parse_error"):
            parse_failures += 1
            row["status"] = "parse_error"
        elif prediction.get("schema_error"):
            schema_failures += 1
            row["status"] = "schema_error"
        else:
            repair = prediction.pop("_evaluation_repair", None)
            if repair is not None:
                repaired_predictions += 1
                row["evaluation_repair"] = repair
            truth_refusal = bool(ground_truth["not_food"])
            predicted_refusal = bool(prediction["not_food"])
            if truth_refusal and predicted_refusal:
                refusal_correct += 1
                row["status"] = "refusal_correct"
            elif truth_refusal:
                refusal_missed += 1
                row["status"] = "refusal_missed"
            elif predicted_refusal:
                false_refusals += 1
                row["status"] = "false_refusal"
            else:
                row["status"] = "ok"
        paired.append(row)
        print(f"[{index + 1}/{len(records)}] {row['id']} — {row['status']}", flush=True)
    return {
        "schema_version": 1,
        "identification_contract": args.contract,
        "samples": len(records),
        "evaluation_seconds": time.perf_counter() - started,
        "parse_failures": parse_failures,
        "schema_failures": schema_failures,
        "repaired_predictions": repaired_predictions,
        "repair_rate": repaired_predictions / len(records) if records else None,
        "post_repair_rejection_rate": (
            (parse_failures + schema_failures) / len(records) if records else None
        ),
        "refusals_correct": refusal_correct,
        "refusals_missed": refusal_missed,
        "false_refusals": false_refusals,
        "paired_results": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="LoRA adapter directory; omit for an untuned base-model baseline.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("~/models/mlx-community/Qwen3.5-4B-MLX-4bit"),
    )
    parser.add_argument(
        "--contract",
        choices=("portion-units", "legacy-shares"),
        default="portion-units",
        help=(
            "Use relative units, or normalize an existing percentage adapter's "
            "positive share values to 100."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "paired_results"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
