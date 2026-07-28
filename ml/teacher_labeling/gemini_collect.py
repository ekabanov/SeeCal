"""Poll, validate, download, and reconcile a submitted Gemini batch."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .budget import BudgetLedger, BudgetPolicy, PricingCatalog, load_secret_env
from .cli import (
    DEFAULT_CONFIG,
    DEFAULT_LEDGER,
    DEFAULT_PRICING,
    DEFAULT_SECRET,
)
from .gemini_submit import BatchSubmissionError, _atomic_json, _job_state


TERMINAL_FAILURES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _field(payload: dict[str, Any], camel: str, snake: str) -> Any:
    return payload.get(camel, payload.get(snake))


def _response_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise BatchSubmissionError("response has no candidate")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text_parts = [part["text"] for part in parts if isinstance(part.get("text"), str)]
    if not text_parts:
        raise BatchSubmissionError("response candidate has no text")
    return "".join(text_parts)


def _validate_semantic(value: Any) -> None:
    if not isinstance(value, dict):
        raise BatchSubmissionError("semantic output must be an object")
    required = {
        "is_food",
        "visible_foods",
        "container",
        "mixed_dish",
        "occlusion",
        "abstain",
        "ambiguity_reason",
    }
    if set(value) != required:
        raise BatchSubmissionError("semantic output fields do not match schema")
    if not isinstance(value["visible_foods"], list):
        raise BatchSubmissionError("visible_foods must be an array")
    for food in value["visible_foods"]:
        if set(food) != {"name", "cooking_state", "visibility"}:
            raise BatchSubmissionError("visible food fields do not match schema")
        if not all(isinstance(item, str) for item in food.values()):
            raise BatchSubmissionError("visible food values must be strings")
    if not all(
        isinstance(value[field], bool)
        for field in ("is_food", "mixed_dish", "abstain")
    ):
        raise BatchSubmissionError("semantic boolean field has wrong type")
    if not all(
        isinstance(value[field], str)
        for field in ("container", "occlusion", "ambiguity_reason")
    ):
        raise BatchSubmissionError("semantic string field has wrong type")


def _validate_nutrition(value: Any) -> None:
    if not isinstance(value, dict):
        raise BatchSubmissionError("nutrition output must be an object")
    required = {"total_calories", "protein_g", "fat_g", "carbs_g"}
    if set(value) != required:
        raise BatchSubmissionError("nutrition output fields do not match schema")
    for field in required:
        number = value[field]
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or number < 0
        ):
            raise BatchSubmissionError(
                f"nutrition field must be a non-negative number: {field}"
            )


def parse_batch_results(
    data: bytes,
    *,
    expected_records: int,
    response_kind: str = "semantic",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    validators = {
        "semantic": _validate_semantic,
        "nutrition": _validate_nutrition,
    }
    if response_kind not in validators:
        raise BatchSubmissionError(f"unknown response kind: {response_kind}")
    parsed: list[dict[str, Any]] = []
    keys: set[str] = set()
    prompt_tokens = candidate_tokens = thought_tokens = 0
    lines = data.decode("utf-8").splitlines()
    if len(lines) != expected_records:
        raise BatchSubmissionError(
            f"expected {expected_records} result lines, found {len(lines)}"
        )
    for line_number, line in enumerate(lines, start=1):
        envelope = json.loads(line)
        key = envelope.get("key")
        if not isinstance(key, str) or not key or key in keys:
            raise BatchSubmissionError(
                f"missing/duplicate result key at line {line_number}"
            )
        keys.add(key)
        error = envelope.get("error")
        response = envelope.get("response")
        if error is not None:
            parsed.append({"key": key, "error": error})
            continue
        if not isinstance(response, dict):
            raise BatchSubmissionError(
                f"line {line_number} has neither response nor error"
            )
        value = json.loads(_response_text(response))
        validators[response_kind](value)
        usage = _field(response, "usageMetadata", "usage_metadata") or {}
        prompt_tokens += int(
            _field(usage, "promptTokenCount", "prompt_token_count") or 0
        )
        candidate_tokens += int(
            _field(usage, "candidatesTokenCount", "candidates_token_count") or 0
        )
        thought_tokens += int(
            _field(usage, "thoughtsTokenCount", "thoughts_token_count") or 0
        )
        parsed.append(
            {
                "key": key,
                response_kind: value,
                "usage": usage,
                "model_version": _field(
                    response,
                    "modelVersion",
                    "model_version",
                ),
            }
        )
    return parsed, {
        "input_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "thought_tokens": thought_tokens,
        "output_tokens": candidate_tokens + thought_tokens,
        "successful_records": sum(response_kind in item for item in parsed),
        "failed_records": sum("error" in item for item in parsed),
    }


def collect_batch(
    *,
    state_path: Path,
    ledger: BudgetLedger,
    pricing: PricingCatalog,
    client: Any,
) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["phase"] == "collected":
        return state
    if state["phase"] != "submitted":
        raise BatchSubmissionError(
            f"cannot collect state in phase {state['phase']!r}"
        )

    job = client.batches.get(name=state["job_name"])
    current_state = _job_state(job)
    state["job_state"] = current_state
    if current_state in TERMINAL_FAILURES:
        state["phase"] = "terminal_failure_unreconciled"
        error = getattr(job, "error", None)
        state["job_error"] = (
            error.model_dump(mode="json", by_alias=True, exclude_none=True)
            if hasattr(error, "model_dump")
            else str(error)
        )
        _atomic_json(state_path, state)
        return state
    if current_state != "JOB_STATE_SUCCEEDED":
        _atomic_json(state_path, state)
        return state

    destination = getattr(job, "dest", None)
    result_file = getattr(destination, "file_name", None)
    if not result_file:
        raise BatchSubmissionError("succeeded job has no result file")
    raw = client.files.download(file=result_file)
    response_kind = "semantic"
    if state.get("plan"):
        plan = json.loads(Path(state["plan"]).read_text(encoding="utf-8"))
        response_kind = plan.get("response_kind", "semantic")
    parsed, usage = parse_batch_results(
        raw,
        expected_records=int(state["records"]),
        response_kind=response_kind,
    )
    results_path = state_path.parent / (
        f"{state['model'].replace('/', '_')}-"
        f"{state['batch_index']:03d}-results.jsonl"
    )
    results_text = "".join(
        json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n"
        for item in parsed
    )
    results_path.write_text(results_text, encoding="utf-8")

    settled = pricing.cost(
        model=state["model"],
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
    )
    ledger.settle(
        request_id=state["request_id"],
        settled_usd=settled,
        usage={
            **usage,
            "result_file": result_file,
            "pricing_snapshot_hash": pricing.snapshot_hash,
        },
    )
    state.update(
        {
            "phase": "collected",
            "result_file_name": result_file,
            "results_path": str(results_path),
            "usage": usage,
            "settled_usd": str(settled),
        }
    )
    _atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--state", type=Path)
    target.add_argument(
        "--state-dir",
        type=Path,
        help="Collect every job-state JSON in this directory in filename order.",
    )
    parser.add_argument("--secret", type=Path, default=DEFAULT_SECRET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    secret_env = load_secret_env(args.secret)
    policy = BudgetPolicy.load(
        secret_env=secret_env,
        run_config_path=args.config,
    )
    pricing = PricingCatalog.load(args.pricing)
    ledger = BudgetLedger(args.ledger, policy)
    try:
        api_key = secret_env["GEMINI_API_KEY"]
    except KeyError as exc:
        raise BatchSubmissionError("GEMINI_API_KEY is missing") from exc

    from google import genai

    client = genai.Client(api_key=api_key)
    states: list[dict[str, Any]] = []
    try:
        if args.state_dir is not None:
            state_paths = sorted(
                path
                for path in args.state_dir.glob("*.json")
                if not path.name.endswith("-results.json")
            )
        else:
            state_paths = [args.state]
        for state_path in state_paths:
            states.append(
                collect_batch(
                    state_path=state_path,
                    ledger=ledger,
                    pricing=pricing,
                    client=client,
                )
            )
    finally:
        client.close()
    summary = {
        "jobs": len(states),
        "phases": dict(
            Counter(state["phase"] for state in states)
        ),
        "job_states": dict(
            Counter(state.get("job_state", "unknown") for state in states)
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(ledger.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
