"""Crash-conscious submission of one prepared Gemini batch.

Batch creation is not idempotent.  This command therefore:

1. validates the prepared request hash;
2. creates a budget reservation;
3. uploads the JSONL input (not billable);
4. persists the uploaded file resource;
5. makes exactly one batch-create attempt and persists its job name.

If the create call raises after it is attempted, the reservation remains open
and the state is marked ``create_ambiguous``.  The command will never retry
that state automatically because doing so could create a duplicate paid job.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .budget import BudgetLedger, BudgetPolicy, PricingCatalog, load_secret_env
from .cli import (
    DEFAULT_CONFIG,
    DEFAULT_LEDGER,
    DEFAULT_PRICING,
    DEFAULT_SECRET,
)


class BatchSubmissionError(RuntimeError):
    """Prepared input or submission state is unsafe."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _job_state(job: Any) -> str | None:
    state = getattr(job, "state", None)
    return getattr(state, "value", state)


def submit_prepared_batch(
    *,
    plan_path: Path,
    model: str,
    batch_index: int,
    ledger: BudgetLedger,
    pricing: PricingCatalog,
    client: Any,
    state_dir: Path | None = None,
) -> Path:
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("paid_calls_submitted") is not False:
        raise BatchSubmissionError("prepared plan is not in offline state")
    if model not in plan["models"]:
        raise BatchSubmissionError(f"model is absent from plan: {model}")
    price = pricing.prices.get(model)
    if price is None or price.provider != "google":
        raise BatchSubmissionError(f"model is not pinned to Google pricing: {model}")
    try:
        batch = next(
            item for item in plan["batches"]
            if item["batch_index"] == batch_index
        )
    except StopIteration as exc:
        raise BatchSubmissionError(
            f"batch index is absent: {batch_index}"
        ) from exc

    request_path = plan_path.parent / batch["request_file"]
    request_bytes = request_path.read_bytes()
    actual_hash = hashlib.sha256(request_bytes).hexdigest()
    if actual_hash != batch["request_sha256"]:
        raise BatchSubmissionError("prepared request hash mismatch")

    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    safe_model = model.replace("/", "_")
    request_id = (
        f"{plan['run_id']}:{safe_model}:{plan_hash[:16]}:"
        f"{batch_index:03d}"
    )
    state_root = state_dir or plan_path.parent / "jobs"
    state_path = state_root / f"{safe_model}-{batch_index:03d}.json"
    if state_path.exists():
        raise BatchSubmissionError(
            f"state already exists; refusing duplicate submission: {state_path}"
        )

    total_records = int(plan["models"][model]["records"])
    batch_records = int(batch["records"])
    model_estimate = Decimal(
        plan["models"][model]["buffered_estimate_usd"]
    )
    reservation = (
        model_estimate * Decimal(batch_records) / Decimal(total_records)
    )
    ledger.reserve(
        request_id=request_id,
        provider="google",
        model=model,
        reserved_usd=reservation,
        metadata={
            "plan_sha256": plan_hash,
            "request_sha256": actual_hash,
            "batch_index": batch_index,
            "records": batch_records,
            "pricing_snapshot_hash": pricing.snapshot_hash,
        },
    )
    state: dict[str, Any] = {
        "schema_version": 1,
        "phase": "reserved",
        "request_id": request_id,
        "model": model,
        "batch_index": batch_index,
        "records": batch_records,
        "reserved_usd": str(reservation),
        "plan": str(plan_path),
        "plan_sha256": plan_hash,
        "request_file": str(request_path),
        "request_sha256": actual_hash,
    }
    _atomic_json(state_path, state)

    display_name = (
        f"{plan['run_id']}-{safe_model}-{plan_hash[:8]}-"
        f"{batch_index:03d}"
    )
    try:
        uploaded = client.files.upload(
            file=request_path,
            config={
                "display_name": display_name,
                "mime_type": "jsonl",
            },
        )
    except Exception:
        ledger.cancel(
            request_id=request_id,
            reason="input upload failed before batch-create attempt",
        )
        state["phase"] = "upload_failed"
        _atomic_json(state_path, state)
        raise

    state.update(
        {
            "phase": "uploaded",
            "uploaded_file_name": uploaded.name,
            "uploaded_file_uri": getattr(uploaded, "uri", None),
        }
    )
    _atomic_json(state_path, state)

    state["phase"] = "create_ambiguous"
    _atomic_json(state_path, state)
    try:
        job = client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": display_name},
        )
    except Exception:
        # Do not cancel or retry: server acceptance is unknown.
        raise

    state.update(
        {
            "phase": "submitted",
            "job_name": job.name,
            "job_state": _job_state(job),
        }
    )
    _atomic_json(state_path, state)
    return state_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument(
        "--all-batches",
        action="store_true",
        help="Submit each prepared batch once, stopping on the first failure.",
    )
    parser.add_argument("--secret", type=Path, default=DEFAULT_SECRET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--confirm-paid-submit",
        action="store_true",
        help="Required acknowledgement that batch creation can incur charges.",
    )
    args = parser.parse_args()
    if not args.confirm_paid_submit:
        raise BatchSubmissionError("--confirm-paid-submit is required")

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
    state_paths: list[Path] = []
    try:
        if args.all_batches:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            batch_indices = [
                int(item["batch_index"]) for item in plan["batches"]
            ]
        else:
            batch_indices = [args.batch_index]
        for batch_index in batch_indices:
            state_paths.append(
                submit_prepared_batch(
                    plan_path=args.plan,
                    model=args.model,
                    batch_index=batch_index,
                    ledger=ledger,
                    pricing=pricing,
                    client=client,
                )
            )
    finally:
        client.close()
    states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in state_paths
    ]
    print(json.dumps(states, indent=2, sort_keys=True))
    print(json.dumps(ledger.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
