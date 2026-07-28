from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from teacher_labeling.budget import (
    BudgetLedger,
    BudgetPolicy,
    ModelPrice,
    PricingCatalog,
)
from teacher_labeling.gemini_submit import (
    BatchSubmissionError,
    submit_prepared_batch,
)


class _Files:
    def upload(self, *, file, config):
        assert Path(file).is_file()
        assert config["mime_type"] == "jsonl"
        return SimpleNamespace(name="files/input-1", uri="file://input-1")


class _Batches:
    def __init__(self):
        self.calls = 0

    def create(self, *, model, src, config):
        self.calls += 1
        assert src == "files/input-1"
        return SimpleNamespace(
            name="batches/job-1",
            state=SimpleNamespace(value="JOB_STATE_QUEUED"),
        )


class _Client:
    def __init__(self):
        self.files = _Files()
        self.batches = _Batches()


def _policy() -> BudgetPolicy:
    return BudgetPolicy(
        run_id="test-run",
        enabled_providers=frozenset({"google"}),
        provider_caps={"google": Decimal("5"), "openai": Decimal("0")},
        automatic_total_cap=Decimal("4"),
        authorized_total_cap=Decimal("5"),
        usage_buffer=Decimal("0.20"),
    )


def _pricing() -> PricingCatalog:
    return PricingCatalog(
        prices={
            "gemini-test": ModelPrice(
                provider="google",
                mode="batch",
                input_per_million_usd=Decimal("1"),
                output_per_million_usd=Decimal("1"),
            )
        },
        snapshot_hash="pricing",
        source_url="https://example.test",
        retrieved_at="2026-07-28",
    )


def _plan(tmp_path: Path) -> Path:
    request = tmp_path / "requests-001.jsonl"
    request.write_text('{"request":{}}\n', encoding="utf-8")
    plan = {
        "paid_calls_submitted": False,
        "run_id": "test-run",
        "batches": [
            {
                "batch_index": 1,
                "records": 2,
                "request_file": request.name,
                "request_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
            }
        ],
        "models": {
            "gemini-test": {
                "records": 2,
                "buffered_estimate_usd": "1.25",
            }
        },
    }
    path = tmp_path / "batch-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_submit_reserves_once_and_persists_job_state(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.jsonl", _policy())
    client = _Client()
    plan = _plan(tmp_path)

    state_path = submit_prepared_batch(
        plan_path=plan,
        model="gemini-test",
        batch_index=1,
        ledger=ledger,
        pricing=_pricing(),
        client=client,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["phase"] == "submitted"
    assert state["job_name"] == "batches/job-1"
    assert client.batches.calls == 1
    assert ledger.summary()["reserved_usd"] == "1.250000"

    with pytest.raises(BatchSubmissionError, match="state already exists"):
        submit_prepared_batch(
            plan_path=plan,
            model="gemini-test",
            batch_index=1,
            ledger=ledger,
            pricing=_pricing(),
            client=client,
        )
    assert client.batches.calls == 1
