from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

from teacher_labeling.budget import (
    BudgetLedger,
    BudgetPolicy,
    ModelPrice,
    PricingCatalog,
)
from teacher_labeling.gemini_collect import collect_batch


def _semantic() -> dict:
    return {
        "is_food": True,
        "visible_foods": [
            {
                "name": "rice",
                "cooking_state": "boiled",
                "visibility": "clear",
            }
        ],
        "container": "plate",
        "mixed_dish": False,
        "occlusion": "none",
        "abstain": False,
        "ambiguity_reason": "",
    }


class _Client:
    def __init__(self, result: bytes):
        self._result = result
        self.batches = self
        self.files = self

    def get(self, *, name):
        assert name == "batches/1"
        return SimpleNamespace(
            state=SimpleNamespace(value="JOB_STATE_SUCCEEDED"),
            dest=SimpleNamespace(file_name="files/results"),
        )

    def download(self, *, file):
        assert file == "files/results"
        return self._result


def _policy() -> BudgetPolicy:
    return BudgetPolicy(
        run_id="test",
        enabled_providers=frozenset({"google"}),
        provider_caps={"google": Decimal("5")},
        automatic_total_cap=Decimal("4"),
        authorized_total_cap=Decimal("5"),
        usage_buffer=Decimal("0.20"),
    )


def test_collect_validates_and_reconciles_actual_usage(tmp_path):
    response = {
        "key": "request-1",
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(_semantic())}]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 20,
                "thoughtsTokenCount": 5,
            },
            "modelVersion": "test-version",
        },
    }
    client = _Client((json.dumps(response) + "\n").encode())
    ledger = BudgetLedger(tmp_path / "ledger.jsonl", _policy())
    ledger.reserve(
        request_id="request-id",
        provider="google",
        model="gemini-test",
        reserved_usd=Decimal("1"),
    )
    state_path = tmp_path / "job.json"
    state_path.write_text(
        json.dumps(
            {
                "phase": "submitted",
                "job_name": "batches/1",
                "model": "gemini-test",
                "batch_index": 1,
                "records": 1,
                "request_id": "request-id",
            }
        ),
        encoding="utf-8",
    )
    pricing = PricingCatalog(
        prices={
            "gemini-test": ModelPrice(
                provider="google",
                mode="batch",
                input_per_million_usd=Decimal("1"),
                output_per_million_usd=Decimal("2"),
            )
        },
        snapshot_hash="pricing",
        source_url="https://example.test",
        retrieved_at="2026-07-28",
    )

    state = collect_batch(
        state_path=state_path,
        ledger=ledger,
        pricing=pricing,
        client=client,
    )

    assert state["phase"] == "collected"
    assert state["usage"]["output_tokens"] == 25
    assert state["settled_usd"] == "0.000150"
    assert ledger.summary()["reserved_usd"] == "0.000000"
    assert ledger.summary()["settled_usd"] == "0.000150"
