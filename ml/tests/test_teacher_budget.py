import json
from decimal import Decimal
from pathlib import Path

import pytest

from teacher_labeling.budget import (
    BudgetExceeded,
    BudgetLedger,
    BudgetPolicy,
    ConfigurationError,
    LedgerCorrupt,
    PricingCatalog,
    load_secret_env,
)


def _secret_values() -> dict[str, str]:
    return {
        "GEMINI_API_KEY": "not-a-real-secret",
        "OPENAI_API_KEY": "also-not-real",
        "TEACHER_BUDGET_TOTAL_USD": "100.00",
        "TEACHER_BUDGET_GOOGLE_USD": "65.00",
        "TEACHER_BUDGET_OPENAI_USD": "35.00",
        "TEACHER_BUDGET_RESERVE_USD": "10.00",
    }


def _write_run_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "test-run",
                "enabled_providers": ["google"],
                "caps_usd": {
                    "total": "25.00",
                    "google": "25.00",
                    "openai": "0.00",
                },
                "safety_reserve_usd": "5.00",
                "reservation_usage_buffer": "0.20",
            }
        ),
        encoding="utf-8",
    )


def _policy(tmp_path: Path) -> BudgetPolicy:
    config = tmp_path / "config.json"
    _write_run_config(config)
    return BudgetPolicy.load(
        secret_env=_secret_values(),
        run_config_path=config,
    )


def test_policy_can_only_narrow_environment_caps(tmp_path):
    policy = _policy(tmp_path)

    assert policy.authorized_total_cap == Decimal("25.000000")
    assert policy.automatic_total_cap == Decimal("20.000000")
    assert policy.provider_caps["google"] == Decimal("25")
    assert policy.provider_caps["openai"] == Decimal("0")
    assert policy.enabled_providers == frozenset({"google"})


def test_secret_loader_requires_owner_only_permissions(tmp_path):
    secret = tmp_path / "teacher.env"
    secret.write_text(
        "GEMINI_API_KEY=secret\nTEACHER_BUDGET_TOTAL_USD=25\n",
        encoding="utf-8",
    )
    secret.chmod(0o644)

    with pytest.raises(ConfigurationError, match="owner-only"):
        load_secret_env(secret)

    secret.chmod(0o600)
    values = load_secret_env(secret)
    assert values["GEMINI_API_KEY"] == "secret"


def test_reserve_settle_and_reload(tmp_path):
    ledger_path = tmp_path / "budget-ledger.jsonl"
    ledger = BudgetLedger(ledger_path, _policy(tmp_path))

    ledger.reserve(
        request_id="batch-001",
        provider="google",
        model="gemini-3.5-flash",
        reserved_usd=Decimal("4.00"),
        metadata={"records": 500},
    )
    assert ledger.summary()["reserved_usd"] == "4.000000"

    ledger.settle(
        request_id="batch-001",
        settled_usd=Decimal("3.25"),
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    reloaded = BudgetLedger(ledger_path, _policy(tmp_path))
    summary = reloaded.summary()
    assert summary["settled_usd"] == "3.250000"
    assert summary["reserved_usd"] == "0.000000"
    assert summary["remaining_automatic_usd"] == "16.750000"
    assert summary["event_count"] == 2


def test_rejects_disabled_provider_and_total_overrun(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.jsonl", _policy(tmp_path))

    with pytest.raises(BudgetExceeded, match="disabled"):
        ledger.reserve(
            request_id="openai-001",
            provider="openai",
            model="unused",
            reserved_usd=Decimal("1"),
        )

    with pytest.raises(BudgetExceeded, match="automatic total cap"):
        ledger.reserve(
            request_id="google-too-large",
            provider="google",
            model="gemini-3.5-flash",
            reserved_usd=Decimal("20.01"),
        )


def test_duplicate_request_id_is_rejected(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.jsonl", _policy(tmp_path))
    ledger.reserve(
        request_id="same-id",
        provider="google",
        model="gemini-3.5-flash",
        reserved_usd=Decimal("1"),
    )
    ledger.cancel(request_id="same-id", reason="test")

    with pytest.raises(Exception, match="already exists"):
        ledger.reserve(
            request_id="same-id",
            provider="google",
            model="gemini-3.5-flash",
            reserved_usd=Decimal("1"),
        )


def test_tampered_ledger_fails_closed(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    policy = _policy(tmp_path)
    ledger = BudgetLedger(ledger_path, policy)
    ledger.reserve(
        request_id="batch-001",
        provider="google",
        model="gemini-3.5-flash",
        reserved_usd=Decimal("1"),
    )

    text = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(text.replace("1.000000", "0.000001"), encoding="utf-8")

    with pytest.raises(LedgerCorrupt, match="event hash mismatch"):
        BudgetLedger(ledger_path, policy)


def test_pricing_catalog_matches_pilot_estimates(tmp_path):
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "retrieved_at": "2026-07-28",
                "source_url": "https://example.test/pricing",
                "models": {
                    "flash": {
                        "provider": "google",
                        "mode": "batch",
                        "input_per_million_usd": "0.75",
                        "output_per_million_usd": "4.50",
                    },
                    "lite": {
                        "provider": "google",
                        "mode": "batch",
                        "input_per_million_usd": "0.15",
                        "output_per_million_usd": "1.25",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    catalog = PricingCatalog.load(pricing_path)

    assert catalog.cost(
        model="flash",
        input_tokens=7_500_000,
        output_tokens=1_500_000,
    ) == Decimal("12.375000")
    assert catalog.cost(
        model="lite",
        input_tokens=7_500_000,
        output_tokens=1_500_000,
    ) == Decimal("3.000000")
    assert catalog.cost(
        model="flash",
        input_tokens=7_500_000,
        output_tokens=1_500_000,
        usage_buffer=Decimal("0.20"),
    ) == Decimal("14.850000")
