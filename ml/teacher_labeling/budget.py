"""Fail-closed budget accounting for paid teacher-labeling calls.

The provider only reports final usage after a request or asynchronous batch
finishes.  A safe runner therefore accounts for both settled cost and open
reservations.  The append-only ledger is hash-chained so a partial/corrupt edit
cannot silently reset spend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


ZERO = Decimal("0")
MILLION = Decimal("1000000")
MONEY_QUANTUM = Decimal("0.000001")
LEDGER_SCHEMA_VERSION = 1


class BudgetError(RuntimeError):
    """Base class for budget failures."""


class BudgetExceeded(BudgetError):
    """A proposed reservation would exceed an enabled budget ceiling."""


class LedgerCorrupt(BudgetError):
    """The ledger cannot be trusted and paid work must stop."""


class ConfigurationError(BudgetError):
    """Budget or pricing configuration is invalid."""


def _decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ConfigurationError(f"{field} must be a decimal number") from exc
    if not result.is_finite():
        raise ConfigurationError(f"{field} must be finite")
    return result


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _event_hash(event_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(event_without_hash)).hexdigest()


def load_secret_env(path: Path) -> dict[str, str]:
    """Load a strict dotenv file without mutating ``os.environ``.

    Secret files must be owner-only.  Values are never included in errors.
    """

    if not path.is_file():
        raise ConfigurationError(f"secret file is missing: {path}")

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ConfigurationError(
            f"secret file must be owner-only (chmod 600): {path}"
        )

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            raise ConfigurationError(
                f"{path}:{line_number}: use NAME=value without export"
            )
        if "=" not in line:
            raise ConfigurationError(
                f"{path}:{line_number}: expected NAME=value"
            )
        name, value = line.split("=", 1)
        if not name or name.strip() != name or not name.replace("_", "").isalnum():
            raise ConfigurationError(
                f"{path}:{line_number}: invalid variable name"
            )
        if name in values:
            raise ConfigurationError(
                f"{path}:{line_number}: duplicate variable {name}"
            )
        if not value:
            raise ConfigurationError(
                f"{path}:{line_number}: {name} must not be empty"
            )
        values[name] = value
    return values


@dataclass(frozen=True)
class BudgetPolicy:
    """Effective run ceilings, narrowed from secret-level authorization."""

    run_id: str
    enabled_providers: frozenset[str]
    provider_caps: Mapping[str, Decimal]
    automatic_total_cap: Decimal
    authorized_total_cap: Decimal
    usage_buffer: Decimal

    @classmethod
    def load(
        cls,
        *,
        secret_env: Mapping[str, str],
        run_config_path: Path,
    ) -> "BudgetPolicy":
        try:
            config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"cannot load run config: {run_config_path}"
            ) from exc

        if config.get("schema_version") != 1:
            raise ConfigurationError("unsupported run-config schema_version")

        run_id = config.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ConfigurationError("run_id must be a non-empty string")

        enabled_raw = config.get("enabled_providers")
        if not isinstance(enabled_raw, list) or not all(
            isinstance(item, str) and item for item in enabled_raw
        ):
            raise ConfigurationError("enabled_providers must be a string list")
        enabled = frozenset(enabled_raw)

        env_total = _decimal(
            secret_env.get("TEACHER_BUDGET_TOTAL_USD"),
            "TEACHER_BUDGET_TOTAL_USD",
        )
        env_reserve = _decimal(
            secret_env.get("TEACHER_BUDGET_RESERVE_USD", "0"),
            "TEACHER_BUDGET_RESERVE_USD",
        )
        run_caps = config.get("caps_usd")
        if not isinstance(run_caps, dict):
            raise ConfigurationError("caps_usd must be an object")
        run_total = _decimal(run_caps.get("total"), "caps_usd.total")
        run_reserve = _decimal(
            config.get("safety_reserve_usd", "0"),
            "safety_reserve_usd",
        )

        if min(env_total, env_reserve, run_total, run_reserve) < ZERO:
            raise ConfigurationError("budget caps and reserves must be non-negative")
        if env_reserve >= env_total:
            raise ConfigurationError("environment reserve must be below total cap")
        if run_reserve >= run_total:
            raise ConfigurationError("run reserve must be below total cap")

        automatic_total = min(
            env_total - env_reserve,
            run_total - run_reserve,
        )
        authorized_total = min(env_total, run_total)
        if automatic_total <= ZERO:
            raise ConfigurationError("automatic total cap must be positive")

        provider_caps: dict[str, Decimal] = {}
        for provider in {"google", "openai"} | set(run_caps):
            if provider == "total":
                continue
            env_name = f"TEACHER_BUDGET_{provider.upper()}_USD"
            env_cap = _decimal(secret_env.get(env_name, "0"), env_name)
            run_cap = _decimal(
                run_caps.get(provider, "0"),
                f"caps_usd.{provider}",
            )
            if min(env_cap, run_cap) < ZERO:
                raise ConfigurationError("provider caps must be non-negative")
            provider_caps[provider] = min(env_cap, run_cap)

        for provider in enabled:
            if provider_caps.get(provider, ZERO) <= ZERO:
                raise ConfigurationError(
                    f"enabled provider {provider!r} has no positive cap"
                )

        usage_buffer = _decimal(
            config.get("reservation_usage_buffer", "0.20"),
            "reservation_usage_buffer",
        )
        if usage_buffer < ZERO:
            raise ConfigurationError("reservation_usage_buffer must be non-negative")

        return cls(
            run_id=run_id,
            enabled_providers=enabled,
            provider_caps=provider_caps,
            automatic_total_cap=_money(automatic_total),
            authorized_total_cap=_money(authorized_total),
            usage_buffer=usage_buffer,
        )


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    mode: str
    input_per_million_usd: Decimal
    output_per_million_usd: Decimal


class PricingCatalog:
    """Pinned token prices used for estimates and ledger reconciliation."""

    def __init__(
        self,
        *,
        prices: Mapping[str, ModelPrice],
        snapshot_hash: str,
        source_url: str,
        retrieved_at: str,
    ) -> None:
        self.prices = dict(prices)
        self.snapshot_hash = snapshot_hash
        self.source_url = source_url
        self.retrieved_at = retrieved_at

    @classmethod
    def load(cls, path: Path) -> "PricingCatalog":
        try:
            raw_bytes = path.read_bytes()
            data = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot load pricing: {path}") from exc

        if data.get("schema_version") != 1:
            raise ConfigurationError("unsupported pricing schema_version")
        models = data.get("models")
        if not isinstance(models, dict) or not models:
            raise ConfigurationError("pricing models must be a non-empty object")

        parsed: dict[str, ModelPrice] = {}
        for model, item in models.items():
            if not isinstance(item, dict):
                raise ConfigurationError(f"pricing for {model} must be an object")
            parsed[model] = ModelPrice(
                provider=str(item["provider"]),
                mode=str(item["mode"]),
                input_per_million_usd=_decimal(
                    item["input_per_million_usd"],
                    f"{model}.input_per_million_usd",
                ),
                output_per_million_usd=_decimal(
                    item["output_per_million_usd"],
                    f"{model}.output_per_million_usd",
                ),
            )

        return cls(
            prices=parsed,
            snapshot_hash=hashlib.sha256(raw_bytes).hexdigest(),
            source_url=str(data["source_url"]),
            retrieved_at=str(data["retrieved_at"]),
        )

    def cost(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        usage_buffer: Decimal = ZERO,
    ) -> Decimal:
        if input_tokens < 0 or output_tokens < 0:
            raise ConfigurationError("token counts must be non-negative")
        if usage_buffer < ZERO:
            raise ConfigurationError("usage_buffer must be non-negative")
        try:
            price = self.prices[model]
        except KeyError as exc:
            raise ConfigurationError(f"unknown model price: {model}") from exc
        base = (
            Decimal(input_tokens) * price.input_per_million_usd
            + Decimal(output_tokens) * price.output_per_million_usd
        ) / MILLION
        return _money(base * (Decimal("1") + usage_buffer))


class BudgetLedger:
    """Hash-chained append-only reservation and settlement ledger."""

    def __init__(self, path: Path, policy: BudgetPolicy) -> None:
        self.path = path
        self.policy = policy
        self._events = self._load_events()
        self._validate_state()

    def _load_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if not self.path.is_file():
            raise LedgerCorrupt(f"ledger path is not a file: {self.path}")

        events: list[dict[str, Any]] = []
        previous_hash = "GENESIS"
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise LedgerCorrupt(f"cannot read ledger: {self.path}") from exc

        for index, line in enumerate(lines, start=1):
            if not line.strip():
                raise LedgerCorrupt(f"blank ledger line at {index}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerCorrupt(f"invalid JSON at ledger line {index}") from exc
            if not isinstance(event, dict):
                raise LedgerCorrupt(f"ledger line {index} is not an object")
            supplied_hash = event.get("event_hash")
            payload = {key: value for key, value in event.items() if key != "event_hash"}
            if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise LedgerCorrupt(f"unsupported schema at ledger line {index}")
            if payload.get("sequence") != index:
                raise LedgerCorrupt(f"sequence mismatch at ledger line {index}")
            if payload.get("previous_hash") != previous_hash:
                raise LedgerCorrupt(f"hash-chain mismatch at ledger line {index}")
            expected_hash = _event_hash(payload)
            if supplied_hash != expected_hash:
                raise LedgerCorrupt(f"event hash mismatch at ledger line {index}")
            previous_hash = supplied_hash
            events.append(event)
        return events

    def _state(
        self,
    ) -> tuple[dict[str, tuple[str, Decimal]], dict[str, Decimal]]:
        open_reservations: dict[str, tuple[str, Decimal]] = {}
        settled: dict[str, Decimal] = {}
        for event in self._events:
            kind = event["kind"]
            request_id = event["request_id"]
            provider = event["provider"]
            if kind == "reserve":
                if request_id in open_reservations:
                    raise LedgerCorrupt(f"duplicate open reservation: {request_id}")
                open_reservations[request_id] = (
                    provider,
                    _decimal(event["reserved_usd"], "reserved_usd"),
                )
            elif kind == "settle":
                current = open_reservations.pop(request_id, None)
                if current is None or current[0] != provider:
                    raise LedgerCorrupt(f"settlement without reservation: {request_id}")
                settled[provider] = settled.get(provider, ZERO) + _decimal(
                    event["settled_usd"], "settled_usd"
                )
            elif kind == "cancel":
                current = open_reservations.pop(request_id, None)
                if current is None or current[0] != provider:
                    raise LedgerCorrupt(f"cancel without reservation: {request_id}")
            else:
                raise LedgerCorrupt(f"unknown ledger event kind: {kind!r}")
        return open_reservations, settled

    def _validate_state(self) -> None:
        open_reservations, settled = self._state()
        for request_id, (provider, amount) in open_reservations.items():
            if amount <= ZERO:
                raise LedgerCorrupt(f"invalid reservation amount: {request_id}")
            if provider not in self.policy.enabled_providers:
                raise LedgerCorrupt(f"reservation uses disabled provider: {provider}")
        for provider, amount in settled.items():
            if amount < ZERO:
                raise LedgerCorrupt(f"negative settled amount: {provider}")

    def summary(self) -> dict[str, Any]:
        open_reservations, settled = self._state()
        reserved_by_provider: dict[str, Decimal] = {}
        for provider, amount in open_reservations.values():
            reserved_by_provider[provider] = (
                reserved_by_provider.get(provider, ZERO) + amount
            )

        providers = sorted(
            set(self.policy.provider_caps)
            | set(settled)
            | set(reserved_by_provider)
        )
        provider_summary: dict[str, Any] = {}
        for provider in providers:
            provider_settled = settled.get(provider, ZERO)
            provider_reserved = reserved_by_provider.get(provider, ZERO)
            provider_summary[provider] = {
                "enabled": provider in self.policy.enabled_providers,
                "cap_usd": str(_money(self.policy.provider_caps.get(provider, ZERO))),
                "settled_usd": str(_money(provider_settled)),
                "reserved_usd": str(_money(provider_reserved)),
                "committed_usd": str(
                    _money(provider_settled + provider_reserved)
                ),
            }

        total_settled = sum(settled.values(), ZERO)
        total_reserved = sum(reserved_by_provider.values(), ZERO)
        total_committed = total_settled + total_reserved
        return {
            "run_id": self.policy.run_id,
            "automatic_total_cap_usd": str(self.policy.automatic_total_cap),
            "authorized_total_cap_usd": str(self.policy.authorized_total_cap),
            "settled_usd": str(_money(total_settled)),
            "reserved_usd": str(_money(total_reserved)),
            "committed_usd": str(_money(total_committed)),
            "remaining_automatic_usd": str(
                _money(self.policy.automatic_total_cap - total_committed)
            ),
            "open_request_ids": sorted(open_reservations),
            "providers": provider_summary,
            "event_count": len(self._events),
        }

    def reserve(
        self,
        *,
        request_id: str,
        provider: str,
        model: str,
        reserved_usd: Decimal,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not request_id:
            raise ConfigurationError("request_id must not be empty")
        amount = _money(_decimal(reserved_usd, "reserved_usd"))
        if amount <= ZERO:
            raise ConfigurationError("reserved_usd must be positive")
        if provider not in self.policy.enabled_providers:
            raise BudgetExceeded(f"provider is disabled for this run: {provider}")

        open_reservations, settled = self._state()
        all_request_ids = {event["request_id"] for event in self._events}
        if request_id in all_request_ids:
            raise BudgetError(f"request_id already exists: {request_id}")

        current_total = sum(settled.values(), ZERO) + sum(
            (item[1] for item in open_reservations.values()), ZERO
        )
        proposed_total = current_total + amount
        if proposed_total > self.policy.automatic_total_cap:
            raise BudgetExceeded(
                "reservation would exceed automatic total cap: "
                f"{_money(proposed_total)} > {self.policy.automatic_total_cap}"
            )

        current_provider = settled.get(provider, ZERO) + sum(
            (
                reservation_amount
                for reservation_provider, reservation_amount
                in open_reservations.values()
                if reservation_provider == provider
            ),
            ZERO,
        )
        provider_cap = self.policy.provider_caps.get(provider, ZERO)
        proposed_provider = current_provider + amount
        if proposed_provider > provider_cap:
            raise BudgetExceeded(
                f"reservation would exceed {provider} cap: "
                f"{_money(proposed_provider)} > {provider_cap}"
            )

        return self._append(
            {
                "kind": "reserve",
                "request_id": request_id,
                "provider": provider,
                "model": model,
                "reserved_usd": str(amount),
                "metadata": dict(metadata or {}),
            }
        )

    def settle(
        self,
        *,
        request_id: str,
        settled_usd: Decimal,
        usage: Mapping[str, Any],
    ) -> dict[str, Any]:
        amount = _money(_decimal(settled_usd, "settled_usd"))
        if amount < ZERO:
            raise ConfigurationError("settled_usd must be non-negative")
        open_reservations, _ = self._state()
        current = open_reservations.get(request_id)
        if current is None:
            raise BudgetError(f"no open reservation: {request_id}")
        provider, _ = current
        return self._append(
            {
                "kind": "settle",
                "request_id": request_id,
                "provider": provider,
                "settled_usd": str(amount),
                "usage": dict(usage),
            }
        )

    def cancel(self, *, request_id: str, reason: str) -> dict[str, Any]:
        open_reservations, _ = self._state()
        current = open_reservations.get(request_id)
        if current is None:
            raise BudgetError(f"no open reservation: {request_id}")
        provider, _ = current
        return self._append(
            {
                "kind": "cancel",
                "request_id": request_id,
                "provider": provider,
                "reason": reason,
            }
        )

    def _append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        previous_hash = (
            self._events[-1]["event_hash"] if self._events else "GENESIS"
        )
        event_without_hash = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": len(self._events) + 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "previous_hash": previous_hash,
            **payload,
        }
        event = {
            **event_without_hash,
            "event_hash": _event_hash(event_without_hash),
        }
        encoded = _canonical_json(event) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        descriptor = os.open(self.path, flags, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._events.append(event)
        self._validate_state()
        return event
