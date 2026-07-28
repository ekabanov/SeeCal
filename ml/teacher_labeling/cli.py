"""Command-line interface for budget preflight and reconciliation."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path

from .budget import BudgetLedger, BudgetPolicy, PricingCatalog, load_secret_env


PACKAGE_DIR = Path(__file__).resolve().parent
ML_DIR = PACKAGE_DIR.parent
REPO_DIR = ML_DIR.parent
DEFAULT_SECRET = REPO_DIR / ".secrets" / "teacher-labeling.env"
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "gemini_5k_pilot.json"
DEFAULT_PRICING = PACKAGE_DIR / "pricing.json"
DEFAULT_LEDGER = (
    ML_DIR
    / "runs"
    / "teacher-labeling"
    / "gemini-5k-pilot"
    / "budget-ledger.jsonl"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secret", type=Path, default=DEFAULT_SECRET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")

    estimate = subparsers.add_parser("estimate")
    estimate.add_argument("--model", required=True)
    estimate.add_argument("--input-tokens", type=int, required=True)
    estimate.add_argument("--output-tokens", type=int, required=True)

    reserve = subparsers.add_parser("reserve")
    reserve.add_argument("--request-id", required=True)
    reserve.add_argument("--model", required=True)
    reserve.add_argument("--input-tokens", type=int, required=True)
    reserve.add_argument("--output-tokens", type=int, required=True)
    reserve.add_argument("--records", type=int, required=True)

    settle = subparsers.add_parser("settle")
    settle.add_argument("--request-id", required=True)
    settle.add_argument("--settled-usd", type=Decimal, required=True)
    settle.add_argument("--input-tokens", type=int, required=True)
    settle.add_argument("--output-tokens", type=int, required=True)

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--request-id", required=True)
    cancel.add_argument("--reason", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    secret_env = load_secret_env(args.secret)
    policy = BudgetPolicy.load(
        secret_env=secret_env,
        run_config_path=args.config,
    )
    pricing = PricingCatalog.load(args.pricing)
    ledger = BudgetLedger(args.ledger, policy)

    if args.command == "status":
        result = ledger.summary()
    elif args.command == "estimate":
        result = {
            "model": args.model,
            "pricing_snapshot_hash": pricing.snapshot_hash,
            "estimated_usd": str(
                pricing.cost(
                    model=args.model,
                    input_tokens=args.input_tokens,
                    output_tokens=args.output_tokens,
                    usage_buffer=policy.usage_buffer,
                )
            ),
        }
    elif args.command == "reserve":
        price = pricing.prices[args.model]
        estimate = pricing.cost(
            model=args.model,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            usage_buffer=policy.usage_buffer,
        )
        event = ledger.reserve(
            request_id=args.request_id,
            provider=price.provider,
            model=args.model,
            reserved_usd=estimate,
            metadata={
                "records": args.records,
                "input_tokens_estimate": args.input_tokens,
                "output_tokens_estimate": args.output_tokens,
                "pricing_snapshot_hash": pricing.snapshot_hash,
            },
        )
        result = {"event": event, "summary": ledger.summary()}
    elif args.command == "settle":
        event = ledger.settle(
            request_id=args.request_id,
            settled_usd=args.settled_usd,
            usage={
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "pricing_snapshot_hash": pricing.snapshot_hash,
            },
        )
        result = {"event": event, "summary": ledger.summary()}
    else:
        event = ledger.cancel(request_id=args.request_id, reason=args.reason)
        result = {"event": event, "summary": ledger.summary()}

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
