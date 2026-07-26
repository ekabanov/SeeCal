#!/usr/bin/env python3
"""Evaluate SeeCal runtime gates from JSON metrics.

Usage:
  python ios/SeeCal/Scripts/run_runtime_gates.py \
    --mlx ios/SeeCal/Docs/metrics_mlx_example.json \
    --mnn ios/SeeCal/Docs/metrics_mnn_example.json \
    --coreml ios/SeeCal/Docs/metrics_coreml_example.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class GateError(Exception):
    pass


def load_metric(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = {
        "runtimeName",
        "modelFamily",
        "medianLatencyMs",
        "p95LatencyMs",
        "maxMemoryMB",
        "validSchemaRate",
        "meanAbsoluteErrorCalories",
    }
    missing = required - set(data)
    if missing:
        raise GateError(f"{path} missing fields: {sorted(missing)}")
    return data


def gate_a(metric: dict) -> tuple[bool, str]:
    passed = (
        metric["modelFamily"] == "qwen3.5-native-multimodal"
        and metric["validSchemaRate"] >= 0.99
        and metric["medianLatencyMs"] <= 6000
    )
    reason = "pass" if passed else "requires native qwen3.5 + schema>=0.99 + median<=6000"
    return passed, reason


def gate_b(primary: dict, fallback: dict) -> tuple[bool, str]:
    passed = (
        fallback["modelFamily"] == primary["modelFamily"]
        and abs(primary["validSchemaRate"] - fallback["validSchemaRate"]) <= 0.01
        and abs(primary["meanAbsoluteErrorCalories"] - fallback["meanAbsoluteErrorCalories"]) <= 25
    )
    reason = "pass" if passed else "requires family parity, schema parity, calorie MAE delta<=25"
    return passed, reason


def gate_c(metric: dict) -> tuple[bool, str]:
    passed = (
        metric["modelFamily"] == "qwen3.5-native-multimodal"
        and metric["validSchemaRate"] >= 0.99
        and metric["medianLatencyMs"] <= 6000
        and metric["maxMemoryMB"] <= 5500
    )
    reason = "pass" if passed else "requires native qwen3.5 + schema>=0.99 + median<=6000 + memory<=5500"
    return passed, reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx", type=Path, required=True)
    parser.add_argument("--mnn", type=Path, required=False)
    parser.add_argument("--coreml", type=Path, required=False)
    args = parser.parse_args()

    mlx = load_metric(args.mlx)
    a_passed, a_reason = gate_a(mlx)
    print(f"Gate A (MLX): {'PASS' if a_passed else 'FAIL'} - {a_reason}")

    if args.mnn:
        mnn = load_metric(args.mnn)
        b_passed, b_reason = gate_b(mlx, mnn)
        print(f"Gate B (MNN): {'PASS' if b_passed else 'FAIL'} - {b_reason}")

    if args.coreml:
        coreml = load_metric(args.coreml)
        c_passed, c_reason = gate_c(coreml)
        print(f"Gate C (CoreML): {'PASS' if c_passed else 'FAIL'} - {c_reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
