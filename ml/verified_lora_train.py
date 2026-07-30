"""Run mlx-vlm LoRA training with a fail-fast trainable-parameter guard."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


TRAINABLE_PATTERN = re.compile(r"#trainable params:\s*([0-9.]+)\s*M")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-millions", type=float, required=True)
    parser.add_argument("--tolerance-millions", type=float, default=0.01)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("training command is required after --")

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    saw_count = False
    log_handle = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("w", encoding="utf-8")
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_handle:
                log_handle.write(line)
                log_handle.flush()
            match = TRAINABLE_PATTERN.search(line)
            if match:
                saw_count = True
                actual = float(match.group(1))
                delta = abs(actual - args.expected_millions)
                if delta > args.tolerance_millions:
                    process.terminate()
                    process.wait(timeout=30)
                    raise SystemExit(
                        "TRAINABLE-PARAMETER GUARD FAILED: "
                        f"expected {args.expected_millions:.6f}M ± "
                        f"{args.tolerance_millions:.6f}M, got {actual:.6f}M"
                    )
            if "Starting training" in line and not saw_count:
                process.terminate()
                process.wait(timeout=30)
                raise SystemExit(
                    "TRAINABLE-PARAMETER GUARD FAILED: training started before "
                    "mlx-vlm reported a trainable-parameter count"
                )
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise
    finally:
        if log_handle:
            log_handle.close()
    return_code = process.wait()
    if not saw_count:
        raise SystemExit(
            "TRAINABLE-PARAMETER GUARD FAILED: mlx-vlm exited without reporting "
            "a trainable-parameter count"
        )
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
