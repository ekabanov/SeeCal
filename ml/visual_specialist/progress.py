"""Report compact progress for all visual-specialist runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m"
    if minutes:
        return f"{minutes}m{second:02d}s"
    return f"{second}s"


def summarize(runs_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(runs_dir.glob("*/status.json")):
        status = json.loads(path.read_text(encoding="utf-8"))
        latest = status.get("latest") or {}
        valid = latest.get("valid") or {}
        rows.append(
            {
                "run": path.parent.name,
                "state": status["state"],
                "epoch": f"{status['epoch']}/{status['epochs']}",
                "progress": f"{status['progress']:.0%}",
                "elapsed": _duration(status.get("elapsed_seconds")),
                "eta": _duration(status.get("eta_seconds")),
                "best_calories_mae": status.get("best_calories_mae"),
                "latest_calories_mae": valid.get("calories_mae"),
                "latest_frb_micro_f1": valid.get(
                    "frb_best_micro_f1", valid.get("frb_micro_f1")
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/visual-specialist"),
    )
    args = parser.parse_args()
    rows = summarize(args.runs_dir)
    if not rows:
        print("no tracked visual-specialist runs")
        return
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
