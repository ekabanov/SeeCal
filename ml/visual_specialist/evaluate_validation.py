"""Recompute full numeric, FRB, and teacher-head validation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SpecialistDataset
from .model import SpecialistConfig, VisualSpecialist
from .train import EXPERIMENTS, choose_device, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    settings = {**EXPERIMENTS[args.experiment], "include_sides": False}
    dataset = SpecialistDataset(
        args.manifest_dir / "valid.jsonl",
        ml_root=args.ml_root,
        train=False,
        **settings,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VisualSpecialist(SpecialistConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    device = choose_device(args.device)
    metrics = evaluate(model.to(device), loader, device)
    result = {
        "schema_version": 1,
        "experiment": args.experiment,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "records": len(dataset),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
