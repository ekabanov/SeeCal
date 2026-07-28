"""Write deterministic specialist predictions for a JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .constants import NUMERIC_FIELDS
from .data import SpecialistDataset
from .model import SpecialistConfig, VisualSpecialist
from .train import _move, choose_device


@torch.no_grad()
def predict(args: argparse.Namespace) -> Path:
    checkpoint_bytes = args.checkpoint.read_bytes()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VisualSpecialist(SpecialistConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    device = choose_device(args.device)
    model = model.to(device).eval()
    dataset = SpecialistDataset(
        args.manifest,
        ml_root=args.ml_root,
        train=False,
        include_sides=False,
        include_frb=False,
        include_teacher=False,
        include_negatives=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    rows = []
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = model(batch["image"])
        numeric = torch.expm1(
            output["numeric_log1p"].clamp(0, 10)
        ).clamp_min(0).cpu()
        food = torch.sigmoid(output["food_logit"]).cpu()
        for index, record_id in enumerate(raw_batch["id"]):
            rows.append(
                {
                    "schema_version": 1,
                    "id": record_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_epoch": payload["epoch"],
                    "food_probability": float(food[index]),
                    "numeric": {
                        field: {
                            "p10": float(numeric[index, field_index, 0]),
                            "p50": float(numeric[index, field_index, 1]),
                            "p90": float(numeric[index, field_index, 2]),
                        }
                        for field_index, field in enumerate(NUMERIC_FIELDS)
                    },
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(rows),
                "checkpoint_sha256": checkpoint_sha256,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ml-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
