"""PyTorch datasets and source-balanced sampling for specialist manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from .constants import (
    CONTAINERS,
    COOKING_STATES,
    FRB_CLASS_COUNT,
    NUMERIC_FIELDS,
    OCCLUSION_STATES,
)

SOURCE_TARGET_MIX = {
    "nutrition_overhead": 0.50,
    "nutrition_side": 0.20,
    "frb": 0.20,
    "negative": 0.10,
}


def transforms(train: bool, image_size: int = 224) -> v2.Compose:
    if train:
        geometry = [
            v2.RandomResizedCrop(
                image_size, scale=(0.72, 1.0), ratio=(0.85, 1.18),
                antialias=True,
            ),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(8),
            v2.ColorJitter(0.18, 0.18, 0.12, 0.04),
        ]
    else:
        geometry = [v2.Resize(232, antialias=True), v2.CenterCrop(image_size)]
    return v2.Compose(
        [
            *geometry,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class SpecialistDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        *,
        ml_root: Path,
        train: bool,
        include_sides: bool,
        include_frb: bool,
        include_teacher: bool,
        include_nutrition: bool = True,
        include_negatives: bool = True,
        image_size: int = 224,
    ) -> None:
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rows = [
            row
            for row in rows
            if not (
                (row["source"] == "nutrition5k" and not include_nutrition)
                or (row["source"] == "nutrition5k" and row["view"] != "overhead"
                    and not include_sides)
                or (row["source"] == "frb" and not include_frb)
                or (row["source"] == "negative" and not include_negatives)
            )
        ]
        self.ml_root = ml_root
        self.transform = transforms(train, image_size)
        self.include_teacher = include_teacher

    def __len__(self) -> int:
        return len(self.rows)

    def source_bucket(self, index: int) -> str:
        row = self.rows[index]
        if row["source"] == "nutrition5k":
            return (
                "nutrition_overhead"
                if row["view"] == "overhead"
                else "nutrition_side"
            )
        return row["source"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = Image.open(self.ml_root / row["image_path"]).convert("RGB")
        target = row["targets"]
        masks = row["loss_mask"]
        numeric = [
            float(target.get("numeric", {}).get(field, 0.0))
            for field in NUMERIC_FIELDS
        ]
        frb = torch.zeros(FRB_CLASS_COUNT)
        for class_id in target.get("frb_class_ids", []):
            frb[int(class_id)] = 1.0
        teacher = target.get("teacher") if self.include_teacher else None
        cooking = torch.zeros(len(COOKING_STATES))
        if teacher:
            for class_id in teacher["cooking_states"]:
                cooking[int(class_id)] = 1.0
        return {
            "id": row["id"],
            "source": self.source_bucket(index),
            "image": self.transform(image),
            "numeric": torch.tensor(numeric, dtype=torch.float32),
            "food": torch.tensor(float(target["food"]), dtype=torch.float32),
            "frb": frb,
            "container": torch.tensor(
                int(teacher["container"]) if teacher else 0, dtype=torch.long
            ),
            "cooking": cooking,
            "mixed": torch.tensor(
                float(teacher["mixed_dish"]) if teacher else 0.0
            ),
            "occlusion": torch.tensor(
                int(teacher["occlusion"]) if teacher else 0, dtype=torch.long
            ),
            "numeric_mask": torch.tensor(bool(masks["numeric"])),
            "food_mask": torch.tensor(bool(masks["food"])),
            "frb_mask": torch.tensor(bool(masks["frb_classes"])),
            "teacher_mask": torch.tensor(
                bool(masks["teacher_attributes"] and teacher is not None)
            ),
        }


def balanced_sampler(
    dataset: SpecialistDataset,
    *,
    seed: int,
    samples_per_epoch: int | None = None,
) -> WeightedRandomSampler:
    bucket_counts: dict[str, int] = {}
    for index in range(len(dataset)):
        bucket = dataset.source_bucket(index)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    active_mix = {
        bucket: weight
        for bucket, weight in SOURCE_TARGET_MIX.items()
        if bucket_counts.get(bucket, 0)
    }
    total_mix = sum(active_mix.values())
    weights = [
        active_mix[dataset.source_bucket(index)]
        / total_mix
        / bucket_counts[dataset.source_bucket(index)]
        for index in range(len(dataset))
    ]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=samples_per_epoch or len(dataset),
        replacement=True,
        generator=generator,
    )
