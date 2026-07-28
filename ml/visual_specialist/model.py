"""MobileNetV3 multi-task specialist with independently maskable heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .constants import (
    CONTAINERS,
    COOKING_STATES,
    FRB_CLASS_COUNT,
    NUMERIC_FIELDS,
    OCCLUSION_STATES,
    QUANTILES,
)


@dataclass(frozen=True)
class SpecialistConfig:
    pretrained: bool = True
    dropout: float = 0.2
    frb_classes: int = FRB_CLASS_COUNT

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VisualSpecialist(nn.Module):
    def __init__(self, config: SpecialistConfig = SpecialistConfig()) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if config.pretrained else None
        base = mobilenet_v3_large(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.embedding = nn.Sequential(*list(base.classifier.children())[:-1])
        embedding_dim = base.classifier[-1].in_features
        self.numeric = nn.Linear(
            embedding_dim, len(NUMERIC_FIELDS) * len(QUANTILES)
        )
        self.food = nn.Linear(embedding_dim, 1)
        self.frb = nn.Linear(embedding_dim, config.frb_classes)
        self.container = nn.Linear(embedding_dim, len(CONTAINERS))
        self.cooking = nn.Linear(embedding_dim, len(COOKING_STATES))
        self.mixed = nn.Linear(embedding_dim, 1)
        self.occlusion = nn.Linear(embedding_dim, len(OCCLUSION_STATES))
        self.config = config

    @torch.no_grad()
    def initialize_numeric_bias(self, log1p_medians: torch.Tensor) -> None:
        """Start quantiles near the measured-data median, not near zero grams."""
        if tuple(log1p_medians.shape) != (len(NUMERIC_FIELDS),):
            raise ValueError("numeric medians have the wrong shape")
        offsets = log1p_medians.new_tensor((-0.35, 0.0, 0.35))
        bias = (log1p_medians[:, None] + offsets[None, :]).reshape(-1)
        self.numeric.weight.zero_()
        self.numeric.bias.copy_(bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(images)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        embedding = self.embedding(features)
        numeric = self.numeric(embedding).reshape(
            -1, len(NUMERIC_FIELDS), len(QUANTILES)
        )
        return {
            "embedding": embedding,
            "numeric_log1p": numeric,
            "food_logit": self.food(embedding).squeeze(-1),
            "frb_logits": self.frb(embedding),
            "container_logits": self.container(embedding),
            "cooking_logits": self.cooking(embedding),
            "mixed_logit": self.mixed(embedding).squeeze(-1),
            "occlusion_logits": self.occlusion(embedding),
        }
