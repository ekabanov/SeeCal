"""Trace and convert the specialist to an iOS 17 Core ML ML Program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .model import SpecialistConfig, VisualSpecialist


OUTPUT_NAMES = (
    "numeric_log1p",
    "food_logit",
    "frb_logits",
    "container_logits",
    "cooking_logits",
    "mixed_logit",
    "occlusion_logits",
)


class ExportWrapper(nn.Module):
    """Remove the training-only embedding and give Core ML a tuple contract."""

    def __init__(self, model: VisualSpecialist) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(image)
        return tuple(output[name] for name in OUTPUT_NAMES)


def load_model(checkpoint: Path | None) -> VisualSpecialist:
    if checkpoint is None:
        return VisualSpecialist(SpecialistConfig(pretrained=False))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = SpecialistConfig(**payload["model_config"])
    model = VisualSpecialist(config)
    model.load_state_dict(payload["model_state"])
    return model


def export(
    *,
    checkpoint: Path | None,
    output: Path,
    image_size: int,
    trace_only: bool,
) -> Path:
    model = ExportWrapper(load_model(checkpoint).eval()).eval()
    example = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    traced = torch.jit.trace(model, example, strict=True)
    if trace_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(output))
        return output

    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(
            "coremltools is required unless --trace-only is used"
        ) from exc
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="image",
                shape=example.shape,
                dtype=float,
            )
        ],
        outputs=[ct.TensorType(name=name) for name in OUTPUT_NAMES],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output))
    metadata = {
        "schema_version": 1,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "image_size": image_size,
        "minimum_deployment_target": "iOS17",
        "compute_precision": "float16",
        "outputs": list(OUTPUT_NAMES),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    path = export(
        checkpoint=args.checkpoint,
        output=args.output,
        image_size=args.image_size,
        trace_only=args.trace_only,
    )
    print(path)


if __name__ == "__main__":
    main()
