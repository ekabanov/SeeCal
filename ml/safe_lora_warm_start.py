"""Warm-start mlx-vlm LoRA training without unfreezing the base model.

mlx-vlm 0.6.7's built-in ``--adapter-path`` branch applies LoRA layers to a
freshly loaded model without first freezing the base model.  For SeeCal's
Qwen3.5-4B model that exposes 366.9M trainable parameters instead of the
expected 32.464896M LoRA parameters.

This wrapper preserves mlx-vlm's normal training implementation while fixing
that ordering:

1. freeze the base model;
2. insert and load the saved LoRA layers;
3. retain the adapter configuration for subsequent checkpoints.

Run it through ``verified_lora_train.py`` so training cannot begin unless the
reported trainable-parameter count matches the expected LoRA shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def configure_model_for_warm_start(model, adapter_path: str):
    """Freeze ``model`` before loading only the adapter's LoRA parameters."""
    from mlx_vlm.trainer.utils import apply_lora_layers, freeze_model

    path = Path(adapter_path)
    config_path = path / "adapter_config.json"
    weights_path = path / "adapters.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter configuration: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze_model(model)
    model = apply_lora_layers(model, str(path))

    # mlx-vlm's save_adapter() reads this attribute when writing checkpoints.
    model.config.lora = config
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    parser.add_argument("--custom-prompt-format", type=str, default=None)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-report", type=int, default=50)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--steps-per-save", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--train-on-completions", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--assistant-id", type=int, default=77091)

    # These describe the already-created adapter. They remain available so the
    # Namespace matches mlx-vlm's normal CLI, but the saved config is authoritative.
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument(
        "--train-mode",
        choices=("sft", "orpo"),
        default="sft",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--train-vision", action="store_true")
    return parser


def main() -> None:
    from mlx_vlm import lora

    args = build_parser().parse_args()
    if args.full_finetune or args.train_vision:
        raise SystemExit(
            "Warm-start continuation supports LoRA-only training; "
            "--full-finetune and --train-vision are forbidden."
        )

    def guarded_setup(model, _args, adapter_path=None):
        if not adapter_path:
            raise ValueError("--adapter-path is required for a warm start")
        return configure_model_for_warm_start(model, adapter_path)

    lora.setup_model_for_training = guarded_setup
    lora.main(args)


if __name__ == "__main__":
    main()
