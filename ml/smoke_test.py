"""
smoke_test.py
-------------
Fast pre-training validation: push real JSONL records through the installed
mlx-vlm batch pipeline (processor + config only — NO model weights) and assert
the training inputs are sane. Runs in ~1 minute and catches, at batch-construction
time, every silent failure mode that previously cost multi-day training runs
(AGENTS.md issues 10, 14, 15, 17, 18).

Run from ml/ (image paths in the JSONL are pipeline-root-relative):

  .venv/bin/python smoke_test.py --data finetune_data_v2/train.jsonl

Exit code 0 = safe to train; 1 = do NOT start a training run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ASSISTANT_ID = 74455       # tokenizer.encode('assistant') for Qwen3.5
DEFAULT_MODEL = "~/models/mlx-community/Qwen3.5-4B-MLX-4bit"


def main():
    ap = argparse.ArgumentParser(description="SeeCal pre-training smoke test")
    ap.add_argument("--data", type=Path, default=Path("finetune_data_v2/train.jsonl"))
    ap.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL).expanduser())
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--records", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    import mlx_vlm
    version = tuple(int(x) for x in mlx_vlm.__version__.split(".")[:2])
    legacy = version < (0, 5)
    print(f"mlx-vlm {mlx_vlm.__version__} ({'legacy assistant-id' if legacy else 'completion_mask'} stack)")

    from datasets import load_dataset
    from transformers import AutoProcessor
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.sft_trainer import iterate_batches

    with open(args.model_dir / "config.json") as f:
        config = json.load(f)
    processor = AutoProcessor.from_pretrained(str(args.model_dir))
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    stop = args.offset + args.records
    hf_ds = load_dataset(
        "json",
        data_files=str(args.data),
        split=f"train[{args.offset}:{stop}]",
    )

    failures = []

    # Pre-flight: on the new stack images MUST be a top-level column, or the
    # vision encoder silently never runs (upstream ignores content-embedded paths).
    if not legacy and "images" not in hf_ds.column_names:
        failures.append(
            "JSONL has no top-level 'images' column — mlx-vlm >= 0.5 will train "
            "text-only WITHOUT ERROR. Regenerate with prepare_finetune.py."
        )
    for rec in hf_ds:
        for p in rec.get("images") or []:
            if p and not Path(p).exists():
                failures.append(f"image path does not resolve from cwd: {p} (run from ml/)")

    if failures:
        report(failures)

    if legacy:
        ds = VisionDataset(hf_ds, config, processor)
    else:
        ds = VisionDataset(hf_ds, config, processor, train_on_completions=True)

    batch = next(iterate_batches(ds, batch_size=args.batch_size,
                                 max_seq_length=args.max_seq_length, train=True))

    ids = np.array(batch["input_ids"])
    pv = batch.get("pixel_values")
    thw = batch.get("image_grid_thw")
    attn = np.array(batch["attention_mask"])
    seq_lens = attn.sum(axis=-1)

    print(f"input_ids shape    : {ids.shape}")
    print(f"pixel_values shape : {None if pv is None else tuple(pv.shape)}")
    print(f"image_grid_thw     : {None if thw is None else np.array(thw).tolist()}")
    print(f"seq lengths        : {seq_lens.tolist()}")

    # 1. Vision must actually run (v2 failure: pixel_values absent → text-only LoRA)
    if pv is None:
        failures.append("pixel_values is None — vision encoder will NOT run during training")

    # 2. Sequences must not be collapsed (bug #17: batches truncated to ~33 tokens)
    if ids.shape[-1] < 200:
        failures.append(f"batched seq len {ids.shape[-1]} — sequences collapsed (length bug)")

    # 3. Image tokens in input_ids must match the vision grid exactly (bug #15 class)
    n_img_tok = int((ids == image_pad_id).sum())
    if thw is not None:
        thw_np = np.array(thw).reshape(-1, 3)
        merge = config.get("vision_config", {}).get("spatial_merge_size", 2)
        expected = int(np.prod(thw_np, axis=1).sum()) // (merge ** 2)
        print(f"image_pad tokens   : {n_img_tok} (expected from grid: {expected})")
        if n_img_tok != expected:
            failures.append(f"image token mismatch: {n_img_tok} in input_ids vs {expected} from grid")

    # 4. Completion masking must be engaged and sane (v1-v3 failure: mask silently all-ones)
    if legacy:
        per_row = [(row == ASSISTANT_ID).sum() for row in ids]
        print(f"assistant/row      : {per_row}")
        if any(c == 0 for c in per_row):
            failures.append("no 'assistant' (74455) token in a row — --assistant-id masking "
                            "will be silently disabled (v1-v3 failure mode)")
        completion_lens = []
        for i, row in enumerate(ids):
            pos = np.where(row == ASSISTANT_ID)[0]
            if len(pos):
                completion_lens.append(int(seq_lens[i] - pos[0] - 1))
    # Per-row decoded completion text, so the short-completion check can exempt
    # the v7 not-food refusal (`{"not_food": true}`), whose target is
    # legitimately ~8 tokens — not a truncation.
    completion_texts = ["" for _ in range(ids.shape[0])]

    if legacy:
        for i, row in enumerate(ids):
            pos = np.where(row == ASSISTANT_ID)[0]
            if len(pos):
                completion_texts[i] = processor.tokenizer.decode(
                    row[pos[0] + 1:int(seq_lens[i])].tolist())
    else:
        cm = batch.get("completion_mask")
        if cm is None:
            failures.append("completion_mask missing from batch despite train_on_completions=True")
            completion_lens = []
        else:
            cm = np.array(cm)
            completion_lens = [int((cm[i] * attn[i]).sum()) for i in range(cm.shape[0])]
            for i in range(cm.shape[0]):
                mask_i = (cm[i] * attn[i]).astype(bool)
                completion_texts[i] = processor.tokenizer.decode(ids[i][mask_i].tolist())
            # The mask must exclude the image tokens (loss over image tokens caused
            # the v1-v3 olive-oil mode collapse).
            img_positions = ids == image_pad_id
            leaked = int((cm * img_positions).sum())
            if leaked > 0:
                failures.append(f"completion_mask covers {leaked} image tokens — loss would "
                                f"be computed on vision tokens")

    for i, clen in enumerate(completion_lens):
        frac = clen / int(seq_lens[i])
        is_refusal = "not_food" in completion_texts[i]
        label = " [not-food refusal]" if is_refusal else ""
        print(f"row {i}: seq={int(seq_lens[i])} completion={clen} "
              f"({frac:.0%} of seq carries loss){label}")
        # A refusal target is ~8 tokens by design; only flag short FOOD targets.
        if clen < 20 and not is_refusal:
            failures.append(f"row {i}: only {clen} completion tokens — target truncated "
                            f"(raise --max-seq-length) or mask broken")
        if frac > 0.95:
            failures.append(f"row {i}: {frac:.0%} of sequence carries loss — masking "
                            f"effectively disabled (v1-v3 failure mode)")

    report(failures)


def report(failures):
    print()
    if failures:
        print("SMOKE TEST FAILED — do NOT start a training run:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("SMOKE TEST PASSED: images flow, token counts match, masking sane.")
    sys.exit(0)


if __name__ == "__main__":
    main()
