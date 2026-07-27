"""
make_v7_data.py
---------------
Assemble the v7 training set = v5's food data (finetune_data_v2) + the not-food
negatives (ml/negatives/, built by make_negatives.py). prepare_finetune.py is
left completely untouched — the food half is reused verbatim, so there is zero
risk of perturbing the 2594 validated food targets.

  finetune_data_v7/train.jsonl = shuffle(finetune_data_v2/train + negatives/train)
  finetune_data_v7/valid.jsonl = shuffle(finetune_data_v2/valid + negatives/valid)

Test sets stay SEPARATE and are consumed directly by eval:
  - finetune_data_v2/test.jsonl   food accuracy + over-refusal gate (325 dishes)
  - ml/negatives/test.jsonl       refusal-recall gate (held-out negatives)

Run from ml/.
"""

import argparse
import json
import random
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
FOOD_DIR = ML_ROOT / "finetune_data_v2"
NEG_DIR = ML_ROOT / "negatives"
DEFAULT_OUT_DIR = ML_ROOT / "finetune_data_v7"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def refusal_count(records: list[dict]) -> int:
    n = 0
    for r in records:
        txt = r["messages"][-1]["content"][0]["text"]
        try:
            if json.loads(txt).get("not_food") is True:
                n += 1
        except (json.JSONDecodeError, AttributeError):
            pass
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Output dir (default finetune_data_v7).")
    p.add_argument("--neg-train-cap", type=int, default=None,
                   help="Cap the number of TRAIN negatives (deterministic subsample). "
                        "v7 used all 221; v7b uses ~100 to reduce the food-accuracy hit. "
                        "valid/test negatives are untouched so the gate stays comparable.")
    args = p.parse_args()
    out_dir = args.out_dir

    for split in ("train", "valid"):
        food_path = FOOD_DIR / f"{split}.jsonl"
        neg_path = NEG_DIR / f"{split}.jsonl"
        if not food_path.exists():
            raise FileNotFoundError(f"{food_path} missing — build finetune_data_v2 first.")
        if not neg_path.exists():
            raise FileNotFoundError(
                f"{neg_path} missing — run make_negatives.py select + jsonl first.")

        food = read_jsonl(food_path)
        neg = read_jsonl(neg_path)
        # Subsample TRAIN negatives only (valid stays whole; it's just training-time
        # validation loss, not a gate).
        if split == "train" and args.neg_train_cap is not None and len(neg) > args.neg_train_cap:
            neg = random.Random(args.seed).sample(neg, args.neg_train_cap)
        combined = food + neg
        # Deterministic interleave so refusal records aren't all clustered at the
        # tail of an epoch.
        random.Random(args.seed).shuffle(combined)

        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{split}.jsonl"
        with out.open("w") as f:
            for r in combined:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(food)} food + {len(neg)} negatives = {len(combined)} "
              f"records ({refusal_count(neg)} refusals) -> {out}")

    print(f"\nv7 training data ready in {out_dir}.")
    print("Test sets stay separate:")
    print("  food     : finetune_data_v2/test.jsonl  (accuracy + over-refusal gate)")
    print("  negatives: negatives/test.jsonl          (refusal-recall gate)")
    print("\nNext: parity gate — python check_prompt_parity.py")


if __name__ == "__main__":
    main()
