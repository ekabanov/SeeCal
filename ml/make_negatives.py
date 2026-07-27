"""
make_negatives.py
-----------------
Build the v7 "not-food" negative examples: real non-food images paired with a
refusal completion, so the v7 LoRA learns to say `{"not_food": true}` instead of
hallucinating a nutrition JSON for a computer mouse, an empty plate, etc.

Two stages:

  select   Filter COCO val2017 to non-food images, sample a balanced set
           (hard-negative empty-tableware vs plain non-food), download them
           into ml/negatives/, and write ml/negatives/manifest.csv.

  jsonl    Turn the manifest into train/valid/test JSONL under ml/negatives/,
           using the *imported* prompt constants and build_record() from
           prepare_finetune.py so every negative's prompt is BYTE-IDENTICAL to
           v5's food prompt (the load-bearing adapter<->prompt parity invariant).
           The only thing that differs from a food record is the assistant
           completion: `{"not_food": true}`.

Run from ml/ (image paths in the JSONL are relative to ml/, like the food data).

Why COCO: it ships instance annotations, so we can *exclude* every image that
contains a food-supercategory object (banana, apple, sandwich, orange, broccoli,
carrot, hot dog, pizza, donut, cake). Training the model to refuse on those
would cause exactly the over-refusal on real food we must avoid. Images are
downloaded per-URL (not the 1GB val2017 zip) and never redistributed.

Usage:
  ./download_negatives.sh                 # fetches annotations, then runs `select`
  python make_negatives.py select [--count 300] [--hard-frac 0.4] [--seed 42]
  python make_negatives.py jsonl  [--seed 42]
"""

import argparse
import csv
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
COCO_ANN = ML_ROOT / "coco" / "annotations" / "instances_val2017.json"
NEG_DIR = ML_ROOT / "negatives"
MANIFEST = NEG_DIR / "manifest.csv"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"

# Tableware / dining objects. An image with these but NO food is a HARD negative:
# it teaches "tableware present != food present" (empty plates, set tables) — the
# framing closest to a real meal photo, where over-refusal risk is highest.
TABLEWARE_NAMES = {
    "dining table", "bowl", "cup", "wine glass", "fork", "knife", "spoon",
}


# ---------------------------------------------------------------------------
# select: filter COCO, sample, download
# ---------------------------------------------------------------------------

def load_coco():
    if not COCO_ANN.exists():
        sys.exit(
            f"error: {COCO_ANN} not found.\n"
            "Run ./download_negatives.sh first (it fetches the COCO val2017 "
            "annotations, then invokes this script's `select` stage)."
        )
    with COCO_ANN.open() as f:
        return json.load(f)


def select(count: int, hard_frac: float, seed: int):
    coco = load_coco()

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    food_cat_ids = {c["id"] for c in coco["categories"] if c["supercategory"] == "food"}
    print(f"COCO food-supercategory categories excluded: "
          f"{sorted(cat_name[i] for i in food_cat_ids)}")

    # image_id -> set of category names present
    cats_by_image: dict[int, set[str]] = {}
    food_images: set[int] = set()
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cid = ann["category_id"]
        cats_by_image.setdefault(img_id, set()).add(cat_name[cid])
        if cid in food_cat_ids:
            food_images.add(img_id)

    images_by_id = {img["id"]: img for img in coco["images"]}

    # Eligible = annotated images with NO food object.
    eligible = [
        img_id for img_id in cats_by_image
        if img_id not in food_images
    ]
    hard = [i for i in eligible if cats_by_image[i] & TABLEWARE_NAMES]
    plain = [i for i in eligible if not (cats_by_image[i] & TABLEWARE_NAMES)]
    print(f"Eligible non-food images: {len(eligible)}  "
          f"(hard/tableware: {len(hard)}, plain: {len(plain)})")

    n_hard = min(round(count * hard_frac), len(hard))
    n_plain = min(count - n_hard, len(plain))

    rng = random.Random(seed)
    # Deterministic: sort candidate ids before sampling so the seed fully
    # determines the picked set across machines.
    pick_hard = rng.sample(sorted(hard), n_hard)
    pick_plain = rng.sample(sorted(plain), n_plain)
    picked = [(i, "hard") for i in pick_hard] + [(i, "plain") for i in pick_plain]
    print(f"Sampled: {n_hard} hard + {n_plain} plain = {len(picked)} negatives")

    NEG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, (img_id, klass) in enumerate(sorted(picked), 1):
        img = images_by_id[img_id]
        file_name = img["file_name"]
        dest = NEG_DIR / file_name
        if not dest.exists():
            url = img.get("coco_url") or f"{IMAGE_BASE_URL}/{file_name}"
            _download(url, dest)
        objects = ";".join(sorted(cats_by_image[img_id]))
        rows.append({
            "coco_id": img_id,
            "file_name": file_name,
            "klass": klass,
            "objects": objects,
        })
        if n % 50 == 0:
            print(f"  fetched {n}/{len(picked)} ...")

    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["coco_id", "file_name", "klass", "objects"])
        w.writeheader()
        for row in sorted(rows, key=lambda r: r["file_name"]):
            w.writerow(row)
    print(f"Wrote manifest: {len(rows)} negatives -> {MANIFEST}")
    print("\nNext: spot-check ml/negatives/ (confirm no actual food slipped through), "
          "then `python make_negatives.py jsonl`.")


def _download(url: str, dest: Path, retries: int = 3):
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SeeCal/v7"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return
        except Exception as e:  # noqa: BLE001 — network flakiness; retry then fail loud
            last = e
    sys.exit(f"error: failed to download {url} after {retries} tries: {last}")


# ---------------------------------------------------------------------------
# jsonl: manifest -> train/valid/test refusal records
# ---------------------------------------------------------------------------

def _load_cull_set() -> set[int]:
    """coco_ids to exclude, from ml/negatives_cull.txt (one id per line; blank
    lines and `#` comments ignored). Lives at the ml/ root (not inside the
    gitignored negatives/) so the curated cull decision is version-controlled.
    Missing file => empty set."""
    path = ML_ROOT / "negatives_cull.txt"
    if not path.exists():
        return set()
    ids: set[int] = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(int(line))
    return ids


def build_jsonl(seed: int, train_frac: float, valid_frac: float):
    # Import the EXACT prompt constants + record builder used for the food data,
    # so the negatives' prompt is byte-identical to v5 (parity invariant).
    import prepare_finetune as pf

    if not MANIFEST.exists():
        sys.exit(f"error: {MANIFEST} not found. Run `make_negatives.py select` first.")

    with MANIFEST.open(newline="") as f:
        entries = list(csv.DictReader(f))
    if not entries:
        sys.exit("error: manifest is empty.")

    # Manual food-leak cull (negatives/cull.txt): COCO's food supercategory is
    # only 10 items, so dining-table scenes with un-annotated food (rice, meat,
    # produce) pass the automated filter. Visual review found ~20 such images in
    # the hard/tableware set; their coco_ids are listed there and excluded here so
    # we never train the model to refuse on a photo that actually contains food
    # (the critical over-refusal failure mode).
    culled = _load_cull_set()
    if culled:
        before = len(entries)
        entries = [e for e in entries if int(e["coco_id"]) not in culled]
        print(f"Culled {before - len(entries)} food-leak negatives "
              f"(from negatives_cull.txt); {len(entries)} remain.")

    # The one thing that differs from a food record: the completion. Same JSON
    # dump style (separators) as format_assistant_response for consistency.
    refusal = json.dumps({"not_food": True}, separators=(", ", ": "))

    rng = random.Random(seed)
    entries = sorted(entries, key=lambda e: e["file_name"])
    rng.shuffle(entries)
    n = len(entries)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)
    splits = {
        "train": entries[:n_train],
        "valid": entries[n_train:n_train + n_valid],
        "test":  entries[n_train + n_valid:],
    }

    sample_record = None
    for split, items in splits.items():
        records = []
        for e in items:
            # Image path relative to ml/ (project root), matching the food JSONL
            # convention: e.g. "negatives/000000123456.jpg".
            img_str = os.path.relpath((NEG_DIR / e["file_name"]).resolve(), ML_ROOT)
            records.append(pf.build_record(img_str, refusal))
        if split == "train" and records:
            sample_record = records[0]
        out = NEG_DIR / f"{split}.jsonl"
        with out.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(records):>4} refusal records -> {out}")

    print(f"\nNegative splits ready. Prompt is byte-identical to v5 by construction "
          f"(imported prepare_finetune.build_record).")
    if sample_record is not None:
        print("\nSample negative record:")
        print(json.dumps(sample_record, indent=2))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    ps = sub.add_parser("select", help="Filter COCO + download negatives + manifest.")
    ps.add_argument("--count", type=int, default=300,
                    help="Total negatives to sample (default 300 ~= 10%% of the "
                         "2594 food train records).")
    ps.add_argument("--hard-frac", type=float, default=0.4,
                    help="Fraction that are hard negatives (empty tableware).")
    ps.add_argument("--seed", type=int, default=42)

    pj = sub.add_parser("jsonl", help="Manifest -> train/valid/test refusal JSONL.")
    pj.add_argument("--seed", type=int, default=42)
    pj.add_argument("--train-frac", type=float, default=0.80)
    pj.add_argument("--valid-frac", type=float, default=0.10)

    args = p.parse_args()
    if args.stage == "select":
        select(args.count, args.hard_frac, args.seed)
    else:
        build_jsonl(args.seed, args.train_frac, args.valid_frac)


if __name__ == "__main__":
    main()
