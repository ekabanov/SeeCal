"""Build labelled contact sheets of the sampled negatives for manual food-leak
review. COCO's food supercategory is only 10 items, so dining-table scenes with
un-annotated food (rice, eggs, ...) pass the filter — a visual pass is required
before training. Each thumbnail is captioned with its coco_id so culls can be
recorded. Run from ml/:  python make_contact_sheet.py [--klass hard|plain|all]
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

ML = Path(__file__).resolve().parent
NEG = ML / "negatives"
COLS, THUMB, PAD, CAP = 6, 200, 6, 16


def build(rows, out: Path):
    per = COLS * 6  # 6 rows per sheet
    sheets = [rows[i:i + per] for i in range(0, len(rows), per)]
    outs = []
    for s, chunk in enumerate(sheets):
        n_rows = (len(chunk) + COLS - 1) // COLS
        W = COLS * (THUMB + PAD) + PAD
        H = n_rows * (THUMB + CAP + PAD) + PAD
        sheet = Image.new("RGB", (W, H), (24, 22, 20))
        d = ImageDraw.Draw(sheet)
        for i, row in enumerate(chunk):
            r, c = divmod(i, COLS)
            x = PAD + c * (THUMB + PAD)
            y = PAD + r * (THUMB + CAP + PAD)
            try:
                im = Image.open(NEG / row["file_name"]).convert("RGB")
                im.thumbnail((THUMB, THUMB))
                sheet.paste(im, (x, y))
            except Exception:
                d.rectangle([x, y, x + THUMB, y + THUMB], fill=(80, 40, 40))
            d.text((x, y + THUMB + 2), f'{row["coco_id"]} {row["klass"]}', fill=(230, 230, 230))
        op = out.with_name(f"{out.stem}_{s+1}{out.suffix}")
        sheet.save(op)
        outs.append(op)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klass", choices=["hard", "plain", "all"], default="all")
    args = ap.parse_args()
    with (NEG / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    if args.klass != "all":
        rows = [r for r in rows if r["klass"] == args.klass]
    rows.sort(key=lambda r: r["file_name"])
    outs = build(rows, NEG / f"contact_{args.klass}.jpg")
    for o in outs:
        print(o)


if __name__ == "__main__":
    main()
