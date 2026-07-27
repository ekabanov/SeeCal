#!/usr/bin/env python3
"""
make-appicon.py — turn an icon *mockup* into a valid iOS AppIcon asset.

Design mockups (and most AI-generated icons) come pre-rendered as a rounded
square with a drop shadow on a white page. iOS wants the opposite: a
full-bleed, fully-opaque 1024x1024 square with square corners and no shadow —
the system applies its own squircle mask. Handing Xcode the mockup as-is gives
you white wedges in the corners and a baked-in shadow.

This script does that conversion:

  1. Detect the icon body (saturated / dark pixels) and crop to its bounding
     box, discarding the white page and the drop shadow around it.
  2. Pad to a perfect square if the crop isn't one.
  3. Fill what's left of the page *inside* the square — the four rounded-corner
     gaps — by extending the nearest icon pixel outward (scipy nearest-neighbour
     inpainting), so the art goes edge to edge and the colour gradient is
     preserved. iOS masks these corners off anyway; this just guarantees no
     white ever shows.
  4. Flatten any alpha onto an opaque background and resize to 1024x1024.

Interior light areas (a white plate, say) are protected: only background
regions that touch the image border are filled, found via connected-component
labelling rather than a naive "is it whitish" test.

Usage:
  scripts/make-appicon.py INPUT [--out PATH] [--preview DIR] [--no-write]

Default --out is the app's icon asset:
  ios/App/Resources/Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png

--preview DIR writes intermediate images (crop, detected background mask,
final) so the crop can be eyeballed before committing to it. --no-write does
everything except overwrite the real asset.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = (REPO_ROOT / "ios/App/Resources/Assets.xcassets"
               / "AppIcon.appiconset/AppIcon-1024.png")
ICON_SIZE = 1024


def body_mask(rgb: np.ndarray, sat_min: float, val_max: float) -> np.ndarray:
    """True where a pixel looks like icon art rather than white page or grey
    shadow: either colourful (saturation above sat_min) or dark (value below
    val_max). Shadows are desaturated *and* light, so they fail both tests."""
    mx = rgb.max(axis=2).astype(np.float32) / 255.0
    mn = rgb.min(axis=2).astype(np.float32) / 255.0
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > sat_min) | (mx < val_max)


def border_connected(mask: np.ndarray) -> np.ndarray:
    """Restrict `mask` to components touching the image border. This is what
    separates 'the page around the icon' from 'a white plate in the middle'."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    keep = set(int(v) for v in np.unique(edge) if v != 0)
    if not keep:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, list(keep))


def nearest_fill(rgb: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Replace pixels where `fill` is True with the colour of the nearest
    pixel where it is False (nearest-neighbour inpainting)."""
    if not fill.any():
        return rgb
    # EDT on the fill region returns, for each filled pixel, the index of the
    # closest *unfilled* pixel — exactly the source colour we want.
    _, (iy, ix) = ndimage.distance_transform_edt(fill, return_indices=True)
    out = rgb.copy()
    out[fill] = rgb[iy[fill], ix[fill]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Source icon image (png/jpg/webp).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--preview", type=Path, default=None,
                    help="Directory for intermediate images (crop / mask / final).")
    ap.add_argument("--no-write", action="store_true",
                    help="Do everything except write --out.")
    ap.add_argument("--sat-min", type=float, default=0.15,
                    help="Min saturation to count as icon art (default 0.15).")
    ap.add_argument("--val-max", type=float, default=0.85,
                    help="Below this brightness a pixel counts as art (default 0.85).")
    ap.add_argument("--pad-pct", type=float, default=0.0,
                    help="Extra inward crop as %% of the detected box, if the "
                         "mockup has a visible outline you want gone.")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    src = Image.open(args.input)
    print(f"input: {args.input}  {src.size[0]}x{src.size[1]}  mode={src.mode}")

    # Flatten onto white so a transparent mockup behaves like a white-page one.
    if src.mode in ("RGBA", "LA", "P"):
        src = src.convert("RGBA")
        flat = Image.new("RGBA", src.size, (255, 255, 255, 255))
        flat.alpha_composite(src)
        src = flat.convert("RGB")
    else:
        src = src.convert("RGB")

    rgb = np.asarray(src)

    # --- 1. crop to the icon body ---
    art = body_mask(rgb, args.sat_min, args.val_max)
    # Drop stray specks so a single JPEG artifact can't blow up the box.
    art = ndimage.binary_opening(art, np.ones((3, 3)))
    if not art.any():
        print("error: no icon body detected — tune --sat-min/--val-max", file=sys.stderr)
        return 1
    ys, xs = np.where(art)
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    print(f"detected icon box: x[{left}:{right}] y[{top}:{bottom}] "
          f"({right-left+1}x{bottom-top+1})")

    if args.pad_pct:
        inset = int(round(min(right - left, bottom - top) * args.pad_pct / 100.0))
        top, bottom, left, right = top + inset, bottom - inset, left + inset, right - inset
        print(f"after --pad-pct {args.pad_pct}: {right-left+1}x{bottom-top+1}")

    crop = rgb[top:bottom + 1, left:right + 1]

    # --- 2. square it up (centre the shorter axis) ---
    h, w = crop.shape[:2]
    if h != w:
        side = max(h, w)
        # Pad with the mean edge colour so padding blends; corners get
        # inpainted below anyway.
        pad_colour = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]) \
            .mean(axis=0).astype(np.uint8)
        square = np.tile(pad_colour, (side, side, 1))
        oy, ox = (side - h) // 2, (side - w) // 2
        square[oy:oy + h, ox:ox + w] = crop
        print(f"padded {w}x{h} -> {side}x{side} (fill rgb{tuple(int(c) for c in pad_colour)})")
        crop = square

    # --- 3. fill the rounded-corner page remnants, edge to edge ---
    bg = border_connected(~body_mask(crop, args.sat_min, args.val_max))
    pct = 100.0 * bg.sum() / bg.size
    print(f"background to fill: {bg.sum()} px ({pct:.2f}% — expect a few % from "
          f"the rounded corners)")
    if pct > 25:
        print("warning: that is a lot of background; check --preview before shipping",
              file=sys.stderr)
    filled = nearest_fill(crop, bg)

    # --- 4. resize to the asset size, no alpha ---
    final = Image.fromarray(filled, "RGB").resize(
        (ICON_SIZE, ICON_SIZE), Image.LANCZOS)

    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop, "RGB").save(args.preview / "1-crop.png")
        Image.fromarray((bg * 255).astype(np.uint8), "L").save(args.preview / "2-bgmask.png")
        final.save(args.preview / "3-final.png")
        print(f"preview written to {args.preview}/")

    if args.no_write:
        print("--no-write: skipping asset write")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    final.save(args.out, "PNG")
    print(f"wrote {args.out}  ({ICON_SIZE}x{ICON_SIZE}, RGB, no alpha)")
    print("Rebuild to pick it up: scripts/build.sh --device")
    return 0


if __name__ == "__main__":
    sys.exit(main())
