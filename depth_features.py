"""
depth_features.py
------------------
Depth-map utilities for the Nutrition5K RealSense overhead captures.

Provides the primitives needed by the depth-input execution plan
(docs/plans/2026-07-26-depth-plan.md, Task D1):

  - load_depth(path)               -> meters, HxW float array, NaN = invalid
  - plane_fit(depth)                -> (normal, d, inlier_mask)  RANSAC table/plate plane
  - food_stats(depth, plane)        -> {"volume_ml", "max_height_mm", "coverage_cm2"}
  - depth_to_height_image(depth, plane) -> 8-bit grayscale PIL.Image (variant A input)

Unit-scale finding (empirical, see docs/plans/2026-07-26-depth-plan.md Task D1 and
tests/test_depth_features.py::test_border_depth_matches_rig_height):
  Nutrition5K's realsense_overhead/<dish>/depth_raw.png files are uint16 PNGs whose
  raw pixel values are in units of 1e-4 m (0.1 mm) -- i.e. meters = raw / 10000.
  Evidence: across a random sample of dishes, the median raw depth value is ~3584-3730,
  and 3584 / 10000 = 0.3584 m, matching the Nutrition5k rig height of 0.359 m (per the
  Nutrition5k paper) to within a millimeter. The candidate "/1000 (mm)" scale would put
  the table ~3.6-4 METERS from an overhead tabletop rig, which is physically impossible.
  No README/docs ship inside Nutrition5K/ (checked: only the three CSVs + imagery/ dirs),
  so this empirical check is the only source of truth.

  One known-bad file was found during this investigation:
  Nutrition5K/imagery/realsense_overhead/dish_1556572657/depth_raw.png has raw values
  ~342-501 (10x smaller than every other sampled dish) -- almost certainly a corrupted
  or mis-captured file, not a different sensor calibration (a 60-dish random sample
  found zero other files like it). load_depth() does not special-case this; downstream
  consumers that need robustness against it should sanity-check the decoded depth
  (see _iter_valid_depth_dishes below, which filters on plausible raw-median range).
  A second bad file was found: dish_1564159636/depth_raw.png is a zero-byte file
  (unreadable by PIL); _iter_valid_depth_dishes skips files it cannot open.

Pinhole camera model:
  The Nutrition5K capture rig uses an Intel RealSense D435 with a ~65 deg horizontal
  FOV over 640 px, giving focal length f = (640 / 2) / tan(65deg / 2) =~ 616 px. This
  focal length is used for both the per-pixel footprint calculation (food_stats) and
  the pixel -> 3D unprojection used for the plane fit (plane_fit).
"""

from __future__ import annotations

import csv
import glob
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Empirically determined (see module docstring + test_border_depth_matches_rig_height):
# raw depth units are 1e-4 m (0.1 mm), i.e. meters = raw_uint16 / 10000.
DEPTH_SCALE_M_PER_UNIT = 1.0 / 10000.0

# RealSense D435: ~65 deg horizontal FOV over 640 px -> focal length in pixels.
FOCAL_PX = 616.0

# Fraction of the shorter image dimension used as the outer "border ring" that is
# assumed to be table/plate (background) for plane-fit candidate selection.
BORDER_FRAC = 0.12

# RANSAC plane-fit parameters.
RANSAC_ITERS = 300
RANSAC_INLIER_THRESH_M = 0.004  # 4 mm
RANSAC_SEED = 0

# Food/noise-floor threshold: pixels within this height of the fitted plane are
# considered table/plate, not food.
HEIGHT_NOISE_FLOOR_M = 0.003  # 3 mm

# depth_to_height_image scaling: 0 mm (plane) -> pixel 0, 60 mm+ -> pixel 255.
HEIGHT_IMAGE_MAX_MM = 60.0

REPO_ROOT = Path(__file__).resolve().parent
NUTRITION5K_DIR = REPO_ROOT / "Nutrition5K"
REALSENSE_OVERHEAD_DIR = NUTRITION5K_DIR / "imagery" / "realsense_overhead"
NUTRITION_CSV = NUTRITION5K_DIR / "dish_nutrition_values.csv"


# ---------------------------------------------------------------------------
# load_depth
# ---------------------------------------------------------------------------

def load_depth(path) -> np.ndarray:
    """
    Load a Nutrition5K realsense_overhead depth_raw.png file.

    Returns a float64 HxW array in meters. Raw value 0 (no reading / invalid) is
    mapped to NaN so downstream code can distinguish "no depth" from "0 meters".
    """
    raw = np.array(Image.open(path))
    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16)
    depth_m = raw.astype(np.float64) * DEPTH_SCALE_M_PER_UNIT
    depth_m[raw == 0] = np.nan
    return depth_m


# ---------------------------------------------------------------------------
# Pinhole helpers
# ---------------------------------------------------------------------------

def _unproject(rows: np.ndarray, cols: np.ndarray, z: np.ndarray, shape: tuple[int, int]):
    """Pixel (row, col) + depth z (meters) -> camera-space (X, Y, Z) in meters."""
    h, w = shape
    cx, cy = w / 2.0, h / 2.0
    x = (cols - cx) * z / FOCAL_PX
    y = (rows - cy) * z / FOCAL_PX
    return x, y, z


# ---------------------------------------------------------------------------
# plane_fit
# ---------------------------------------------------------------------------

def plane_fit(depth: np.ndarray, seed: int = RANSAC_SEED):
    """
    RANSAC-fit the table/plate plane using border-ring pixels (assumed background).

    Returns (normal, d, inlier_mask):
      - normal: unit np.ndarray shape (3,), the plane's unit normal in camera space.
      - d: float, plane offset such that for a point p=(X,Y,Z), normal . p + d == 0
           for points exactly on the plane.
      - inlier_mask: HxW bool array, True for border-ring pixels that were RANSAC
        inliers of the final fitted plane (False everywhere else, including
        non-border/interior pixels).

    Sign convention: normal . p + d is the signed perpendicular distance (meters)
    from point p to the plane, positive when p is displaced from the plane in the
    same direction as `normal`. Because the fit orients the plane as Z = A*X + B*Y + C
    with normal = normalize([A, B, -1]), `normal` points back toward the camera
    (negative Z component), so height_above_plane(p) = normal . p + d is positive
    for food (which sits between the camera and the table, i.e. at smaller Z).
    """
    h, w = depth.shape
    border = max(1, int(BORDER_FRAC * min(h, w)))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    valid_mask = ~np.isnan(depth)
    candidate_mask = border_mask & valid_mask
    rows, cols = np.nonzero(candidate_mask)
    if rows.size < 3:
        raise ValueError("plane_fit: fewer than 3 valid border pixels to fit a plane")

    z = depth[rows, cols]
    x, y, zc = _unproject(rows, cols, z, depth.shape)
    pts = np.stack([x, y, zc], axis=1)  # (N, 3)
    n_pts = pts.shape[0]

    rng = np.random.default_rng(seed)
    best_inlier_bool = None
    best_count = -1

    for _ in range(RANSAC_ITERS):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        norm_len = np.linalg.norm(cross)
        if norm_len < 1e-12:
            continue
        cand_normal = cross / norm_len
        cand_d = -np.dot(cand_normal, p0)
        dist = np.abs(pts @ cand_normal + cand_d)
        inlier_bool = dist < RANSAC_INLIER_THRESH_M
        count = int(np.sum(inlier_bool))
        if count > best_count:
            best_count = count
            best_inlier_bool = inlier_bool

    if best_inlier_bool is None or best_count < 3:
        raise RuntimeError("plane_fit: RANSAC failed to find a valid plane")

    # Refine with a least-squares fit of Z = A*X + B*Y + C over all inliers found by
    # RANSAC. This form is valid because the rig looks straight down at the table, so
    # the plane is never (near-)vertical in camera space.
    inlier_pts = pts[best_inlier_bool]
    design = np.stack(
        [inlier_pts[:, 0], inlier_pts[:, 1], np.ones(len(inlier_pts))], axis=1
    )
    target = inlier_pts[:, 2]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    a, b, c = coeffs
    norm_factor = np.sqrt(a ** 2 + b ** 2 + 1.0)
    normal = np.array([a, b, -1.0]) / norm_factor
    d = c / norm_factor

    inlier_mask = np.zeros((h, w), dtype=bool)
    inlier_rows = rows[best_inlier_bool]
    inlier_cols = cols[best_inlier_bool]
    inlier_mask[inlier_rows, inlier_cols] = True

    return normal, float(d), inlier_mask


# ---------------------------------------------------------------------------
# food_stats
# ---------------------------------------------------------------------------

def _height_above_plane(depth: np.ndarray, plane):
    """Return (rows, cols, height_m) for every valid (non-NaN) pixel."""
    normal, d, _inlier_mask = plane
    h, w = depth.shape
    valid = ~np.isnan(depth)
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    x, y, zc = _unproject(rows, cols, z, depth.shape)
    height = normal[0] * x + normal[1] * y + normal[2] * zc + d
    return rows, cols, z, height


def food_stats(depth: np.ndarray, plane) -> dict:
    """
    Compute volume/height/coverage stats for the food sitting on the fitted plane.

    - volume_ml: sum over food pixels of height_above_plane * per-pixel footprint
      area, where footprint = (z / FOCAL_PX)^2 (pinhole model), converted to mL.
      "Food" pixels are those with height_above_plane > HEIGHT_NOISE_FLOOR_M (3 mm);
      this also naturally excludes the surrounding table/plate (which sits ~0 mm
      above the fitted plane by construction).
    - max_height_mm: tallest food pixel above the plane, in millimeters.
    - coverage_cm2: total footprint area of food pixels, in cm^2.
    """
    rows, cols, z, height = _height_above_plane(depth, plane)
    footprint_m2 = (z / FOCAL_PX) ** 2

    food_mask = height > HEIGHT_NOISE_FLOOR_M
    food_heights = np.clip(height[food_mask], 0.0, None)
    food_footprints = footprint_m2[food_mask]

    volume_m3 = float(np.sum(food_heights * food_footprints))
    volume_ml = volume_m3 * 1.0e6

    max_height_mm = float(np.max(food_heights) * 1000.0) if food_heights.size else 0.0
    coverage_cm2 = float(np.sum(food_footprints) * 1.0e4)

    return {
        "volume_ml": volume_ml,
        "max_height_mm": max_height_mm,
        "coverage_cm2": coverage_cm2,
    }


# ---------------------------------------------------------------------------
# depth_to_height_image
# ---------------------------------------------------------------------------

def depth_to_height_image(depth: np.ndarray, plane) -> Image.Image:
    """
    Render a height-above-plane map as an 8-bit grayscale PIL image (variant A input).

    0   -> at or below the fitted plane (or invalid depth)
    255 -> at or above HEIGHT_IMAGE_MAX_MM (60 mm) above the plane
    Linear in between.
    """
    h, w = depth.shape
    rows, cols, _z, height = _height_above_plane(depth, plane)

    height_mm_full = np.zeros((h, w), dtype=np.float64)
    height_mm_full[rows, cols] = height * 1000.0
    height_mm_full = np.clip(height_mm_full, 0.0, HEIGHT_IMAGE_MAX_MM)

    img_arr = np.round(height_mm_full / HEIGHT_IMAGE_MAX_MM * 255.0).astype(np.uint8)
    return Image.fromarray(img_arr, mode="L")


# ---------------------------------------------------------------------------
# Sanity harness (also used by tests/test_depth_features.py)
# ---------------------------------------------------------------------------

def _iter_valid_depth_dishes():
    """
    Yield dish_ids under Nutrition5K/imagery/realsense_overhead/ that have a
    depth_raw.png whose raw median falls in a plausible range (guards against the
    one known-corrupt file described in the module docstring).
    """
    for depth_path in sorted(glob.glob(str(REALSENSE_OVERHEAD_DIR / "*" / "depth_raw.png"))):
        dish_id = Path(depth_path).parent.name
        try:
            raw = np.array(Image.open(depth_path))
        except Exception:
            # e.g. a zero-byte / truncated file (one such case is known to exist
            # in Nutrition5K: dish_1564159636/depth_raw.png). Skip it.
            continue
        valid = raw[raw > 0]
        if valid.size == 0:
            continue
        median_raw = float(np.median(valid))
        # Plausible table distance is roughly 0.2m-0.6m -> raw 2000-6000.
        if not (1000.0 <= median_raw <= 8000.0):
            continue
        yield dish_id


def _load_nutrition_table() -> dict:
    table = {}
    with open(NUTRITION_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[row["dish_id"]] = row
    return table


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient (no scipy dependency)."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    corr_matrix = np.corrcoef(x, y)
    return float(corr_matrix[0, 1])


def run_sanity_harness(n_dishes: int = 20, seed: int = 42, verbose: bool = True) -> dict:
    """
    Join food_stats(volume_ml) against dish_nutrition_values.csv mass for a random
    sample of dishes with valid overhead depth, report Pearson r, and print a table.

    Returns {"r": float, "rows": [(dish_id, volume_ml, max_height_mm, mass_g), ...]}.
    """
    nutrition = _load_nutrition_table()
    candidate_dishes = [d for d in _iter_valid_depth_dishes() if d in nutrition]

    rng = random.Random(seed)
    sample = candidate_dishes[:]
    rng.shuffle(sample)
    sample = sample[:n_dishes]

    rows = []
    for dish_id in sample:
        depth_path = REALSENSE_OVERHEAD_DIR / dish_id / "depth_raw.png"
        depth = load_depth(depth_path)
        plane = plane_fit(depth)
        stats = food_stats(depth, plane)
        mass_g = float(nutrition[dish_id]["mass"])
        rows.append((dish_id, stats["volume_ml"], stats["max_height_mm"], mass_g))

    volumes = [r[1] for r in rows]
    masses = [r[3] for r in rows]
    r = pearson_r(volumes, masses)

    if verbose:
        print(f"Sanity harness: {len(rows)} dishes, volume_ml vs mass Pearson r = {r:.3f}")
        print(f"{'dish_id':<18}{'volume_ml':>12}{'max_height_mm':>16}{'mass_g':>10}")
        for dish_id, volume_ml, max_height_mm, mass_g in rows:
            print(f"{dish_id:<18}{volume_ml:>12.1f}{max_height_mm:>16.1f}{mass_g:>10.1f}")

    return {"r": r, "rows": rows}


if __name__ == "__main__":
    run_sanity_harness(n_dishes=20, seed=42, verbose=True)
