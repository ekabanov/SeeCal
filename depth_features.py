"""
depth_features.py
------------------
Depth-map utilities for the Nutrition5K RealSense overhead captures.

Provides the primitives needed by the depth-input execution plan
(docs/plans/2026-07-26-depth-plan.md, Task D1), as corrected by the research pass
in docs/design/2026-07-26-depth-design-brief.md (that brief is the authority for
every constant/algorithm choice below -- see its section (e) checklist):

  - load_depth(path)               -> meters, HxW float array, NaN = invalid
  - plane_fit(depth)                -> (normal, d, inlier_mask)  support-surface plane
  - food_stats(depth, plane)        -> {"volume_ml", "max_height_mm", "coverage_cm2"}
  - depth_to_height_image(depth, plane) -> 8-bit grayscale PIL.Image (variant A input)

Unit-scale finding (empirical, see docs/plans/2026-07-26-depth-plan.md Task D1 and
tests/test_depth_features.py::test_plate_mode_depth_matches_rig_height):
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

GLASS PLATFORM TRAP (why the reference surface is NOT a border-pixel plane fit):
  The dish sits on a glass platform; IR depth sees THROUGH the glass to the table
  ~4.8 cm below (border median ~0.407 m). The true capture plane the food actually
  rests on (~0.359 m) is invisible at the image border. A border-ring plane fit
  therefore references the wrong surface and inflates every height by ~48 mm, which
  is a measured failure mode (1.6-3.3 L volumes; see design brief section (a)).
  The fix is to find the support surface from the CENTRAL disk (where the plate/food
  sits, not the glass rim) via a histogram mode restricted to a z-range that excludes
  the below-glass table -- see plane_fit() below.

Pinhole camera model:
  No RealSense intrinsics are published for this rig. The Nutrition5k paper reports
  a per-pixel footprint area of 5.957e-3 cm^2 at the measured plate-mode depth of
  0.359 m; solving the pinhole footprint formula (z/f)^2 for f at that depth/area
  gives f = 465.1 px (this is the D435 COLOR-sensor focal length at 640x480; the
  previously used 616 px was the D415 value -- a wrong-camera mixup that biased
  every volume by a factor of (616/465.1)^2 =~ 1.75). f = 465.1 is used for both the
  per-pixel footprint calculation (food_stats) and the pixel -> 3D unprojection used
  for the plane fit (plane_fit). c = (w/2, h/2); depth is RGB-aligned.
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

# Empirically determined (see module docstring + test_plate_mode_depth_matches_rig_height):
# raw depth units are 1e-4 m (0.1 mm), i.e. meters = raw_uint16 / 10000.
DEPTH_SCALE_M_PER_UNIT = 1.0 / 10000.0

# Invalid raw depth: 0 (no reading) OR > 4200 (98.5% of dishes carry >4,200-unit
# speckle noise well beyond any plausible table/food depth -- see design brief (a)).
RAW_INVALID_MAX = 4200

# f = 465.1 px, derived from the Nutrition5k paper's published per-pixel footprint
# area (5.957e-3 cm^2 @ 0.359 m) via the pinhole footprint formula -- see module
# docstring. This is the D435 color-sensor focal length at 640x480 (NOT the D415's
# ~616 px, which was used previously and biased every volume by ~1.75x).
FOCAL_PX = 465.1

# --- Support-surface reference (replaces the old border-ring RANSAC plane fit) ---
# The dish sits on a glass platform that IR depth sees through to the table below,
# so the true capture plane must be found from the CENTRAL disk, not the border
# (see GLASS PLATFORM TRAP in the module docstring).
CENTRAL_DISK_RADIUS_PX = 240.0

# z-range used when building the histogram used to locate the support-surface
# mode. The upper cut (0.398 m) is essential: without it the mode snaps to the
# below-glass table (~0.407 m) and volumes explode. Median plate-mode depth
# measured at 0.358 m.
REFERENCE_MODE_ZMIN_M = 0.30
REFERENCE_MODE_ZMAX_M = 0.398
REFERENCE_MODE_BIN_M = 0.001  # 1 mm histogram bins

# Robust plane fit: least-squares fit over central-disk pixels within this
# distance of the histogram mode. Constant-z suffices on Nutrition5k (table std
# 3.3 mm); the least-squares fit path exists so the same function generalizes to
# handheld-tilt captures (e.g. iPhone).
PLANE_FIT_INLIER_THRESH_M = 0.004  # 4 mm

RANSAC_SEED = 0  # retained for API compatibility; unused (mode-finding is deterministic).

# Food/noise-floor threshold: pixels within this height of the fitted plane are
# considered table/plate, not food.
HEIGHT_NOISE_FLOOR_M = 0.004  # 4 mm

# Upper food-height bound: pixels taller than this above the plane are excluded
# from the food mask (implausible for a plated meal; guards against residual
# speckle/artifact pixels dominating the volume sum).
HEIGHT_FOOD_MAX_M = 0.150  # 150 mm

# Percentile used for max_height_mm (not the true max -- speckle noise can produce
# a handful of implausibly tall outlier pixels; p99 is a robust "tallest" estimate).
MAX_HEIGHT_PERCENTILE = 99.0

# depth_to_height_image scaling: 0 mm (plane) -> pixel 0, 120 mm+ -> pixel 255.
# (Not 60 mm: p99 heights hit 113 mm across the dataset, so 60 mm saturates
# 10-15% of dishes.)
HEIGHT_IMAGE_MAX_MM = 120.0

# depth_to_height_image output resolution (variant A input) -- full 640x480 res
# roughly doubles VLM sequence length for no information gain.
HEIGHT_IMAGE_SIZE = (320, 240)  # (width, height), PIL convention

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

    Returns a float64 HxW array in meters. Raw value 0 (no reading) OR > 4200
    (speckle noise -- see RAW_INVALID_MAX) is mapped to NaN so downstream code can
    distinguish "no depth" from a real (small) depth reading.
    """
    raw = np.array(Image.open(path))
    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16)
    depth_m = raw.astype(np.float64) * DEPTH_SCALE_M_PER_UNIT
    invalid = (raw == 0) | (raw > RAW_INVALID_MAX)
    depth_m[invalid] = np.nan
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

def _central_disk_mask(shape: tuple[int, int]) -> np.ndarray:
    """HxW bool mask, True within CENTRAL_DISK_RADIUS_PX of the image center."""
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    rows, cols = np.indices((h, w))
    r = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    return r < CENTRAL_DISK_RADIUS_PX


def plane_fit(depth: np.ndarray, seed: int = RANSAC_SEED):
    """
    Fit the support-surface plane (the surface the dish/food actually rests on).

    Reference-surface finding (see GLASS PLATFORM TRAP in the module docstring for
    why this is NOT a border-pixel plane fit): pixels are restricted to a central
    disk (the plate/food sits in the middle of the frame, away from the glass rim
    where depth sees through to the table below). Within that disk, only depths in
    [REFERENCE_MODE_ZMIN_M, REFERENCE_MODE_ZMAX_M] are histogrammed (1mm bins); the
    upper cut excludes the below-glass table so the mode lands on the true capture
    plane. The plane itself is then a robust least-squares fit over central-disk
    pixels within PLANE_FIT_INLIER_THRESH_M (4mm) of that histogram mode.

    Returns (normal, d, inlier_mask):
      - normal: unit np.ndarray shape (3,), the plane's unit normal in camera space.
      - d: float, plane offset such that for a point p=(X,Y,Z), normal . p + d == 0
           for points exactly on the plane.
      - inlier_mask: HxW bool array, True for the central-disk, near-mode pixels
        used in the robust fit (False everywhere else).

    Sign convention: normal . p + d is the signed perpendicular distance (meters)
    from point p to the plane, positive when p is displaced from the plane in the
    same direction as `normal`. Because the fit orients the plane as Z = A*X + B*Y + C
    with normal = normalize([A, B, -1]), `normal` points back toward the camera
    (negative Z component), so height_above_plane(p) = normal . p + d is positive
    for food (which sits between the camera and the table, i.e. at smaller Z).

    `seed` is accepted for API compatibility with the previous RANSAC-based
    implementation but is unused: mode-finding is deterministic.
    """
    h, w = depth.shape
    valid_mask = ~np.isnan(depth)
    central_mask = _central_disk_mask(depth.shape)

    range_mask = (depth >= REFERENCE_MODE_ZMIN_M) & (depth <= REFERENCE_MODE_ZMAX_M)
    mode_candidate_mask = valid_mask & central_mask & range_mask
    mode_z = depth[mode_candidate_mask]
    if mode_z.size < 3:
        raise ValueError(
            "plane_fit: fewer than 3 valid central-disk pixels in "
            f"[{REFERENCE_MODE_ZMIN_M}, {REFERENCE_MODE_ZMAX_M}] m to locate the "
            "support-surface mode"
        )

    n_bins = max(1, int(round((REFERENCE_MODE_ZMAX_M - REFERENCE_MODE_ZMIN_M) / REFERENCE_MODE_BIN_M)))
    counts, edges = np.histogram(mode_z, bins=n_bins, range=(REFERENCE_MODE_ZMIN_M, REFERENCE_MODE_ZMAX_M))
    best_bin = int(np.argmax(counts))
    mode_value = float((edges[best_bin] + edges[best_bin + 1]) / 2.0)

    inlier_mask = valid_mask & central_mask & (np.abs(depth - mode_value) <= PLANE_FIT_INLIER_THRESH_M)
    rows, cols = np.nonzero(inlier_mask)
    if rows.size < 3:
        raise RuntimeError("plane_fit: fewer than 3 pixels within the robust-fit window of the mode")

    z = depth[rows, cols]
    x, y, zc = _unproject(rows, cols, z, depth.shape)
    pts = np.stack([x, y, zc], axis=1)  # (N, 3)

    # Least-squares fit of Z = A*X + B*Y + C over the near-mode inliers. This form
    # is valid because the rig looks straight down at the table, so the plane is
    # never (near-)vertical in camera space. On Nutrition5k this is effectively a
    # constant-z fit (table std 3.3mm); the fit path exists so this function
    # generalizes to handheld-tilt captures.
    design = np.stack([pts[:, 0], pts[:, 1], np.ones(len(pts))], axis=1)
    target = pts[:, 2]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    a, b, c = coeffs
    norm_factor = np.sqrt(a ** 2 + b ** 2 + 1.0)
    normal = np.array([a, b, -1.0]) / norm_factor
    d = c / norm_factor

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
      "Food" pixels are those within the central disk AND with height_above_plane
      in (HEIGHT_NOISE_FLOOR_M, HEIGHT_FOOD_MAX_M] = (4 mm, 150 mm]; this excludes
      the surrounding table/plate (~0 mm above the fitted plane by construction),
      the glass-rim border region (never a food location), and implausibly tall
      outlier/speckle pixels.
    - max_height_mm: p99 (not true max -- speckle can produce a few implausibly
      tall outlier pixels) of food-pixel heights above the plane, in millimeters.
    - coverage_cm2: total footprint area of food pixels, in cm^2.
    """
    rows, cols, z, height = _height_above_plane(depth, plane)
    footprint_m2 = (z / FOCAL_PX) ** 2

    central = _central_disk_mask(depth.shape)[rows, cols]
    food_mask = central & (height > HEIGHT_NOISE_FLOOR_M) & (height <= HEIGHT_FOOD_MAX_M)
    food_heights = np.clip(height[food_mask], 0.0, None)
    food_footprints = footprint_m2[food_mask]

    volume_m3 = float(np.sum(food_heights * food_footprints))
    volume_ml = volume_m3 * 1.0e6

    max_height_mm = (
        float(np.percentile(food_heights, MAX_HEIGHT_PERCENTILE) * 1000.0)
        if food_heights.size
        else 0.0
    )
    coverage_cm2 = float(np.sum(food_footprints) * 1.0e4)

    return {
        "volume_ml": volume_ml,
        "max_height_mm": max_height_mm,
        "coverage_cm2": coverage_cm2,
    }


# ---------------------------------------------------------------------------
# depth_to_height_image
# ---------------------------------------------------------------------------

def _height_image_full_res(depth: np.ndarray, plane) -> np.ndarray:
    """
    Native-resolution (depth.shape) uint8 height-above-plane render.

    0   -> at or below the fitted plane (or invalid depth)
    255 -> at or above HEIGHT_IMAGE_MAX_MM (120 mm) above the plane
    Linear in between. Split out from depth_to_height_image() so the pixel-level
    mapping can be tested independently of the final output-size resize.
    """
    h, w = depth.shape
    rows, cols, _z, height = _height_above_plane(depth, plane)

    height_mm_full = np.zeros((h, w), dtype=np.float64)
    height_mm_full[rows, cols] = height * 1000.0
    height_mm_full = np.clip(height_mm_full, 0.0, HEIGHT_IMAGE_MAX_MM)

    return np.round(height_mm_full / HEIGHT_IMAGE_MAX_MM * 255.0).astype(np.uint8)


def depth_to_height_image(depth: np.ndarray, plane) -> Image.Image:
    """
    Render a height-above-plane map as an 8-bit grayscale PIL image (variant A input).

    0   -> at or below the fitted plane (or invalid depth)
    255 -> at or above HEIGHT_IMAGE_MAX_MM (120 mm) above the plane
    Linear in between. Output is resized to HEIGHT_IMAGE_SIZE (~320x240): full
    640x480 resolution roughly doubles VLM sequence length for no information gain.
    """
    img_arr = _height_image_full_res(depth, plane)
    img = Image.fromarray(img_arr, mode="L")
    return img.resize(HEIGHT_IMAGE_SIZE, resample=Image.BILINEAR)


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
    sample of dishes with valid overhead depth, report Pearson r and implied median
    density (mass_g / volume_ml), and print a table.

    The density figure is the decisive check (design brief section (b).6): a biased
    reference plane can still produce a plausible-looking r while the implied
    density is nonsense (e.g. the border-fit bug produced r=0.57 with density
    0.06 g/ml). r alone does not discriminate correct from correlated-but-biased.

    Returns {"r": float, "density_g_per_ml": float,
             "rows": [(dish_id, volume_ml, max_height_mm, mass_g), ...]}.
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
    densities = [m / v for v, m in zip(volumes, masses) if v > 0]
    density = float(np.median(densities)) if densities else float("nan")

    if verbose:
        print(
            f"Sanity harness: {len(rows)} dishes, volume_ml vs mass Pearson r = {r:.3f}, "
            f"median implied density = {density:.3f} g/ml"
        )
        print(f"{'dish_id':<18}{'volume_ml':>12}{'max_height_mm':>16}{'mass_g':>10}")
        for dish_id, volume_ml, max_height_mm, mass_g in rows:
            print(f"{dish_id:<18}{volume_ml:>12.1f}{max_height_mm:>16.1f}{mass_g:>10.1f}")

    return {"r": r, "density_g_per_ml": density, "rows": rows}


if __name__ == "__main__":
    run_sanity_harness(n_dishes=20, seed=42, verbose=True)
