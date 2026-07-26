"""
tests/test_depth_features.py
-----------------------------
Tests for depth_features.py (Task D1 of docs/plans/2026-07-26-depth-plan.md).

Run with: .venv/bin/python -m pytest tests/test_depth_features.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import depth_features as df

RIG_HEIGHT_M = 0.359  # Nutrition5k paper: RealSense rig height above the table.

# Five dishes known (from manual inspection, see module docstring in
# depth_features.py) to have plausible, non-corrupted depth_raw.png files --
# deliberately skips dish_1556572657 (10x-too-small raw values) and
# dish_1564159636 (zero-byte file).
GOOD_DISH_IDS = [
    "dish_1556573514",
    "dish_1556575014",
    "dish_1556575083",
    "dish_1556575124",
    "dish_1556575273",
]


def _depth_path(dish_id: str) -> Path:
    return df.REALSENSE_OVERHEAD_DIR / dish_id / "depth_raw.png"


@pytest.fixture(scope="module")
def good_depths():
    depths = {}
    for dish_id in GOOD_DISH_IDS:
        path = _depth_path(dish_id)
        assert path.exists(), f"expected fixture dish missing: {path}"
        depths[dish_id] = df.load_depth(path)
    return depths


# ---------------------------------------------------------------------------
# Unit-scale pin: this is the empirical decision from Task D1's step 1.
# ---------------------------------------------------------------------------

def test_border_depth_matches_rig_height(good_depths):
    """
    Pins the DEPTH_SCALE_M_PER_UNIT decision (raw units = 1e-4 m). The outer
    border ring of an overhead depth map is background table, so its median
    depth (in the chosen scale) should be close to the rig height (0.359 m,
    per the Nutrition5k paper).
    """
    border_medians = []
    for dish_id, depth in good_depths.items():
        h, w = depth.shape
        border = max(1, int(df.BORDER_FRAC * min(h, w)))
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:border, :] = True
        border_mask[-border:, :] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True
        valid = border_mask & ~np.isnan(depth)
        assert valid.sum() > 0, f"{dish_id}: no valid border pixels"
        median_m = float(np.median(depth[valid]))
        border_medians.append(median_m)

    overall_median = float(np.median(border_medians))
    assert overall_median == pytest.approx(RIG_HEIGHT_M, abs=0.05), (
        f"median border depth {overall_median:.4f} m is not within 0.05 m of "
        f"rig height {RIG_HEIGHT_M} m -- wrong unit scale? "
        f"per-dish medians: {border_medians}"
    )


# ---------------------------------------------------------------------------
# load_depth
# ---------------------------------------------------------------------------

def test_load_depth_shape_and_dtype(good_depths):
    for dish_id, depth in good_depths.items():
        assert depth.shape == (480, 640), dish_id
        assert depth.dtype == np.float64, dish_id


def test_load_depth_invalid_pixels_are_nan(good_depths):
    depth = next(iter(good_depths.values()))
    raw = np.array(Image.open(_depth_path(next(iter(good_depths)))))
    zero_mask = raw == 0
    if zero_mask.any():
        assert np.all(np.isnan(depth[zero_mask]))
    nonzero_mask = raw != 0
    assert np.all(~np.isnan(depth[nonzero_mask]))


def test_load_depth_plausible_range(good_depths):
    for dish_id, depth in good_depths.items():
        valid = depth[~np.isnan(depth)]
        assert valid.size > 0, dish_id
        # A tabletop overhead rig should be well under 1 meter from the food.
        assert np.nanmedian(valid) < 1.0, dish_id
        assert np.nanmedian(valid) > 0.1, dish_id


# ---------------------------------------------------------------------------
# plane_fit
# ---------------------------------------------------------------------------

def test_plane_fit_returns_expected_shapes(good_depths):
    depth = next(iter(good_depths.values()))
    normal, d, inlier_mask = df.plane_fit(depth)
    assert normal.shape == (3,)
    assert np.isfinite(normal).all()
    assert isinstance(d, float)
    assert inlier_mask.shape == depth.shape
    assert inlier_mask.dtype == bool
    assert inlier_mask.sum() > 0


def test_plane_fit_normal_is_unit_length(good_depths):
    depth = next(iter(good_depths.values()))
    normal, _d, _inlier_mask = df.plane_fit(depth)
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)


def test_plane_fit_inliers_are_near_zero_height(good_depths):
    """
    Points RANSAC calls plane inliers should sit close to the fitted plane. Some
    slack beyond RANSAC_INLIER_THRESH_M is expected: the returned plane is a
    least-squares refit over the RANSAC-selected inlier set, which shifts the
    plane slightly from the exact 3-point model used during the RANSAC search
    itself, so a handful of original inliers end up a bit further from the
    final (refined) plane.
    """
    depth = next(iter(good_depths.values()))
    normal, d, inlier_mask = df.plane_fit(depth)
    rows, cols = np.nonzero(inlier_mask)
    z = depth[rows, cols]
    h, w = depth.shape
    x, y, zc = df._unproject(rows, cols, z, depth.shape)
    height = normal[0] * x + normal[1] * y + normal[2] * zc + d
    assert np.max(np.abs(height)) < 2 * df.RANSAC_INLIER_THRESH_M


def test_plane_fit_table_is_roughly_rig_height_away(good_depths):
    """The fitted plane's distance from the camera origin should be ~ rig height."""
    for dish_id, depth in good_depths.items():
        normal, d, _inlier_mask = df.plane_fit(depth)
        # Signed distance from the camera origin (0,0,0) to the plane.
        dist_from_origin = abs(normal @ np.zeros(3) + d)
        assert dist_from_origin == pytest.approx(RIG_HEIGHT_M, abs=0.1), dish_id


# ---------------------------------------------------------------------------
# food_stats
# ---------------------------------------------------------------------------

def test_food_stats_keys_and_types(good_depths):
    depth = next(iter(good_depths.values()))
    plane = df.plane_fit(depth)
    stats = df.food_stats(depth, plane)
    assert set(stats.keys()) == {"volume_ml", "max_height_mm", "coverage_cm2"}
    for v in stats.values():
        assert isinstance(v, float)
        assert v >= 0.0
        assert np.isfinite(v)


def test_food_stats_plausible_magnitudes(good_depths):
    for dish_id, depth in good_depths.items():
        plane = df.plane_fit(depth)
        stats = df.food_stats(depth, plane)
        # A dinner-plate-sized meal shouldn't exceed a few liters or ~10cm tall.
        assert stats["volume_ml"] < 5000.0, dish_id
        assert stats["max_height_mm"] < 200.0, dish_id
        assert stats["coverage_cm2"] < 2000.0, dish_id


# ---------------------------------------------------------------------------
# depth_to_height_image
# ---------------------------------------------------------------------------

def test_depth_to_height_image_basic(good_depths):
    depth = next(iter(good_depths.values()))
    plane = df.plane_fit(depth)
    img = df.depth_to_height_image(depth, plane)
    assert isinstance(img, Image.Image)
    assert img.mode == "L"
    assert img.size == (depth.shape[1], depth.shape[0])  # PIL size is (W, H)
    arr = np.array(img)
    assert arr.dtype == np.uint8
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_depth_to_height_image_plane_pixels_are_zero(good_depths):
    """Pixels the plane fit calls inliers should render as (near) 0 in the height image."""
    depth = next(iter(good_depths.values()))
    plane = df.plane_fit(depth)
    _normal, _d, inlier_mask = plane
    img_arr = np.array(df.depth_to_height_image(depth, plane))
    inlier_values = img_arr[inlier_mask]
    # Inliers are within roughly RANSAC_INLIER_THRESH_M (4mm, plus refit slack --
    # see test_plane_fit_inliers_are_near_zero_height) of the plane; at the
    # 0-60mm mapping that's a small slice of the 0-255 range.
    max_expected = round(2 * df.RANSAC_INLIER_THRESH_M * 1000.0 / df.HEIGHT_IMAGE_MAX_MM * 255.0) + 1
    assert inlier_values.max() <= max_expected


def test_depth_to_height_image_saturates_at_60mm(good_depths):
    depth = next(iter(good_depths.values())).copy()
    plane = df.plane_fit(depth)
    normal, d, _inlier_mask = plane
    # Fabricate a pixel far above the plane (100mm) to check saturation to 255.
    h, w = depth.shape
    row, col = h // 2, w // 2
    # Solve for a Z that gives height ~= 0.10m at this pixel's (X, Y) ray.
    # height = normal[0]*X + normal[1]*Y + normal[2]*Z + d, X=(col-cx)*Z/f, Y=(row-cy)*Z/f
    cx, cy = w / 2.0, h / 2.0
    a = normal[0] * (col - cx) / df.FOCAL_PX + normal[1] * (row - cy) / df.FOCAL_PX + normal[2]
    # Solve a*Z + d = 0.10 -> Z = (0.10 - d) / a
    target_height = 0.10
    z_new = (target_height - d) / a
    depth[row, col] = z_new
    img_arr = np.array(df.depth_to_height_image(depth, plane))
    assert img_arr[row, col] == 255


# ---------------------------------------------------------------------------
# Sanity harness: volume_ml vs dish mass correlation (regression floor).
# ---------------------------------------------------------------------------

def test_sanity_harness_correlation_regression_floor():
    result = df.run_sanity_harness(n_dishes=20, seed=42, verbose=True)
    assert len(result["rows"]) >= 15, "too few dishes had both depth and nutrition data"
    assert result["r"] > 0.3, f"volume_ml vs mass correlation too low: r={result['r']:.3f}"
