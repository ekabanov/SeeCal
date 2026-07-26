"""
tests/test_depth_features.py
-----------------------------
Tests for depth_features.py (Task D1 of docs/plans/2026-07-26-depth-plan.md, as
corrected by docs/design/2026-07-26-depth-design-brief.md).

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

# Design brief section (e), checklist item 1: plate-mode depth must land in this
# range (not the below-glass table's ~0.407 m, which a border-ring fit would find).
PLATE_MODE_MIN_M = 0.33
PLATE_MODE_MAX_M = 0.37

# Design brief section (b).6 / (e) checklist item 6: the decisive test that
# discriminates a correct reference plane from a correlated-but-biased one.
DENSITY_MIN_G_PER_ML = 0.3
DENSITY_MAX_G_PER_ML = 1.2

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
# Support-surface reference pin: this is the empirical decision from the design
# brief's section (b)/(e) -- central-disk histogram mode, NOT a border-ring fit.
# ---------------------------------------------------------------------------

def test_plate_mode_depth_matches_rig_height(good_depths):
    """
    Pins the support-surface reference used by plane_fit(): the central-disk
    histogram mode (restricted to z in [0.30, 0.398] m) must land on the true
    capture plane, in [0.33, 0.37] m -- NOT the below-glass table (~0.407 m) that
    a border-ring plane fit finds (see the GLASS PLATFORM TRAP module docstring).

    Checklist item 1 (design brief section (e)): asserted per-dish over >= 5
    dishes, replacing the old border-based, +/-0.05 m tolerance check.
    """
    assert len(good_depths) >= 5
    plate_depths = []
    for dish_id, depth in good_depths.items():
        normal, d, _inlier_mask = df.plane_fit(depth)
        # normal is unit-length and, for the near-flat Nutrition5k table, points
        # along -Z, so |normal . 0 + d| reduces to the plane's Z depth (see
        # plane_fit's sign-convention docstring).
        plate_depth = abs(float(normal @ np.zeros(3) + d))
        plate_depths.append(plate_depth)
        assert PLATE_MODE_MIN_M <= plate_depth <= PLATE_MODE_MAX_M, (
            f"{dish_id}: plate-mode depth {plate_depth:.4f} m outside "
            f"[{PLATE_MODE_MIN_M}, {PLATE_MODE_MAX_M}] m"
        )

    overall_median = float(np.median(plate_depths))
    assert overall_median == pytest.approx(RIG_HEIGHT_M, abs=0.03), (
        f"median plate-mode depth {overall_median:.4f} m is not close to rig "
        f"height {RIG_HEIGHT_M} m -- per-dish depths: {plate_depths}"
    )


# ---------------------------------------------------------------------------
# load_depth
# ---------------------------------------------------------------------------

def test_load_depth_shape_and_dtype(good_depths):
    for dish_id, depth in good_depths.items():
        assert depth.shape == (480, 640), dish_id
        assert depth.dtype == np.float64, dish_id


def test_load_depth_invalid_pixels_are_nan(good_depths):
    """raw == 0 (no reading) maps to NaN; ordinary in-range readings do not."""
    depth = next(iter(good_depths.values()))
    raw = np.array(Image.open(_depth_path(next(iter(good_depths)))))
    zero_mask = raw == 0
    if zero_mask.any():
        assert np.all(np.isnan(depth[zero_mask]))
    in_range_mask = (raw != 0) & (raw <= df.RAW_INVALID_MAX)
    assert np.all(~np.isnan(depth[in_range_mask]))


def test_load_depth_speckle_cut_is_nan(good_depths):
    """raw > RAW_INVALID_MAX (4200, speckle noise) also maps to NaN."""
    found_any_speckle = False
    for dish_id, depth in good_depths.items():
        raw = np.array(Image.open(_depth_path(dish_id)))
        speckle_mask = raw > df.RAW_INVALID_MAX
        if speckle_mask.any():
            found_any_speckle = True
            assert np.all(np.isnan(depth[speckle_mask])), dish_id
    # Per the design brief, 98.5% of dishes carry some >4200 speckle, so this
    # should be true for at least one of the fixture dishes.
    assert found_any_speckle, "expected at least one fixture dish with >4200 speckle"


def test_load_depth_invalid_cut_synthetic(tmp_path):
    """Exact boundary behavior of the raw==0 OR raw>4200 invalid cut."""
    raw = np.array([[0, 100, 4200, 4201, 65535]], dtype=np.uint16)
    path = tmp_path / "synthetic_depth.png"
    Image.fromarray(raw).save(path)
    depth = df.load_depth(path)
    assert np.isnan(depth[0, 0]), "raw == 0 must be invalid"
    assert not np.isnan(depth[0, 1]), "ordinary in-range raw must be valid"
    assert not np.isnan(depth[0, 2]), "raw == 4200 (the boundary) must be valid"
    assert np.isnan(depth[0, 3]), "raw == 4201 (just past the boundary) must be invalid"
    assert np.isnan(depth[0, 4]), "raw == 65535 (speckle) must be invalid"


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


def test_plane_fit_inliers_are_within_central_disk(good_depths):
    """The robust-fit inlier set must come entirely from the central disk."""
    depth = next(iter(good_depths.values()))
    _normal, _d, inlier_mask = df.plane_fit(depth)
    central = df._central_disk_mask(depth.shape)
    assert np.all(central[inlier_mask])


def test_plane_fit_inliers_are_near_zero_height(good_depths):
    """
    Points plane_fit calls inliers (within PLANE_FIT_INLIER_THRESH_M of the
    histogram mode) should sit close to the final fitted plane. Some slack beyond
    PLANE_FIT_INLIER_THRESH_M is expected: the returned plane is a least-squares
    refit over the near-mode inlier set, which shifts the plane slightly from the
    per-pixel mode-distance criterion used to select those inliers, so a handful
    end up a bit further from the final (refined) plane.
    """
    depth = next(iter(good_depths.values()))
    normal, d, inlier_mask = df.plane_fit(depth)
    rows, cols = np.nonzero(inlier_mask)
    z = depth[rows, cols]
    x, y, zc = df._unproject(rows, cols, z, depth.shape)
    height = normal[0] * x + normal[1] * y + normal[2] * zc + d
    assert np.max(np.abs(height)) < 2 * df.PLANE_FIT_INLIER_THRESH_M


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
        # A dinner-plate-sized meal shouldn't exceed a few liters or ~15cm tall
        # (food_stats caps the food mask at HEIGHT_FOOD_MAX_M = 150mm).
        assert stats["volume_ml"] < 5000.0, dish_id
        assert stats["max_height_mm"] <= 150.0, dish_id
        assert stats["coverage_cm2"] < 2000.0, dish_id


def test_food_stats_max_height_is_not_raw_max():
    """
    max_height_mm must be the p99 of food-pixel heights, not the true max --
    a single speckle outlier pixel should not move it (design brief checklist
    item 7).
    """
    for dish_id in GOOD_DISH_IDS:
        depth = df.load_depth(_depth_path(dish_id))
        plane = df.plane_fit(depth)
        rows, cols, z, height = df._height_above_plane(depth, plane)
        central = df._central_disk_mask(depth.shape)[rows, cols]
        food_mask = central & (height > df.HEIGHT_NOISE_FLOOR_M) & (height <= df.HEIGHT_FOOD_MAX_M)
        food_heights_mm = height[food_mask] * 1000.0
        if food_heights_mm.size == 0:
            continue
        expected_p99 = float(np.percentile(food_heights_mm, df.MAX_HEIGHT_PERCENTILE))
        stats = df.food_stats(depth, plane)
        assert stats["max_height_mm"] == pytest.approx(expected_p99, abs=1e-6)
        # p99 must never exceed the true max.
        assert stats["max_height_mm"] <= float(np.max(food_heights_mm)) + 1e-6
        return
    pytest.fail("no fixture dish produced a non-empty food mask")


# ---------------------------------------------------------------------------
# depth_to_height_image
# ---------------------------------------------------------------------------

def test_depth_to_height_image_basic(good_depths):
    depth = next(iter(good_depths.values()))
    plane = df.plane_fit(depth)
    img = df.depth_to_height_image(depth, plane)
    assert isinstance(img, Image.Image)
    assert img.mode == "L"
    assert img.size == df.HEIGHT_IMAGE_SIZE  # ~320x240, not native 640x480.
    arr = np.array(img)
    assert arr.dtype == np.uint8
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_depth_to_height_image_plane_pixels_are_zero(good_depths):
    """
    Pixels the plane fit calls inliers should render as (near) 0 in the
    NATIVE-resolution height render (_height_image_full_res), before the final
    resize to HEIGHT_IMAGE_SIZE.
    """
    depth = next(iter(good_depths.values()))
    plane = df.plane_fit(depth)
    _normal, _d, inlier_mask = plane
    img_arr = df._height_image_full_res(depth, plane)
    inlier_values = img_arr[inlier_mask]
    # Inliers are within roughly PLANE_FIT_INLIER_THRESH_M (4mm, plus refit slack
    # -- see test_plane_fit_inliers_are_near_zero_height) of the plane; at the
    # 0-120mm mapping that's a small slice of the 0-255 range.
    max_expected = round(2 * df.PLANE_FIT_INLIER_THRESH_M * 1000.0 / df.HEIGHT_IMAGE_MAX_MM * 255.0) + 1
    assert inlier_values.max() <= max_expected


def test_depth_to_height_image_saturates_at_120mm(good_depths):
    depth = next(iter(good_depths.values())).copy()
    plane = df.plane_fit(depth)
    normal, d, _inlier_mask = plane
    # Fabricate a pixel far above the plane (200mm) to check saturation to 255.
    h, w = depth.shape
    row, col = h // 2, w // 2
    # Solve for a Z that gives height ~= 0.20m at this pixel's (X, Y) ray.
    # height = normal[0]*X + normal[1]*Y + normal[2]*Z + d, X=(col-cx)*Z/f, Y=(row-cy)*Z/f
    cx, cy = w / 2.0, h / 2.0
    a = normal[0] * (col - cx) / df.FOCAL_PX + normal[1] * (row - cy) / df.FOCAL_PX + normal[2]
    # Solve a*Z + d = 0.20 -> Z = (0.20 - d) / a
    target_height = 0.20
    z_new = (target_height - d) / a
    depth[row, col] = z_new
    img_arr = df._height_image_full_res(depth, plane)
    assert img_arr[row, col] == 255


# ---------------------------------------------------------------------------
# Sanity harness: volume_ml vs dish mass correlation + density (regression floor).
# ---------------------------------------------------------------------------

def test_sanity_harness_correlation_regression_floor():
    result = df.run_sanity_harness(n_dishes=20, seed=42, verbose=True)
    assert len(result["rows"]) >= 15, "too few dishes had both depth and nutrition data"
    assert result["r"] > 0.3, f"volume_ml vs mass correlation too low: r={result['r']:.3f}"


def test_sanity_harness_density_regression_floor():
    """
    Design brief section (b).6 / (e) checklist item 6: the decisive test. r alone
    passes even on the buggy border-fit reference (r=0.57 with implied density
    0.06 g/ml, i.e. off by ~10x from real food density); the median implied
    density (mass_g / volume_ml) over >= 20 dishes catches that a correct
    correlation still requires a physically plausible reference surface.
    """
    result = df.run_sanity_harness(n_dishes=20, seed=42, verbose=True)
    assert len(result["rows"]) >= 15, "too few dishes had both depth and nutrition data"
    density = result["density_g_per_ml"]
    assert DENSITY_MIN_G_PER_ML <= density <= DENSITY_MAX_G_PER_ML, (
        f"median implied density {density:.3f} g/ml outside plausible "
        f"[{DENSITY_MIN_G_PER_ML}, {DENSITY_MAX_G_PER_ML}] g/ml range -- "
        "reference plane is likely biased (see GLASS PLATFORM TRAP)"
    )
