from scale_occupancy_audit import (
    _correlation,
    _rectangle_union_area,
    _transform_point,
)


def test_rectangle_union_area_handles_overlap_and_clipping():
    area = _rectangle_union_area(
        [
            (-0.1, 0.0, 0.6, 0.5),
            (0.4, 0.0, 1.1, 0.5),
        ]
    )
    assert abs(area - 0.5) < 1e-9


def test_correlation_reports_linear_and_rank_response():
    result = _correlation([1, 2, 3], [2, 4, 6])
    assert abs(result["pearson_r"] - 1) < 1e-9
    assert abs(result["spearman_rho"] - 1) < 1e-9


def test_geometry_transform_center_crop_and_letterbox():
    # Portrait image: center crop removes vertical context; letterbox pads width.
    center = _transform_point(
        50, 100, width=100, height=200, mode="center_crop"
    )
    letterbox = _transform_point(
        50, 100, width=100, height=200, mode="letterbox"
    )
    assert center == (112.0, 112.0)
    assert letterbox == (112.0, 112.0)
