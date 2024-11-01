import math

import numpy as np
import pytest

from mapa.algorithm import (
    ModelSize,
    _compute_triangles_of_3d_surface,
    _compute_triangles_of_body_side,
    _compute_triangles_of_bottom,
    _create_raster,
    compute_all_triangles,
)


def test_create_raster() -> None:
    with pytest.raises(
        ValueError, match="not enough values to unpack \\(expected 2, got 0\\)"
    ):
        one = np.array(1)
        max_x, max_y = one.shape
        _create_raster(one, max_x, max_y)

    two_by_two = np.random.random((2, 2))
    max_x, max_y = two_by_two.shape
    res = _create_raster(two_by_two, max_x, max_y)
    assert res.shape == (3, 3)

    assert res[0, 0] == two_by_two[0, 0]
    assert res[0, 1] == two_by_two[0, 1]
    assert res[1, 0] == two_by_two[1, 0]
    assert res[2, 0] == two_by_two[1, 0]
    assert res[0, 2] == two_by_two[0, 1]
    assert res[2, 2] == two_by_two[1, 1]
    assert res[1, 2] == two_by_two[1, 1]
    assert res[2, 1] == two_by_two[1, 1]

    assert round(res[1, 1], 5) == round(two_by_two.mean(), 5)
    assert math.isclose(res[1, 1], two_by_two.mean())


def test_compute_triangles_of_3d_surface() -> None:
    array = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    max_x, max_y = array.shape
    raster = _create_raster(array, max_x, max_y)
    max_x, max_y = array.shape
    model_size = 100
    x_scale = model_size / max_x
    y_scale = model_size / max_x
    z_scale = model_size / 6000
    triangles = _compute_triangles_of_3d_surface(
        raster=raster,
        array=array,
        max_x=max_x,
        max_y=max_y,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        z_offset=0.0,
    )
    expected = np.array(
        [
            [
                [2.50000000e01, 2.50000000e01, 1.66666667e-02],
                [0.00000000e00, 0.00000000e00, 1.66666667e-02],
                [5.00000000e01, 0.00000000e00, 5.00000000e-02],
            ],
            [
                [0.00000000e00, 5.00000000e01, 3.33333333e-02],
                [0.00000000e00, 0.00000000e00, 1.66666667e-02],
                [2.50000000e01, 2.50000000e01, 1.66666667e-02],
            ],
            [
                [5.00000000e01, 5.00000000e01, 4.16666667e-02],
                [0.00000000e00, 5.00000000e01, 3.33333333e-02],
                [2.50000000e01, 2.50000000e01, 1.66666667e-02],
            ],
            [
                [2.50000000e01, 2.50000000e01, 1.66666667e-02],
                [5.00000000e01, 0.00000000e00, 5.00000000e-02],
                [5.00000000e01, 5.00000000e01, 4.16666667e-02],
            ],
            [
                [2.50000000e01, 7.50000000e01, 3.33333333e-02],
                [0.00000000e00, 5.00000000e01, 3.33333333e-02],
                [5.00000000e01, 5.00000000e01, 4.16666667e-02],
            ],
            [
                [0.00000000e00, 1.00000000e02, 3.33333333e-02],
                [0.00000000e00, 5.00000000e01, 3.33333333e-02],
                [2.50000000e01, 7.50000000e01, 3.33333333e-02],
            ],
            [
                [5.00000000e01, 1.00000000e02, 6.66666667e-02],
                [0.00000000e00, 1.00000000e02, 3.33333333e-02],
                [2.50000000e01, 7.50000000e01, 3.33333333e-02],
            ],
            [
                [2.50000000e01, 7.50000000e01, 3.33333333e-02],
                [5.00000000e01, 5.00000000e01, 4.16666667e-02],
                [5.00000000e01, 1.00000000e02, 6.66666667e-02],
            ],
            [
                [7.50000000e01, 2.50000000e01, 5.00000000e-02],
                [5.00000000e01, 0.00000000e00, 5.00000000e-02],
                [1.00000000e02, 0.00000000e00, 5.00000000e-02],
            ],
            [
                [5.00000000e01, 5.00000000e01, 4.16666667e-02],
                [5.00000000e01, 0.00000000e00, 5.00000000e-02],
                [7.50000000e01, 2.50000000e01, 5.00000000e-02],
            ],
            [
                [1.00000000e02, 5.00000000e01, 6.66666667e-02],
                [5.00000000e01, 5.00000000e01, 4.16666667e-02],
                [7.50000000e01, 2.50000000e01, 5.00000000e-02],
            ],
            [
                [7.50000000e01, 2.50000000e01, 5.00000000e-02],
                [1.00000000e02, 0.00000000e00, 5.00000000e-02],
                [1.00000000e02, 5.00000000e01, 6.66666667e-02],
            ],
            [
                [7.50000000e01, 7.50000000e01, 6.66666667e-02],
                [5.00000000e01, 5.00000000e01, 4.16666667e-02],
                [1.00000000e02, 5.00000000e01, 6.66666667e-02],
            ],
            [
                [5.00000000e01, 1.00000000e02, 6.66666667e-02],
                [5.00000000e01, 5.00000000e01, 4.16666667e-02],
                [7.50000000e01, 7.50000000e01, 6.66666667e-02],
            ],
            [
                [1.00000000e02, 1.00000000e02, 6.66666667e-02],
                [5.00000000e01, 1.00000000e02, 6.66666667e-02],
                [7.50000000e01, 7.50000000e01, 6.66666667e-02],
            ],
            [
                [7.50000000e01, 7.50000000e01, 6.66666667e-02],
                [1.00000000e02, 5.00000000e01, 6.66666667e-02],
                [1.00000000e02, 1.00000000e02, 6.66666667e-02],
            ],
        ]
    )
    np.testing.assert_array_almost_equal(expected, np.array(triangles))


def test__compute_triangles_of_body_side() -> None:
    # note, this is just a smoke test to detect changes in the output of the function
    array = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    max_x, max_y = array.shape
    raster = _create_raster(array, max_x, max_y)
    side_triangles = _compute_triangles_of_body_side(
        raster=raster,
        max_x=max_x,
        max_y=max_y,
        x_scale=1.0,
        y_scale=1.0,
        z_scale=2.0,
        z_offset=3.0,
    )
    expected = np.array(
        [
            [[0.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 7.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 9.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 9.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 7.0], [0.0, 2.0, 7.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 7.0], [0.0, 2.0, 0.0]],
            [[0.0, 2.0, 7.0], [1.0, 2.0, 11.0], [0.0, 2.0, 0.0]],
            [[0.0, 2.0, 0.0], [1.0, 2.0, 11.0], [1.0, 2.0, 0.0]],
            [[2.0, 1.0, 11.0], [2.0, 0.0, 9.0], [2.0, 0.0, 0.0]],
            [[2.0, 1.0, 11.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            [[2.0, 0.0, 9.0], [1.0, 0.0, 9.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 9.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 2.0, 11.0], [2.0, 1.0, 11.0], [2.0, 1.0, 0.0]],
            [[2.0, 2.0, 11.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
            [[1.0, 2.0, 11.0], [2.0, 2.0, 11.0], [1.0, 2.0, 0.0]],
            [[1.0, 2.0, 0.0], [2.0, 2.0, 11.0], [2.0, 2.0, 0.0]],
        ]
    )
    np.testing.assert_array_equal(expected, side_triangles)


def test_compute_all_triangles__min_occurrences() -> None:
    array = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )
    triangles = compute_all_triangles(
        array,
        desired_size=ModelSize(200, 200),
        z_offset=1.0,
        z_scale=2.0,
        elevation_scale=10.0,
    )

    # verify every every point occurs at least 4 times
    unique, counts = np.unique(triangles, return_counts=True)
    occurrences = dict(zip(unique, counts)).values()
    assert min(occurrences) >= 4


def test__compute_triangles_of_bottom() -> None:
    array = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )

    max_x, max_y = array.shape
    bottom_triangles = _compute_triangles_of_bottom(
        max_x=max_x, max_y=max_y, x_scale=1.0, y_scale=1.0
    )

    expected = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 2.0, 0.0]],
            [[2.0, 3.0, 0.0], [3.0, 3.0, 0.0], [3.0, 2.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 3.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [1.0, 3.0, 0.0]],
            [[3.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
            [[3.0, 1.0, 0.0], [2.0, 0.0, 0.0], [3.0, 2.0, 0.0]],
            [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [3.0, 2.0, 0.0]],
            [[1.0, 3.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    np.testing.assert_array_equal(expected, bottom_triangles)
