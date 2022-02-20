import math

import numpy as np
import pytest

from mapa.algorithm import _compute_triangles_of_3d_surface, _compute_triangles_of_body_side, _create_raster
from mapa.conf import NO_DATA


def test_create_raster() -> None:
    with pytest.raises(ValueError, match="not enough values to unpack \\(expected 2, got 0\\)"):
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
    target_size = 100
    x_scale = target_size / max_x
    y_scale = target_size / max_x
    z_scale = target_size / 6000
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
        array=array,
        max_x=max_x,
        max_y=max_y,
        x_scale=1.0,
        y_scale=1.0,
        z_scale=2.0,
        z_offset=3.0,
    )
    expected = [
        [[0, 0.0, 5.0], [0, 1.0, 7.0], [0, 0.0, 0]],
        [[0, 0.0, 0], [0, 1.0, 7.0], [0, 1.0, 0]],
        [[1.0, 0, 9.0], [0.0, 0, 5.0], [0.0, 0, 0]],
        [[1.0, 0, 9.0], [0.0, 0, 0], [1.0, 0, 0]],
        [[0, 1.0, 7.0], [0, 2.0, 7.0], [0, 1.0, 0]],
        [[0, 1.0, 0], [0, 2.0, 7.0], [0, 2.0, 0]],
        [[0.0, 2.0, 7.0], [1.0, 2.0, 8.0], [0.0, 2.0, 0]],
        [[0.0, 2.0, 0], [1.0, 2.0, 8.0], [1.0, 2.0, 0]],
        [[2.0, 1.0, 8.0], [2.0, 0.0, 9.0], [2.0, 0.0, 0]],
        [[2.0, 1.0, 8.0], [2.0, 0.0, 0], [2.0, 1.0, 0]],
        [[2.0, 0, 9.0], [1.0, 0, 9.0], [1.0, 0, 0]],
        [[2.0, 0, 9.0], [1.0, 0, 0], [2.0, 0, 0]],
        [[2.0, 2.0, 11.0], [2.0, 1.0, 8.0], [2.0, 1.0, 0]],
        [[2.0, 2.0, 11.0], [2.0, 1.0, 0], [2.0, 2.0, 0]],
        [[1.0, 2.0, 8.0], [2.0, 2.0, 11.0], [1.0, 2.0, 0]],
        [[1.0, 2.0, 0], [2.0, 2.0, 11.0], [2.0, 2.0, 0]],
    ]
    assert side_triangles == expected


def test__compute_triangles_of_body_side__cropped_with_no_data() -> None:
    # note, this is just a smoke test to detect changes in the output of the function
    array = np.array(
        [
            [NO_DATA, NO_DATA, NO_DATA, NO_DATA],
            [NO_DATA, 1, 2, NO_DATA],
            [NO_DATA, 3, 4, NO_DATA],
            [NO_DATA, NO_DATA, NO_DATA, NO_DATA],
        ]
    )
    max_x, max_y = array.shape
    raster = _create_raster(array, max_x, max_y)
    side_triangles = _compute_triangles_of_body_side(
        raster=raster,
        array=array,
        max_x=max_x,
        max_y=max_y,
        x_scale=1.0,
        y_scale=1.0,
        z_scale=1.0,
        z_offset=0.0,
    )
    expected = [
        [[0, 0.0, 5.0], [0, 1.0, 7.0], [0, 0.0, 0]],
        [[0, 0.0, 0], [0, 1.0, 7.0], [0, 1.0, 0]],
        [[1.0, 0, 9.0], [0.0, 0, 5.0], [0.0, 0, 0]],
        [[1.0, 0, 9.0], [0.0, 0, 0], [1.0, 0, 0]],
        [[0, 1.0, 7.0], [0, 2.0, 7.0], [0, 1.0, 0]],
        [[0, 1.0, 0], [0, 2.0, 7.0], [0, 2.0, 0]],
        [[0.0, 2.0, 7.0], [1.0, 2.0, 8.0], [0.0, 2.0, 0]],
        [[0.0, 2.0, 0], [1.0, 2.0, 8.0], [1.0, 2.0, 0]],
        [[2.0, 1.0, 8.0], [2.0, 0.0, 9.0], [2.0, 0.0, 0]],
        [[2.0, 1.0, 8.0], [2.0, 0.0, 0], [2.0, 1.0, 0]],
        [[2.0, 0, 9.0], [1.0, 0, 9.0], [1.0, 0, 0]],
        [[2.0, 0, 9.0], [1.0, 0, 0], [2.0, 0, 0]],
        [[2.0, 2.0, 11.0], [2.0, 1.0, 8.0], [2.0, 1.0, 0]],
        [[2.0, 2.0, 11.0], [2.0, 1.0, 0], [2.0, 2.0, 0]],
        [[1.0, 2.0, 8.0], [2.0, 2.0, 11.0], [1.0, 2.0, 0]],
        [[1.0, 2.0, 0], [2.0, 2.0, 11.0], [2.0, 2.0, 0]],
    ]
    assert side_triangles == expected
