import numpy as np
import pytest
import rasterio as rio
from haversine import haversine

from mapa.raster import (
    _cut_array_to_rectangular_shape,
    _cut_array_to_square,
    _get_coordinate_of_pixel,
    clip_tiff_to_bbox,
    cut_array_to_format,
    determine_elevation_scale,
    remove_empty_first_and_last_rows_and_cols,
    tiff_to_array,
)
from mapa.stac import fetch_stac_items_for_bbox


def test_tiff_to_2_dimensional_array(test_tiff) -> None:
    tiff = rio.open(test_tiff)
    array = tiff_to_array(tiff)
    assert len(array.shape) == 2


def test_cut_array_to_square() -> None:
    # test input square, shape should not change
    square = np.random.random((10, 10))
    assert square.shape == (10, 10)
    result = _cut_array_to_square(square)
    assert result.shape == (10, 10)
    np.testing.assert_array_equal(square, result)

    # test non square rect with more rows than cols
    rect = np.random.random((10, 5))
    assert rect.shape == (10, 5)
    result = _cut_array_to_square(rect)
    assert result.shape == (5, 5)

    # test non square rect with more cols than rows
    rect = np.random.random((2, 4))
    assert rect.shape == (2, 4)
    result = _cut_array_to_square(rect)
    assert result.shape == (2, 2)


def test_clip_tiff_to_bbox(test_tiff) -> None:
    # bbox does not overlap tiff raster coordinates
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [18.401292, -33.930818],
                [18.401292, -33.91059],
                [18.426691, -33.91059],
                [18.426691, -33.930818],
                [18.401292, -33.930818],
            ]
        ],
    }
    with pytest.raises(ValueError, match="Input shapes do not overlap raster."):
        clip_tiff_to_bbox(test_tiff, bbox, "foo")

    # bbox does overlap tiff raster coordinates
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [-156.0, 20.0],
                [-156.0, 18.9],
                [-154.8, 18.9],
                [-154.8, 20.0],
                [-156.0, 20.0],
            ]
        ],
    }
    clipped_tiff = clip_tiff_to_bbox(test_tiff, bbox, "foo")
    test_array = tiff_to_array(rio.open(test_tiff))
    clipped_array = tiff_to_array(rio.open(clipped_tiff))

    assert test_array.shape > clipped_array.shape
    overlap = np.isin(clipped_array, test_array)
    # remove all False entries
    overlap = overlap[np.ix_(~np.all(overlap == False, axis=1), ~np.all(overlap == False, axis=0))]  # noqa: E712
    # need to add +1 since rasterio seems to add additional row and col with all zeros...
    assert overlap.shape[0] + 1 == clipped_array.shape[0]
    assert overlap.shape[1] + 1 == clipped_array.shape[1]


def test_remove_empty_first_and_last_rows_and_cols() -> None:
    # array without all zero rows and cols
    array = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6],
        ]
    )
    result = remove_empty_first_and_last_rows_and_cols(array)
    np.testing.assert_array_equal(array, result)

    # array with all zero first row
    array = np.array(
        [
            [0, 0, 0, 0],
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6],
        ]
    )
    result = remove_empty_first_and_last_rows_and_cols(array)
    expected = np.array(
        [
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6],
        ]
    )
    np.testing.assert_array_equal(expected, result)

    # array with all zero last row
    array = np.array(
        [
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6],
            [0, 0, 0, 0],
        ]
    )
    result = remove_empty_first_and_last_rows_and_cols(array)
    expected = np.array(
        [
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6],
        ]
    )
    np.testing.assert_array_equal(expected, result)

    # array with all zero first and last row
    array = np.array(
        [
            [0, 0, 0, 0],
            [9, 0, 1, 2],
            [3, 4, 5, 6],
            [0, 0, 0, 0],
        ]
    )
    result = remove_empty_first_and_last_rows_and_cols(array)
    expected = np.array(
        [
            [9, 0, 1, 2],
            [3, 4, 5, 6],
        ]
    )
    np.testing.assert_array_equal(expected, result)

    # array with all zero last col
    array = np.array(
        [
            [1, 2, 3, 0],
            [5, 6, 7, 0],
            [9, 0, 1, 0],
            [3, 4, 5, 0],
        ]
    )
    result = remove_empty_first_and_last_rows_and_cols(array)
    expected = np.array(
        [
            [1, 2, 3],
            [5, 6, 7],
            [9, 0, 1],
            [3, 4, 5],
        ]
    )
    np.testing.assert_array_equal(expected, result)

    # array with all zero first col
    array = np.array(
        [
            [0, 1, 2, 3],
            [0, 5, 6, 7],
            [0, 9, 0, 1],
            [0, 3, 4, 5],
        ]
    )
    result = remove_empty_first_and_last_rows_and_cols(array)
    expected = np.array(
        [
            [1, 2, 3],
            [5, 6, 7],
            [9, 0, 1],
            [3, 4, 5],
        ]
    )
    np.testing.assert_array_equal(expected, result)

    # array with all zero first and last col
    array = np.array(
        [
            [0, 1, 2, 0],
            [0, 5, 6, 0],
            [0, 9, 0, 0],
            [0, 3, 4, 0],
        ]
    )
    result = remove_empty_first_and_last_rows_and_cols(array)
    expected = np.array(
        [
            [1, 2],
            [5, 6],
            [9, 0],
            [3, 4],
        ]
    )
    np.testing.assert_array_equal(expected, result)

    # array with all zero first and last col and first and last row
    array = np.array(
        [
            [0, 0, 0, 0],
            [0, 5, 6, 0],
            [0, 9, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    result = remove_empty_first_and_last_rows_and_cols(array)
    expected = np.array(
        [
            [5, 6],
            [9, 0],
        ]
    )
    np.testing.assert_array_equal(expected, result)


def test_determine_z_scale(geojson_bbox, mock_file_download) -> None:
    tiff_path = fetch_stac_items_for_bbox(geojson_bbox, allow_caching=False)
    tiff = rio.open(tiff_path[0])
    scale = determine_elevation_scale(tiff, model_size=200)
    expected_scale = 0.0013001543499978835
    assert expected_scale == scale


def test__get_coordinate_of_pixel(test_tiff) -> None:
    tiff = rio.open(test_tiff)
    c1 = _get_coordinate_of_pixel(0, 0, tiff)
    c2 = _get_coordinate_of_pixel(0, 1, tiff)
    assert c1 == (-156.14972920662672, 20.30046320308474)
    assert c2 == (-156.13176290094432, 20.30046320308474)
    assert c1 != c2
    assert haversine(c1, c2) == 1.997764801854062

    c3 = _get_coordinate_of_pixel(0, 0, tiff)
    c4 = _get_coordinate_of_pixel(0, 80, tiff)
    assert haversine(c3, c4) == 159.82118414828818


def test__cut_array_to_rectangular_shape() -> None:
    # test array which does already have desired output ratio
    arr = np.ones((2, 3))
    ratio = 2 / 3
    result = _cut_array_to_rectangular_shape(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(arr, result)

    result = _cut_array_to_rectangular_shape(array=arr, cut_to_format_ratio=3 / 2)
    np.testing.assert_array_equal(arr, result)

    # test array which has less cols than rows
    ratio = 1 / 3
    arr = np.ones((3, 2))
    expected = np.ones((3, 1))
    result = _cut_array_to_rectangular_shape(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(expected, result)

    # test array which has less rows than cols
    ratio = 1 / 3
    arr = np.ones((2, 3))
    expected = np.ones((1, 3))
    result = _cut_array_to_rectangular_shape(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(expected, result)


def test_cut_array_to_format() -> None:
    # input is square and desired output ratio corresponds to square
    ratio = 1.0
    arr = np.ones((3, 3))
    result = cut_array_to_format(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(arr, result)

    # input is rectangular and desired output ratio is square
    ratio = 1.0
    arr = np.ones((3, 4))
    expected = np.ones((3, 3))
    result = cut_array_to_format(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(expected, result)

    # input rectangular but already in desired output ratio
    ratio = 3 / 2
    arr = np.ones((3, 2))
    expected = np.ones((3, 2))
    result = cut_array_to_format(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(expected, result)

    # same behavior is achieved when input ratio is transposed
    ratio = 2 / 3
    arr = np.ones((3, 2))
    expected = np.ones((3, 2))
    result = cut_array_to_format(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(expected, result)

    # number of rows are less than number of cols, thus rows are cut to achieve the desired output ratio
    ratio = 1 / 3
    arr = np.ones((2, 3))
    expected = np.ones((1, 3))
    result = cut_array_to_format(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(expected, result)

    # number of cols are less than number of rows, thus cols are cut ro achieve the desired output radio
    ratio = 3 / 5
    arr = np.ones((10, 9))
    expected = np.ones((10, 6))
    result = cut_array_to_format(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(expected, result)

    # same for ratio transposed
    ratio = 5 / 3
    arr = np.ones((10, 9))
    expected = np.ones((10, 6))
    result = cut_array_to_format(array=arr, cut_to_format_ratio=ratio)
    np.testing.assert_array_equal(expected, result)
