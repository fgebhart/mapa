import numpy as np
import pytest
import rasterio as rio
from haversine import haversine

from mapa.raster import (
    _get_coordinate_of_pixel,
    clip_tiff_to_bbox,
    cut_array_to_square,
    determine_elevation_scale,
    read_tiff,
    remove_empty_first_and_last_rows_and_cols,
)
from mapa.stac import fetch_stac_items_for_bbox


def test_read_tiff(test_tiff) -> None:
    array = read_tiff(test_tiff)
    assert len(array.shape) == 2


def test_cut_array_to_square() -> None:
    # test input square, shape should not change
    square = np.random.random((10, 10))
    assert square.shape == (10, 10)
    result = cut_array_to_square(square)
    assert result.shape == (10, 10)
    np.testing.assert_array_equal(square, result)

    # test non square rect with more rows than cols
    rect = np.random.random((10, 5))
    assert rect.shape == (10, 5)
    result = cut_array_to_square(rect)
    assert result.shape == (5, 5)

    # test non square rect with more cols than rows
    rect = np.random.random((2, 4))
    assert rect.shape == (2, 4)
    result = cut_array_to_square(rect)
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
    test_array = read_tiff(test_tiff)
    clipped_array = read_tiff(clipped_tiff)

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


def test_determine_z_scale(geojson_bbox) -> None:
    tiff_path = fetch_stac_items_for_bbox(geojson_bbox, allow_caching=True)
    tiff = rio.open(tiff_path[0])
    scale = determine_elevation_scale(tiff, model_size=200)
    expected_scale = 0.0017986407274490762
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
