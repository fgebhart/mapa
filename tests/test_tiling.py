import numpy as np
import pytest

from mapa.tiling import TileFormat, _divide_array_into_tiles, get_x_y_from_tiles_format, split_array_into_tiles


def test__ensure_array_is_evenly_divisible_and_get_blocks__square() -> None:
    # square shape into 2x2
    four_by_four = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6],
        ]
    )
    blocks = split_array_into_tiles(four_by_four, tiles_format=TileFormat(x=2, y=2))
    expected = np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 0], [3, 4]], [[1, 2], [5, 6]]])
    np.testing.assert_array_equal(expected, blocks)

    # square shape into 3x3
    three_by_three = np.array(
        [
            [1, 2, 3],
            [5, 6, 7],
            [9, 0, 1],
        ]
    )
    blocks = split_array_into_tiles(three_by_three, tiles_format=TileFormat(x=3, y=3))
    expected = list(
        [
            np.array([[1]]),
            np.array([[2]]),
            np.array([[3]]),
            np.array([[5]]),
            np.array([[6]]),
            np.array([[7]]),
            np.array([[9]]),
            np.array([[0]]),
            np.array([[1]]),
        ]
    )

    np.testing.assert_array_equal(expected, blocks)

    # four by four into 3x3
    with pytest.raises(AssertionError):
        _divide_array_into_tiles(four_by_four, tiles_format=TileFormat(x=3, y=3))
    blocks = split_array_into_tiles(four_by_four, tiles_format=TileFormat(x=3, y=3))
    np.testing.assert_array_equal(expected, blocks)

    # too little array
    two_by_two = np.array(
        [
            [1, 2],
            [5, 6],
        ]
    )
    with pytest.raises(ValueError, match="Input array is too small to be split into tiles."):
        split_array_into_tiles(two_by_two, tiles_format=TileFormat(x=3, y=3))


def test__ensure_array_is_evenly_divisible_and_get_blocks__rect() -> None:
    # rect shape into 2x2
    array = np.array(
        [
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9],
            [9, 0, 1, 2, 3],
            [3, 4, 5, 6, 7],
        ]
    )
    blocks = split_array_into_tiles(array, tiles_format=TileFormat(x=2, y=2))
    expected = np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 0], [3, 4]], [[1, 2], [5, 6]]])
    np.testing.assert_array_equal(expected, blocks)

    blocks = split_array_into_tiles(array, tiles_format=TileFormat(x=3, y=3))
    expected = list(
        [
            np.array([[1]]),
            np.array([[2]]),
            np.array([[3]]),
            np.array([[5]]),
            np.array([[6]]),
            np.array([[7]]),
            np.array([[9]]),
            np.array([[0]]),
            np.array([[1]]),
        ]
    )
    np.testing.assert_array_equal(expected, blocks)

    # array being too small
    with pytest.raises(ValueError, match="Input array is too small to be split into tiles."):
        split_array_into_tiles(array, tiles_format=TileFormat(x=10, y=10))


def test_get_x_y_from_tiles_format() -> None:
    # valid examples
    assert get_x_y_from_tiles_format("3*3") == TileFormat(x=3, y=3)
    assert get_x_y_from_tiles_format("2*10") == TileFormat(x=2, y=10)

    # invalid examples
    error_msg = "Invalid format"
    with pytest.raises(ValueError, match=error_msg):
        assert get_x_y_from_tiles_format("foo")

    with pytest.raises(ValueError, match=error_msg):
        assert get_x_y_from_tiles_format("1*2*3")

    with pytest.raises(ValueError, match="invalid literal for int"):
        assert get_x_y_from_tiles_format("a*f")
