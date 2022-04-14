from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TileFormat:
    x: int
    y: int


def split_array_into_tiles(array: np.ndarray, tiles_format: TileFormat) -> List[np.ndarray]:
    # ensure array is evenly divisible by the corresponding number of the tile format
    x, y = array.shape

    if tiles_format.x > x or tiles_format.y > y:
        raise ValueError("Input array is too small to be split into tiles.")

    n_rows = x % tiles_format.x
    n_cols = y % tiles_format.y

    if n_rows != 0:
        # drop number of rows
        array = array[:-n_rows]

    if n_cols != 0:
        # drop number of cols
        array = array[:, :-n_cols]

    return _divide_array_into_tiles(array, tiles_format=tiles_format)


def _divide_array_into_tiles(array, tiles_format: TileFormat) -> List[np.ndarray]:
    x, y = array.shape

    assert x % tiles_format.x == 0
    assert y % tiles_format.y == 0

    n_rows_per_tile = x // tiles_format.x
    n_cols_per_tile = y // tiles_format.y

    h_tiles = []
    # add horizontal/row tiles
    for i in range(tiles_format.x):
        h_tiles.append(array[i * n_rows_per_tile : (i * n_rows_per_tile) + n_rows_per_tile])  # noqa: E203
    tiles = []
    # split row tiles also vertically
    for tile in h_tiles:
        for i in range(tiles_format.y):
            tiles.append(tile[:, i * n_cols_per_tile : (i * n_cols_per_tile) + n_cols_per_tile])  # noqa: E203
    return tiles


def get_x_y_from_tiles_format(tiles_format: str) -> TileFormat:
    error_msg = (
        "Invalid format of `split_area_in_tiles`. Input value needs to be of format `n*m`, where `n` and `m` "
        "are integers."
    )
    if "*" not in tiles_format:
        raise ValueError(error_msg)

    format_list = tiles_format.split("*")

    if len(format_list) != 2:
        raise ValueError(error_msg)

    return TileFormat(x=int(format_list[0]), y=int(format_list[1]))
