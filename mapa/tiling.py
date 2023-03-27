from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TileFormat:
    x: int
    y: int


def split_array_into_tiles(array: np.ndarray, tiles_format: TileFormat) -> List[np.ndarray]:
    x, y = array.shape
    if tiles_format.x > x or tiles_format.y > y:
        raise ValueError("Input array is too small to be split into tiles.")

    h_tiles = np.array_split(array, tiles_format.x, axis=0)
    tiles = []
    for tile in h_tiles:
        tiles += np.array_split(tile, tiles_format.y, axis=1)

    return tiles


def get_x_y_from_tiles_format(tiles_format: str) -> TileFormat:
    error_msg = (
        "Invalid format of `split_area_in_tiles`. Input value needs to be of format `nxm`, where `n` and `m` "
        "are integers greater than zero."
    )
    if "x" not in tiles_format:
        raise ValueError(error_msg)

    format_list = tiles_format.split("x")

    if "0" in format_list:
        raise ValueError(error_msg)

    if len(format_list) != 2:
        raise ValueError(error_msg)

    return TileFormat(x=int(format_list[0]), y=int(format_list[1]))
