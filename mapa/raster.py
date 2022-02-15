from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import numpy.typing as npt
import rasterio as rio
from haversine import haversine
from rasterio.io import DatasetReader
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.windows import Window

from mapa.utils import _path_to_clipped_tiff, _path_to_merged_tiff, timing


@timing
def clip_tiff_to_bbox(input_tiff: Path, bbox_geometry: dict, bbox_hash: str) -> Path:
    click.echo(f"{'🔪  clipping region of interest...':<50s}", nl=False)
    data = rio.open(input_tiff)
    out_img, out_transform = mask(data, shapes=[bbox_geometry], crop=True)
    out_meta = data.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "crs": data.crs,
        }
    )
    clipped_tiff = _path_to_clipped_tiff(bbox_hash)
    with rio.open(clipped_tiff, "w", **out_meta) as file:
        file.write(out_img)
    return clipped_tiff


def read_tiff(path: Path) -> np.ndarray:
    array = rio.open(path).read()
    # drop higher dimension to get 2-dimensional (x * y) array
    return array[0, :, :]


def remove_empty_first_and_last_rows_and_cols(array: npt.ArrayLike) -> np.ndarray:
    # remove first and last cols in case of all zero
    if not array[:, 0].any():
        array = np.delete(array, (0), axis=1)
    if not array[:, -1].any():
        array = np.delete(array, (-1), axis=1)

    # remove first and last rows in case of all zero
    if not array[0].any():
        array = np.delete(array, (0), axis=0)
    if not array[-1].any():
        array = np.delete(array, (-1), axis=0)

    return array


def _cut_array_to_square(array: npt.ArrayLike) -> np.ndarray:
    rows, cols = array.shape
    if rows > cols:
        diff = rows - cols
        # remove last n=diff rows
        return array[:-diff, :]
    elif cols > rows:
        diff = cols - rows
        return array[:, :-diff]
    else:
        return array


def _cut_array_to_rectangular_shape(array: npt.ArrayLike, cut_to_format_ratio: float) -> np.ndarray:
    rows, cols = array.shape
    if rows / cols == cut_to_format_ratio or cols / rows == cut_to_format_ratio:
        # input array has already desired format ratio
        return array
    elif rows > cols:
        # cut cols
        desired_n_cols = int(rows * cut_to_format_ratio)
        return array[:, :desired_n_cols]
    elif cols > rows:
        # cut rows
        desired_n_rows = int(cols * cut_to_format_ratio)
        return array[:desired_n_rows, :]
    else:
        # cut cols anyway
        desired_n_cols = int(rows * cut_to_format_ratio)
        return array[:, :desired_n_cols]


def cut_array_to_format(array: npt.ArrayLike, cut_to_format_ratio: float) -> np.ndarray:
    if cut_to_format_ratio == 1.0:
        return _cut_array_to_square(array)
    if cut_to_format_ratio == 0.0:
        raise ValueError("Cannot cut array to format with ratio 0.0. Choose a format ratio between 0.0 and 1.0")
    else:
        if cut_to_format_ratio > 1.0:
            # ensure ratio is between 0.0 and 1.0 and transpose ratio
            cut_to_format_ratio = cut_to_format_ratio**-1
        return _cut_array_to_rectangular_shape(array, cut_to_format_ratio)


def _get_coordinate_of_pixel(row: int, col: int, tiff) -> Tuple[float]:
    meta = tiff.meta
    window = Window(0, 0, meta["width"], meta["height"])
    meta["transform"] = rio.windows.transform(window, tiff.transform)
    return rio.transform.xy(meta["transform"], row, col, offset="center")


def determine_elevation_scale(tiff: DatasetReader, model_size: int) -> float:
    array = tiff.read()[0, :, :]
    _, cols = array.shape

    # get lat lon coordinate of top left and top right pixel
    top_left_coor = _get_coordinate_of_pixel(0, 0, tiff)
    top_right_coor = _get_coordinate_of_pixel(0, cols, tiff)
    # get distance in meter between the two coordinates
    distance = haversine(top_left_coor, top_right_coor, unit="m")

    # to find out what 1 meter in reality corresponds to in the model, we need to divide the model size by the distance
    one_meter_in_model = model_size / distance
    return one_meter_in_model


def merge_tiffs(tiffs: List[Path], bbox_hash: str) -> Path:
    datasets = []
    for tiff in tiffs:
        data = rio.open(tiff)
        datasets.append(data)
    mosaic, out_trans = merge(datasets)
    out_meta = datasets[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": data.crs,
        }
    )
    tiff = _path_to_merged_tiff(bbox_hash)
    with rio.open(tiff, "w", **out_meta) as dest:
        dest.write(mosaic)
    return tiff
