import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import rasterio as rio

from mapa import conf
from mapa.algorithm import ModelSize, compute_all_triangles, reduce_resolution
from mapa.caching import get_hash_of_geojson, tiff_for_bbox_is_cached
from mapa.raster import (
    clip_tiff_to_bbox,
    cut_array_to_square,
    determine_elevation_scale,
    merge_tiffs,
    remove_empty_first_and_last_rows_and_cols,
    tiff_to_array,
)
from mapa.stac import fetch_stac_items_for_bbox
from mapa.stl_file import save_to_stl_file
from mapa.tiling import get_x_y_from_tiles_format, split_array_into_tiles
from mapa.utils import TMPDIR, ProgressBar, path_to_clipped_tiff
from mapa.verification import verify_input_and_output_are_valid
from mapa.zip import create_zip_archive

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(os.getenv("MAPA_LOG_LEVEL", "INFO"))


def convert_array_to_stl(
    array: np.ndarray,
    as_ascii: bool,
    desired_size: ModelSize,
    max_res: bool,
    z_offset: Union[None, float],
    z_scale: float,
    elevation_scale: float,
    output_file: Path,
) -> Path:
    x, y = array.shape
    # when merging tiffs, sometimes an empty row/col is added, which should be dropped (in case the array size suffices)
    if x > 1 and y > 1:
        array = remove_empty_first_and_last_rows_and_cols(array)

    if max_res:
        if x * y > conf.PERFORMANCE_WARNING_THRESHOLD:
            log.warning(
                "⛔️  Warning: Using max_res=True on the given bounding box might consume a lot of time and memory. "
                "Consider setting max_res=False."
            )
    else:
        bin_fac = round((x / conf.MAXIMUM_RESOLUTION + y / conf.MAXIMUM_RESOLUTION) / 2)
        if bin_fac > 1:
            log.debug("🔍  reducing image resolution...")
            array = reduce_resolution(array, bin_factor=bin_fac)

    triangles = compute_all_triangles(array, desired_size, z_offset, z_scale, elevation_scale)
    log.debug("💾  saving data to stl file...")

    output_file = save_to_stl_file(triangles, output_file, as_ascii)
    log.info(f"🎉  successfully generated STL file: {Path(output_file).absolute()}")
    return Path(output_file)


def _get_desired_size(array: np.ndarray, x: float, y: float, ensure_squared: bool) -> ModelSize:
    if ensure_squared:
        return ModelSize(x=x, y=y)
    else:
        rows, cols = array.shape
        return ModelSize(x=x, y=y / rows * cols)


def convert_tiff_to_stl(
    input_file: str,
    as_ascii: bool,
    model_size: int,
    output_file: Union[str, None],
    max_res: bool,
    z_offset: Union[None, float],
    z_scale: float,
    ensure_squared: bool = False,
) -> Path:
    output_file = verify_input_and_output_are_valid(input=input_file, output=output_file)

    tiff = rio.open(input_file)
    elevation_scale = determine_elevation_scale(tiff, model_size)
    array = tiff_to_array(tiff)

    if ensure_squared:
        array = cut_array_to_square(array)
    desired_size = _get_desired_size(array=array, x=model_size, y=model_size, ensure_squared=ensure_squared)

    return convert_array_to_stl(
        array=array,
        as_ascii=as_ascii,
        desired_size=desired_size,
        max_res=max_res,
        z_offset=z_offset,
        z_scale=z_scale,
        elevation_scale=elevation_scale,
        output_file=output_file,
    )


def _fetch_merge_and_clip_tiffs(
    bbox_geojson: dict,
    bbox_hash: str,
    allow_caching: bool,
    cache_dir: Path,
    progress_bar: Union[None, ProgressBar] = None,
) -> Path:
    tiffs = fetch_stac_items_for_bbox(bbox_geojson, allow_caching, cache_dir, progress_bar)
    if len(tiffs) > 1:
        merged_tiff = merge_tiffs(tiffs, bbox_hash, cache_dir)
    else:
        merged_tiff = tiffs[0]
    return clip_tiff_to_bbox(merged_tiff, bbox_geojson, bbox_hash, cache_dir)


def _get_tiff_for_bbox(
    bbox_geojson: dict, allow_caching: bool, cache_dir: Path, progress_bar: Union[None, ProgressBar] = None
) -> Path:
    bbox_hash = get_hash_of_geojson(bbox_geojson)
    if tiff_for_bbox_is_cached(bbox_hash, cache_dir) and allow_caching:
        log.info("🚀  using cached tiff!")
        return path_to_clipped_tiff(bbox_hash, cache_dir)
    else:
        return _fetch_merge_and_clip_tiffs(bbox_geojson, bbox_hash, allow_caching, cache_dir, progress_bar)


def convert_bbox_to_stl(
    bbox_geometry: dict,
    as_ascii: bool = False,
    model_size: int = 200,
    output_file: str = "output",
    max_res: bool = False,
    z_offset: Union[None, float] = 0.0,
    z_scale: float = 1.0,
    ensure_squared: bool = False,
    split_area_in_tiles: str = "1x1",
    compress: bool = True,
    allow_caching: bool = True,
    cache_dir: Union[Path, str] = TMPDIR(),
    progress_bar: Union[None, object] = None,
) -> Union[Path, List[Path]]:
    """
    Takes a GeoJSON containing a bounding box as input, fetches the required STAC GeoTIFFs for the
    given bounding box and creates a STL file with elevation data from the GeoTIFFs.

    Parameters
    ----------
    bbox_geometry : dict
        GeoJSON containing the coordinates of the bounding box, selected on the ipyleaflet widget. Usually the
        value of `drawer.last_draw["geometry"]` is used for this.
    as_ascii : bool, optional
        Save output STL as ascii file. If False, output file will be binary. By default False
    model_size : int, optional
        Desired size of the (larger side of the) generated 3d model in millimeter. By default 200
    output_file : str, optional
        Name and path to output file. File ending should not be provided. Mapa will add .zip or .stl depending
        on the settings. By default "output"
    max_res : bool, optional
        Whether maximum resolution should be used. Note, that this flag potentially increases compute time
        and memory consumption dramatically. The default behavior (i.e. max_res=False) should return 3d models
        with sufficient resolution, while the output stl file should be < ~300 MB. By default False
    z_offset : Union[None, float], optional
        Offset distance in millimeter to be put below the 3d model. Is not influenced by z-scale. Set to None
        if you want your model to have the natural offset, corresponding to height above mean sea level.
        By default 0.0
    z_scale : float, optional
        Value to be multiplied to the z-axis elevation data to scale up the height of the model. By default 1.0
    ensure_squared : bool, optional
        Boolean flag to toggle whether the output model should be squared in x- and y-dimension. When enabled
        it will remove pixels from one side to ensure same length for both sides. By default False
    split_area_in_tiles : str, optional
        Split the selected bounding box into tiles with this option. The allowed format of a given string is
        "nxm" e.g. "1x1", "2x3", "4x4" or similar, where "1x1" would not split at all and result in only
        one stl file. If an allowed tile format is specified, `nxm` stl files will be computed. By default "1x1"
    compress : bool, optional
        If enabled, the output stl file(s) will be compressed to a zip file. Compressing is recommended as it
        reduces the data volume of typical stl files by a factor of ~4.
    allow_caching : bool, optional
        Whether caching previous downloaded GeoTIFF files should be enabled/disabled. By default True
    cache_dir: Union[Path, str]
        Path to a directory which should be used as local cache. This is helpful when intermediary tiff files
        should be persisted even after the temp directory gets cleaned-up by e.g. a restart. By default TMPDIR
    progress_bar : Union[None, object], optional
        A streamlit progress bar object can be used to indicate the progress of downloading the STAC items. By
        default None

    Returns
    -------
    Union[Path, List[Path]]
        Path or list of paths to the resulting output file(s) on your local machine.
    """

    # fail early in case of missing requirements
    if bbox_geometry is None:
        raise ValueError("⛔️  ERROR: make sure to draw a rectangle on the map first!")

    # evaluate tile format to fail early in case of invalid input value
    tiles = get_x_y_from_tiles_format(split_area_in_tiles)

    args = locals().copy()
    args.pop("progress_bar", None)
    log.info(f"⏳  converting bounding box to STL file with arguments: {args}")

    if progress_bar:
        steps = tiles.x * tiles.y * 2 if compress else tiles.x * tiles.y
        progress_bar = ProgressBar(progress_bar=progress_bar, steps=steps)

    path_to_tiff = _get_tiff_for_bbox(bbox_geometry, allow_caching, Path(cache_dir), progress_bar)
    tiff = rio.open(path_to_tiff)
    elevation_scale = determine_elevation_scale(tiff, model_size)
    array = tiff_to_array(tiff)
    if ensure_squared:
        array = cut_array_to_square(array)

    desired_size = _get_desired_size(
        array=array,
        x=model_size / tiles.x,
        y=model_size / tiles.y,
        ensure_squared=ensure_squared,
    )

    tiled_arrays = split_array_into_tiles(array, tiles)
    stl_files = []
    for i, array in enumerate(tiled_arrays):
        stl_files.append(
            convert_array_to_stl(
                array=array,
                as_ascii=as_ascii,
                desired_size=desired_size,
                max_res=max_res,
                z_offset=z_offset,
                z_scale=z_scale,
                elevation_scale=elevation_scale,
                output_file=f"{output_file}_{i+1}.stl" if len(tiled_arrays) > 1 else f"{output_file}.stl",
            )
        )
        if progress_bar:
            progress_bar.step()
    if compress:
        return create_zip_archive(files=stl_files, output_file=f"{output_file}.zip", progress_bar=progress_bar)
    else:
        return stl_files[0] if len(stl_files) == 1 else stl_files
