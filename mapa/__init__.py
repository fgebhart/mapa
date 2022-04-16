import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import rasterio as rio

from mapa import conf
from mapa.algorithm import compute_all_triangles, reduce_resolution
from mapa.caching import get_hash_of_geojson, tiff_for_bbox_is_cached
from mapa.raster import (
    clip_tiff_to_bbox,
    cut_array_to_format,
    determine_elevation_scale,
    merge_tiffs,
    remove_empty_first_and_last_rows_and_cols,
    tiff_to_two_dimensional_array,
)
from mapa.stac import fetch_stac_items_for_bbox
from mapa.stl_file import save_to_stl_file
from mapa.tiling import TileFormat, get_x_y_from_tiles_format, split_array_into_tiles
from mapa.utils import path_to_clipped_tiff
from mapa.verification import verify_input_and_output_are_valid
from mapa.zip import create_zip_archive

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(os.getenv("MAPA_LOG_LEVEL", "INFO"))


def convert_array_to_stl(
    array: np.ndarray,
    as_ascii: bool,
    model_size: int,
    max_res: bool,
    z_offset: Union[None, float],
    z_scale: float,
    cut_to_format_ratio: Union[None, float],
    elevation_scale: float,
    output_file: Path,
    tiles_format: TileFormat = TileFormat(x=1, y=1),
) -> Path:
    x, y = array.shape
    # when merging tiffs, sometimes an empty row/col is added, which should be dropped (in case the array size suffices)
    if x > 1 and y > 1:
        array = remove_empty_first_and_last_rows_and_cols(array)
    if cut_to_format_ratio:
        array = cut_array_to_format(array, cut_to_format_ratio)

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

    triangles = compute_all_triangles(
        array, model_size, z_offset, z_scale, elevation_scale, cut_to_format_ratio, tiles_format
    )
    log.debug("💾  saving data to stl file...")

    output_file = save_to_stl_file(triangles, output_file, as_ascii)
    log.info(f"🎉  successfully generated STL file: {Path(output_file).absolute()}")
    return Path(output_file)


def convert_tiff_to_stl(
    input_file: str,
    as_ascii: bool,
    model_size: int,
    output_file: Union[str, None],
    max_res: bool,
    z_offset: Union[None, float],
    z_scale: float,
    cut_to_format_ratio: Union[None, float],
) -> Path:

    output_file = verify_input_and_output_are_valid(input=input_file, output=output_file)

    tiff = rio.open(input_file)
    elevation_scale = determine_elevation_scale(tiff, model_size)
    array = tiff_to_two_dimensional_array(tiff)

    return convert_array_to_stl(
        array=array,
        as_ascii=as_ascii,
        model_size=model_size,
        max_res=max_res,
        z_offset=z_offset,
        z_scale=z_scale,
        cut_to_format_ratio=cut_to_format_ratio,
        elevation_scale=elevation_scale,
        output_file=output_file,
    )


def _fetch_merge_and_clip_tiffs(
    bbox_geojson: dict,
    bbox_hash: str,
    allow_caching: bool,
    progress_bar: Union[None, object] = None,
) -> Path:
    tiffs = fetch_stac_items_for_bbox(bbox_geojson, allow_caching, progress_bar)
    if len(tiffs) > 1:
        merged_tiff = merge_tiffs(tiffs, bbox_hash)
    else:
        merged_tiff = tiffs[0]
    return clip_tiff_to_bbox(merged_tiff, bbox_geojson, bbox_hash)


def _get_tiff_for_bbox(bbox_geojson: dict, allow_caching: bool, progress_bar: Union[None, object] = None) -> Path:
    bbox_hash = get_hash_of_geojson(bbox_geojson)
    if tiff_for_bbox_is_cached(bbox_hash) and allow_caching:
        log.info("🚀  using cached tiff!")
        return path_to_clipped_tiff(bbox_hash)
    else:
        return _fetch_merge_and_clip_tiffs(bbox_geojson, bbox_hash, allow_caching, progress_bar)


def convert_bbox_to_stl(
    bbox_geometry: dict,
    as_ascii: bool = False,
    model_size: int = 200,
    output_file: str = "output",
    max_res: bool = False,
    z_offset: Union[None, float] = 0.0,
    z_scale: float = 1.0,
    cut_to_format_ratio: Union[None, float] = None,
    split_area_in_tiles: str = "1*1",
    compress: bool = True,
    allow_caching: bool = True,
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
    cut_to_format_ratio : Union[None, float], optional
        Cut the input tiff file to a specified format. Set to `1` if you want the output model to be squared.
        Set to `0.5` if you want one side to be half the length of the other side. Omit this flag to keep the
        input format. This option is particularly useful when an exact output format ratio is required for
        example when planning to put the 3d printed model into a picture frame. Using this option will always
        try to cut the shorter side of the input tiff. By default None
    split_area_in_tiles: str, optional
        Split the selected bounding box into tiles with this option. The allowed format of a given string is
        "n*m" e.g. "1*1`, "2*3", "4*4" or similar, where "1*1" would not split at all and result in only
        one stl file. If an allowed tile format is specified, `n*m` stl files will be computed. By default "1*1"
    compress: bool, optional
        If enabled, the output stl file(s) will be compressed to a zip file. Compressing is recommended as it
        reduces the data volume of typical stl files by a factor of ~4.
    allow_caching : bool, optional
        Whether caching previous downloaded GeoTIFF files should be enabled/disabled. By default True
    progress_bar : Union[None, object], optional
        A streamlit progress bar object can be used to indicate the progress of downloading the STAC items.

    Returns
    -------
    Union[Path, List[Path]]
        Path or list of paths to the resulting output file(s) on your local machine.
    """

    # fail early in case of missing requirements
    if bbox_geometry is None:
        raise ValueError("⛔️  ERROR: make sure to draw a rectangle on the map first!")

    # evaluate tile format to fail early in case of invalid input value
    tiles_format = get_x_y_from_tiles_format(split_area_in_tiles)

    args = locals().copy()
    args.pop("progress_bar", None)

    log.info(f"⏳  converting bounding box to STL file with arguments: {args}")
    path_to_tiff = _get_tiff_for_bbox(bbox_geometry, allow_caching, progress_bar)
    tiff = rio.open(path_to_tiff)
    elevation_scale = determine_elevation_scale(tiff, model_size)
    array = tiff_to_two_dimensional_array(tiff)
    if cut_to_format_ratio:
        array = cut_array_to_format(array, cut_to_format_ratio)

    breakpoint()
    list_of_tiled_arrays = split_array_into_tiles(array, tiles_format)
    stl_files = []
    for i, array in enumerate(list_of_tiled_arrays):
        stl_files.append(
            convert_array_to_stl(
                array=array,
                as_ascii=as_ascii,
                model_size=model_size,
                max_res=max_res,
                z_offset=z_offset,
                z_scale=z_scale,
                cut_to_format_ratio=None,
                elevation_scale=elevation_scale,
                output_file=f"{output_file}_{i+1}.stl" if len(list_of_tiled_arrays) > 1 else f"{output_file}.stl",
                tiles_format=tiles_format,
            )
        )
    if compress:
        return create_zip_archive(files=stl_files, output_file=f"{output_file}.zip")
    else:
        return stl_files[0] if len(stl_files) == 1 else stl_files
