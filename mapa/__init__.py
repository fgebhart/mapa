from pathlib import Path
from typing import Union

import click
import numpy as np
import rasterio as rio
from stl import Mode, mesh

from mapa import conf
from mapa.caching import get_hash_of_geojson, tiff_for_bbox_is_cached
from mapa.geometry import compute_all_triangles, reduce_resolution
from mapa.raster import (
    clip_tiff_to_bbox,
    cut_array_to_format,
    determine_elevation_scale,
    merge_tiffs,
    remove_empty_first_and_last_rows_and_cols,
)
from mapa.stac import fetch_stac_items_for_bbox
from mapa.utils import _path_to_clipped_tiff, timing


def _verify_input_is_valid(input: str):
    input_path = Path(input)
    if not input_path.is_file():
        raise FileNotFoundError(f"input file: '{input}' does not seem to be a file")
    if input_path.suffix in conf.SUPPORTED_INPUT_FORMAT:
        pass  # ok
    else:
        raise IOError(
            f"input file '{input}' does not seem to be a tiff file, only {conf.SUPPORTED_INPUT_FORMAT} are supported."
        )


def _verify_output_is_valid(output: str):
    output_path = Path(output)
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"parent directory of output file '{output_path.parent}' does not seem to be a valid directory."
        )


@timing
def _save_to_stl_file(triangles: np.ndarray, output_file: str, as_ascii: bool) -> str:
    stl = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
    stl.vectors = triangles
    if as_ascii:
        stl.save(output_file, mode=Mode.ASCII)
    else:
        stl.save(output_file, mode=Mode.BINARY)
    return output_file


def convert_array_to_stl(
    array: np.ndarray,
    as_ascii: bool,
    model_size: int,
    max_res: bool,
    z_offset: float,
    z_scale: float,
    cut_to_format_ratio: Union[None, float],
    elevation_scale: float,
    output_file: Path,
) -> Path:
    # drop higher dimension to get 2-dimensional (x * y) array
    array = array[0, :, :]
    array = remove_empty_first_and_last_rows_and_cols(array)
    if cut_to_format_ratio:
        array = cut_array_to_format(array, cut_to_format_ratio)

    x, y = array.shape
    if max_res:
        if x * y > conf.PERFORMANCE_WARNING_THRESHOLD:
            click.echo(
                "⛔️  Warning: Using max_res=True on the given bounding box might consume a lot of time and memory. "
                "Consider setting max_res=False."
            )
    else:
        bin_fac = round((x / conf.MAXIMUM_RESOLUTION + y / conf.MAXIMUM_RESOLUTION) / 2)
        if bin_fac > 1:
            click.echo(f"{'🔍  reducing image resolution...':<50s}", nl=False)
            array = reduce_resolution(array, bin_factor=bin_fac)
    combined_z_scale = z_scale * elevation_scale

    triangles = compute_all_triangles(array, model_size, z_offset, combined_z_scale, cut_to_format_ratio)
    click.echo(f"{'💾  saving data to stl file...':<50s}", nl=False)

    output_file = _save_to_stl_file(triangles, output_file, as_ascii)
    click.echo(f"\n🎉  successfully generated STL file: {Path(output_file).absolute()}")
    return Path(output_file)


def convert_tiff_to_stl(
    input_file: str,
    as_ascii: bool,
    model_size: int,
    output_file: Union[str, None],
    max_res: bool,
    z_offset: float,
    z_scale: float,
    cut_to_format_ratio: Union[None, float],
) -> Path:
    _verify_input_is_valid(input_file)
    if output_file is None:
        output_file = Path.home() / str(Path(input_file).name).replace(".tiff", ".stl").replace(".tif", ".stl")
    _verify_output_is_valid(output_file)

    tiff = rio.open(input_file)
    elevation_scale = determine_elevation_scale(tiff, model_size)
    array = tiff.read()

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


def _fetch_merge_and_clip_tiffs(bbox_geojson: dict, bbox_hash: str, allow_caching: bool) -> Path:
    tiffs = fetch_stac_items_for_bbox(bbox_geojson, allow_caching)
    if len(tiffs) > 1:
        merged_tiff = merge_tiffs(tiffs, bbox_hash)
    else:
        merged_tiff = tiffs[0]
    return clip_tiff_to_bbox(merged_tiff, bbox_geojson, bbox_hash)


def _get_tiff_for_bbox(bbox_geojson: dict, allow_caching: bool) -> Path:
    bbox_hash = get_hash_of_geojson(bbox_geojson)
    if tiff_for_bbox_is_cached(bbox_hash) and allow_caching:
        click.echo("🚀  using cached tiff...                           ✅ (0.0s)")
        return _path_to_clipped_tiff(bbox_hash)
    else:
        return _fetch_merge_and_clip_tiffs(bbox_geojson, bbox_hash, allow_caching)


def convert_bbox_to_stl(
    bbox_geometry: dict,
    as_ascii: bool = False,
    model_size: int = 200,
    output_file: str = "output.stl",
    max_res: bool = False,
    z_offset: float = 0.0,
    z_scale: float = 1.0,
    cut_to_format_ratio: Union[None, float] = None,
    allow_caching: bool = True,
) -> Path:
    """
    Takes a GeoJSON containing a bounding box as input, fetches the required STAC GeoTIFFs for the
    given bounding box and creates a STL file with elevation data from the GeoTIFFs.

    Parameters
    ----------
    bbox_geometry : dict
        GeoJSON containing the coordinates of the bounding box, selected on the ipyleaflet widget.
        Usually the value of `drawer.last_draw["geometry"]` is used for this.
    as_ascii : bool, optional
        Save output STL as ascii file. If False, output file will be binary. By default False
    model_size : int, optional
        Desired size of the (larger side of the) generated 3d model in millimeter. By default 200
    output_file : str, optional
        Path to output STL file. By default "output.stl"
    max_res : bool, optional
        Whether maximum resolution should be used. Note, that this flag potentially increases compute time
        and memory consumption dramatically. The default behavior (i.e. max_res=False) should return 3d models
        with sufficient resolution, while the output stl file should be < ~300 MB. By default False
    z_offset : float, optional
        Offset distance in millimeter to be put below the 3d model. Is not influenced by z-scale.
        By default 0.0
    z_scale : float, optional
        Value to be multiplied to the z-axis elevation data to scale up the height of the model.
        By default 1.0
    cut_to_format_ratio : Union[None, float], optional
        Cut the input tiff file to a specified format. Set to `1` if you want the output model to be squared.
        Set to `0.5` if you want one side to be half the length of the other side. Omit this flag to keep the
        input format. This option is particularly useful when an exact output format ratio is required for
        example when planning to put the 3d printed model into a picture frame. Using this option will always
        try to cut the shorter side of the input tiff. By default None
    allow_caching : bool, optional
        Whether caching previous downloaded GeoTIFF files should be enabled/disabled. By default True

    Returns
    -------
    Path
        Path to the resulting STL file on your local machine.
    """

    if bbox_geometry is None:
        click.echo("⛔️  ERROR: make sure to draw a rectangle on the map first!")
        return

    click.echo("⏳  converting bounding box to STL file... \n")

    tiff = _get_tiff_for_bbox(bbox_geometry, allow_caching)
    output_file = convert_tiff_to_stl(
        input_file=tiff,
        as_ascii=as_ascii,
        model_size=model_size,
        output_file=output_file,
        max_res=max_res,
        z_offset=z_offset,
        z_scale=z_scale,
        cut_to_format_ratio=cut_to_format_ratio,
    )
    return output_file
