from pathlib import Path

import click
import numpy as np
import rasterio as rio
from stl import Mode, mesh

from mapa import conf
from mapa.caching import _get_hash_of_geojson, _tiff_for_bbox_is_cached
from mapa.geometry import compute_all_triangles, reduce_resolution
from mapa.raster import (
    clip_tiff_to_bbox,
    cut_array_to_square,
    determine_elevation_scale,
    merge_tiffs,
    remove_empty_first_and_last_rows_and_cols,
)
from mapa.stac import fetch_stac_items_for_bbox
from mapa.utils import _path_to_clipped_tiff, debug_image, timing


def _verify_input_is_valid(input: str):
    input_path = Path(input)
    if not input_path.is_file():
        click.echo(f"input file: '{input}' does not seem to be a file")
        exit(1)
    if input_path.suffix in conf.SUPPORTED_INPUT_FORMAT:
        pass  # ok
    else:
        click.echo(
            f"input file '{input}' does not seem to be a tiff file, only {conf.SUPPORTED_INPUT_FORMAT} are supported."
        )
        exit(1)


def _verify_output_is_valid(output: str):
    output_path = Path(output)
    if not output_path.parent.is_dir():
        click.echo(f"parent of output file '{output_path.parent}' does not seem to be a valid directory.")
        exit(1)


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
    make_square: bool,
    elevation_scale: float,
    output_file: Path,
    debug: bool,
) -> Path:
    # drop higher dimension to get 2-dimensional (x * y) array
    array = array[0, :, :]
    debug_image(debug, array, "input array:")
    array = remove_empty_first_and_last_rows_and_cols(array)
    if make_square:
        array = cut_array_to_square(array)
        debug_image(debug, array, "array cut to square: ")

    x, y = array.shape
    if max_res:
        if x * y > conf.PERFORMANCE_WARNING_THRESHOLD:
            click.warning(
                "⛔️  Warning: Using max_res=True on the selected region might cause performance issues. "
                "Consider setting max_res=False."
            )
    else:
        bin_fac = round((x / conf.MAXIMUM_RESOLUTION + y / conf.MAXIMUM_RESOLUTION) / 2)
        if bin_fac > 1:
            click.echo(f"{'🔍  reducing image resolution...':<50s}", nl=False)
            array = reduce_resolution(array, bin_factor=bin_fac)
            debug_image(debug, array, "reduced resolution of array: ")
    combined_z_scale = z_scale * elevation_scale
    triangles = compute_all_triangles(array, model_size, z_offset, combined_z_scale)
    click.echo(f"{'💾  saving data to stl file...':<50s}", nl=False)
    output_file = _save_to_stl_file(triangles, output_file, as_ascii)
    click.echo(f"\n🎉  successfully generated STL file: {Path(output_file).absolute()}")
    return Path(output_file)


def convert_tif_to_stl(
    input_file: str,
    as_ascii: bool,
    model_size: int,
    output_file: str,
    max_res: bool,
    z_offset: float,
    z_scale: float,
    make_square: bool,
    debug: bool = False,
) -> Path:
    _verify_input_is_valid(input_file)
    if output_file is None:
        output_file = Path.cwd() / str(Path(input_file).name).replace(".tiff", ".stl").replace(".tif", ".stl")
    _verify_output_is_valid(output_file)

    debug_image(debug, input_file, "clipped tiff: ")
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
        make_square=make_square,
        elevation_scale=elevation_scale,
        output_file=output_file,
        debug=debug,
    )


def _fetch_merge_and_clip_tiffs(bbox_geojson: dict, debug: bool, bbox_hash: str) -> Path:
    tiffs = fetch_stac_items_for_bbox(bbox_geojson, debug)
    if len(tiffs) > 1:
        merged_tiff = merge_tiffs(tiffs, bbox_hash)
    else:
        merged_tiff = tiffs[0]
    debug_image(debug, merged_tiff, "merged tiff: ")
    return clip_tiff_to_bbox(merged_tiff, bbox_geojson, bbox_hash)


def _get_tiff_for_bbox(bbox_geojson: dict, debug: bool) -> Path:
    bbox_hash = _get_hash_of_geojson(bbox_geojson)
    if _tiff_for_bbox_is_cached(bbox_hash):
        click.echo("🚀  using cached tiff...                           ✅ (0.0s)")
        return _path_to_clipped_tiff(bbox_hash)
    else:
        return _fetch_merge_and_clip_tiffs(bbox_geojson, debug, bbox_hash)


def create_stl_for_bbox(
    bbox_geometry: dict,
    as_ascii: bool = False,
    model_size: int = 200,
    output_file: str = "output.stl",
    max_res: bool = False,
    z_offset: float = 0.0,
    z_scale: float = 1.0,
    make_square: bool = False,
    debug: bool = False,
) -> Path:
    if bbox_geometry is None:
        print("ERROR: make sure to draw a rectangle on the map first!")
        return

    click.echo("⏳  converting bounding box to STL file... \n")

    tiff = _get_tiff_for_bbox(bbox_geometry, debug)
    output_file = convert_tif_to_stl(
        input_file=tiff,
        as_ascii=as_ascii,
        model_size=model_size,
        output_file=output_file,
        max_res=max_res,
        z_offset=z_offset,
        z_scale=z_scale,
        make_square=make_square,
        debug=debug,
    )
    return output_file
