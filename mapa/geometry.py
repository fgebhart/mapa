from pathlib import Path
from typing import Tuple, Union

import click
import numba as nb
import numpy as np
import numpy.typing as npt
import stl
from numpy.lib.stride_tricks import as_strided
from stl import mesh

from mapa.utils import timing


@timing
@nb.njit(fastmath=True, cache=True)
def _create_raster(array: npt.ArrayLike, max_x: int, max_y: int) -> np.ndarray:
    max_x, max_y = array.shape
    raster = np.zeros((max_x + 1, max_y + 1))
    # loop over raster elements to determine z value of raster elements
    for ix in range(0, raster.shape[0]):
        for iy in range(0, raster.shape[1]):
            # special treatment of first and last rows/cols
            if ix >= max_x or iy >= max_y:
                if ix >= max_x and iy < max_y:
                    raster[ix][iy] = array[ix - 1][iy]
                elif iy >= max_y and ix < max_x:
                    raster[ix][iy] = array[ix][iy - 1]
                else:
                    raster[ix][iy] = array[ix - 1][iy - 1]
            elif ix == 0 or iy == 0:
                raster[ix][iy] = array[ix][iy]
            else:
                # z value in raster is average of four neighbors
                raster[ix][iy] = (array[ix][iy] + array[ix - 1][iy] + array[ix][iy - 1] + array[ix - 1][iy - 1]) / 4
    return raster


@timing
@nb.njit(fastmath=True, cache=True)
def _compute_triangles_of_3d_surface(
    raster: npt.ArrayLike,
    array: npt.ArrayLike,
    max_x: int,
    max_y: int,
    x_scale: float,
    y_scale: float,
    z_scale: float,
    z_offset: float,
) -> np.ndarray:
    triangles = np.full((max_x, max_y, 4, 3, 3), -1.0)
    for ix in range(0, max_x):
        for iy in range(0, max_y):
            if ix > max_x or iy > max_y:
                continue
            else:
                # top triangle
                # first vertex
                triangles[ix, iy, 0, 0, 0] = (ix + 1 / 2) * x_scale
                triangles[ix, iy, 0, 0, 1] = (iy + 1 / 2) * y_scale
                triangles[ix, iy, 0, 0, 2] = (array[ix, iy]) * z_scale + z_offset
                # second vertex
                triangles[ix, iy, 0, 1, 0] = ix * x_scale
                triangles[ix, iy, 0, 1, 1] = iy * y_scale
                triangles[ix, iy, 0, 1, 2] = (raster[ix, iy]) * z_scale + z_offset
                # third vertex
                triangles[ix, iy, 0, 2, 0] = (ix + 1) * x_scale
                triangles[ix, iy, 0, 2, 1] = iy * y_scale
                triangles[ix, iy, 0, 2, 2] = (raster[ix + 1, iy]) * z_scale + z_offset

                # left triangle
                # first vertex
                triangles[ix, iy, 1, 0, 0] = ix * x_scale
                triangles[ix, iy, 1, 0, 1] = (iy + 1) * y_scale
                triangles[ix, iy, 1, 0, 2] = (raster[ix, iy + 1]) * z_scale + z_offset
                # second vertex
                triangles[ix, iy, 1, 1, 0] = ix * x_scale
                triangles[ix, iy, 1, 1, 1] = iy * y_scale
                triangles[ix, iy, 1, 1, 2] = (raster[ix, iy]) * z_scale + z_offset
                # third vertex
                triangles[ix, iy, 1, 2, 0] = (ix + 1 / 2) * x_scale
                triangles[ix, iy, 1, 2, 1] = (iy + 1 / 2) * y_scale
                triangles[ix, iy, 1, 2, 2] = (array[ix, iy]) * z_scale + z_offset

                # bottom triangle
                # first vertex
                triangles[ix, iy, 2, 0, 0] = (ix + 1) * x_scale
                triangles[ix, iy, 2, 0, 1] = (iy + 1) * y_scale
                triangles[ix, iy, 2, 0, 2] = (raster[ix + 1, iy + 1]) * z_scale + z_offset
                # second vertex
                triangles[ix, iy, 2, 1, 0] = ix * x_scale
                triangles[ix, iy, 2, 1, 1] = (iy + 1) * y_scale
                triangles[ix, iy, 2, 1, 2] = (raster[ix, iy + 1]) * z_scale + z_offset
                # third vertex
                triangles[ix, iy, 2, 2, 0] = (ix + 1 / 2) * x_scale
                triangles[ix, iy, 2, 2, 1] = (iy + 1 / 2) * y_scale
                triangles[ix, iy, 2, 2, 2] = (array[ix, iy]) * z_scale + z_offset

                # right triangle
                # first vertex
                triangles[ix, iy, 3, 0, 0] = (ix + 1 / 2) * x_scale
                triangles[ix, iy, 3, 0, 1] = (iy + 1 / 2) * y_scale
                triangles[ix, iy, 3, 0, 2] = (array[ix, iy]) * z_scale + z_offset
                # second vertex
                triangles[ix, iy, 3, 1, 0] = (ix + 1) * x_scale
                triangles[ix, iy, 3, 1, 1] = iy * y_scale
                triangles[ix, iy, 3, 1, 2] = (raster[ix + 1, iy]) * z_scale + z_offset
                # third vertex
                triangles[ix, iy, 3, 2, 0] = (ix + 1) * x_scale
                triangles[ix, iy, 3, 2, 1] = (iy + 1) * y_scale
                triangles[ix, iy, 3, 2, 2] = (raster[ix + 1, iy + 1]) * z_scale + z_offset

    return triangles.reshape((max_x * max_y * 4, 3, 3))


@timing
def _compute_triangles_of_body_side(
    raster: npt.ArrayLike, max_x: int, max_y: int, x_scale: float, y_scale: float, z_scale: float, z_offset: float
) -> np.ndarray:
    # loop over raster and build triangles when in first and last col and row
    triangles = []
    for ix, row in enumerate(raster):
        if ix >= max_x:
            continue
        for iy, _ in enumerate(row):
            if iy >= max_y:
                continue
            if ix == 0:  # first row
                triangles.append(  # triangle with two points at top of mesh
                    [
                        [0, iy * y_scale, raster[ix][iy] * z_scale + z_offset],  # first point in col
                        [0, (iy + 1) * y_scale, raster[ix][iy + 1] * z_scale + z_offset],  # second point in col
                        [0, iy * y_scale, 0],  # first point on ground
                    ]
                )
                triangles.append(  # triangle with two points at ground
                    [
                        [0, iy * y_scale, 0],
                        [0, (iy + 1) * y_scale, raster[ix][iy + 1] * z_scale + z_offset],
                        [0, (iy + 1) * y_scale, 0],
                    ]
                )
            if ix == max_x - 1:  # last row
                triangles.append(  # two points at top 3d mesh
                    [
                        [
                            max_x * x_scale,
                            (iy + 1) * y_scale,
                            raster[ix][iy + 1] * z_scale + z_offset,
                        ],  # second point in col
                        [max_x * x_scale, iy * y_scale, raster[ix][iy] * z_scale + z_offset],  # first point in col
                        [max_x * x_scale, iy * y_scale, 0],  # first point on ground
                    ]
                )
                triangles.append(  # two points at ground
                    [
                        [max_x * x_scale, (iy + 1) * y_scale, raster[ix][iy + 1] * z_scale + z_offset],
                        [max_x * x_scale, iy * y_scale, 0],
                        [max_x * x_scale, (iy + 1) * y_scale, 0],
                    ]
                )
            if iy == 0:  # first col
                # two points at top 3d mesh
                triangles.append(
                    [
                        [(ix + 1) * x_scale, 0, raster[ix + 1][iy] * z_scale + z_offset],  # second point in col
                        [ix * x_scale, 0, raster[ix][iy] * z_scale + z_offset],  # first point in col
                        [ix * x_scale, 0, 0],  # first point on ground
                    ]
                )
                # two points at ground
                triangles.append(
                    [
                        [(ix + 1) * x_scale, 0, raster[ix + 1][iy] * z_scale + z_offset],
                        [ix * x_scale, 0, 0],
                        [(ix + 1) * x_scale, 0, 0],
                    ]
                )
            if iy == max_y - 1:  # last col
                # two points at top 3d mesh
                triangles.append(
                    [
                        [ix * x_scale, max_y * y_scale, raster[ix][iy] * z_scale + z_offset],  # first point in col
                        [
                            (ix + 1) * x_scale,
                            max_y * y_scale,
                            raster[ix + 1][iy] * z_scale + z_offset,
                        ],  # second point in col
                        [ix * x_scale, max_y * y_scale, 0],  # first point on ground
                    ]
                )
                # two points at ground
                triangles.append(
                    [
                        [ix * x_scale, max_y * y_scale, 0],
                        [(ix + 1) * x_scale, max_y * y_scale, raster[ix + 1][iy] * z_scale + z_offset],
                        [(ix + 1) * x_scale, max_y * y_scale, 0],
                    ]
                )
    return triangles


def _compute_triangles_of_bottom(max_x: int, max_y: int, x_scale: float, y_scale: float) -> np.ndarray:
    triangles = []
    triangles.append(
        [
            [0, 0, 0],
            [0, max_y * y_scale, 0],
            [max_x * x_scale, 0, 0],
        ]
    )
    triangles.append(
        [
            [0, max_y * y_scale, 0],
            [max_x * x_scale, max_y * y_scale, 0],
            [max_x * x_scale, 0, 0],
        ]
    )
    return triangles


def compute_all_triangles(
    array: npt.ArrayLike,
    target_size: int,
    z_offset: float,
    z_scale: float,
    cut_to_format_ratio: Union[float, None],
) -> np.ndarray:

    # determine scales
    max_x, max_y = array.shape
    x_scale = target_size / max_x
    if cut_to_format_ratio:
        if cut_to_format_ratio > 1.0:
            # ensure ratio is between 0.0 and 1.0 and transpose ratio
            cut_to_format_ratio = cut_to_format_ratio**-1
        y_scale = target_size * cut_to_format_ratio / max_y
    else:
        y_scale = target_size / max_x

    # create raster
    click.echo(f"{'🗺  creating base raster for tiff...':<50s}", nl=False)
    raster = _create_raster(array, max_x, max_y)

    # determine z_offset
    if z_offset is None:
        # using the natural height, i.e. islands will have an z_offset of ~0 and mountains will have a larger z_offset
        z_offset = raster.min()
        z_offset = 0 if z_offset > 0 else z_offset
    else:
        # use given input offset as height to ground
        z_offset = z_offset
        if z_offset < 0:
            click.echo("☝️  Warning: Be careful using negative z_offsets, as it might break your 3D model.")

    # compute triangles
    click.echo(f"{'⛰  computing triangles of 3d surface...':<50s}", nl=False)
    dem_triangles = _compute_triangles_of_3d_surface(
        raster=raster,
        array=array,
        max_x=max_x,
        max_y=max_y,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        z_offset=z_offset,
    )
    click.echo(f"{'📐  computing triangles of body sides...':<50s}", nl=False)
    side_triangles = _compute_triangles_of_body_side(
        raster=raster,
        max_x=max_x,
        max_y=max_y,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        z_offset=z_offset,
    )
    bottom_triangles = _compute_triangles_of_bottom(max_x=max_x, max_y=max_y, x_scale=x_scale, y_scale=y_scale)
    return np.vstack((dem_triangles, side_triangles, bottom_triangles))


def _find_dimensions_of_mesh(mesh_obj) -> Tuple[float]:
    minx = maxx = miny = maxy = minz = maxz = None
    for p in mesh_obj.points:
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    x = maxx - minx
    y = maxy - miny
    z = maxz - minz
    return x, y, z


def get_dimensions_of_stl_file(stl_path: Path) -> Tuple[float]:
    main_body = mesh.Mesh.from_file(stl_path)
    return _find_dimensions_of_mesh(main_body)


@timing
def reduce_resolution(array: npt.ArrayLike, bin_factor: int) -> np.ndarray:
    strided = as_strided(
        array,
        shape=(array.shape[0] // bin_factor, array.shape[1] // bin_factor, bin_factor, bin_factor),
        strides=((array.strides[0] * bin_factor, array.strides[1] * bin_factor) + array.strides),
    )
    return strided.mean(axis=-1).mean(axis=-1)
