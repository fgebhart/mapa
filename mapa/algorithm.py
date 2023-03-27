"""
Explanation of the Algorithm
============================

First of all it is crucial to understand how a STL file is constructed. For this purpose I would recommend reading the
following articles, since this explanation aims on outlining the implementation of the below algorithm only:
* http://www.fabbers.com/tech/STL_Format
* https://danbscott.ghost.io/writing-an-stl-file-from-scratch/

With that in mind, we basically only need to accomplish the transformation of pixels into facets (triangles). This is
achieved by the following steps:


0. Reading the GeoTiff and turning it into a 2d array where each pixel value equals the altitude
------------------------------------------------------------------------------------------------

Reading the GeoTiff and turning it into a numpy array is not considered part of the below algorithm. It is taken care of
by `mapa/__init__.py` and not described in greater detail here.


1. Creating a 2d array (raster) given the input GeoTIFF data
------------------------------------------------------------

A 2d numpy array is created within `_create_raster`. It has one more row and one more column compared to the input
array of the GeoTIFF. That is because we are going to describe each input pixel by four triangles.

A pixel with center C and corners 1,2,3 and 4 represented by four triangles:

1-----------------2
| x             x |
|   x         x   |
|     x     x     |
|      x   x      |
|        C        |
|      x   x      |
|     x     x     |
|   x         x   |
| x             x |
3-----------------4



All four triangles are connected to each other in the center C of the pixel. Imaging we are moving the raster slightly
across/above the input array.

r   r   r   r   r   r
  a   a   a   a   a
r   r   r   r   r   r
  a   a   a   a   a
r   r   r   r   r   r
  a   a   a   a   a
r   r   r   r   r   r
  a   a   a   a   a
r   r   r   r   r   r
  a   a   a   a   a
r   r   r   r   r   r

The values of the raster r can then be used to describe the altitude at the corners (1,2,3 and 4) of a pixel, while the
values of the array a describe the altitude at the center C of the pixel. With this approach the information for
computing the position of the required triangles representing the 3d surface of the desired output model can easily be
determined by looking into the input array and the constructed raster.


2. Computing the triangles representing the elevation data
----------------------------------------------------------

With the above step in mind, determining the altitude values basically consists of a lookup into the given raster and
input array. The function `_compute_triangles_of_3d_surface` takes care of this while at the same time adding offset
values and scaling the x, y and z-axis to achieve the desired model size. The return value of the function is a numpy
array containing the computed triangles. Each triangle T consists of 3 vertices V where each vertex V corresponds to one
coordinates C, where each coordinate C in turn consists of a X, Y and Z value:

T = V1, V2, V3
V = C
C = X, Y, Z

This means each triangle consists of nine values. The `_compute_triangles_of_3d_surface` therefore needs to iterate over
all pixels and compute the positions of the four triangles for each pixel. The triangles per pixel are described as top,
left, bottom and right triangles.


3. Computing the triangles representing the side and bottom of the output 3d model
----------------------------------------------------------------------------------

With the above steps we manage to compute the triangles for describing the 3d surface which we derived from the input
GeoTIFF file. However, in order to make a 3d-printable STL file, we need to ensure that the resulting mesh is actually
closed i.e. watertight. This is done by `_compute_triangles_of_body_side` and `_compute_triangles_of_bottom`. The
vertices of the triangles for the side of the resulting STL model are computed in the following fashion. Imaging moving
along one side (x or y) of the computed 3d surface. Each coordinate along this side will be the considered the vertex of
two triangles, while the next coordinate along the side of the surface is the second vertex and one coordinate at the
bottom of the model will be the third vertex of the triangle. Or one vertex at the surface and two at the bottom.

        s
s   s       s       s           s       s   s
                s       s   s       s           s

b   b   b   b   b   b   b   b   b   b   b   b   b

s illustrates a coordinate at the side of the 3d surface and b a coordinate at the bottom of the model. Imaging drawing
triangles between the s's and the b's. This approach is used to compute the side triangles within
`_compute_triangles_of_body_side`.

Computing the triangles of the bottom in the scope of `_compute_triangles_of_bottom` is super straight forward, as it
only needs to compute two triangles, where the altitude value z is zero and all other x and y values correspond the
desired model dimensions.


4. Write triangles to STL file
------------------------------

This is not considered part of the algorithm and is taken care of by `mapa/__init__._save_to_stl_file`. It boils down to
the usage of numpy-stl, which has a super convenient and efficient interface for writing triangles to a (binary of ascii)
STL file.
"""

import logging
from dataclasses import dataclass
from typing import Union

import numba as nb
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided

log = logging.getLogger(__name__)


@dataclass
class ModelSize:
    x: float
    y: float


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
    triangles = np.full((max_x, max_y, 4, 3, 3), -1.0, dtype=np.float64)
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


def _compute_triangles_of_body_side(
    raster: npt.ArrayLike, max_x: int, max_y: int, x_scale: float, y_scale: float, z_scale: float, z_offset: float
) -> np.ndarray:
    # loop over raster and build triangles when in first and last col and row
    triangles = np.full((max_x * 4 + max_y * 4, 3, 3), -1.0, dtype=np.float64)
    cnt = 0

    for ix in range(0, max_x):
        for iy in range(0, max_y):
            if ix == 0:  # first row
                # triangle with two points at top of mesh
                # first vertex
                triangles[cnt + 0, 0, 0] = 0
                triangles[cnt + 0, 0, 1] = iy * y_scale
                triangles[cnt + 0, 0, 2] = raster[ix][iy] * z_scale + z_offset
                # second vertex
                triangles[cnt + 0, 1, 0] = 0
                triangles[cnt + 0, 1, 1] = (iy + 1) * y_scale
                triangles[cnt + 0, 1, 2] = raster[ix][iy + 1] * z_scale + z_offset
                # third vertex
                triangles[cnt + 0, 2, 0] = 0
                triangles[cnt + 0, 2, 1] = iy * y_scale
                triangles[cnt + 0, 2, 2] = 0
                # triangle with two points at ground
                # first vertex
                triangles[cnt + 1, 0, 0] = 0
                triangles[cnt + 1, 0, 1] = iy * y_scale
                triangles[cnt + 1, 0, 2] = 0
                # second vertex
                triangles[cnt + 1, 1, 0] = 0
                triangles[cnt + 1, 1, 1] = (iy + 1) * y_scale
                triangles[cnt + 1, 1, 2] = raster[ix][iy + 1] * z_scale + z_offset
                # third vertex
                triangles[cnt + 1, 2, 0] = 0
                triangles[cnt + 1, 2, 1] = (iy + 1) * y_scale
                triangles[cnt + 1, 2, 2] = 0
                cnt += 2
            if ix == max_x - 1:  # last row
                # two points at top 3d mesh
                # first vertex
                triangles[cnt + 0, 0, 0] = max_x * x_scale
                triangles[cnt + 0, 0, 1] = (iy + 1) * y_scale
                triangles[cnt + 0, 0, 2] = raster[ix + 1][iy + 1] * z_scale + z_offset
                # second vertex
                triangles[cnt + 0, 1, 0] = max_x * x_scale
                triangles[cnt + 0, 1, 1] = iy * y_scale
                triangles[cnt + 0, 1, 2] = raster[ix + 1][iy] * z_scale + z_offset
                # third vertex
                triangles[cnt + 0, 2, 0] = max_x * x_scale
                triangles[cnt + 0, 2, 1] = iy * y_scale
                triangles[cnt + 0, 2, 2] = 0
                # two points at ground
                # first vertex
                triangles[cnt + 1, 0, 0] = max_x * x_scale
                triangles[cnt + 1, 0, 1] = (iy + 1) * y_scale
                triangles[cnt + 1, 0, 2] = raster[ix + 1][iy + 1] * z_scale + z_offset
                # second vertex
                triangles[cnt + 1, 1, 0] = max_x * x_scale
                triangles[cnt + 1, 1, 1] = iy * y_scale
                triangles[cnt + 1, 1, 2] = 0
                # third vertex
                triangles[cnt + 1, 2, 0] = max_x * x_scale
                triangles[cnt + 1, 2, 1] = (iy + 1) * y_scale
                triangles[cnt + 1, 2, 2] = 0
                cnt += 2
            if iy == 0:  # first col
                # two points at top 3d mesh
                # first vertex
                triangles[cnt + 0, 0, 0] = (ix + 1) * x_scale
                triangles[cnt + 0, 0, 1] = 0
                triangles[cnt + 0, 0, 2] = raster[ix + 1][iy] * z_scale + z_offset
                # second vertex
                triangles[cnt + 0, 1, 0] = ix * x_scale
                triangles[cnt + 0, 1, 1] = 0
                triangles[cnt + 0, 1, 2] = raster[ix][iy] * z_scale + z_offset
                # third vertex
                triangles[cnt + 0, 2, 0] = ix * x_scale
                triangles[cnt + 0, 2, 1] = 0
                triangles[cnt + 0, 2, 2] = 0
                # two points at ground
                # first vertex
                triangles[cnt + 1, 0, 0] = (ix + 1) * x_scale
                triangles[cnt + 1, 0, 1] = 0
                triangles[cnt + 1, 0, 2] = raster[ix + 1][iy] * z_scale + z_offset
                # second vertex
                triangles[cnt + 1, 1, 0] = ix * x_scale
                triangles[cnt + 1, 1, 1] = 0
                triangles[cnt + 1, 1, 2] = 0
                # third vertex
                triangles[cnt + 1, 2, 0] = (ix + 1) * x_scale
                triangles[cnt + 1, 2, 1] = 0
                triangles[cnt + 1, 2, 2] = 0
                cnt += 2
            if iy == max_y - 1:  # last col
                # two points at top 3d mesh
                # first vertex
                triangles[cnt + 0, 0, 0] = ix * x_scale
                triangles[cnt + 0, 0, 1] = max_y * y_scale
                triangles[cnt + 0, 0, 2] = raster[ix][iy + 1] * z_scale + z_offset
                # second vertex
                triangles[cnt + 0, 1, 0] = (ix + 1) * x_scale
                triangles[cnt + 0, 1, 1] = max_y * y_scale
                triangles[cnt + 0, 1, 2] = raster[ix + 1][iy + 1] * z_scale + z_offset
                # third vertex
                triangles[cnt + 0, 2, 0] = ix * x_scale
                triangles[cnt + 0, 2, 1] = max_y * y_scale
                triangles[cnt + 0, 2, 2] = 0
                # two points at ground
                # first vertex
                triangles[cnt + 1, 0, 0] = ix * x_scale
                triangles[cnt + 1, 0, 1] = max_y * y_scale
                triangles[cnt + 1, 0, 2] = 0
                # second vertex
                triangles[cnt + 1, 1, 0] = (ix + 1) * x_scale
                triangles[cnt + 1, 1, 1] = max_y * y_scale
                triangles[cnt + 1, 1, 2] = raster[ix + 1][iy + 1] * z_scale + z_offset
                # third vertex
                triangles[cnt + 1, 2, 0] = (ix + 1) * x_scale
                triangles[cnt + 1, 2, 1] = max_y * y_scale
                triangles[cnt + 1, 2, 2] = 0
                cnt += 2
    return triangles


def _compute_triangles_of_bottom(max_x: int, max_y: int, x_scale: float, y_scale: float) -> np.ndarray:
    # first row
    fr_triangles = np.full((max_x - 1, 3, 3), -1.0, dtype=np.float64)
    for i, cnt in enumerate(range(0, max_x - 1)):
        fr_triangles[i, 0, 0] = cnt * x_scale
        fr_triangles[i, 0, 1] = 0
        fr_triangles[i, 0, 2] = 0
        fr_triangles[i, 1, 0] = 0
        fr_triangles[i, 1, 1] = 1 * y_scale
        fr_triangles[i, 1, 2] = 0
        fr_triangles[i, 2, 0] = (cnt + 1) * x_scale
        fr_triangles[i, 2, 1] = 0
        fr_triangles[i, 2, 2] = 0

    # first col
    fc_triangles = np.full((max_y - 1, 3, 3), -1.0, dtype=np.float64)
    for i, cnt in enumerate(range(1, max_y)):
        fc_triangles[i, 0, 0] = 0
        fc_triangles[i, 0, 1] = cnt * y_scale
        fc_triangles[i, 0, 2] = 0
        fc_triangles[i, 1, 0] = 0
        fc_triangles[i, 1, 1] = (cnt + 1) * y_scale
        fc_triangles[i, 1, 2] = 0
        fc_triangles[i, 2, 0] = 1 * x_scale
        fc_triangles[i, 2, 1] = max_y * y_scale
        fc_triangles[i, 2, 2] = 0

    # last row
    lr_triangles = np.full((max_x - 1, 3, 3), -1.0, dtype=np.float64)
    for i, cnt in enumerate(range(1, max_x)):
        lr_triangles[i, 0, 0] = cnt * x_scale
        lr_triangles[i, 0, 1] = max_y * y_scale
        lr_triangles[i, 0, 2] = 0
        lr_triangles[i, 1, 0] = (cnt + 1) * x_scale
        lr_triangles[i, 1, 1] = max_y * y_scale
        lr_triangles[i, 1, 2] = 0
        lr_triangles[i, 2, 0] = max_x * x_scale
        lr_triangles[i, 2, 1] = (max_y - 1) * y_scale
        lr_triangles[i, 2, 2] = 0

    # last col
    lc_triangles = np.full((max_y - 1, 3, 3), -1.0, dtype=np.float64)
    for i, cnt in enumerate(range(0, max_y - 1)):
        lc_triangles[i, 0, 0] = max_x * x_scale
        lc_triangles[i, 0, 1] = cnt * y_scale
        lc_triangles[i, 0, 2] = 0
        lc_triangles[i, 1, 0] = (max_x - 1) * x_scale
        lc_triangles[i, 1, 1] = 0
        lc_triangles[i, 1, 2] = 0
        lc_triangles[i, 2, 0] = max_x * x_scale
        lc_triangles[i, 2, 1] = (cnt + 1) * y_scale
        lc_triangles[i, 2, 2] = 0

    center_triangles = np.full((2, 3, 3), -1.0, dtype=np.float64)
    center_triangles[0, 0, 0] = (max_x - 1) * x_scale
    center_triangles[0, 0, 1] = 0 * y_scale
    center_triangles[0, 0, 2] = 0
    center_triangles[0, 1, 0] = 1 * x_scale
    center_triangles[0, 1, 1] = max_y * y_scale
    center_triangles[0, 1, 2] = 0
    center_triangles[0, 2, 0] = max_x * x_scale
    center_triangles[0, 2, 1] = (max_y - 1) * y_scale
    center_triangles[0, 2, 2] = 0

    center_triangles[1, 0, 0] = 1 * x_scale
    center_triangles[1, 0, 1] = max_y * y_scale
    center_triangles[1, 0, 2] = 0
    center_triangles[1, 1, 0] = (max_x - 1) * x_scale
    center_triangles[1, 1, 1] = 0 * y_scale
    center_triangles[1, 1, 2] = 0
    center_triangles[1, 2, 0] = 0 * x_scale
    center_triangles[1, 2, 1] = 1 * y_scale
    center_triangles[1, 2, 2] = 0

    return np.vstack((fr_triangles, lr_triangles, fc_triangles, lc_triangles, center_triangles))


def _determine_z_offset(z_offset: Union[None, float], minimum: float, elevation_scale: float) -> float:
    if z_offset is None:
        # using the natural height, i.e. islands will have an z_offset of ~0 and mountains will have a larger z_offset
        return minimum * elevation_scale
    else:
        if z_offset < 0:
            log.warning("☝️  Warning: Be careful using negative z_offsets, as it might break your 3D model.")
        # subtract scaled minimum from z_offset to ensure the input z_offset will remain
        return z_offset - minimum * elevation_scale


def compute_all_triangles(
    array: npt.ArrayLike,
    desired_size: ModelSize,
    z_offset: Union[None, float],
    z_scale: float,
    elevation_scale: float,
) -> np.ndarray:
    max_x, max_y = array.shape

    log.debug("🗺  creating base raster for tiff...")
    raster = _create_raster(array, max_x, max_y)

    x_scale, y_scale = desired_size.x / max_x, desired_size.y / max_y
    z_offset = _determine_z_offset(z_offset, raster.min(), elevation_scale)
    combined_z_scale = elevation_scale * z_scale

    # compute triangles for 3d surface, sides and bottom
    log.debug("⛰  computing triangles of 3d surface...")
    dem_triangles = _compute_triangles_of_3d_surface(
        raster=raster,
        array=array,
        max_x=max_x,
        max_y=max_y,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=combined_z_scale,
        z_offset=z_offset,
    )
    log.debug("📐  computing triangles of body sides...")
    side_triangles = _compute_triangles_of_body_side(
        raster=raster,
        max_x=max_x,
        max_y=max_y,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=combined_z_scale,
        z_offset=z_offset,
    )
    bottom_triangles = _compute_triangles_of_bottom(max_x=max_x, max_y=max_y, x_scale=x_scale, y_scale=y_scale)
    return np.vstack((dem_triangles, side_triangles, bottom_triangles))


def reduce_resolution(array: npt.ArrayLike, bin_factor: int) -> np.ndarray:
    strided = as_strided(
        array,
        shape=(array.shape[0] // bin_factor, array.shape[1] // bin_factor, bin_factor, bin_factor),
        strides=((array.strides[0] * bin_factor, array.strides[1] * bin_factor) + array.strides),
    )
    return strided.mean(axis=-1).mean(axis=-1)
