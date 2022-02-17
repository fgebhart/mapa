from pathlib import Path
from typing import Tuple

import numpy as np
from stl import Dimension, Mode, mesh

from mapa.utils import timing


@timing
def save_to_stl_file(triangles: np.ndarray, output_file: str, as_ascii: bool) -> str:
    stl = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
    stl.vectors = triangles
    if as_ascii:
        stl.save(output_file, mode=Mode.ASCII)
    else:
        stl.save(output_file, mode=Mode.BINARY)
    return output_file


def _find_dimensions_of_mesh(mesh_obj) -> Tuple[float]:
    minx = maxx = miny = maxy = minz = maxz = None
    for p in mesh_obj.points:
        if minx is None:
            minx = p[Dimension.X]
            maxx = p[Dimension.X]
            miny = p[Dimension.Y]
            maxy = p[Dimension.Y]
            minz = p[Dimension.Z]
            maxz = p[Dimension.Z]
        else:
            maxx = max(p[Dimension.X], maxx)
            minx = min(p[Dimension.X], minx)
            maxy = max(p[Dimension.Y], maxy)
            miny = min(p[Dimension.Y], miny)
            maxz = max(p[Dimension.Z], maxz)
            minz = min(p[Dimension.Z], minz)
    x = maxx - minx
    y = maxy - miny
    z = maxz - minz
    return x, y, z


def get_dimensions_of_stl_file(stl_path: Path) -> Tuple[float]:
    main_body = mesh.Mesh.from_file(stl_path)
    return _find_dimensions_of_mesh(main_body)
