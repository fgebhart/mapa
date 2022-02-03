import tempfile
from functools import wraps
from pathlib import Path
from time import time
from typing import Union

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import requests
from rasterio.plot import show


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        # start_msg = f"{f.__name__} ..."
        # click.echo(f"{start_msg:<40s}", nl=False)
        result = f(*args, **kw)
        te = time()
        end_msg = f"✅ ({round(te - ts, 1)}s)"
        click.echo(f"{end_msg:<20s}")
        return result

    return wrap


def download_file(url: str, local_file: Path) -> Path:
    data = requests.get(url, stream=True)
    with open(local_file, "wb") as file:
        file.write(data.content)
    return local_file


def TMPDIR() -> Path:
    tmpdir = Path(tempfile.gettempdir()) / "map2stl"
    if not tmpdir.is_dir():
        tmpdir.mkdir()
    return tmpdir


def _path_to_merged_tiff(bbox_hash: str) -> Path:
    return TMPDIR() / f"merged_{bbox_hash}.tiff"


def _path_to_clipped_tiff(bbox_hash: str) -> Path:
    return TMPDIR() / f"clipped_{bbox_hash}.tiff"


def show_array(array: np.ndarray):
    plt.imshow(array, interpolation="none")
    plt.show()


def show_tiff(path: Path) -> None:
    tiff = rio.open(path)
    show((tiff, 1), cmap="terrain")


def debug_image(debug: bool, image: Union[np.ndarray, Path], message: str) -> None:
    if debug:
        click.echo(message)
        if isinstance(image, np.ndarray):
            show_array(image)
        elif isinstance(image, Path):
            show_tiff(image)
        else:
            click.echo(f"ERROR: cannot show image of type: {type(image)}")
