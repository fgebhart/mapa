import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def TMPDIR() -> Path:
    tmpdir = Path(tempfile.gettempdir()) / "mapa"
    if not tmpdir.is_dir():
        tmpdir.mkdir()
    return tmpdir


def _path_to_merged_tiff(bbox_hash: str) -> Path:
    return TMPDIR() / f"merged_{bbox_hash}.tiff"


def _path_to_clipped_tiff(bbox_hash: str) -> Path:
    return TMPDIR() / f"clipped_{bbox_hash}.tiff"


def plot_bottom_triangles(triangles, max_x, max_y):
    for triangle in triangles:
        plt.plot([triangle[0][0], triangle[1][0]], [triangle[0][1], triangle[1][1]], "b--", linewidth=4)
        plt.plot([triangle[1][0], triangle[2][0]], [triangle[1][1], triangle[2][1]], "b:", linewidth=4)
        plt.plot([triangle[2][0], triangle[0][0]], [triangle[2][1], triangle[0][1]], "b-", linewidth=4)
    plt.axis([0, max_x, 0, max_y])
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    ax.yaxis.set_ticks(np.arange(0, max_y, 1))
    ax.yaxis.tick_left()
    plt.show()
