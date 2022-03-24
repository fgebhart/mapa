import logging
import tempfile
from pathlib import Path

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
