import logging
import tempfile
from functools import wraps
from pathlib import Path
from time import time

log = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        # start_msg = f"{f.__name__} ..."
        # log.debug(f"{start_msg:<40s}")
        result = f(*args, **kw)
        te = time()
        end_msg = f"✅ ({round(te - ts, 1)}s)"
        log.debug(end_msg)
        return result

    return wrap


def TMPDIR() -> Path:
    tmpdir = Path(tempfile.gettempdir()) / "mapa"
    if not tmpdir.is_dir():
        tmpdir.mkdir()
    return tmpdir


def _path_to_merged_tiff(bbox_hash: str) -> Path:
    return TMPDIR() / f"merged_{bbox_hash}.tiff"


def _path_to_clipped_tiff(bbox_hash: str) -> Path:
    return TMPDIR() / f"clipped_{bbox_hash}.tiff"
