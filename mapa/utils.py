import hashlib
import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def TMPDIR() -> Path:
    tmpdir = Path(tempfile.gettempdir()) / "mapa"
    if not tmpdir.is_dir():
        tmpdir.mkdir()
    return tmpdir


def path_to_merged_tiff(bbox_hash: str) -> Path:
    return TMPDIR() / f"merged_{bbox_hash}.tiff"


def path_to_clipped_tiff(bbox_hash: str) -> Path:
    return TMPDIR() / f"clipped_{bbox_hash}.tiff"


def md5_sum(path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
