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


def path_to_merged_tiff(bbox_hash: str, cache_dir: Path) -> Path:
    return cache_dir / f"merged_{bbox_hash}.tiff"


def path_to_clipped_tiff(bbox_hash: str, cache_dir: Path) -> Path:
    return cache_dir / f"clipped_{bbox_hash}.tiff"


def md5_sum(path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class ProgressBar:
    def __init__(self, progress_bar: object, steps: int = 0) -> None:
        self.progress_bar = progress_bar  # streamlit st.progress_bar object
        self.steps: int = steps
        self.counter: int = 0

    def step(self) -> None:
        progress = 100 // (self.steps - 1) * (self.counter + 1)
        progress = progress if progress <= 100 else 100
        self.progress_bar.progress(progress)
        self.counter += 1
