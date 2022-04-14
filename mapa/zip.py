import logging
import zipfile
from pathlib import Path
from typing import List, Union

log = logging.getLogger(__name__)


def create_zip_archive(files: List[Path], output_file: Union[str, Path]) -> Path:
    log.info(f"📦  compressing stl files: {[f.name for f in files]}")
    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for f in files:
            zip_file.write(f, f.name)
    log.info(f"✅  finished compressing STL files into: {output_file}")
    return Path(output_file)
