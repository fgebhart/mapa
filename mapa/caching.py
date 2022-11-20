import json
from hashlib import md5
from pathlib import Path

from mapa.utils import path_to_clipped_tiff


def get_hash_of_geojson(bbox_geojson: dict) -> str:
    return md5(json.dumps(bbox_geojson, sort_keys=True).encode()).hexdigest()


def tiff_for_bbox_is_cached(bbox_hash: str, cache_dir: Path) -> bool:
    return (path_to_clipped_tiff(bbox_hash, cache_dir)).is_file()
