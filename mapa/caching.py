import json
from hashlib import md5

from mapa.utils import _path_to_clipped_tiff


def get_hash_of_geojson(bbox_geojson: dict) -> str:
    return md5(json.dumps(bbox_geojson, sort_keys=True).encode()).hexdigest()


def tiff_for_bbox_is_cached(bbox_hash: str) -> bool:
    return (_path_to_clipped_tiff(bbox_hash)).is_file()
