import logging
from pathlib import Path
from typing import List, Union
from urllib import request

import geojson
from pystac.item import Item
from pystac_client import Client

from mapa import conf
from mapa.exceptions import NoSTACItemFound
from mapa.utils import ProgressBar

log = logging.getLogger(__name__)


def _download_file(url: str, local_file: Path) -> Path:
    request.urlretrieve(url, local_file)
    return local_file


def _bbox(coord_list):
    box = []
    for i in (0, 1):
        res = sorted(coord_list, key=lambda x: x[i])
        box.append((res[0][i], res[-1][i]))
    return [box[0][0], box[1][0], box[0][1], box[1][1]]


def _turn_geojson_into_bbox(geojson_bbox: dict) -> List[float]:
    coordinates = geojson_bbox["coordinates"]
    return _bbox(list(geojson.utils.coords(geojson.Polygon(coordinates))))


def _get_tiff_file(stac_item: Item, allow_caching: bool, cache_dir: Path, count: int, max: int) -> Path:
    tiff = cache_dir / f"{stac_item.id}.tiff"
    if tiff.is_file() and allow_caching:
        log.info(f"🚀  {count}/{max} using cached stac item {stac_item.id}")
        return tiff
    else:
        log.info(f"🏞  {count}/{max} downloading stac item {stac_item.id}")
        return _download_file(stac_item.assets["data"].href, tiff)


def fetch_stac_items_for_bbox(
    geojson: dict, allow_caching: bool, cache_dir: Path, progress_bar: Union[None, ProgressBar] = None
) -> List[Path]:
    bbox = _turn_geojson_into_bbox(geojson)
    client = Client.open(conf.PLANETARY_COMPUTER_API_URL, ignore_conformance=True)
    search = client.search(collections=[conf.PLANETARY_COMPUTER_COLLECTION], bbox=bbox)
    items = list(search.get_items())
    n = len(items)
    if progress_bar:
        progress_bar.steps += n
    if n > 0:
        log.info(f"⬇️  fetching {n} stac items...")
        files = []
        for cnt, item in enumerate(items):
            files.append(_get_tiff_file(item, allow_caching, cache_dir, cnt + 1, n))
            if progress_bar:
                progress_bar.step()
        return files
    else:
        raise NoSTACItemFound("Could not find the desired STAC item for the given bounding box.")
