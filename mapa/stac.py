from pathlib import Path
from typing import List

import click
import geojson
from pystac.item import Item
from pystac_client import Client

from mapa import conf
from mapa.utils import TMPDIR, download_file


def _bbox(coord_list):
    box = []
    for i in (0, 1):
        res = sorted(coord_list, key=lambda x: x[i])
        box.append((res[0][i], res[-1][i]))
    return [box[0][0], box[1][0], box[0][1], box[1][1]]


def _turn_geojson_into_bbox(geojson_bbox: dict) -> List[float]:
    poly = geojson.Polygon(geojson_bbox["coordinates"])
    return _bbox(list(geojson.utils.coords(poly)))


def _get_tiff_file(stac_item: Item, allow_caching: bool) -> Path:
    tiff = TMPDIR() / f"{stac_item.id}.tiff"
    if tiff.is_file() and allow_caching:
        click.echo(f"🚀  using cached stac item {stac_item.id}  ✅ (0.0s)")
        return tiff
    else:
        click.echo(f"{f'🏞  downloading stac item {stac_item.id} ':<50s}", nl=False)
        return download_file(stac_item.assets["data"].href, tiff)


def fetch_stac_items_for_bbox(geojson: dict, allow_caching: bool) -> List[Path]:
    bbox = _turn_geojson_into_bbox(geojson)
    client = Client.open(conf.PLANETARY_COMPUTER_API_URL, ignore_conformance=True)
    search = client.search(collections=[conf.PLANETARY_COMPUTER_COLLECTION], bbox=bbox)
    items = list(search.get_items())
    if len(items) > 0:
        click.echo(f"⬇️  fetching {len(items)} stac items...")
        files = []
        for item in items:
            files.append(_get_tiff_file(item, allow_caching))
        return files
    else:
        raise ValueError("Could not find the desired STAC item for the given bounding box.")
