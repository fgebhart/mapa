from pathlib import Path
from typing import List

import click
import geojson
from pystac_client import Client

from mapa import conf
from mapa.utils import TMPDIR, download_file, timing


def _bbox(coord_list):
    box = []
    for i in (0, 1):
        res = sorted(coord_list, key=lambda x: x[i])
        box.append((res[0][i], res[-1][i]))
    return [box[0][0], box[1][0], box[0][1], box[1][1]]


def _turn_geojson_into_bbox(geojson_bbox: dict) -> List[float]:
    poly = geojson.Polygon(geojson_bbox["coordinates"])
    return _bbox(list(geojson.utils.coords(poly)))


@timing
def fetch_stac_items_for_bbox(geojson: dict) -> List[Path]:
    click.echo(f"{'🏞  fetching tiff '}", nl=False)
    bbox = _turn_geojson_into_bbox(geojson)
    client = Client.open(conf.PLANETARY_COMPUTER_API_URL, ignore_conformance=True)
    search = client.search(collections=[conf.PLANETARY_COMPUTER_COLLECTION], bbox=bbox)
    items = list(search.get_items())
    msg = f"from {len(items)} stac item(s)..."
    click.echo(f"{msg:<33s}", nl=False)
    if len(items) > 0:
        files = []
        for i, item in enumerate(items):
            url = item.assets["data"].href
            alos_dem_stac_tif = TMPDIR() / f"alos_dem_stac_{i}.tiff"
            files.append(download_file(url, alos_dem_stac_tif))
        return files
    else:
        raise ValueError("Could not find the desired STAC item for the given bounding box.")
