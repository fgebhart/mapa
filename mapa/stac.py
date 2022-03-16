from pathlib import Path
from typing import List, Union
from urllib import request

import click
import geojson
from pystac.item import Item
from pystac_client import Client

from mapa import conf
from mapa.utils import TMPDIR, timing


@timing
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
    poly = geojson.Polygon(geojson_bbox["coordinates"])
    return _bbox(list(geojson.utils.coords(poly)))


def _get_tiff_file(stac_item: Item, allow_caching: bool) -> Path:
    tiff = TMPDIR() / f"{stac_item.id}.tiff"
    if tiff.is_file() and allow_caching:
        click.echo(f"🚀  using cached stac item {stac_item.id}  ✅ (0.0s)")
        return tiff
    else:
        click.echo(f"{f'🏞  downloading stac item {stac_item.id} ':<50s}", nl=False)
        return _download_file(stac_item.assets["data"].href, tiff)


def fetch_stac_items_for_bbox(
    geojson: dict, allow_caching: bool, max_number_of_stac_items: int, progress_bar: Union[None, object] = None
) -> List[Path]:
    bbox = _turn_geojson_into_bbox(geojson)
    client = Client.open(conf.PLANETARY_COMPUTER_API_URL, ignore_conformance=True)
    search = client.search(collections=[conf.PLANETARY_COMPUTER_COLLECTION], bbox=bbox)
    items = list(search.get_items())
    n = len(items)
    if n > 0:
        if n < max_number_of_stac_items or max_number_of_stac_items < 0:
            click.echo(f"⬇️  fetching {n} stac items...")
            files = []
            for i, item in enumerate(items):
                files.append(_get_tiff_file(item, allow_caching))
                if progress_bar:
                    progress_bar.progress(int(100 / n * (i + 1)))
            return files
        else:
            raise ValueError(
                f"Given area of input geometry exceeds the maximal number of stac items ({max_number_of_stac_items})"
            )
    else:
        raise ValueError("Could not find the desired STAC item for the given bounding box.")
