import math

import rasterio as rio

from mapa import _fetch_merge_and_clip_tiffs, _get_tiff_for_bbox, convert_bbox_to_stl
from mapa.caching import get_hash_of_geojson
from mapa.stac import _turn_geojson_into_bbox
from mapa.stl_file import get_dimensions_of_stl_file
from mapa.utils import _path_to_clipped_tiff, _path_to_merged_tiff


def test_create_stl_for_bbox__success(output_file, geojson_bbox) -> None:
    output_file = convert_bbox_to_stl(
        bbox_geometry=geojson_bbox,
        output_file=output_file,
    )
    assert output_file.is_file()


def test_create_stl_for_bbox__z_scale_from_geotiff(output_file):
    geojson_bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [-73.491314, -50.735036],
                [-73.491314, -50.611451],
                [-73.244186, -50.611451],
                [-73.244186, -50.735036],
                [-73.491314, -50.735036],
            ]
        ],
    }
    convert_bbox_to_stl(
        bbox_geometry=geojson_bbox,
        as_ascii=False,
        model_size=200,
        output_file=output_file,
        max_res=True,
        z_offset=5,
        z_scale=0.3,
        cut_to_format_ratio=True,
    )
    x, y, z = get_dimensions_of_stl_file(output_file)
    assert x == 200.0
    assert y == 200.0
    assert math.isclose(z, 10.44, rel_tol=0.1)

    # again get dimensions of a model with 10 instead of 5mm z-offset
    z_5 = z
    convert_bbox_to_stl(
        bbox_geometry=geojson_bbox,
        as_ascii=False,
        model_size=200,
        output_file=output_file,
        max_res=True,
        z_offset=10,
        z_scale=0.3,
        cut_to_format_ratio=True,
    )
    x, y, z_10 = get_dimensions_of_stl_file(output_file)
    assert x == 200.0
    assert y == 200.0
    assert math.isclose(z_10, 15.44, rel_tol=0.1)
    assert z_5 + 5 == z_10


def test_convert_bbox_to_stl__verify_x_y_dimensions(output_file) -> None:
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [10.724114, 45.746092],
                [10.724114, 46.065259],
                [11.171691, 46.065259],
                [11.171691, 45.746092],
                [10.724114, 45.746092],
            ]
        ],
    }

    path = convert_bbox_to_stl(
        bbox_geometry=bbox,
        model_size=200,
        output_file=output_file,
        cut_to_format_ratio=1.0,  # format ratio should equal a square
    )

    x, y, z1 = get_dimensions_of_stl_file(path)
    assert x == y == 200

    path = convert_bbox_to_stl(
        bbox_geometry=bbox,
        model_size=200,
        output_file=output_file,
        cut_to_format_ratio=1 / 2,  # one side should be half the length of the second side
    )

    x, y, z2 = get_dimensions_of_stl_file(path)
    assert x == 2 * y == 200
    # z dimension should stay the same
    assert z1 == z2


def test__get_tiff_for_bbox(geojson_bbox) -> None:
    tiff = _get_tiff_for_bbox(geojson_bbox, allow_caching=True)
    assert tiff.is_file()


def test__fetch_merge_and_clip_tiffs(geojson_bbox_two_stac_items) -> None:
    bbox_hash = get_hash_of_geojson(geojson_bbox_two_stac_items)
    assert isinstance(bbox_hash, str)
    tiff = _fetch_merge_and_clip_tiffs(geojson_bbox_two_stac_items, bbox_hash, allow_caching=True)
    assert tiff.is_file()
    assert _path_to_merged_tiff(bbox_hash).is_file()
    assert _path_to_clipped_tiff(bbox_hash).is_file()

    # verify coordinates of resulting tiff is in line with the coordinates of the input bbox
    bbox = _turn_geojson_into_bbox(geojson_bbox_two_stac_items)
    left, bottom, right, top = bbox[0], bbox[1], bbox[2], bbox[3]
    data = rio.open(tiff)
    assert math.isclose(data.bounds.left, left, rel_tol=0.0001)
    assert math.isclose(data.bounds.bottom, bottom, rel_tol=0.0001)
    assert math.isclose(data.bounds.right, right, rel_tol=0.0001)
    assert math.isclose(data.bounds.top, top, rel_tol=0.0001)
