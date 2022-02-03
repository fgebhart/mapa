import math

from mapa import create_stl_for_bbox
from mapa.geometry import get_dimensions_of_stl_file


def test_create_stl_for_bbox__success(tmpdir, geojson_bbox) -> None:
    output_file = tmpdir / "output.stl"
    output_file = create_stl_for_bbox(
        bbox_geometry=geojson_bbox,
        output_file=output_file,
    )
    assert output_file.is_file()


def test_create_stl_for_bbox__z_scale_from_geotiff(tmpdir):
    output_file = tmpdir / "test.stl"
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
    create_stl_for_bbox(
        bbox_geometry=geojson_bbox,
        as_ascii=False,
        model_size=200,
        output_file=output_file,
        max_res=True,
        z_offset=5,
        z_scale=0.3,
        make_square=True,
    )
    x, y, z = get_dimensions_of_stl_file(output_file)
    assert x == 200.0
    assert y == 200.0
    assert math.isclose(z, 10.44, rel_tol=0.1)

    # again get dimensions of a model with 10 instead of 5mm z-offset
    z_5 = z
    create_stl_for_bbox(
        bbox_geometry=geojson_bbox,
        as_ascii=False,
        model_size=200,
        output_file=output_file,
        max_res=True,
        z_offset=10,
        z_scale=0.3,
        make_square=True,
    )
    x, y, z_10 = get_dimensions_of_stl_file(output_file)
    assert x == 200.0
    assert y == 200.0
    assert math.isclose(z_10, 15.44, rel_tol=0.1)
    assert z_5 + 5 == z_10
