import math
import time
from pathlib import Path
from zipfile import ZipFile

import pytest
import rasterio as rio

from mapa import _fetch_merge_and_clip_tiffs, _get_tiff_for_bbox, caching, convert_bbox_to_stl
from mapa.stac import _turn_geojson_into_bbox
from mapa.stl_file import get_dimensions_of_stl_file
from mapa.utils import TMPDIR, path_to_clipped_tiff, path_to_merged_tiff


def test_create_stl_for_bbox__success(output_file, hawaii_bbox) -> None:
    output_file = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        allow_caching=False,
        compress=False,
    )
    assert output_file.is_file()
    # assert mesh.Mesh.from_file(output_file).is_closed()   # TODO


def test_create_stl_for_bbox__z_scale_from_geotiff(hawaii_bbox, output_file):
    output = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        as_ascii=False,
        model_size=200,
        output_file=output_file,
        max_res=True,
        z_offset=5,
        z_scale=0.3,
        ensure_squared=True,
        compress=False,
        allow_caching=False,
    )
    x, y, z = get_dimensions_of_stl_file(output)
    assert x == 200.0
    assert y == 200.0
    assert math.isclose(z, 87.96, rel_tol=0.1)

    # again get dimensions of a model with 10 instead of 5mm z-offset
    z_5 = z
    output = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        as_ascii=False,
        model_size=200,
        output_file=output_file,
        max_res=True,
        z_offset=10,
        z_scale=0.3,
        ensure_squared=True,
        compress=False,
        allow_caching=False,
    )
    x, y, z_10 = get_dimensions_of_stl_file(output)
    assert x == 200.0
    assert y == 200.0
    assert math.isclose(z_10, 92.96, rel_tol=0.1)
    assert z_5 + 5 == z_10


def test_convert_bbox_to_stl__verify_x_y_dimensions(output_file, hawaii_bbox) -> None:
    size = 200
    path = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        model_size=size,
        z_offset=None,
        output_file=output_file,
        ensure_squared=True,  # format ratio should equal a square
        allow_caching=False,
        compress=False,
    )

    x, y, z1 = get_dimensions_of_stl_file(path)
    assert x == y == size


def test__get_tiff_for_bbox(hawaii_bbox) -> None:
    tiff = _get_tiff_for_bbox(hawaii_bbox, allow_caching=False, cache_dir=TMPDIR())
    assert tiff.is_file()


def test__fetch_merge_and_clip_tiffs(geojson_bbox_two_stac_items) -> None:
    cache_dir = TMPDIR()
    bbox_hash = caching.get_hash_of_geojson(geojson_bbox_two_stac_items)
    assert isinstance(bbox_hash, str)
    tiff = _fetch_merge_and_clip_tiffs(geojson_bbox_two_stac_items, bbox_hash, allow_caching=True, cache_dir=cache_dir)
    assert tiff.is_file()
    assert path_to_merged_tiff(bbox_hash, cache_dir=cache_dir).is_file()
    assert path_to_clipped_tiff(bbox_hash, cache_dir=cache_dir).is_file()

    # verify coordinates of resulting tiff is in line with the coordinates of the input bbox
    bbox = _turn_geojson_into_bbox(geojson_bbox_two_stac_items)
    left, bottom, right, top = bbox[0], bbox[1], bbox[2], bbox[3]
    data = rio.open(tiff)
    assert math.isclose(data.bounds.left, left, rel_tol=0.0001)
    assert math.isclose(data.bounds.bottom, bottom, rel_tol=0.0001)
    assert math.isclose(data.bounds.right, right, rel_tol=0.0001)
    assert math.isclose(data.bounds.top, top, rel_tol=0.0001)


def test_convert_bbox_to_stl__ensure_z_offset_is_correct(output_file, hawaii_bbox) -> None:
    path1 = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        z_offset=None,  # setting z_offset=None will use natural offset, i.e. height above sea level
        ensure_squared=True,
        allow_caching=False,
        compress=False,
    )
    x1, y1, z1 = get_dimensions_of_stl_file(path1)

    path2 = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        z_offset=10.0,  # setting z_offset=10.0 will ensure an offset of 10.0mm
        ensure_squared=True,
        allow_caching=False,
        compress=False,
    )
    x2, y2, z2 = get_dimensions_of_stl_file(path2)

    path3 = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        z_offset=0.0,  # setting z_offset=0.0 will ensure an offset of 0.0mm
        ensure_squared=True,
        allow_caching=False,
        compress=False,
    )
    x3, y3, z3 = get_dimensions_of_stl_file(path3)

    assert x1 == x2 == x3
    assert y1 == y2 == y3
    assert z1 > z2 > z3


def test_convert_bbox_to_stl__progress_bar(output_file, geojson_bbox_two_stac_items, progress_bar, mock_max_res) -> None:
    convert_bbox_to_stl(
        bbox_geometry=geojson_bbox_two_stac_items,
        output_file=output_file,
        allow_caching=False,  # TODO: allow caching to speed up test execution
        progress_bar=progress_bar,
        split_area_in_tiles="2x2",
        compress=True,
    )
    # we expect 10 steps, 2 for fetching stac items, 4 for generating
    # the 4 stl files and another 4 for compressing the tiles
    assert len(progress_bar.progress_track) == 10
    assert progress_bar.progress_track == [11, 22, 33, 44, 55, 66, 77, 88, 99, 100]


def test_mapa__index_error(output_file) -> None:
    # chose a very tiny area, which in turn will create a very tiny array, i.e. array with one element only
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [6.164654, 45.89936],
                [6.164654, 45.899404],
                [6.164713, 45.899404],
                [6.164713, 45.89936],
                [6.164654, 45.89936],
            ]
        ],
    }
    # converting such a bbox into an STL caused an index error when trying to drop first/last row/col, but the fix to
    # this avoids dropping rows and cols in case there are not sufficient rows/cols. Thus the computing such a STL should
    # work out of the box, even though a STL with only one elevation data point is of course questionable.
    convert_bbox_to_stl(
        bbox_geometry=bbox,
        output_file=output_file,
        compress=False,
    )


@pytest.mark.parametrize("compress", (False, True))
def test_mapa__split_area_into_tiles__success(hawaii_bbox, tmp_path, compress) -> None:
    size = 100
    output_path = Path(tmp_path, "foo")
    output = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_path,
        split_area_in_tiles="2x2",
        model_size=size,
        ensure_squared=True,
        compress=compress,
    )

    if compress:
        assert output.suffix == ".zip"
        assert output.is_file()

        with ZipFile(output) as zip_file:
            stl_1 = Path(zip_file.extract("foo_1.stl", tmp_path))
            stl_2 = Path(zip_file.extract("foo_2.stl", tmp_path))
            stl_3 = Path(zip_file.extract("foo_3.stl", tmp_path))
            stl_4 = Path(zip_file.extract("foo_4.stl", tmp_path))
        stls = [stl_1, stl_2, stl_3, stl_4]
    else:
        assert isinstance(output, list)
        assert len(output) == 4
        stls = output

    for stl in stls:
        assert stl.is_file()
        x, y, _ = get_dimensions_of_stl_file(stl)
        assert x == size / 2
        assert y == size / 2


def test_mapa__split_area_into_tiles__one_by_two(hawaii_bbox, output_file) -> None:
    size = 100
    # first round we have 1 by 2
    output = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        split_area_in_tiles="1x2",
        model_size=size,
        ensure_squared=True,
        compress=False,
    )
    assert isinstance(output, list)
    assert len(output) == 2

    x1, y1, _ = get_dimensions_of_stl_file(output[0])
    x2, y2, _ = get_dimensions_of_stl_file(output[1])

    assert x1 == size
    assert x2 == size
    assert y1 == y2
    assert math.isclose(y1, size / 2, rel_tol=0.03)
    assert math.isclose(y2, size / 2, rel_tol=0.03)

    # second round we have 2 by 1
    output = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        split_area_in_tiles="2x1",
        model_size=size,
        ensure_squared=True,
        compress=False,
    )
    assert isinstance(output, list)
    assert len(output) == 2

    x1, y1, _ = get_dimensions_of_stl_file(output[0])
    x2, y2, _ = get_dimensions_of_stl_file(output[1])
    assert x1 == size / 2
    assert x2 == size / 2
    assert y1 == y2
    assert math.isclose(y1, size, rel_tol=0.03)
    assert math.isclose(y2, size, rel_tol=0.03)


def test_mapa__split_area_into_tiles__area_too_small(output_file) -> None:
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [25.30000, 25.30000],
                [25.30000, 25.30000],
                [25.30000, 25.30001],
                [25.30001, 25.30000],
                [25.30000, 25.30000],
            ]
        ],
    }
    with pytest.raises(ValueError, match="Input array is too small to be split into tiles."):
        convert_bbox_to_stl(
            bbox_geometry=bbox,
            output_file=output_file,
            split_area_in_tiles="5x5",
            model_size=100,
            z_scale=3.0,
            compress=False,
            allow_caching=False,
        )


@pytest.mark.parametrize("ensure_squared", (False, True))
def test_mapa__ensure_squared__two_by_two(ensure_squared, output_file, hawaii_bbox) -> None:
    # creating 2 by 2 tiles
    size = 100
    output = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        split_area_in_tiles="2x2",
        model_size=size,
        ensure_squared=ensure_squared,
        compress=False,
    )
    assert isinstance(output, list)
    assert len(output) == 4

    x1, y1, _ = get_dimensions_of_stl_file(output[0])
    x2, y2, _ = get_dimensions_of_stl_file(output[1])
    x3, y3, _ = get_dimensions_of_stl_file(output[2])
    x4, y4, _ = get_dimensions_of_stl_file(output[3])

    # adding two sides will result in the desired model size
    assert x1 + x2 == size
    assert x3 + x4 == size
    assert y1 + y2 == size
    assert y3 + y4 == size


def test_mapa__interplay_of_ensure_squared_with_tiling(output_file, hawaii_bbox) -> None:
    size = 100
    output = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        split_area_in_tiles="1x2",
        model_size=size,
        compress=False,
    )
    assert isinstance(output, list)
    assert len(output) == 2

    x1, y1, _ = get_dimensions_of_stl_file(output[0])
    x2, y2, _ = get_dimensions_of_stl_file(output[1])
    assert x1 == size
    assert x2 == size
    assert y1 == size / 2
    assert y2 == size / 2

    # now setting ensure_squared to True means we want the combined stl file to be squared, since bbox is already
    # squared we expect the same results
    output = convert_bbox_to_stl(
        bbox_geometry=hawaii_bbox,
        output_file=output_file,
        split_area_in_tiles="1x2",
        model_size=size,
        ensure_squared=True,
        compress=False,
    )
    assert isinstance(output, list)
    assert len(output) == 2

    x1, y1, _ = get_dimensions_of_stl_file(output[0])
    x2, y2, _ = get_dimensions_of_stl_file(output[1])
    assert x1 == size
    assert x2 == size
    assert y1 == size / 2
    assert y2 == size / 2


def test_mapa__tiling_with_rectangular_bbox(geojson_bbox_two_stac_items, output_file, mock_max_res) -> None:
    # don't use squared output
    size = 100
    tiling = "2x3"
    output = convert_bbox_to_stl(
        bbox_geometry=geojson_bbox_two_stac_items,
        output_file=output_file,
        split_area_in_tiles=tiling,
        model_size=size,
        ensure_squared=False,
        compress=False,
    )
    assert isinstance(output, list)
    assert len(output) == 6

    x1, y1, _ = get_dimensions_of_stl_file(output[0])
    x2, y2, _ = get_dimensions_of_stl_file(output[1])
    x3, y3, _ = get_dimensions_of_stl_file(output[2])
    x4, y4, _ = get_dimensions_of_stl_file(output[3])
    x5, y5, _ = get_dimensions_of_stl_file(output[4])
    x6, y6, _ = get_dimensions_of_stl_file(output[5])

    assert x1 == x2 == x3 == x4 == x5 == x6
    assert x1 + x2 == size
    assert x3 + x4 == size
    assert x5 + x6 == size
    assert y1 == y2 == y3 == y4 == y5 == y6
    assert y1 != size
    assert y1 != x1

    # now setting output shape should to be squares
    output = convert_bbox_to_stl(
        bbox_geometry=geojson_bbox_two_stac_items,
        output_file=output_file,
        split_area_in_tiles=tiling,
        model_size=size,
        ensure_squared=True,
        compress=False,
    )

    x1, y1, _ = get_dimensions_of_stl_file(output[0])
    x2, y2, _ = get_dimensions_of_stl_file(output[1])
    x3, y3, _ = get_dimensions_of_stl_file(output[2])
    x4, y4, _ = get_dimensions_of_stl_file(output[3])
    x5, y5, _ = get_dimensions_of_stl_file(output[4])
    x6, y6, _ = get_dimensions_of_stl_file(output[5])

    assert x1 == x2 == x3 == x4 == x5 == x6
    assert x1 + x2 == size
    assert x3 + x4 == size
    assert x5 + x6 == size
    assert y1 == y2 == y3 == y4 == y5 == y6
    assert y1 + y2 + y3 == size
    assert y4 + y5 + y6 == size


def test_custom_caching_path(output_file, geojson_bbox, tmp_path) -> None:
    start = time.time()
    output_file = convert_bbox_to_stl(
        bbox_geometry=geojson_bbox,
        output_file=output_file,
        allow_caching=True,
        cache_dir=tmp_path,
        compress=False,
    )
    duration_uncached = time.time() - start
    assert output_file.is_file()

    start = time.time()
    output_file = convert_bbox_to_stl(
        bbox_geometry=geojson_bbox,
        output_file=output_file,
        allow_caching=True,
        cache_dir=tmp_path,
        compress=False,
    )
    duration_cached = time.time() - start
    assert output_file.is_file()

    # show second attempt is significantly faster because of caching
    assert duration_cached < duration_uncached / 2

    # verify certain files are present in the custom cache_dir
    assert len([stl for stl in Path(tmp_path).glob("*.stl")]) == 2
    assert len([stl for stl in Path(tmp_path).glob("*.tiff")]) == 2
