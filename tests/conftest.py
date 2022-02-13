from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def test_stl_binary():
    yield Path(__file__).parent / "stl" / "hawaii_binary.stl"


@pytest.fixture
def test_stl_ascii():
    yield Path(__file__).parent / "stl" / "hawaii_ascii.stl"


@pytest.fixture
def test_tiff():
    yield Path(__file__).parent / "tif" / "hawaii_low_res.tif"


@pytest.fixture
def clipped_tiff():
    yield Path(__file__).parent / "tif" / "clipped.tif"


@pytest.fixture
def output_file(tmpdir):
    yield tmpdir / "test.stl"


@pytest.fixture
def corrupted_tiff(tmp_path):
    p = tmp_path / "hello.tiff"
    p.write_text("foo")
    yield p


@pytest.fixture
def input_array():
    yield np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )


@pytest.fixture
def geojson_bbox():
    yield {
        "type": "Polygon",
        "coordinates": [
            [
                [8.076906, 48.098505],
                [8.076906, 48.115011],
                [8.107111, 48.115011],
                [8.107111, 48.098505],
                [8.076906, 48.098505],
            ]
        ],
    }
