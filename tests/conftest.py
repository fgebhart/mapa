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
    yield Path(__file__).parent / "tiff" / "hawaii_low_res.tiff"


@pytest.fixture
def clipped_tiff():
    yield Path(__file__).parent / "tiff" / "clipped.tiff"


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


@pytest.fixture
def geojson_bbox_two_stac_items():
    yield {
        "type": "Polygon",
        "coordinates": [
            [
                [8.631413, 41.318388],
                [8.631413, 41.762435],
                [9.685828, 41.762435],
                [9.685828, 41.318388],
                [8.631413, 41.318388],
            ]
        ],
    }


@pytest.fixture
def progress_bar():
    class ProgressBar:
        def __init__(self) -> None:
            self.progress_track = []

        def progress(self, value: int) -> None:
            self.progress_track.append(value)

    yield ProgressBar()
