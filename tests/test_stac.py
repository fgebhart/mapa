from mapa.stac import _download_file, _turn_geojson_into_bbox, fetch_stac_items_for_bbox


def test__turn_geojson_into_bbox(geojson_bbox):
    assert _turn_geojson_into_bbox(geojson_bbox) == [8.076906, 48.098505, 8.107111, 48.115011]


def test_fetch_stac_items_for_bbox(mock_file_download):
    multiple_stac_items_bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [18.289063, -34.260546],
                [18.289063, -33.87148],
                [18.817643, -33.87148],
                [18.817643, -34.260546],
                [18.289063, -34.260546],
            ]
        ],
    }
    tiffs = fetch_stac_items_for_bbox(geojson=multiple_stac_items_bbox, allow_caching=False, max_number_of_stac_items=-1)
    assert len(tiffs) == 2


def test__download_file(tmp_path) -> None:
    # little test to verify downloading file works, everywhere else the _download_file func will be mocked
    file = tmp_path / "foo.txt"
    url = "https://raw.githubusercontent.com/fgebhart/mapa/main/README.md"
    path = _download_file(url, local_file=file)
    assert path.is_file()
