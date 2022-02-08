from mapa.caching import get_hash_of_geojson


def test__get_hash_of_geojson(geojson_bbox) -> None:
    md5sum = get_hash_of_geojson(geojson_bbox)
    assert md5sum == "286fa8d103174f6299f994c4ab69ba94"
