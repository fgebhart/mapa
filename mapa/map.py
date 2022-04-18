from typing import List, Tuple

from ipyleaflet import DrawControl, Map, Rectangle, ScaleControl, basemap_to_tiles, basemaps
from ipywidgets import Layout

CENTER = [40.5566, 23.4660]
ZOOM = 4


def show_map(center: List[float] = CENTER, zoom: int = ZOOM) -> Tuple[Map, DrawControl]:
    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, layout=Layout(height="600px"))
    m.add_control(ScaleControl(position="bottomleft"))
    m.add_layer(basemap_to_tiles(basemaps.OpenTopoMap))

    dc = DrawControl(rectangle={"shapeOptions": {"color": "#0000FF"}}, polyline={}, polygon={}, circlemarker={})

    def handle_draw(target, action, geo_json):
        print("Rectangle detected, execute next cells to continue!")

    dc.on_draw(handle_draw)
    m.add_control(dc)
    return m, dc


def draw_rect_on_map(rect_coordinates: List[List[float]]) -> Map:
    coords = [i[::-1] for i in rect_coordinates]  # swap position because of GeoJSON
    m = Map(center=coords[0], zoom=5, scroll_wheel_zoom=True)
    rectangle = Rectangle(bounds=coords)
    m.add_layer(rectangle)
    return m
