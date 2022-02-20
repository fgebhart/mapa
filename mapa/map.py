from typing import List, Tuple

from ipyleaflet import DrawControl, Map, ScaleControl, basemap_to_tiles, basemaps
from ipywidgets import Layout

CENTER = [40.5566, 23.4660]
ZOOM = 4
OPACITY = 0.3
COLOR = "#0000FF"
SHAPE_OPTIONS = {"fillColor": COLOR, "color": COLOR, "fillOpacity": OPACITY}


def show_map(center: List[float] = CENTER, zoom: int = ZOOM) -> Tuple[Map, DrawControl]:
    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, layout=Layout(height="600px"))
    m.add_control(ScaleControl(position="bottomleft"))
    m.add_layer(basemap_to_tiles(basemaps.OpenTopoMap))

    dc = DrawControl(circlemarker={}, polyline={})
    dc.rectangle = {"shapeOptions": SHAPE_OPTIONS}
    dc.circle = {"shapeOptions": SHAPE_OPTIONS}
    dc.polygon = {"shapeOptions": SHAPE_OPTIONS, "allowIntersection": False}
    m.add_control(dc)

    def handle_draw(target, action, geo_json):
        print("Shape detected, execute next cells to continue!")

    dc.on_draw(handle_draw)
    return m, dc
