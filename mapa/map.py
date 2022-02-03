from typing import List, Tuple

from ipyleaflet import DrawControl, LayersControl, Map, basemap_to_tiles, basemaps
from ipywidgets import Layout

CENTER = [40.5566, 23.4660]
ZOOM = 4


def show_map(center: List[float] = CENTER, zoom: int = ZOOM) -> Tuple[Map, DrawControl]:
    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, layout=Layout(height="600px"))
    m.add_layer(basemap_to_tiles(basemaps.Gaode.Satellite))
    m.add_layer(basemap_to_tiles(basemaps.OpenTopoMap))
    control = LayersControl(position="topright")
    m.add_control(control)

    dc = DrawControl(rectangle={"shapeOptions": {"color": "#0000FF"}}, polyline={}, polygon={}, circlemarker={})

    def handle_draw(target, action, geo_json):
        print("Rectangle detected, execute next cells to continue!")

    dc.on_draw(handle_draw)
    m.add_control(dc)
    return m, dc
