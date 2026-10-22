"""Land mask in image coordinates from Natural Earth 10 m land polygons.

Natural Earth is coarse along fjords and small islands, so the mask is
dilated by a buffer and fixed objects that survive it are handled later by
cross-pass persistence.
"""
import zipfile
import io

import numpy as np
import shapefile
from shapely.geometry import shape, box
from shapely.ops import unary_union
from rasterio.features import rasterize
from scipy import ndimage

_LAND = {}


def land_polygons(zip_path, bbox):
    key = (zip_path, tuple(np.round(bbox, 3)))
    if key in _LAND:
        return _LAND[key]
    z = zipfile.ZipFile(zip_path)
    names = {n.rsplit(".", 1)[1]: n for n in z.namelist() if n.startswith("ne_10m_land.")}
    sf = shapefile.Reader(shp=io.BytesIO(z.read(names["shp"])), shx=io.BytesIO(z.read(names["shx"])),
                          dbf=io.BytesIO(z.read(names["dbf"])))
    clip = box(*bbox)
    geoms = [shape(s.__geo_interface__).intersection(clip) for s in sf.shapes()
             if box(*s.bbox).intersects(clip)]
    land = unary_union([g for g in geoms if not g.is_empty])
    _LAND[key] = land
    return land


def _rings(geom):
    polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for p in polys:
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        yield p


def tile_mask(scene, land, line0, pixel0, n_lines, n_pixels, buffer_px=60):
    """Boolean mask (True = land or within buffer) for one tile."""
    from shapely.geometry import Polygon

    m = 2 * buffer_px
    lat, lon = scene.to_latlon(np.array([line0 - m, line0 - m, line0 + n_lines + m, line0 + n_lines + m]),
                               np.array([pixel0 - m, pixel0 + n_pixels + m, pixel0 - m, pixel0 + n_pixels + m]))
    local = land.intersection(box(lon.min(), lat.min(), lon.max(), lat.max()))
    shapes = []
    for p in _rings(local):
        def to_px(coords):
            c = np.asarray(coords)
            L, P = scene.to_pixel(c[:, 1], c[:, 0])
            return list(zip(P - pixel0, L - line0))
        shapes.append(Polygon(to_px(p.exterior.coords), [to_px(r.coords) for r in p.interiors]))
    if not shapes:
        return np.zeros((n_lines, n_pixels), bool)
    mask = rasterize([(s, 1) for s in shapes], out_shape=(n_lines, n_pixels), fill=0,
                     all_touched=True).astype(bool)
    if buffer_px:
        mask = ndimage.binary_dilation(mask, iterations=buffer_px)
    return mask
