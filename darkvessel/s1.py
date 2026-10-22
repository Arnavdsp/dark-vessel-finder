"""Sentinel-1 IW GRD access and geometry from the product annotation.

Image axes follow the product: line = azimuth (increases with time, i.e. in
the flight direction), pixel = ground range (pixel 0 is near range). Pixels
are 10 m in both directions for GRDH, so x_m = 10 * pixel, y_m = 10 * line.

Everything the physics needs (slant range, incidence, azimuth time, PRF per
sub-swath, platform speed, wavelength) is read from the annotation rather
than assumed.
"""
from dataclasses import dataclass, field
from datetime import datetime
import time
import xml.etree.ElementTree as ET

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator

C = 299_792_458.0
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-grd/items/"
EARTH_R = 6_371_000.0


def _get(url, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=90)
            if r.ok:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"GET failed: {url.split('?')[0]}")


def _t(s):
    return datetime.fromisoformat(s)


@dataclass
class Swath:
    name: str
    first_pixel: int
    last_pixel: int
    prf_hz: float


@dataclass
class S1Scene:
    item_id: str
    vv_href: str
    pass_dir: str
    t0: datetime                   # first line time
    line_dt_s: float               # azimuth time interval
    pixel_m: float
    n_lines: int
    n_pixels: int
    wavelength_m: float
    speed_ms: float                # mean platform speed over the scene
    swaths: list
    grid_lines: np.ndarray
    grid_pixels: np.ndarray
    grids: dict = field(repr=False)  # name -> (n_lines, n_pixels) array
    _interp: dict = field(default_factory=dict, repr=False)

    # --- interpolation on the geolocation grid -------------------------
    def at(self, name, line, pixel):
        if name not in self._interp:
            self._interp[name] = RegularGridInterpolator(
                (self.grid_lines, self.grid_pixels), self.grids[name],
                method="cubic", bounds_error=False, fill_value=None)
        line, pixel = np.broadcast_arrays(np.asarray(line, float), np.asarray(pixel, float))
        return self._interp[name](np.stack([line.ravel(), pixel.ravel()], -1)).reshape(line.shape)

    def to_latlon(self, line, pixel):
        return self.at("lat", line, pixel), self.at("lon", line, pixel)

    def slant_range_m(self, line, pixel):
        return self.at("srt", line, pixel) * C / 2.0

    def incidence_deg(self, line, pixel):
        return self.at("inc", line, pixel)

    def az_time_s(self, line, pixel):
        """Seconds after t0 at which (line, pixel) was imaged."""
        return self.at("azt", line, pixel)

    def to_pixel(self, lat, lon, iters=8):
        """Invert the geolocation grid with Newton steps. Vectorised."""
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        gl, gp = np.meshgrid(self.grid_lines, self.grid_pixels, indexing="ij")
        glat, glon = self.grids["lat"].ravel(), self.grids["lon"].ravel()
        coslat = np.cos(np.radians(lat))[:, None]
        d2 = (lat[:, None] - glat) ** 2 + ((lon[:, None] - glon) * coslat) ** 2
        k = d2.argmin(1)
        L, P = gl.ravel()[k].astype(float), gp.ravel()[k].astype(float)
        h = 5.0
        for _ in range(iters):
            la, lo = self.to_latlon(L, P)
            la_l, lo_l = self.to_latlon(L + h, P)
            la_p, lo_p = self.to_latlon(L, P + h)
            a, b = (la_l - la) / h, (la_p - la) / h
            c, d = (lo_l - lo) / h, (lo_p - lo) / h
            det = a * d - b * c
            rlat, rlon = lat - la, lon - lo
            L = L + (d * rlat - b * rlon) / det
            P = P + (-c * rlat + a * rlon) / det
        return L, P

    def axes_en(self, line, pixel):
        """East/north metres moved per +1 line and per +1 pixel."""
        la, lo = self.to_latlon(line, pixel)
        la_l, lo_l = self.to_latlon(line + 1, pixel)
        la_p, lo_p = self.to_latlon(line, pixel + 1)
        k = np.radians(1) * EARTH_R
        cl = np.cos(np.radians(la))
        e_line = np.stack([(lo_l - lo) * k * cl, (la_l - la) * k], -1)
        e_pix = np.stack([(lo_p - lo) * k * cl, (la_p - la) * k], -1)
        return e_line, e_pix

    def velocity_image(self, line, pixel, v_east, v_north):
        """Project a ground velocity (m/s) onto the image axes.

        Returns (v_azimuth, v_range) in m/s; v_range > 0 means moving away
        from the satellite.
        """
        e_line, e_pix = self.axes_en(line, pixel)
        # solve [e_line e_pix] @ [n_line, n_pix] = v  (n in pixels per second)
        a, c = e_line[..., 0], e_line[..., 1]
        b, d = e_pix[..., 0], e_pix[..., 1]
        det = a * d - b * c
        n_line = (d * v_east - b * v_north) / det
        n_pix = (-c * v_east + a * v_north) / det
        return n_line * self.pixel_m, n_pix * self.pixel_m

    # --- radar effects --------------------------------------------------
    def azimuth_shift_m(self, line, pixel, v_range_ms):
        """Displacement along the flight direction of a target with ground
        range velocity v_range: -(R / V) * v_range * sin(incidence)."""
        R = self.slant_range_m(line, pixel)
        v_r = v_range_ms * np.sin(np.radians(self.incidence_deg(line, pixel)))
        return -R * v_r / self.speed_ms

    def swath_of(self, pixel):
        pixel = np.asarray(pixel)
        out = np.empty(pixel.shape, dtype=object)
        for s in self.swaths:
            out[(pixel >= s.first_pixel) & (pixel <= s.last_pixel)] = s.name
        return out

    def prf_at(self, pixel):
        pixel = np.atleast_1d(np.asarray(pixel, float))
        prf = np.full(pixel.shape, self.swaths[-1].prf_hz)
        for s in reversed(self.swaths):
            prf[pixel <= s.last_pixel] = s.prf_hz
        return prf

    def ambiguity_offset_m(self, line, pixel):
        """Along-track distance from a target to its first azimuth ambiguity."""
        R = self.slant_range_m(line, pixel)
        return self.wavelength_m * R * self.prf_at(pixel) / (2.0 * self.speed_ms)


def parse_annotation(xml_text, noise_xml_text):
    r = ET.fromstring(xml_text)
    f = lambda p: r.find(p).text
    pts = r.findall(".//geolocationGridPoint")
    lines = sorted({int(p.find("line").text) for p in pts})
    pixels = sorted({int(p.find("pixel").text) for p in pts})
    li = {v: i for i, v in enumerate(lines)}
    pi = {v: i for i, v in enumerate(pixels)}
    t0 = _t(f(".//imageInformation/productFirstLineUtcTime"))
    grids = {k: np.full((len(lines), len(pixels)), np.nan) for k in ("lat", "lon", "srt", "inc", "azt")}
    for p in pts:
        i, j = li[int(p.find("line").text)], pi[int(p.find("pixel").text)]
        grids["lat"][i, j] = float(p.find("latitude").text)
        grids["lon"][i, j] = float(p.find("longitude").text)
        grids["srt"][i, j] = float(p.find("slantRangeTime").text)
        grids["inc"][i, j] = float(p.find("incidenceAngle").text)
        grids["azt"][i, j] = (_t(p.find("azimuthTime").text) - t0).total_seconds()

    vel = np.array([[float(o.find(f"velocity/{a}").text) for a in "xyz"] for o in r.findall(".//orbit")])
    prf = {d.find("swath").text: float(d.find("prf").text) for d in r.findall(".//downlinkInformation")}

    nr = ET.fromstring(noise_xml_text)
    bounds = {}
    for v in nr.findall(".//noiseAzimuthVector"):
        name = v.find("swath").text
        a, b = int(v.find("firstRangeSample").text), int(v.find("lastRangeSample").text)
        lo, hi = bounds.get(name, (a, b))
        bounds[name] = (min(lo, a), max(hi, b))
    swaths = [Swath(n, *bounds[n], prf[n]) for n in sorted(bounds)]

    return dict(
        pass_dir=f(".//productInformation/pass"),
        t0=t0,
        line_dt_s=float(f(".//imageInformation/azimuthTimeInterval")),
        pixel_m=float(f(".//imageInformation/rangePixelSpacing")),
        n_lines=int(f(".//imageInformation/numberOfLines")),
        n_pixels=int(f(".//imageInformation/numberOfSamples")),
        wavelength_m=C / float(f(".//productInformation/radarFrequency")),
        speed_ms=float(np.linalg.norm(vel, axis=1).mean()),
        swaths=swaths,
        grid_lines=np.array(lines, float),
        grid_pixels=np.array(pixels, float),
        grids=grids,
    )


def load_scene(item_id) -> S1Scene:
    import planetary_computer as pc
    import pystac

    item = pc.sign(pystac.Item.from_dict(_get(STAC + item_id).json()))
    vv = item.assets["vv"].href
    base, query = vv.split("?", 1)
    root = base.rsplit("/measurement/", 1)[0]
    ann = _get(f"{root}/annotation/iw-vv.xml?{query}").text
    noise = _get(item.assets["schema-noise-vv"].href).text
    return S1Scene(item_id=item_id, vv_href=vv, **parse_annotation(ann, noise))
