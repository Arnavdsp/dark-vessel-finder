"""Danish Maritime Authority AIS: load a time window and place each vessel
where it was when the radar imaged it.

Columns follow the aisdk CSV (web.ais.dk/aisdata). SOG is in knots, COG and
heading in degrees, A/B/C/D are antenna distances to bow/stern/port/starboard.
"""
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

KN = 0.514444
EARTH_R = 6_371_000.0

COLS = {
    "# Timestamp": "t", "Type of mobile": "kind", "MMSI": "mmsi", "Latitude": "lat",
    "Longitude": "lon", "Navigational status": "status", "SOG": "sog", "COG": "cog",
    "Heading": "heading", "Name": "name", "Ship type": "ship_type", "Width": "width",
    "Length": "length", "A": "a", "B": "b", "C": "c", "D": "d",
}


def load_window(csv_path, t_center, minutes=20, bbox=None):
    """Rows within +-minutes of t_center (naive UTC datetime), cleaned."""
    lo, hi = t_center - timedelta(minutes=minutes), t_center + timedelta(minutes=minutes)
    keep = []
    for chunk in pd.read_csv(csv_path, usecols=list(COLS), chunksize=1_000_000,
                             dtype={"MMSI": "int64"}, low_memory=False):
        chunk = chunk.rename(columns=COLS)
        chunk["t"] = pd.to_datetime(chunk["t"], format="%d/%m/%Y %H:%M:%S")
        chunk = chunk[(chunk.t >= lo) & (chunk.t <= hi)]
        if bbox is not None:
            w, s, e, n = bbox
            chunk = chunk[chunk.lon.between(w, e) & chunk.lat.between(s, n)]
        keep.append(chunk)
    df = pd.concat(keep, ignore_index=True)
    df = df[df.lat.between(-90, 90) & df.lon.between(-180, 180)]
    df = df[(df.mmsi >= 100_000_000) & (df.mmsi <= 999_999_999)]
    return df.drop_duplicates(["mmsi", "t", "lat", "lon"]).sort_values(["mmsi", "t"])


def _offset(lat, lon, de, dn):
    lat2 = lat + np.degrees(dn / EARTH_R)
    lon2 = lon + np.degrees(de / (EARTH_R * np.cos(np.radians(lat))))
    return lat2, lon2


@dataclass
class Fix:
    mmsi: int
    kind: str
    lat: float
    lon: float
    v_east: float
    v_north: float
    sog_kn: float
    age_s: float          # time from the nearest report used to the image time
    method: str           # "interp", "dr" (dead reckoning) or "last"
    length_m: float
    width_m: float
    ship_type: str
    name: str
    status: str


def _hull_centre(lat, lon, row, heading_deg):
    """Move the GPS antenna position to the hull centre."""
    a, b, c, d = (row.get(k) for k in "abcd")
    if any(pd.isna(v) for v in (a, b, c, d)):
        return lat, lon
    fwd = (a - b) / 2.0          # metres towards the bow
    stbd = (c - d) / 2.0         # metres towards starboard
    h = np.radians(heading_deg)
    de = fwd * np.sin(h) + stbd * np.cos(h)
    dn = fwd * np.cos(h) - stbd * np.sin(h)
    return _offset(lat, lon, de, dn)


def fix_at(track: pd.DataFrame, t, mode="interp", max_gap_s=180, max_age_s=900, min_age_s=0):
    """Position of one vessel at time t.

    mode="interp": interpolate between the reports either side when both are
        close, else dead-reckon from the nearest earlier report.
    mode="dr":     dead-reckon from the latest report at or before t.
    mode="last":   the latest report before t, unmoved (what a naive matcher does).
    min_age_s ignores reports newer than t - min_age_s, to mimic AIS heard from
    orbit, where message collisions leave minutes between received reports.
    """
    if min_age_s:
        track = track[track.t <= t - timedelta(seconds=min_age_s)]
    before = track[track.t <= t]
    after = track[track.t > t]
    p = before.iloc[-1] if len(before) else None
    q = after.iloc[0] if len(after) else None
    if p is None:
        if q is None or mode != "interp":
            return None
        p, q = q, None          # only a later report: dead-reckon backwards
    age = (t - p.t).total_seconds()
    if abs(age) > max_age_s:
        return None

    sog = 0.0 if pd.isna(p.sog) or p.sog > 102 else float(p.sog)
    cog = 0.0 if pd.isna(p.cog) or p.cog >= 360 else float(p.cog)
    ve, vn = sog * KN * np.sin(np.radians(cog)), sog * KN * np.cos(np.radians(cog))

    if mode == "interp" and q is not None and age <= max_gap_s and (q.t - t).total_seconds() <= max_gap_s:
        w = age / (q.t - p.t).total_seconds()
        lat, lon = p.lat + w * (q.lat - p.lat), p.lon + w * (q.lon - p.lon)
        method, used_age = "interp", min(age, (q.t - t).total_seconds())
    elif mode == "last":
        lat, lon, method, used_age = p.lat, p.lon, "last", age
    else:
        lat, lon = _offset(p.lat, p.lon, ve * age, vn * age)
        method, used_age = "dr", abs(age)

    hdg = p.heading if not pd.isna(p.heading) and p.heading < 360 else cog
    lat, lon = _hull_centre(lat, lon, p, hdg)
    val = lambda k: float(p[k]) if not pd.isna(p[k]) else np.nan
    return Fix(int(p.mmsi), str(p.kind), float(lat), float(lon), ve, vn, sog, used_age, method,
               val("length"), val("width"), str(p.ship_type), str(p["name"]), str(p.status))


def vessels_at_image_time(df, scene, mode="interp", **kw):
    """Fix every vessel at the moment the radar saw it.

    The imaging time depends on where the ship is in the frame, so the line
    is found from a first guess and the fix is redone at that line's time.
    """
    mid = scene.t0 + timedelta(seconds=scene.n_lines * scene.line_dt_s / 2)
    out = []
    for mmsi, tr in df[df.kind.isin(["Class A", "Class B"])].groupby("mmsi"):
        f = fix_at(tr, mid, mode, **kw)
        if f is None:
            continue
        for _ in range(2):
            L, P = scene.to_pixel(f.lat, f.lon)
            t = scene.t0 + timedelta(seconds=float(scene.az_time_s(L, P)[0]))
            g = fix_at(tr, t, mode, **kw)
            if g is None:
                break
            f = g
        out.append(f)
    return out


def aids_to_navigation(df):
    """Latest position of each AtoN (buoys, lights, platforms that broadcast)."""
    a = df[df.kind == "AtoN"].groupby("mmsi").last().reset_index()
    return a[["mmsi", "lat", "lon", "name"]]
