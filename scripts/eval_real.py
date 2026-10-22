"""Score every processed Sentinel-1 frame and write results/real_metrics.json.

usage: python scripts/eval_real.py
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
import shapely

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from darkvessel.landmask import land_polygons
from darkvessel.real import (REAL_PRESETS, LENGTH_BINS, run_frame, orphan_alerts,
                             ghost_offset_control)

OPEN_WATER_M = 2000.0     # scored zone: at least this far from the coastline
DANISH_M = 40000.0        # ...and in Danish AIS reception: within this distance of
                          # Denmark and closer to Denmark than to any other country
RESIGHT_M = 50.0
TURBINE_M = 150.0
UTM = Transformer.from_crs(4326, 25832, always_xy=True)


def load_frames():
    frames = []
    for p in sorted(glob.glob("results/real/*.json")):
        fr = json.load(open(p))
        fr["t"] = datetime.fromisoformat(fr["meta"]["t0"])
        frames.append(fr)
    return frames


def coast_distance(land_utm, lon, lat):
    x, y = UTM.transform(np.asarray(lon), np.asarray(lat))
    return shapely.distance(land_utm, shapely.points(x, y))


def load_turbines(path="data/osm_wind.json"):
    if not os.path.exists(path):
        return None
    pts = [UTM.transform(e["lon"], e["lat"]) for e in json.load(open(path))["elements"]]
    return cKDTree(pts)


def fixed_structure_masks(frames):
    """Known structures: OpenStreetMap wind turbines and AIS aids to navigation
    within 150 m. Returns per-frame removal masks, per-frame 'seen on two
    other days' flags, and counts.

    Re-sighting is only a flag. Tested on these frames, a rule that removed
    any target seen at the same spot on another day also removed 41% of
    AIS-paired ships (anchorages, ferry routes), so it lowers an alert's
    priority instead of deleting it."""
    pts, owner = [], []
    for k, fr in enumerate(frames):
        for d in fr["detections"]:
            pts.append(UTM.transform(d["lon"], d["lat"]))
            owner.append(k)
    pts, owner = np.array(pts), np.array(owner)
    day = np.array([fr["t"].date().toordinal() for fr in frames])
    tree = cKDTree(pts)
    aton = [UTM.transform(a["lon"], a["lat"]) for fr in frames for a in fr["aton"]]
    aton_tree = cKDTree(aton) if aton else None
    turbines = load_turbines()
    masks, resight, why, i = [], [], Counter(), 0
    for k, fr in enumerate(frames):
        m = np.zeros(len(fr["detections"]), bool)
        r = np.zeros(len(fr["detections"]), bool)
        for j in range(len(fr["detections"])):
            if turbines is not None and turbines.query(pts[i])[0] < TURBINE_M:
                m[j] = True; why["wind_turbine_osm"] += 1
            elif aton_tree is not None and aton_tree.query(pts[i])[0] < 150:
                m[j] = True; why["aton"] += 1
            near = tree.query_ball_point(pts[i], RESIGHT_M)
            r[j] = len({day[owner[n]] for n in near if day[owner[n]] != day[k]}) >= 2
            i += 1
        masks.append(m)
        resight.append(r)
    return masks, resight, why


def length_bin(L):
    for lo, hi in LENGTH_BINS:
        if lo <= L < hi:
            return f"{lo}-{hi}" if hi < 1e9 else f">{lo}"
    return "unknown"


def doppler_points(frames, coast):
    """(predicted shift, measured along-track offset, sog) for isolated moving
    AIS vessels with a fresh report and exactly one detection nearby."""
    rows = []
    for fr, (_, ais_d) in zip(frames, coast):
        D = fr["detections"]
        if not D:
            continue
        dl = np.array([[d["line"], d["pixel"]] for d in D])
        A = fr["ais"]["interp"]
        tA = cKDTree([[a["line"], a["pixel"]] for a in A])
        for a, cd in zip(A, ais_d):
            if a["method"] != "interp" or a["age_s"] > 60 or cd < OPEN_WATER_M:
                continue
            if len(tA.query_ball_point([a["line"], a["pixel"]], 150)) > 1:
                continue
            m = (np.abs(dl[:, 1] - a["pixel"]) < 15) & (np.abs(dl[:, 0] - a["line"]) < 80)
            if m.sum() != 1:
                continue
            d = D[np.flatnonzero(m)[0]]
            rows.append(dict(pred=a["shift_m"], meas=(d["line"] - a["line"]) * 10,
                             cross=(d["pixel"] - a["pixel"]) * 10, sog=a["sog_kn"],
                             v_rg=a["v_rg"], length=a["length_m"], frame=fr["meta"]["item_id"]))
    return rows


def land_in_utm(frames):
    allb = [(min(d["lon"] for d in f["detections"]), min(d["lat"] for d in f["detections"]),
             max(d["lon"] for d in f["detections"]), max(d["lat"] for d in f["detections"]))
            for f in frames if f["detections"]]
    bbox = (min(b[0] for b in allb) - .5, min(b[1] for b in allb) - .5,
            max(b[2] for b in allb) + .5, max(b[3] for b in allb) + .5)
    land = land_polygons("data/ne_10m_land.zip", bbox)
    land_utm = shapely.transform(land, lambda c: np.column_stack(UTM.transform(c[:, 0], c[:, 1])))
    shapely.prepare(land_utm)
    return land_utm


def _countries_utm():
    import io, zipfile, shapefile
    from shapely.geometry import shape
    z = zipfile.ZipFile("data/ne_10m_admin_0_countries.zip")
    n = {k: z.read(f"ne_10m_admin_0_countries.{k}") for k in ("shp", "shx", "dbf")}
    sf = shapefile.Reader(shp=io.BytesIO(n["shp"]), shx=io.BytesIO(n["shx"]), dbf=io.BytesIO(n["dbf"]))
    names = [f.name for f in sf.fields[1:]]
    out = {}
    for rec, shp in zip(sf.records(), sf.shapes()):
        name = dict(zip(names, rec))["ADMIN"]
        if name in ("Denmark", "Germany", "Sweden", "Poland", "Norway", "Netherlands"):
            g = shape(shp.__geo_interface__)
            g = shapely.transform(g, lambda c: np.column_stack(UTM.transform(c[:, 0], c[:, 1])))
            shapely.prepare(g)
            out[name] = g
    return out


def in_danish_coverage(lon, lat, countries):
    x, y = UTM.transform(np.asarray(lon), np.asarray(lat))
    p = shapely.points(x, y)
    dk = shapely.distance(countries["Denmark"], p)
    other = np.min([shapely.distance(g, p) for k, g in countries.items() if k != "Denmark"], axis=0)
    return (dk <= DANISH_M) & (dk < other)


def coast_and_coverage(frames, land_utm):
    """Per frame: (detection distance to coast, AIS distance to coast).
    Detections outside AIS coverage get -1 so they never count as open water."""
    countries = _countries_utm()
    coast = []
    for fr in frames:
        D, A = fr["detections"], fr["ais"]["interp"]
        dd = coast_distance(land_utm, [d["lon"] for d in D], [d["lat"] for d in D]) if D else np.zeros(0)
        ad = coast_distance(land_utm, [a["lon"] for a in A], [a["lat"] for a in A]) if A else np.zeros(0)
        # outside Danish AIS reception a detection looks dark for want of a receiver
        if D:
            cov = in_danish_coverage([d["lon"] for d in D], [d["lat"] for d in D], countries)
            dd = np.where(cov, dd, -1.0)
        if A:
            cov = in_danish_coverage([a["lon"] for a in A], [a["lat"] for a in A], countries)
            ad = np.where(cov, ad, -1.0)
        coast.append((dd, ad))
    return coast


def main():
    frames = load_frames()
    coast = coast_and_coverage(frames, land_in_utm(frames))
    fixed, resight, fixed_why = fixed_structure_masks(frames)

    sea_km2 = sum(fr["meta"]["sea_km2"] for fr in frames)
    out = dict(frames=len(frames), sea_km2_processed=sea_km2, fixed_reasons=dict(fixed_why), variants={})

    for name, cfg in REAL_PRESETS.items():
        tot = Counter()
        by_len = defaultdict(lambda: [0, 0])
        for fr, (dd, ad), fm in zip(frames, coast, fixed):
            res = run_frame(fr, cfg, fm)
            open_det = dd >= OPEN_WATER_M
            dark_open = [i for i in res["dark"] if open_det[i]]
            orph = set(orphan_alerts(fr, res))
            tot["alerts_open"] += len(dark_open)
            tot["alerts_coastal"] += len(res["dark"]) - len(dark_open)
            tot["orphan_open"] += sum(1 for i in dark_open if i in orph)
            tot["ghosts_removed_open"] += sum(1 for i in res["ghosts"] if open_det[i])
            tot["fixed_removed_open"] += sum(1 for i in res["fixed"] if open_det[i])
            # AIS pairing on open water, using the interpolated fix as the reference set
            paired = set(res["pairs"].values())
            in_img = {a["mmsi"]: a for a in fr["ais"][cfg.fix]
                      if 0 <= a["line"] < fr["meta"]["n_lines"] and 0 <= a["pixel"] < fr["meta"]["n_pixels"]}
            for a, cd in zip(fr["ais"]["interp"], ad):
                if cd < OPEN_WATER_M or a["mmsi"] not in in_img:
                    continue
                tot["ais_open"] += 1
                hit = a["mmsi"] in paired
                tot["ais_open_paired"] += hit
                b = by_len[length_bin(a["length_m"]) if a["length_m"] == a["length_m"] else "unknown"]
                b[0] += 1
                b[1] += hit
        v = dict(tot)
        v["orphan_share_open"] = tot["orphan_open"] / max(tot["alerts_open"], 1)
        v["pairing_rate_open"] = tot["ais_open_paired"] / max(tot["ais_open"], 1)
        v["pairing_by_length"] = {k: dict(n=n, paired=p, rate=p / max(n, 1)) for k, (n, p) in sorted(by_len.items())}
        out["variants"][name] = v

    pts = doppler_points(frames, coast)
    pred = np.array([p["pred"] for p in pts]); meas = np.array([p["meas"] for p in pts])
    mov = np.array([p["sog"] > 2 for p in pts])
    k, b = np.polyfit(pred[mov], meas[mov], 1)
    out["doppler_check"] = dict(
        n=int(mov.sum()), n_stationary=int((~mov).sum()), slope=float(k), intercept_m=float(b),
        r=float(np.corrcoef(pred[mov], meas[mov])[0, 1]),
        median_abs_error_naive_m=float(np.median(np.abs(meas[mov]))),
        median_abs_error_corrected_m=float(np.median(np.abs(meas[mov] - pred[mov]))),
        stationary_median_az_m=float(np.median(meas[~mov])),
        cross_range_median_m=float(np.median([p["cross"] for p in pts])))
    out["doppler_points"] = pts
    out["ghost_control"] = {str(k): v for k, v in ghost_offset_control(frames).items()}

    # what is left after the full pipeline: open-water candidates for review
    cands = []
    for fr, (dd, _), fm, rs in zip(frames, coast, fixed, resight):
        res = run_frame(fr, REAL_PRESETS["+fixed_structures"], fm)
        orph = set(orphan_alerts(fr, res))
        for i in res["dark"]:
            if dd[i] >= OPEN_WATER_M:  # open water inside AIS coverage
                d = fr["detections"][i]
                cands.append(dict(frame=fr["meta"]["item_id"], idx=i, lat=d["lat"], lon=d["lon"],
                                  peak=d["peak"], length_m=d["length_m"], coast_km=float(dd[i] / 1000),
                                  near_unpaired_ais=i in orph, seen_other_days=bool(rs[i])))
    out["candidates"] = sorted(cands, key=lambda c: -c["peak"])
    json.dump(out, open("results/real_metrics.json", "w"), indent=1)

    print(f"{len(frames)} frames, {sea_km2:,.0f} km2 of sea processed")
    print(f"{'variant':20s} {'alerts':>7s} {'orphans':>8s} {'ghosts-':>8s} {'fixed-':>7s} {'AIS paired':>10s}")
    for name, v in out["variants"].items():
        print(f"{name:20s} {v['alerts_open']:7d} {v['orphan_open']:8d} {v['ghosts_removed_open']:8d} "
              f"{v['fixed_removed_open']:7d} {100 * v['pairing_rate_open']:9.0f}%")
    print("fixed structures by rule:", out["fixed_reasons"])
    print("pairing by length (full):", {k: round(x["rate"], 2) for k, x in out["variants"]["+fixed_structures"]["pairing_by_length"].items()})
    print("doppler:", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["doppler_check"].items()})
    print("ghost control:", out["ghost_control"])


if __name__ == "__main__":
    main()
