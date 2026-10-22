"""Detect ships in Sentinel-1 frames and place AIS vessels on the same image.

For each frame this writes results/real/<item_id>.json with every CFAR
detection on open water and every AIS vessel fix (interpolated, dead-reckoned
and last-report), plus image chips in results/real/<item_id>_chips.npz.
Association and scoring happen in scripts/eval_real.py so they can be rerun
without touching the rasters.

usage: python scripts/run_real.py <item_id> [<item_id> ...]
       python scripts/run_real.py --frames data/frames.json [--jobs 4]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from darkvessel import ais
from darkvessel.detect import cfar
from darkvessel.landmask import land_polygons, tile_mask
from darkvessel.s1 import load_scene

TILE, PAD = 4096, 128
ENL = 4.4                 # equivalent number of looks, IW GRDH
CHIP = 48
OUT = "results/real"


def ais_csv(scene):
    return f"data/ais/ais-{scene.t0:%Y-%m-%d}-passes.csv"


def fixes_table(scene, fixes):
    if not fixes:
        return []
    L, P = scene.to_pixel([f.lat for f in fixes], [f.lon for f in fixes])
    v_az, v_rg = scene.velocity_image(L, P, np.array([f.v_east for f in fixes]),
                                      np.array([f.v_north for f in fixes]))
    shift = scene.azimuth_shift_m(L, P, v_rg)
    rows = []
    for f, l, p, va, vr, sh in zip(fixes, L, P, v_az, v_rg, shift):
        r = dict(f.__dict__)
        r.update(line=float(l), pixel=float(p), v_az=float(va), v_rg=float(vr), shift_m=float(sh))
        rows.append(r)
    return rows


def process(item_id):
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/{item_id}.json"
    if os.path.exists(out_json):
        return item_id, "cached"
    s = load_scene(item_id)
    bbox = (s.grids["lon"].min() - 0.1, s.grids["lat"].min() - 0.1,
            s.grids["lon"].max() + 0.1, s.grids["lat"].max() + 0.1)
    land = land_polygons("data/ne_10m_land.zip", bbox)
    mid = s.t0 + timedelta(seconds=s.n_lines * s.line_dt_s / 2)
    df = ais.load_window(ais_csv(s), mid, 20, bbox)

    fixes = {m: fixes_table(s, ais.vessels_at_image_time(df, s, mode=m))
             for m in ("interp", "dr", "last")}
    aton = ais.aids_to_navigation(df)
    if len(aton):
        aL, aP = s.to_pixel(aton.lat.values, aton.lon.values)
        aton = [dict(mmsi=int(m), lat=float(a), lon=float(o), line=float(l), pixel=float(p), name=str(n))
                for m, a, o, l, p, n in zip(aton.mmsi, aton.lat, aton.lon, aL, aP, aton["name"])]
    else:
        aton = []
    ais_lp = np.array([[r["line"], r["pixel"]] for r in fixes["interp"]]) if fixes["interp"] else np.zeros((0, 2))

    dets, chips, sea_px = [], [], 0
    with rasterio.open(s.vv_href) as ds:
        for L0 in range(0, s.n_lines, TILE):
            for P0 in range(0, s.n_pixels, TILE):
                # only tiles with AIS vessels nearby can be scored
                near = ((ais_lp[:, 0] > L0 - 2000) & (ais_lp[:, 0] < L0 + TILE + 2000) &
                        (ais_lp[:, 1] > P0 - 2000) & (ais_lp[:, 1] < P0 + TILE + 2000)).sum()
                if near == 0:
                    continue
                l0, p0 = max(L0 - PAD, 0), max(P0 - PAD, 0)
                l1, p1 = min(L0 + TILE + PAD, s.n_lines), min(P0 + TILE + PAD, s.n_pixels)
                mask = tile_mask(s, land, l0, p0, l1 - l0, p1 - p0)
                if mask.mean() > 0.97:
                    continue
                dn = ds.read(1, window=Window(p0, l0, p1 - p0, l1 - l0)).astype(np.float32)
                sea = ~mask & (dn > 0)
                if sea.sum() < 50_000:
                    continue
                I = dn ** 2
                med = float(np.median(I[sea]))
                I[~sea] = med
                found, _ = cfar(I / med, s.pixel_m, guard_px=41, train_px=101, pfa=1e-7,
                                looks=ENL, presum=1, group_px=2, min_area_px=3)
                core = sea[PAD if l0 else 0: (L0 + TILE) - l0, PAD if p0 else 0: (P0 + TILE) - p0]
                sea_px += int(core.sum())
                for d in found:
                    yl, xp = d.y_m / s.pixel_m, d.x_m / s.pixel_m
                    gl, gp = l0 + yl, p0 + xp
                    if not (L0 <= gl < L0 + TILE and P0 <= gp < P0 + TILE):
                        continue
                    iy, ix = int(round(yl)), int(round(xp))
                    if not sea[min(iy, sea.shape[0] - 1), min(ix, sea.shape[1] - 1)]:
                        continue
                    h = CHIP // 2
                    chip = np.zeros((CHIP, CHIP), np.uint16)
                    sub = dn[max(iy - h, 0): iy + h, max(ix - h, 0): ix + h].astype(np.uint16)
                    chip[: sub.shape[0], : sub.shape[1]] = sub
                    chips.append(chip)
                    dets.append(dict(line=float(gl), pixel=float(gp), peak=d.peak,
                                     peak_db=float(10 * np.log10(d.peak_raw * med)),
                                     area_px=d.area_px, length_m=d.length_m,
                                     clutter_db=float(10 * np.log10(med))))
    if dets:
        lat, lon = s.to_latlon(np.array([d["line"] for d in dets]), np.array([d["pixel"] for d in dets]))
        amb = s.ambiguity_offset_m(np.array([d["line"] for d in dets]), np.array([d["pixel"] for d in dets]))
        sw = s.swath_of(np.array([d["pixel"] for d in dets]))
        for d, a, o, am, w in zip(dets, lat, lon, amb, sw):
            d.update(lat=float(a), lon=float(o), amb_m=float(am), swath=str(w))
    meta = dict(item_id=item_id, pass_dir=s.pass_dir, t0=s.t0.isoformat(), speed_ms=s.speed_ms,
                wavelength_m=s.wavelength_m, pixel_m=s.pixel_m, n_lines=s.n_lines, n_pixels=s.n_pixels,
                swaths=[w.__dict__ for w in s.swaths], sea_km2=sea_px * s.pixel_m ** 2 / 1e6)
    json.dump(dict(meta=meta, detections=dets, ais=fixes, aton=aton), open(out_json, "w"))
    np.savez_compressed(f"{OUT}/{item_id}_chips.npz", chips=np.array(chips, np.uint16).reshape(-1, CHIP, CHIP))
    return item_id, f"{len(dets)} detections, {len(fixes['interp'])} AIS, {meta['sea_km2']:.0f} km2 sea"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*")
    ap.add_argument("--frames")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    items = a.items or json.load(open(a.frames))
    if a.jobs == 1:
        for it in items:
            print(*process(it), flush=True)
    else:
        with ProcessPoolExecutor(a.jobs) as ex:
            for it, msg in ex.map(process, items):
                print(it, msg, flush=True)
