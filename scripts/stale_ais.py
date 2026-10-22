"""How matching holds up when the AIS report is minutes old.

Coastal AIS is seconds old; AIS received by a satellite over busy water can
be several minutes old. For each processed frame the AIS fixes are rebuilt
from reports at least N minutes older than the image, then matched with a
fixed 300 m gate and with the age-dependent gate.

usage: python scripts/stale_ais.py
"""
import glob
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from darkvessel import ais
from darkvessel.real import REAL_PRESETS, run_frame
from darkvessel.s1 import load_scene
from run_real import ais_csv, fixes_table

AGES_MIN = [0, 2, 5, 10]
VARIANTS = ["+doppler", "+adaptive_gate"]


def main():
    out = {str(a): {v: [0, 0] for v in VARIANTS} for a in AGES_MIN}
    for p in sorted(glob.glob("results/real/*.json")):
        fr = json.load(open(p))
        if not fr["detections"]:
            continue
        s = load_scene(fr["meta"]["item_id"])
        bbox = (s.grids["lon"].min() - 0.1, s.grids["lat"].min() - 0.1,
                s.grids["lon"].max() + 0.1, s.grids["lat"].max() + 0.1)
        mid = s.t0 + timedelta(seconds=s.n_lines * s.line_dt_s / 2)
        df = ais.load_window(ais_csv(s), mid, 30, bbox)
        # reference set: moving vessels with a fresh interpolated fix that the
        # full pipeline pairs on fresh data, i.e. ships the radar clearly saw
        ref = run_frame(fr, REAL_PRESETS["+adaptive_gate"])
        seen = set(ref["pairs"].values())
        moving = {f["mmsi"] for f in fr["ais"]["interp"] if f["sog_kn"] > 2 and f["mmsi"] in seen}
        for age in AGES_MIN:
            fixes = fixes_table(s, ais.vessels_at_image_time(df, s, mode="dr", min_age_s=60 * age))
            fixes = [f for f in fixes if f["mmsi"] in moving]
            frame = dict(fr, ais=dict(fr["ais"], dr=fixes))
            for v in VARIANTS:
                res = run_frame(frame, REAL_PRESETS[v])
                out[str(age)][v][0] += len(fixes)
                out[str(age)][v][1] += len(set(res["pairs"].values()) & moving)
        print(fr["meta"]["item_id"], {a: out[a]["+adaptive_gate"] for a in out}, flush=True)
    rates = {a: {v: n_p[1] / max(n_p[0], 1) for v, n_p in d.items()} | {"n": d[VARIANTS[0]][0]}
             for a, d in out.items()}
    json.dump(rates, open("results/stale_ais.json", "w"), indent=1)
    for a, r in rates.items():
        print(f"AIS >= {a:>2} min old: n={r['n']:5d}  fixed 300 m gate {r['+doppler']:.0%}  age-based gate {r['+adaptive_gate']:.0%}")


if __name__ == "__main__":
    main()
