"""Alert cards for the held-out day: results/alerts.geojson and chip PNGs.

Each card says why the target was flagged: estimated length, brightness over
the local sea, the nearest AIS vessel and how far it was from where the radar
would draw it, the gate it had to fall inside, whether a ghost was ruled
out, distance to the coast and to the nearest charted wind turbine.

usage: python scripts/build_cards.py [n]   (after scripts/train_triage.py)
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from darkvessel.real import REAL_PRESETS, predictions
import eval_real as E


def chip_png(chip, path):
    a = chip.astype(float) ** 2
    db = 10 * np.log10(np.maximum(a, 1))
    lo, hi = np.percentile(db, 5), np.percentile(db, 99.9)
    img = (np.clip((db - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).resize((144, 144), Image.NEAREST).save(path)


def main(n=25):
    t = json.load(open("results/triage.json"))
    frames = {fr["meta"]["item_id"]: fr for fr in E.load_frames()}
    turbines = E.load_turbines()
    land_utm = E.land_in_utm(list(frames.values()))
    os.makedirs("results/cards", exist_ok=True)
    cfg = REAL_PRESETS["+fixed_structures"]
    feats = []
    for rank, a in enumerate(t["ranked_alerts"][:n], start=1):
        fr = frames[a["frame"]]
        d = fr["detections"][a["idx"]]
        fixes = fr["ais"][cfg.fix]
        preds = predictions(fixes, cfg)
        x, y = d["pixel"] * 10, d["line"] * 10
        dist = [math.hypot(x - px, y - py) for px, py, _ in preds]
        k = int(np.argmin(dist))
        gate = cfg.gate_sigmas * math.hypot(preds[k][2], 0.25 * d["length_m"])
        ux, uy = E.UTM.transform(d["lon"], d["lat"])
        chips = np.load(f"results/real/{a['frame']}_chips.npz")["chips"]
        png = f"results/cards/alert_{rank:02d}.png"
        chip_png(chips[a["idx"]], png)
        props = dict(
            rank=rank, score=round(a["score"], 3), frame=a["frame"],
            time_utc=fr["meta"]["t0"][:19], lat=round(d["lat"], 5), lon=round(d["lon"], 5),
            length_m=round(d["length_m"]), brightness_db=round(10 * math.log10(d["peak"]), 1),
            nearest_ais_km=round(dist[k] / 1000, 2), gate_m=round(gate),
            ghost_check_km=round(d["amb_m"] / 1000, 2),
            coast_km=round(float(E.coast_distance(land_utm, [d["lon"]], [d["lat"]])[0]) / 1000, 1),
            turbine_km=round(float(turbines.query([ux, uy])[0]) / 1000, 1) if turbines else None,
            chip=png,
        )
        feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[d["lon"], d["lat"]]),
                          properties=props))
    json.dump(dict(type="FeatureCollection", features=feats), open("results/alerts.geojson", "w"), indent=1)
    for f in feats[:10]:
        p = f["properties"]
        print(p["rank"], p["score"], p["length_m"], p["brightness_db"], p["nearest_ais_km"], p["coast_km"], p["turbine_km"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
