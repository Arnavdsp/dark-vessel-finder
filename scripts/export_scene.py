"""Export one scene + all pipeline outputs for the interactive viewer."""
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darkvessel.simulate import make_scene                  # noqa: E402
from darkvessel.evaluate import run_scene                   # noqa: E402
from darkvessel.associate import PRESETS                    # noqa: E402

SHOWN = ("naive", "dead_reckoning", "full")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1001
sea = sys.argv[2] if len(sys.argv) > 2 else "moderate"

sc = make_scene(seed=seed, sea_state=sea)
dets, labels, out = run_scene(sc, {k: PRESETS[k] for k in SHOWN})

db = 10 * np.log10(np.maximum(sc.image, 1e-3))
lo, hi = np.percentile(db, 2), np.percentile(db, 99.97)
u8 = (np.clip((db - lo) / (hi - lo), 0, 1) ** 0.9 * 255).astype(np.uint8)
buf = io.BytesIO()
Image.fromarray(u8).save(buf, "JPEG", quality=72)
img_b64 = base64.b64encode(buf.getvalue()).decode()

ships = {s.id: s for s in sc.ships}


def r1(v):
    return round(float(v), 1)


det_rows = []
for i, d in enumerate(dets):
    kind, ref = labels[i]
    row = dict(i=i, x=r1(d.x_m), y=r1(d.y_m), peak=r1(10 * math.log10(d.peak)),
               len=r1(d.length_m), truth=kind, ref=ref,
               truth_dark=bool(kind == "ship" and ships[ref].dark), out={})
    for k in SHOWN:
        res = out[k][0]
        if i in res.ghosts:
            row["out"][k] = {"c": "ghost"}
        elif i in res.pairs:
            row["out"][k] = {"c": "matched", "ship": res.pairs[i]}
        else:
            # nearest AIS prediction, to explain why it was flagged
            best = None
            for sid, (px, py, sig) in res.predicted.items():
                dist = math.hypot(d.x_m - px, d.y_m - py)
                if best is None or dist < best[1]:
                    best = (sid, dist, sig)
            row["out"][k] = {"c": "dark"}
            if best:
                row["out"][k].update(near=best[0], dist=round(best[1]),
                                     gate=round(best[2] * PRESETS[k].gate_sigmas))
    det_rows.append(row)

ais_rows = []
for a in sc.ais:
    row = dict(ship=a.ship_id, x=r1(a.x_m), y=r1(a.y_m), dt=round(a.t_offset_s),
               sog_kn=round(a.sog_ms / 0.514444, 1), cog=round(a.cog_deg), pred={})
    for k in SHOWN:
        px, py, sig = out[k][0].predicted[a.ship_id]
        row["pred"][k] = [r1(px), r1(py), round(sig * PRESETS[k].gate_sigmas)]
    ais_rows.append(row)

ship_rows = [dict(id=s.id, kind=s.kind, len=round(s.length_m), kn=round(s.speed_ms / 0.514444, 1),
                  course=round(s.course_deg), x=r1(s.x_m), y=r1(s.y_m), ax=r1(s.app_x_m),
                  ay=r1(s.app_y_m), ais=s.has_ais, spoofed=s.spoofed, dark=s.dark,
                  shift=r1(s.app_y_m - s.y_m), inimg=s.in_image) for s in sc.ships]

payload = dict(
    seed=seed, sea=sea, size_m=sc.size_m[0], pixel_m=sc.geom.pixel_m,
    geom=dict(R=round(sc.geom.slant_range_m), V=sc.geom.platform_speed_ms,
              inc=sc.geom.incidence_deg, amb=round(sc.geom.ambiguity_offset_m),
              lam=sc.geom.wavelength_m, prf=sc.geom.prf_hz),
    scores={k: out[k][1] for k in SHOWN},
    dets=det_rows, ais=ais_rows, ships=ship_rows,
    ghosts=[[r1(x), r1(y), p] for x, y, p in sc.ghosts],
)
res_dir = Path(__file__).resolve().parents[1] / "results"
res_dir.mkdir(exist_ok=True)
(res_dir / "scene.json").write_text(json.dumps(payload))
(res_dir / "scene.jpg.b64").write_text(img_b64)
print(json.dumps(payload["scores"], indent=1))
print("dets", len(dets), "jpeg kB", len(buf.getvalue()) // 1024)
