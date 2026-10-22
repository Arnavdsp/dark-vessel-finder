"""Fill web/results_template.html with the numbers in results/ -> web/results.html.

usage: python scripts/build_results_page.py
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")


def chip_b64(path):
    buf = io.BytesIO()
    Image.open(path).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def low_chips(triage, n=8):
    out = []
    for a in triage["ranked_alerts"][-n:]:
        chips = np.load(os.path.join(ROOT, f"results/real/{a['frame']}_chips.npz"))["chips"]
        c = chips[a["idx"]].astype(float) ** 2
        db = 10 * np.log10(np.maximum(c, 1))
        lo, hi = np.percentile(db, 5), np.percentile(db, 99.9)
        img = Image.fromarray((np.clip((db - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8))
        buf = io.BytesIO()
        img.resize((144, 144), Image.NEAREST).save(buf, "PNG", optimize=True)
        out.append(dict(score=round(a["score"], 3),
                        src="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()))
    return out


def main():
    r = lambda p: json.load(open(os.path.join(ROOT, p)))
    m, t = r("results/real_metrics.json"), r("results/triage.json")
    sim = r("results/metrics.json")
    cards = r("results/alerts.geojson")["features"][:12]
    stale = r("results/stale_ais.json") if os.path.exists(os.path.join(ROOT, "results/stale_ais.json")) else None
    data = dict(
        frames=m["frames"], sea_km2=m["sea_km2_processed"],
        variants=m["variants"], doppler=m["doppler_check"],
        points=[[round(p["pred"]), round(p["meas"]), round(p["sog"], 1)] for p in m["doppler_points"] if p["sog"] > 2],
        ghost=m["ghost_control"], triage={k: v for k, v in t.items() if k != "ranked_alerts"},
        cards=[dict(f["properties"], src=chip_b64(os.path.join(ROOT, f["properties"]["chip"]))) for f in cards],
        low=low_chips(t), stale=stale,
        sim={k: v for k, v in sim.get("moderate", {}).items()} if isinstance(sim, dict) else None,
    )
    tpl = open(os.path.join(ROOT, "web/results_template.html")).read()
    html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    open(os.path.join(ROOT, "web/results.html"), "w").write(html)
    print(f"web/results.html {len(html) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
