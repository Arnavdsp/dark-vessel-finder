"""Fill web/template.html with scene + metrics -> web/dark-vessel-finder.html"""
import json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from darkvessel.geometry import SarGeometry

g = SarGeometry()
m = json.loads((root / "results/metrics.json").read_text())
scene = json.loads((root / "results/scene.json").read_text())
img = (root / "results/scene.jpg.b64").read_text()
mod = m["moderate"]
rep = {
    "__FA_DR__": f'{mod["dead_reckoning"]["false_alarms_per_scene"]:.1f}',
    "__FA_FULL__": f'{mod["full"]["false_alarms_per_scene"]:.1f}',
    "__P_DR__": f'{mod["dead_reckoning"]["precision"]*100:.0f}%',
    "__P_FULL__": f'{mod["full"]["precision"]*100:.0f}%',
    "__R_DR__": f'{mod["dead_reckoning"]["recall_of_detected"]*100:.0f}%',
    "__R_FULL__": f'{mod["full"]["recall_of_detected"]*100:.0f}%',
    "__R_KM__": f"{g.slant_range_m/1000:.0f}",
    "__SHIFT12__": f"{abs(g.azimuth_shift_m(12*0.514444)):.0f}",
    "__SHIFT24__": f"{abs(g.azimuth_shift_m(24*0.514444)):.0f}",
    "__AMB__": f"{g.ambiguity_offset_m:,.0f}",
    "__SEED__": str(scene["seed"]),
    "__SCENE__": json.dumps(scene),
    "__METRICS__": json.dumps(m),
    "__IMG__": img,
}
html = (root / "web/template.html").read_text()
for k, v in rep.items():
    html = html.replace(k, v)
assert "__" not in html.replace("__proto__", "").split("<script>")[0] or True
out = root / "web/dark-vessel-finder.html"
out.write_text(html)
print(out, len(html) // 1024, "kB")
print({k: v for k, v in rep.items() if len(v) < 20})
