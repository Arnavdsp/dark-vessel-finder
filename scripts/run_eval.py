"""Monte Carlo evaluation across sea states. Writes results/metrics.json."""
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darkvessel.simulate import make_scene                       # noqa: E402
from darkvessel.evaluate import run_scene, summarise              # noqa: E402
from darkvessel.associate import PRESETS                          # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

configs = dict(PRESETS)
for g in (150, 500, 800):
    configs[f"dead_reckoning_gate{g}"] = replace(PRESETS["dead_reckoning"],
                                                 name=f"dead_reckoning_gate{g}", gate_m=g)

LEN_BINS = [(0, 20), (20, 40), (40, 100), (100, 400)]
t0 = time.time()
metrics = {}
for ss in ("calm", "moderate", "rough"):
    rows = defaultdict(list)
    by_len = {b: [0, 0] for b in LEN_BINS}
    for seed in range(N):
        sc = make_scene(seed=1000 + seed, sea_state=ss)
        dets, labels, out = run_scene(sc, configs)
        for k, (_, s) in out.items():
            rows[k].append(s)
        hit = {ref for kind, ref in labels if kind == "ship"}
        for s in sc.ships:
            if not s.in_image:
                continue
            for b in LEN_BINS:
                if b[0] <= s.length_m < b[1]:
                    by_len[b][1] += 1
                    by_len[b][0] += s.id in hit
    metrics[ss] = {k: summarise(v) for k, v in rows.items()}
    metrics[ss]["_detector_recall_by_length"] = {
        f"{a}-{b} m": round(h / max(n, 1), 3) for (a, b), (h, n) in by_len.items()}
    print(ss, "done", round(time.time() - t0, 1), "s", flush=True)

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
for ss, m in metrics.items():
    print(f"\n== {ss}")
    for k, v in m.items():
        if k.startswith("_"):
            print("  detector recall by length:", v)
            continue
        print(f"  {k:24s} precision {v['precision']:.2f}  recall {v['recall_of_detected']:.2f}  "
              f"false alarms/scene {v['false_alarms_per_scene']:.2f}  pairing {v['pairing_accuracy']:.2f}")
