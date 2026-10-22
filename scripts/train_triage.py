"""Train the alert-ranking model and score it on a held-out day.

Training frames: 12-14 Oct 2026. Test frames: 19 Oct 2026, so no scene (and
no day's weather) appears on both sides. Positives are AIS-paired detections,
negatives the unexplained alerts, both on open water inside AIS coverage.

usage: python scripts/train_triage.py   (after scripts/eval_real.py)
"""
import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from darkvessel.real import REAL_PRESETS, run_frame
from darkvessel.triage import FEATURES, feature_row
import eval_real as E

TEST_DAY = "2026-10-19"
KEEP_SHIPS = 0.95


def build_rows():
    frames = E.load_frames()
    land_utm = E.land_in_utm(frames)
    coast = E.coast_and_coverage(frames, land_utm)
    fixed, resight, _ = E.fixed_structure_masks(frames)
    X, y, day, ref, lens = [], [], [], [], []
    for fr, (dd, _), fm, rs in zip(frames, coast, fixed, resight):
        chips = np.load(f"results/real/{fr['meta']['item_id']}_chips.npz")["chips"]
        res = run_frame(fr, REAL_PRESETS["+fixed_structures"], fm)
        ais_len = {a["mmsi"]: a["length_m"] for a in fr["ais"]["dr"]}
        for label, idx in ((1, res["pairs"].keys()), (0, res["dark"])):
            for i in idx:
                if dd[i] < E.OPEN_WATER_M:
                    continue
                X.append(feature_row(fr["detections"][i], chips[i], dd[i] / 1000, rs[i]))
                y.append(label)
                day.append(fr["meta"]["t0"][:10])
                ref.append((fr["meta"]["item_id"], i))
                lens.append(ais_len.get(res["pairs"].get(i), np.nan) if label else np.nan)
    return np.array(X), np.array(y), np.array(day), ref, np.array(lens, float)


def main():
    X, y, day, ref, lens = build_rows()
    tr, te = day != TEST_DAY, day == TEST_DAY
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=0)
    model.fit(X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    yt = y[te]
    auc = roc_auc_score(yt, p)
    # threshold that keeps KEEP_SHIPS of the held-out AIS-confirmed ships
    thr = np.quantile(p[yt == 1], 1 - KEEP_SHIPS)
    alerts_kept = float((p[yt == 0] >= thr).mean())

    # the same budget with brightness alone, the obvious baseline
    peak = X[te][:, FEATURES.index("log_peak")]
    thr_pk = np.quantile(peak[yt == 1], 1 - KEEP_SHIPS)
    alerts_kept_peak = float((peak[yt == 0] >= thr_pk).mean())

    out = dict(test_day=TEST_DAY, n_train=int(tr.sum()), n_test=int(te.sum()),
               test_ships=int((yt == 1).sum()), test_alerts=int((yt == 0).sum()),
               auc=float(auc), auc_peak_only=float(roc_auc_score(yt, peak)),
               ships_kept=KEEP_SHIPS, alerts_kept=alerts_kept, alerts_kept_peak_only=alerts_kept_peak,
               threshold=float(thr))
    # rank the held-out alerts for the results page
    alerts = [(float(s), ref_) for s, ref_, lab in zip(p, np.array(ref, dtype=object)[te], yt) if lab == 0]
    ranked = sorted(alerts, key=lambda a: -a[0])
    out["ranked_alerts"] = [dict(frame=r[0], idx=int(r[1]), score=s) for s, r in ranked]
    out["alerts_above_threshold"] = int(sum(s >= thr for s, _ in ranked))
    # ship retention by AIS length at the chosen threshold (are small boats ranked down?)
    lengths = lens[te]
    by_len = {}
    for lo, hi in [(0, 20), (20, 40), (40, 100), (100, 1e9)]:
        m = (yt == 1) & (lengths >= lo) & (lengths < hi)
        if m.sum():
            by_len[f"{lo}-{hi}" if hi < 1e9 else f">{lo}"] = dict(n=int(m.sum()), kept=float((p[m] >= thr).mean()))
    out["ships_kept_by_length"] = by_len
    json.dump(out, open("results/triage.json", "w"), indent=1)
    pickle.dump(model, open("results/triage_model.pkl", "wb"))   # trained on 12-14 Oct only
    print(json.dumps({k: v for k, v in out.items() if k != "ranked_alerts"}, indent=1))


if __name__ == "__main__":
    main()
