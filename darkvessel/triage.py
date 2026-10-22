"""Rank dark-vessel alerts by how much they look like a ship.

There are no labels saying which unexplained radar targets are real dark
vessels. There are labels for the opposite question: every detection that
pairs with an AIS report is a ship. A classifier trained to separate
AIS-paired detections from unexplained alerts learns P(paired | x), which
under positive-unlabelled learning (Elkan & Noto, 2008) ranks alerts in the
same order as P(ship | x). Coastline edges, land texture and wakes score low;
compact point targets on homogeneous sea score high.
"""
import numpy as np

FEATURES = ["log_peak", "log_length", "log_area", "ring_cv", "edge", "core_share",
            "ring_spiky", "log_coast_km", "seen_other_days"]


def chip_features(chip):
    """Texture and shape of a 48 x 48 DN chip centred on a detection."""
    I = chip.astype(np.float64) ** 2 + 1.0
    n = I.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    ring = np.maximum(abs(yy - c), abs(xx - c)) > 7
    core = ~ring
    r = I[ring]
    med = np.median(r)
    halves = [I[:c][ring[:c]].mean(), I[c:][ring[c:]].mean(),
              I[:, :c][ring[:, :c]].mean(), I[:, c:][ring[:, c:]].mean()]
    top = np.sort(I[core])[::-1]
    return dict(
        ring_cv=float(r.std() / r.mean()),                       # ~0.5 on open sea at 4.4 looks
        edge=float(max(abs(np.log(halves[0] / halves[1])), abs(np.log(halves[2] / halves[3])))),
        core_share=float(top[:5].sum() / top.sum()),             # point target -> high
        ring_spiky=float((r > 6 * med).mean()),                  # other bright things nearby
    )


def feature_row(det, chip, coast_km, seen_other_days):
    f = chip_features(chip)
    f.update(log_peak=np.log10(max(det["peak"], 1.0)), log_length=np.log10(det["length_m"]),
             log_area=np.log10(det["area_px"]), log_coast_km=np.log10(max(coast_km, 0.1)),
             seen_other_days=float(seen_other_days))
    return [f[k] for k in FEATURES]
